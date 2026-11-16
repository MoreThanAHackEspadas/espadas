"""Ball-blind dead evidence: dead or live from the players alone, with NumPy only.

A network that never reads the ball (its position, whether it is tracked, how long it has been
missing) nor anything measured from it, so it cannot learn "no ball = stoppage" (evidence.py,
DEAD_SPLIT). Per frame it reads every tracked player's velocity and pitch position (a player
encoder, pooled per team by max and mean) and the player-motion clues of evidence.motion_design.
The same stack of dilated temporal convolutions as neural.py follows (~10 s each side).

Players are given in track-slot order, not nearest-the-ball first: the pooling does not care
about order, and this keeps the model strictly independent of the ball.

The network is evaluated on the frames as they are, team-swapped, pitch-mirrored and both, and
the four answers are averaged: the output is the same for a match and its swapped / mirrored
copies. Training needs PyTorch (tools/fit_blind.py); this module does not: the fitted weights
are plain numbers (espadas/blind_weights.py).
"""

import numpy as np

from . import config, evidence, neural, prep

K = neural.K_PLAYERS
FEATURES = ("s_vx", "y_vy", "s_px", "y_py")
SWAP_SIGN = np.array([-1.0, 1.0, -1.0, 1.0], dtype=np.float32)    # the pitch turns round
FLIP_SIGN = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32)    # mirror across the pitch


def inputs(tracking, c=None):
    """{"players": (n, 2, K, 4) float32, "mask": (n, 2, K) bool, "globals": (n, g) float32,
    "names": [g names], "half_blocks": ...} for one match; team axis home first."""
    c = evidence.clues(tracking) if c is None else c
    half_blocks = tracking.get("half_blocks", config.HALF_BLOCKS)
    kickoffs = tracking.get("kickoffs", config.KICKOFFS)
    fps = tracking.get("fps", config.FPS)
    length, width = tracking.get("pitch", config.PITCH)
    players, team_of_slot = tracking["players"], np.asarray(tracking["team_of_slot"])
    attack = prep.attacking_direction(players, team_of_slot, half_blocks, kickoffs)
    sx = np.where(attack[:, evidence.HOME] == 0, 1, attack[:, evidence.HOME]).astype(float)
    velocity = evidence._velocity(players, fps)

    n = len(players)
    p = np.zeros((n, 2, K, len(FEATURES)), dtype=np.float32)
    mask = np.zeros((n, 2, K), dtype=bool)
    for team in (evidence.HOME, evidence.AWAY):
        pos, vel = players[:, team_of_slot == team], velocity[:, team_of_slot == team]
        k = min(K, pos.shape[1])
        if k == 0:
            continue
        tracked = ~np.isnan(pos).any(axis=2)
        order = np.argsort(~tracked, axis=1, kind="stable")[:, :k]      # tracked, in slot order

        def take(a):
            return np.take_along_axis(a, order, axis=1)
        ok = take(tracked)
        vx, vy = np.nan_to_num(take(vel[..., 0])) * sx[:, None], np.nan_to_num(take(vel[..., 1]))
        px = (take(pos[..., 0]) - length / 2.0) * sx[:, None]
        py = take(pos[..., 1]) - width / 2.0
        f = np.stack([np.clip(vx / 5.0, -3, 3), np.clip(vy / 5.0, -3, 3),
                      np.clip(px / 50.0, -2, 2), np.clip(py / 35.0, -2, 2)], axis=-1)
        p[:, team, :k] = np.where(ok[..., None], np.nan_to_num(f), 0.0)
        mask[:, team, :k] = ok
    names, g = evidence.motion_design(c)
    return {"players": p, "mask": mask, "globals": g.astype(np.float32), "names": names,
            "half_blocks": half_blocks}


def forward(w, p, mask, g):
    """(n, 2) raw outputs (dead, change) for one run of frames; g already scaled."""
    n = len(p)
    hidden = w["p2.bias"].shape[0]
    enc = np.empty((n, 4 * hidden), dtype=np.float32)
    for a in range(0, n, neural.CHUNK):
        b = min(n, a + neural.CHUNK)
        q, m = p[a:b], mask[a:b][..., None]
        speed = np.sqrt(q[..., 0] ** 2 + q[..., 1] ** 2 + 1e-6)[..., None]
        h = neural._relu(np.concatenate([q, speed], axis=-1) @ w["p1.weight"].T + w["p1.bias"])
        h = neural._relu(h @ w["p2.weight"].T + w["p2.bias"]) * m
        count = np.maximum(m.sum(axis=2), 1).astype(np.float32)
        enc[a:b] = np.concatenate([h.max(axis=2), h.sum(axis=2) / count], axis=-1).reshape(b - a, -1)
    return neural.trunk(w, np.concatenate([enc, g], axis=1))


def apply(ws, inp):
    """Dead log-odds per frame (0 outside the halves): the mean over the nets and over the
    frames plain / team-swapped x plain / pitch-mirrored. Rounded once, at the end."""
    ws = [ws] if isinstance(ws, dict) else list(ws)
    names = list(inp["names"])
    out = np.zeros(len(inp["globals"]))
    for w in ws:
        if names != list(w["names"]):
            raise RuntimeError("blind model was fitted on different inputs; re-run "
                               "tools/fit_blind.py")
        for _, first, last in inp["half_blocks"]:
            s = slice(first, last + 1)
            g = inp["globals"][s] * w["gscale"]
            for swap in (False, True):
                p, m = inp["players"][s], inp["mask"][s]
                if swap:
                    p, m = p[:, ::-1] * SWAP_SIGN, m[:, ::-1]
                for mirror in (False, True):
                    out[s] += forward(w, p * FLIP_SIGN if mirror else p, m, g)[:, 0]
    return np.round(out / (4 * len(ws)), neural.ROUND)


def pack(w):
    """Weights dict as tools/fit_blind.train returns it ("names" of all frame-level inputs, "cols"
    of the motion ones, "gscale" for those) -> packed net for blind_weights.PACKED["nets"]."""
    names = [w["names"][i][len("dead:"):] for i in w["cols"]]
    return neural.pack({k: v for k, v in w.items() if k != "cols"} | {"names": names})


_CACHE = {}


def weights(model=None):
    """List of weight dicts: from `model["blind_w"]` (one dict or a list; tools and tests), else
    blind_weights.PACKED = {"nets": [packed, ...]}."""
    if model is not None and "blind_w" in model:
        w = model["blind_w"]
        return list(w) if isinstance(w, (list, tuple)) else [w]
    if "shipped" not in _CACHE:
        from . import blind_weights
        _CACHE["shipped"] = [neural.unpack(p) for p in blind_weights.PACKED["nets"]]
    return _CACHE["shipped"]
