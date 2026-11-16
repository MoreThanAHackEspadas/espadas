"""Neural team evidence: tracking -> home log-odds per frame, with NumPy only.

One network replaces the tree model for home-or-away of evidence.py. Per frame it reads

- every tracked player (up to K_PLAYERS per team, nearest to the last-seen ball first): where
  they are relative to the ball, how they move, where they stand on the pitch. One small network
  is applied to each player and the results are pooled per team (player encoder);
- frame-level inputs: the ball (speed, direction, gap lengths, kinks) and every clue the tree
  models read (evidence.ball_design, evidence.player_design).

A stack of dilated temporal convolutions then looks about 10 s each side and gives three numbers
per frame: home log-odds (the one the labeler uses), dead log-odds and "possession changes
here". The last two are training targets only: the change target rewards the network for the
timing of a change, and the dead evidence stays with the tree models (see DEAD below).

The network is evaluated twice, on the frames as they are and with the two teams swapped, and
the two answers are combined so that swapping the teams exactly flips the home log-odds and
leaves the dead log-odds unchanged: no home bias by construction.

Training needs PyTorch (tools/fit_neural.py, external games only). This module does not: the
fitted weights are plain numbers (espadas/neural_weights.py) and forward() is NumPy.
"""

import base64

import numpy as np

from . import config, evidence, prep, touches

HOME, AWAY = evidence.HOME, evidence.AWAY
K_PLAYERS = 11             # players per team given to the network, nearest first
TOP = 3                    # the nearest TOP players per team are also passed on in order
PLAYER_FEATURES = ("s_dx", "y_dy", "dist", "close", "s_vx", "y_vy", "closing", "s_px", "y_py")
OWN_GLOBALS = ("g_present", "g_gap_since", "g_gap_until", "g_ball_speed", "s_g_ball_vx",
               "y_g_ball_vy", "s_g_ref_x", "y_g_ref_y", "g_out", "g_kink2", "g_kink6")
GAP_CLIP = 250             # frames
ROUND = 3                  # decimals kept of the log-odds (steadier across machines)
CHUNK = 16384              # frames per block in the player encoder (memory only)


def _sign(names, prefix):
    return np.array([-1.0 if n.startswith(prefix) else 1.0 for n in names], dtype=np.float32)


PLAYER_SWAP_SIGN = _sign(PLAYER_FEATURES, "s_")     # team swap: the pitch turns round
PLAYER_FLIP_SIGN = _sign(PLAYER_FEATURES, "y_")     # mirror across the pitch


def global_names(c):
    names = list(OWN_GLOBALS)
    names += ["dead:" + n for n in evidence.ball_design(c)[0]]
    names += evidence.player_design(c)[0]
    if len(set(names)) != len(names):
        raise RuntimeError("two frame-level inputs share a name")
    return names


def global_swap(names):
    """(perm, sign) so that g[:, perm] * sign is the same frames with the teams swapped."""
    perm, sign = evidence.player_swap(names)
    return np.asarray(perm), np.asarray(sign, dtype=np.float32)


def global_flip(names):
    return _sign(names, "y_")


def inputs(tracking, c=None):
    """{"players": (n, 2, K_PLAYERS, features) float32, "mask": (n, 2, K_PLAYERS) bool,
    "globals": (n, g) float32, "names": [g names], "half_blocks": ...} for one match."""
    c = evidence.clues(tracking) if c is None else c
    half_blocks = tracking.get("half_blocks", config.HALF_BLOCKS)
    kickoffs = tracking.get("kickoffs", config.KICKOFFS)
    fps = tracking.get("fps", config.FPS)
    length, width = tracking.get("pitch", config.PITCH)
    players, team_of_slot = tracking["players"], np.asarray(tracking["team_of_slot"])
    base = prep.prepare(tracking, half_blocks=half_blocks, kickoffs=kickoffs, fps=fps)
    ball, present = base["ball"], base["ball_present"]
    n = len(ball)

    # x always runs towards the goal home attacks, so "home" means the same in both halves
    sx = np.where(base["attack"][:, HOME] == 0, 1, base["attack"][:, HOME]).astype(float)
    last = evidence._last_seen(ball, present)
    ref = np.where(np.isnan(last), np.array([length / 2.0, width / 2.0]), last)
    velocity = evidence._velocity(players, fps)

    p = np.zeros((n, 2, K_PLAYERS, len(PLAYER_FEATURES)), dtype=np.float32)
    mask = np.zeros((n, 2, K_PLAYERS), dtype=bool)
    for team in (HOME, AWAY):
        pos = players[:, team_of_slot == team]
        vel = velocity[:, team_of_slot == team]
        k = min(K_PLAYERS, pos.shape[1])
        if k == 0:
            continue
        diff = pos - ref[:, None, :]
        dist = np.hypot(diff[..., 0], diff[..., 1])
        dist = np.where(np.isnan(dist), np.inf, dist)
        order = np.argsort(dist, axis=1, kind="stable")[:, :k]

        def take(a):
            return np.take_along_axis(a, order, axis=1)
        d = take(dist)
        ok = np.isfinite(d)
        d = np.where(ok, d, 0.0)
        dx, dy = take(diff[..., 0]) * sx[:, None], take(diff[..., 1])
        vx, vy = np.nan_to_num(take(vel[..., 0])) * sx[:, None], np.nan_to_num(take(vel[..., 1]))
        px = (take(pos[..., 0]) - length / 2.0) * sx[:, None]
        py = take(pos[..., 1]) - width / 2.0
        with np.errstate(invalid="ignore"):
            closing = -(dx * vx + dy * vy) / np.maximum(d, 0.5)
            f = np.stack([np.clip(dx / 20.0, -3, 3), np.clip(dy / 20.0, -3, 3),
                          np.clip(d / 20.0, 0, 3), 1.0 / (1.0 + d), np.clip(vx / 5.0, -3, 3),
                          np.clip(vy / 5.0, -3, 3), np.clip(closing / 5.0, -3, 3),
                          np.clip(px / 50.0, -2, 2), np.clip(py / 35.0, -2, 2)], axis=-1)
        p[:, team, :k] = np.where(ok[..., None], np.nan_to_num(f), 0.0)
        mask[:, team, :k] = ok

    ball_v = evidence._velocity(ball, fps)
    with np.errstate(invalid="ignore"):
        own = [present.astype(float),
               np.minimum(base["ball_gap_since"], GAP_CLIP) / 50.0,
               evidence._since(present[::-1], GAP_CLIP)[::-1] / 50.0,
               np.clip(c["ball_speed"], 0, 30) / 10.0,
               np.clip(ball_v[:, 0] * sx, -30, 30) / 10.0,
               np.clip(ball_v[:, 1], -30, 30) / 10.0,
               np.clip((ref[:, 0] - length / 2.0) * sx / 50.0, -2, 2),
               np.clip((ref[:, 1] - width / 2.0) / 35.0, -2, 2),
               np.clip(c["out_dist"], -5, 10) / 5.0,
               np.clip(touches.kink(ball, fps, 2), 0, 30) / 10.0,
               np.clip(touches.kink(ball, fps, 6), 0, 30) / 10.0]
    own = np.stack([np.nan_to_num(np.asarray(x, dtype=float)) for x in own], axis=1)
    g = np.hstack([own, evidence.ball_design(c)[1], evidence.player_design(c)[1]])
    return {"players": p, "mask": mask, "globals": g.astype(np.float32),
            "names": global_names(c), "half_blocks": half_blocks}


# --------------------------------------------------------------------------- forward pass

def _relu(x):
    return np.maximum(x, 0.0, out=x)


def _norm(x, gain, bias, eps=1e-5):
    mean = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gain + bias


def _conv(x, weight, bias, dilation):
    """Three-tap dilated convolution over time, zero padded. weight: (out, in, 3) as PyTorch."""
    out = x @ weight[:, :, 1].T + bias
    d = dilation
    if d < len(x):
        out[d:] += x[:-d] @ weight[:, :, 0].T
        out[:-d] += x[d:] @ weight[:, :, 2].T
    return out


def _encode(w, p, mask):
    """(n, 2 * (2 + TOP) * hidden): per team the max, the mean and the TOP nearest players."""
    n = len(p)
    hidden = w["p2.bias"].shape[0]
    out = np.empty((n, 2 * (2 + TOP) * hidden), dtype=np.float32)
    for a in range(0, n, CHUNK):
        b = min(n, a + CHUNK)
        m = mask[a:b][..., None]
        h = _relu(p[a:b] @ w["p1.weight"].T + w["p1.bias"])
        h = _relu(h @ w["p2.weight"].T + w["p2.bias"]) * m
        count = np.maximum(m.sum(axis=2), 1).astype(np.float32)
        team = np.concatenate([h.max(axis=2), h.sum(axis=2) / count,
                               h[:, :, :TOP].reshape(b - a, 2, TOP * hidden)], axis=-1)
        out[a:b] = team.reshape(b - a, -1)
    return out


def trunk(w, x):
    """What follows the player encoder: frame layer, residual dilated convolutions, output layer.
    x: (n, encoder + frame-level inputs), the frame-level part already scaled."""
    x = _relu(x @ w["frame.weight"].T + w["frame.bias"])
    for i, d in enumerate(dilations(w)):
        y = _norm(x, w["norms.%d.weight" % i], w["norms.%d.bias" % i])
        y = _relu(_conv(y, w["convs.%d.weight" % i], w["convs.%d.bias" % i], int(d)))
        x = x + y @ w["mix.%d.weight" % i].T + w["mix.%d.bias" % i]
    x = _norm(x, w["out_norm.weight"], w["out_norm.bias"])
    return x @ w["out.weight"].T + w["out.bias"]


def forward(w, p, mask, g):
    """(n, 3) raw outputs (dead, home, change) for one unbroken run of frames (a half).
    g must already be scaled (g * w["gscale"])."""
    return trunk(w, np.concatenate([_encode(w, p, mask), g], axis=1))


def dilations(w):
    return [int(d) for d in w["dilations"]]


def swap_inputs(p, mask, g, perm, sign):
    return p[:, ::-1] * PLAYER_SWAP_SIGN, mask[:, ::-1], g[:, perm] * sign


def _sym(w, p, mask, g, perm, sign):
    """(dead, home, change) of one net on one half: the frames as they are and team-swapped."""
    a = forward(w, p, mask, g)
    b = forward(w, *swap_inputs(p, mask, g, perm, sign))
    return (a[:, 0] + b[:, 0]) / 2.0, (a[:, 1] - b[:, 1]) / 2.0, (a[:, 2] + b[:, 2]) / 2.0


def apply(ws, inp, mirror=False):
    """(dead, home, change) log-odds per frame of the match; 0 outside the halves.

    ws: one weights dict or a list of them (an ensemble, averaged). Each net is evaluated on the
    frames as they are and team-swapped (home flips sign exactly, dead and change stay) and, with
    `mirror`, also on the pitch mirrored (y flipped; the teams stay). Rounded once, at the end."""
    ws = [ws] if isinstance(ws, dict) else list(ws)
    names = list(inp["names"])
    perm, sign = global_swap(names)
    flip = global_flip(names)
    out = np.zeros((3, len(inp["globals"])))
    for w in ws:
        if names != list(w["names"]):
            raise RuntimeError("neural model was fitted on different inputs; re-run "
                               "tools/fit_neural.py")
        for _, first, last in inp["half_blocks"]:
            s = slice(first, last + 1)
            p, mask = inp["players"][s], inp["mask"][s]
            g = inp["globals"][s] * w["gscale"]
            for k, x in enumerate(_sym(w, p, mask, g, perm, sign)):
                out[k, s] += x
            if mirror:
                for k, x in enumerate(_sym(w, p * PLAYER_FLIP_SIGN, mask, g * flip, perm, sign)):
                    out[k, s] += x
    out /= len(ws) * (2 if mirror else 1)
    return tuple(np.round(out, ROUND))


def outputs(c, model, tracking):
    """(dead, home, change) log-odds per frame from the network(s) of `model` (see weights)."""
    ws, mirror = weights(model)
    return apply(ws, inputs(tracking, c), mirror)


def team_log_odds(c, model, tracking):
    """Home log-odds per frame. The network's dead output is not used: see DEAD below."""
    return outputs(c, model, tracking)[1]


# DEAD: the network also gives dead log-odds, and on the external games they are very good
# (about 97-99% of frames right on a game it never saw). They are not used. The network can see
# whether the ball is tracked and for how long it has been missing, and in the external games a
# long gap nearly always means a stoppage; the organisers say that does not hold on our match
# (evidence.py, DEAD_SPLIT). The dead evidence therefore stays with the two tree models that
# cannot use that shortcut, and the network supplies the home-or-away evidence only. The dead
# target is kept in training because it helps the shared layers.


# --------------------------------------------------------------------------- weights as text

def pack(w):
    """Weights dict -> plain Python data (lists and base64 strings) for neural_weights.py."""
    out = {"names": list(w["names"]), "dilations": [int(d) for d in w["dilations"]], "arrays": {}}
    for key in sorted(w):
        if key in ("names", "dilations"):
            continue
        a = np.asarray(w[key], dtype="<f4")
        out["arrays"][key] = (list(a.shape), base64.b64encode(a.tobytes()).decode("ascii"))
    return out


def unpack(packed):
    w = {"names": list(packed["names"]), "dilations": list(packed["dilations"])}
    for key, (shape, text) in packed["arrays"].items():
        w[key] = np.frombuffer(base64.b64decode(text), dtype="<f4").reshape(shape)
    return w


def pack_many(ws, mirror=False):
    """An ensemble of nets and the mirror flag, for neural_weights.PACKED."""
    return {"nets": [pack(w) for w in ws], "mirror": bool(mirror)}


_CACHE = {}


def weights(model=None):
    """(list of weight dicts, mirror flag): from `model` if it carries them ("w": one dict or a
    list, optional "mirror"; tools), else neural_weights.PACKED: one packed net (the first
    version) or {"nets": [packed, ...], "mirror": bool}."""
    if model is not None and "w" in model:
        w = model["w"]
        return (list(w) if isinstance(w, (list, tuple)) else [w]), bool(model.get("mirror", False))
    if "shipped" not in _CACHE:
        from . import neural_weights
        packed = neural_weights.PACKED
        if "nets" in packed:
            _CACHE["shipped"] = ([unpack(p) for p in packed["nets"]],
                                 bool(packed.get("mirror", False)))
        else:
            _CACHE["shipped"] = ([unpack(packed)], False)
    return _CACHE["shipped"]
