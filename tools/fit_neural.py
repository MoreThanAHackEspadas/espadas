"""Fit the neural evidence model (espadas/neural.py) on external games only. Needs PyTorch.

    python tools/fit_neural.py prep                   # build and cache the training views
    python tools/fit_neural.py train --games 1 3 12   # one network on these games (cached)
    python tools/fit_neural.py export                 # fit on the pool, write neural_weights.py

Views of a game (what the network is trained and scored on):
- standard: the four tracker levels with the ball hidden on degrade.LIVE_MISSING of live frames,
  exactly what tools/tune.py and tools/logo.py score on;
- extra (training only): the four levels again with other random seeds and 15% / 35% hidden, so
  the network does not see one fixed pattern of noise per match;
- robust (scoring only): the four levels at 15% and 35% hidden, as logo.ROBUSTNESS.

Training: random 82 s windows, each randomly team-swapped and mirrored across the pitch. Three
targets per frame: dead or live, home or away (live frames), and a bump at every change of the
reference label (training only: it rewards getting the moment of a change right).

Everything is cached under ESPADAS_NEURAL_CACHE (default ~/.cache/espadas_neural), outside the
repo. Our match is never read here.
"""

import argparse
import hashlib
import json
import multiprocessing
import os
import pickle
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from espadas import config, evidence, neural, v4  # noqa: E402
from espadas.external import bench, degrade  # noqa: E402

CACHE = os.environ.get("ESPADAS_NEURAL_CACHE") or os.path.expanduser("~/.cache/espadas_neural")
OUT = os.path.join(ROOT, "espadas", "neural_weights.py")
MODELS_OUT = os.path.join(ROOT, "espadas", "evidence_models.py")

LEVELS = degrade.TRACKER_LEVELS
STANDARD = [("std_" + level, level, degrade.LIVE_MISSING, 0) for level in LEVELS]
EXTRA = [("extra_" + level, level, hide, 100 + i)
         for i, (level, hide) in enumerate(zip(LEVELS, (0.15, 0.35, 0.35, 0.15)))]
ROBUST = [("rob{:02.0f}_{}".format(100 * hide, level), level, hide, 0)
          for hide in (0.15, 0.35) for level in LEVELS]
VIEWS = {tag: (level, hide, seed) for tag, level, hide, seed in STANDARD + EXTRA + ROBUST}

CFG = {"player_hidden": 48, "channels": 64, "dilations": [1, 2, 4, 8, 16, 32, 64, 128],
       "dropout": 0.25, "steps": 2000, "batch": 24, "window": 2048, "edge": 256,
       "lr": 2e-3, "weight_decay": 1e-2, "change_weight": 0.5, "change_sigma": 5.0, "seed": 0}


# --------------------------------------------------------------------------- views

def make_view(game, level, hide, seed):
    """bench.setup() with the random seeds shifted by `seed` (0 = exactly bench.setup)."""
    if seed == 0:
        gt_codes, tracking, graded = bench.setup(game, hide, level)
        return bench.reference(game)[0], gt_codes, tracking, graded
    gt, half_blocks, kickoffs = bench.reference(game)
    tracking = dict(game["tracking"], half_blocks=half_blocks, kickoffs=kickoffs,
                    fps=game["fps"], pitch=game["pitch"])
    graded = config.half_of_frame(game["n_frames"], half_blocks) > 0
    if degrade.BLUR[level] is not None:
        tracking["ball"] = degrade.blur_ball(tracking["ball"], *degrade.BLUR[level],
                                             seed=degrade.BLUR_SEED + seed)
    live = (gt == "H") | (gt == "A")
    tracking["ball"], _ = degrade.hide_ball(tracking["ball"], tracking["players"], live, hide,
                                            seed=degrade.SEED + seed)
    if degrade.PLAYER_ERROR.get(level, 0.0) > 0:
        tracking["players"] = degrade.jitter_players(tracking["players"],
                                                     degrade.PLAYER_ERROR[level],
                                                     seed=degrade.PLAYER_SEED + seed)
    return gt, config.pm.label_codes(gt), tracking, graded


def view_path(number, tag):
    return os.path.join(CACHE, "views", "g{}_{}.npz".format(number, tag))


def _prep_one(job):
    number, tag, spent = job
    path = view_path(number, tag)
    if os.path.exists(path):
        return path
    level, hide, seed = VIEWS[tag]
    (_, game), = bench.available([number], heldout=spent)
    gt, gt_codes, tracking, graded = make_view(game, level, hide, seed)
    c = evidence.clues(tracking)
    inp = neural.inputs(tracking, c)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    extra = {}
    if not tag.startswith("extra"):      # scored views also need the decoder's other inputs
        f = v4.features(tracking)
        extra = {"features": np.frombuffer(pickle.dumps(f), dtype=np.uint8)}
    np.savez(path + ".tmp.npz", players=inp["players"], mask=inp["mask"], globals=inp["globals"],
             names=np.array(inp["names"]), half_blocks=np.array(inp["half_blocks"]), gt=gt,
             gt_codes=gt_codes, graded=graded, name=np.array(game["name"]), **extra)
    os.replace(path + ".tmp.npz", path)
    return path


def prep(numbers, tags, workers=5, spent=False):
    jobs = [(n, tag, spent) for n in numbers for tag in tags]
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        for path in pool.imap_unordered(_prep_one, jobs):
            print("  ready", os.path.basename(path), flush=True)


def load_view(number, tag, features=False):
    z = np.load(view_path(number, tag), allow_pickle=False)
    view = {"players": z["players"], "mask": z["mask"], "globals": z["globals"],
            "names": list(neural.OWN_GLOBALS)
            + [str(n) for n in z["names"][len(neural.OWN_GLOBALS):]],
            "half_blocks": [tuple(int(v) for v in b) for b in z["half_blocks"]],
            "gt": z["gt"], "gt_codes": z["gt_codes"], "graded": z["graded"],
            "name": str(z["name"]), "number": number, "tag": tag}
    if features:
        view["features"] = pickle.loads(z["features"].tobytes())
    return view


# --------------------------------------------------------------------------- the network

def build(n_globals, cfg):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            hp, c = cfg["player_hidden"], cfg["channels"]
            self.p1 = nn.Linear(len(neural.PLAYER_FEATURES), hp)
            self.p2 = nn.Linear(hp, hp)
            self.frame = nn.Linear(2 * (2 + neural.TOP) * hp + n_globals, c)
            self.norms = nn.ModuleList(nn.LayerNorm(c) for _ in cfg["dilations"])
            self.convs = nn.ModuleList(nn.Conv1d(c, c, 3, dilation=d, padding=d)
                                       for d in cfg["dilations"])
            self.mix = nn.ModuleList(nn.Linear(c, c) for _ in cfg["dilations"])
            self.out_norm = nn.LayerNorm(c)
            self.out = nn.Linear(c, 3)
            self.drop = nn.Dropout(cfg["dropout"])

        def forward(self, p, mask, g):
            """p (B, T, 2, K, F), mask (B, T, 2, K) bool, g (B, T, G) scaled -> (B, T, 3)."""
            m = mask.unsqueeze(-1).to(p.dtype)
            h = torch.relu(self.p2(torch.relu(self.p1(p)))) * m
            count = m.sum(dim=3).clamp(min=1)
            team = torch.cat([h.max(dim=3).values, h.sum(dim=3) / count,
                              h[:, :, :, :neural.TOP].flatten(3)], dim=-1)
            x = torch.relu(self.frame(torch.cat([team.flatten(2), g], dim=-1)))
            x = self.drop(x)
            for norm, conv, mix in zip(self.norms, self.convs, self.mix):
                y = torch.relu(conv(norm(x).transpose(1, 2)).transpose(1, 2))
                x = x + mix(self.drop(y))
            return self.out(self.out_norm(x))

    return Net()


def change_target(gt, graded, sigma):
    """A bump (peak 1) at every frame where the reference label changes inside a half."""
    change = np.zeros(len(gt))
    change[1:] = (gt[1:] != gt[:-1]) & graded[1:] & graded[:-1]
    radius = int(4 * sigma)
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    return np.minimum(np.convolve(change, kernel, "same"), 1.0).astype(np.float32)


def global_scale(views, names):
    """1 / RMS of every frame-level input over the training views; columns that trade places
    when the teams swap share one scale, so scaling commutes with the swap."""
    total, count = np.zeros(len(names)), 0
    for v in views:
        ok = v["graded"]
        total += np.square(v["globals"][ok].astype(np.float64)).sum(axis=0)
        count += int(ok.sum())
    ms = total / max(count, 1)
    perm, _ = neural.global_swap(names)
    ms = (ms + ms[perm]) / 2.0
    return (1.0 / np.sqrt(ms + 1e-6)).astype(np.float32)


def train(views, cfg=CFG, log=print):
    """Weights dict (NumPy) of a network trained on these views."""
    import torch
    import torch.nn.functional as F
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    names = views[0]["names"]
    assert all(v["names"] == names for v in views)
    gscale = global_scale(views, names)
    perm, sign = neural.global_swap(names)
    flip = neural.global_flip(names)

    data = []
    for v in views:
        gt, graded = v["gt"], v["graded"]
        ok = graded & np.isin(gt, ["H", "A", "D"])
        data.append({
            "p": torch.from_numpy(v["players"]), "m": torch.from_numpy(v["mask"]),
            "g": torch.from_numpy(v["globals"] * gscale),
            "y": torch.from_numpy(np.stack([gt == "D", gt == "H",
                                            change_target(gt, graded, cfg["change_sigma"])],
                                           axis=1).astype(np.float32)),
            "w": torch.from_numpy(np.stack([ok, ok & (gt != "D"), ok], axis=1)
                                  .astype(np.float32)),
            "blocks": v["half_blocks"]})

    net = build(len(names), cfg).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps, warm = cfg["steps"], 100
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * s / steps)))
    t_perm = torch.as_tensor(perm, device=device)
    t_sign = torch.as_tensor(sign, device=device)
    t_flip = torch.as_tensor(flip, device=device)
    p_swap = torch.as_tensor(neural.PLAYER_SWAP_SIGN, device=device)
    p_flip = torch.as_tensor(neural.PLAYER_FLIP_SIGN, device=device)
    loss_w = torch.tensor([1.0, 1.0, cfg["change_weight"]], device=device)
    window, edge, batch = cfg["window"], cfg["edge"], cfg["batch"]

    net.train()
    t0, running = time.time(), []
    for step in range(steps):
        parts = {k: [] for k in "pmgyw"}
        for _ in range(batch):
            d = data[rng.integers(len(data))]
            _, first, last = d["blocks"][rng.integers(len(d["blocks"]))]
            stop = last + 1
            length = min(window, stop - first)
            start = int(np.clip(rng.integers(first - length // 2, stop - length // 2),
                                first, stop - length))
            w = d["w"][start:start + length].clone()
            if start > first:
                w[:edge] = 0
            if start + length < stop:
                w[-edge:] = 0
            for k, x in (("p", d["p"]), ("m", d["m"]), ("g", d["g"]), ("y", d["y"])):
                parts[k].append(x[start:start + length])
            parts["w"].append(w)
        p, m, g, y, w = (torch.stack(parts[k]).to(device, non_blocking=True) for k in "pmgyw")
        swap = torch.rand(batch, device=device) < 0.5
        mirror = torch.rand(batch, device=device) < 0.5
        s4, s3 = swap.view(-1, 1, 1, 1, 1), swap.view(-1, 1, 1)
        p = torch.where(s4, p.flip(2) * p_swap, p)
        m = torch.where(s4.squeeze(-1), m.flip(2), m)
        g = torch.where(s3, g[..., t_perm] * t_sign, g)
        y = torch.where(s3, torch.stack([y[..., 0], 1 - y[..., 1], y[..., 2]], dim=-1), y)
        p = torch.where(mirror.view(-1, 1, 1, 1, 1), p * p_flip, p)
        g = torch.where(mirror.view(-1, 1, 1), g * t_flip, g)

        out = net(p, m, g)
        each = F.binary_cross_entropy_with_logits(out, y, reduction="none") * w
        loss = ((each.sum(dim=(0, 1)) / w.sum(dim=(0, 1)).clamp(min=1)) * loss_w).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        running.append(loss.item())
        if (step + 1) % 250 == 0 or step == 0:
            log("    step {:>5}  loss {:.4f}  ({:.0f} s)".format(
                step + 1, float(np.mean(running[-250:])), time.time() - t0))

    net.eval()
    w = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in net.state_dict().items()}
    w.update(names=list(names), dilations=list(cfg["dilations"]), gscale=gscale)
    check_numpy(net.cpu(), w, views[0])
    return w


def check_numpy(net, w, view, frames=3000):
    """The NumPy forward pass must give what PyTorch gives."""
    import torch
    _, first, _ = view["half_blocks"][0]
    s = slice(first + 1000, first + 1000 + frames)
    p, m, g = view["players"][s], view["mask"][s], view["globals"][s] * w["gscale"]
    with torch.no_grad():
        want = net(torch.from_numpy(p)[None], torch.from_numpy(m)[None],
                   torch.from_numpy(g)[None])[0].numpy()
    got = neural.forward(neural._double(w), p, m, g)
    worst = float(np.abs(want - got).max())
    if worst > 1e-3:
        raise RuntimeError("NumPy and PyTorch disagree by {:.5f}".format(worst))


# --------------------------------------------------------------------------- cached fits

def fit_key(numbers, cfg=CFG):
    text = json.dumps([sorted(numbers), cfg, list(neural.PLAYER_FEATURES),
                       list(neural.OWN_GLOBALS), neural.K_PLAYERS, neural.TOP], sort_keys=True)
    return "net_{}_{}".format("-".join(map(str, sorted(numbers))),
                              hashlib.sha256(text.encode()).hexdigest()[:10])


def fitted(numbers, cfg=CFG, log=print):
    """Weights of the network trained on these games (standard + extra views); cached."""
    path = os.path.join(CACHE, "nets", fit_key(numbers, cfg) + ".pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    log("  training on games {}".format(" ".join(map(str, sorted(numbers)))))
    views = [load_view(n, tag) for n in sorted(numbers) for tag, *_ in STANDARD + EXTRA]
    w = train(views, cfg, log)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "wb") as f:
        pickle.dump(w, f)
    os.replace(path + ".tmp", path)
    return w


def dead_design(view):
    """(motion names, motion x, ball names, ball x, ball seen) of evidence.motion_design and
    ball_design, read back from the cached frame-level inputs of a view."""
    names = view["names"]
    cols = [i for i, n in enumerate(names) if n.startswith("dead:")]
    ball_names = [names[i][5:] for i in cols]
    x = view["globals"][:, cols].astype(float)
    k = sum(1 for n in ball_names if not n.startswith(("ball_speed", "out")))
    seen = view["globals"][:, names.index("g_present")] > 0.5
    return ball_names[:k], x[:, :k], ball_names, x, seen


def dead_models(numbers):
    """The two boosted dead models (evidence.py, DEAD_SPLIT) fitted on these games' standard
    views, as tools/fit_evidence.py does; cached."""
    path = os.path.join(CACHE, "nets", "dead_{}.pkl".format("-".join(map(str, sorted(numbers)))))
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    import fit_evidence
    xm, ym, xb, yb = [], [], [], []
    for n in sorted(numbers):
        for tag, *_ in STANDARD:
            v = load_view(n, tag)
            m_names, m, b_names, b, seen = dead_design(v)
            ok = v["graded"] & np.isin(v["gt"], ["H", "A", "D"])
            xm.append(m[ok]), ym.append(v["gt"][ok] == "D")
            xb.append(b[ok & seen]), yb.append(v["gt"][ok & seen] == "D")
    model = {"motion": fit_evidence.fit_boosted(m_names, np.vstack(xm), np.concatenate(ym), False),
             "ball": fit_evidence.fit_boosted(b_names, np.vstack(xb), np.concatenate(yb), False)}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "wb") as f:
        pickle.dump(model, f)
    os.replace(path + ".tmp", path)
    return model


def dead_log_odds(model, view):
    m_names, m, b_names, b, seen = dead_design(view)
    evidence._check(m_names, model["motion"]), evidence._check(b_names, model["ball"])
    dead = evidence.apply_model(m, model["motion"])
    dead[seen] = evidence.apply_model(b[seen], model["ball"])
    return dead


def frame_report(w, view):
    """Per-frame quality of one network on one view: dict of log-loss and accuracy."""
    dead, team, _ = neural.apply(w, view)
    gt = view["gt"]
    ok = view["graded"] & np.isin(gt, ["H", "A", "D"])
    live = ok & (gt != "D")

    def loss(lo, y):
        return float(np.mean(np.logaddexp(0, np.where(y, -lo, lo))))
    return {"dead_loss": loss(dead[ok], gt[ok] == "D"),
            "dead_acc": float(np.mean((dead[ok] > 0) == (gt[ok] == "D"))),
            "team_loss": loss(team[live], gt[live] == "H"),
            "team_acc": float(np.mean((team[live] > 0) == (gt[live] == "H")))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("prep", "train", "export"))
    parser.add_argument("--games", nargs="+", type=int, default=list(bench.DEV_TUNING))
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args(argv)
    config.seed_everything()
    if args.command == "prep":
        prep(args.games, list(VIEWS), args.workers)
        return 0
    w = fitted(args.games)
    if args.command == "export":
        import fit_evidence
        games = fit_evidence.rows(args.games)
        dead_model = {
            "motion": fit_evidence.fit(games, evidence.motion_design, "all", True, "boosted")[0],
            "ball": fit_evidence.fit(games, evidence.ball_design, "seen", True, "boosted")[0]}
        which = " ".join(map(str, sorted(args.games)))
        with open(OUT, "w") as f:
            f.write('"""Generated by tools/fit_neural.py; do not edit by hand.\n\n'
                    "Neural team evidence model fitted on external games {} only.\n"
                    '"""\n\nPACKED = {!r}\n'.format(which, neural.pack(w)))
        with open(MODELS_OUT, "w") as f:
            f.write('"""Generated by tools/fit_neural.py; do not edit by hand.\n\n'
                    "Fitted on external games {} only. Dead: boosted trees, as "
                    "tools/fit_evidence.py --model boosted. Team: the network of "
                    'espadas/neural.py (weights in neural_weights.py).\n"""\n\n'
                    'DEAD_MODEL = {!r}\n\nTEAM_MODEL = {{"design": "neural"}}\n'
                    .format(which, dead_model))
        print("wrote", os.path.relpath(OUT, ROOT), "and", os.path.relpath(MODELS_OUT, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
