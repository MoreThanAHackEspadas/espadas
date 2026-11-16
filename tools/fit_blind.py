"""Fit the ball-blind dead network (espadas/blind.py) on external games only. Needs PyTorch.

    python tools/fit_blind.py train --games 1 3 12   # one network on these games (cached)

Same views, windows and augmentation as tools/fit_neural.py (it reuses its cached views), but two
targets per frame only: dead or live, and "dead / live changes here" (training only). The inputs
are the players' velocity and pitch position and the player-motion clues; never the ball.
Writing blind_weights.py is blind.pack() on each net: PACKED = {"nets": [blind.pack(w), ...]}.
"""

import argparse
import hashlib
import json
import os
import pickle
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import fit_neural  # noqa: E402
from espadas import blind, config, neural  # noqa: E402
from espadas.external import bench  # noqa: E402

# indices into neural.PLAYER_FEATURES of the features that do not depend on the ball
BLIND_PLAYER = [neural.PLAYER_FEATURES.index(n) for n in blind.FEATURES]
CFG = dict(fit_neural.CFG)


def motion_cols(names):
    """Columns of the cached frame-level inputs that are evidence.motion_design (no ball)."""
    return [i for i, n in enumerate(names)
            if n.startswith("dead:") and not n[5:].startswith(("ball_speed", "out"))]


def build(n_globals, cfg):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            hp, c = cfg["player_hidden"], cfg["channels"]
            self.p1 = nn.Linear(len(BLIND_PLAYER) + 1, hp)      # + speed
            self.p2 = nn.Linear(hp, hp)
            self.frame = nn.Linear(2 * 2 * hp + n_globals, c)
            self.norms = nn.ModuleList(nn.LayerNorm(c) for _ in cfg["dilations"])
            self.convs = nn.ModuleList(nn.Conv1d(c, c, 3, dilation=d, padding=d)
                                       for d in cfg["dilations"])
            self.mix = nn.ModuleList(nn.Linear(c, c) for _ in cfg["dilations"])
            self.out_norm = nn.LayerNorm(c)
            self.out = nn.Linear(c, 2)                           # dead, dead<->live change
            self.drop = nn.Dropout(cfg["dropout"])

        def forward(self, p, mask, g):
            """p (B, T, 2, K, 4), mask (B, T, 2, K), g (B, T, G) scaled -> (B, T, 2)."""
            speed = torch.sqrt(p[..., 0] ** 2 + p[..., 1] ** 2 + 1e-6).unsqueeze(-1)
            m = mask.unsqueeze(-1).to(p.dtype)
            h = torch.relu(self.p2(torch.relu(self.p1(torch.cat([p, speed], dim=-1))))) * m
            count = m.sum(dim=3).clamp(min=1)
            team = torch.cat([h.max(dim=3).values, h.sum(dim=3) / count], dim=-1)
            x = torch.relu(self.frame(torch.cat([team.flatten(2), g], dim=-1)))
            x = self.drop(x)
            for norm, conv, mix in zip(self.norms, self.convs, self.mix):
                y = torch.relu(conv(norm(x).transpose(1, 2)).transpose(1, 2))
                x = x + mix(self.drop(y))
            return self.out(self.out_norm(x))

    return Net()


def dead_change(gt, graded, sigma):
    """A bump (peak 1) at every frame where the reference changes between dead and live."""
    dead = gt == "D"
    change = np.zeros(len(gt))
    change[1:] = (dead[1:] != dead[:-1]) & graded[1:] & graded[:-1]
    radius = int(4 * sigma)
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    return np.minimum(np.convolve(change, kernel, "same"), 1.0).astype(np.float32)


def train(views, cfg=CFG, log=print):
    """Weights dict (NumPy; names of all frame-level inputs, cols = the motion ones) of a network
    trained on these views (fit_neural.load_view)."""
    import torch
    import torch.nn.functional as F
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    names = views[0]["names"]
    cols = motion_cols(names)
    gscale = fit_neural.global_scale(views, names)[cols]
    data = []
    for v in views:
        gt, graded = v["gt"], v["graded"]
        ok = graded & np.isin(gt, ["H", "A", "D"])
        data.append({"p": v["players"], "m": v["mask"], "g": v["globals"],
                     "y": np.stack([gt == "D", dead_change(gt, graded, cfg["change_sigma"])],
                                   axis=1).astype(np.float32),
                     "w": np.stack([ok, ok], axis=1).astype(np.float32),
                     "blocks": v["half_blocks"]})
    net = build(len(cols), cfg).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                            fused=amp)
    scaler = torch.amp.GradScaler(device, enabled=amp)
    steps, warm = cfg["steps"], 100
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * s / steps)))
    t_scale = torch.as_tensor(gscale, device=device)
    sx = torch.as_tensor(blind.SWAP_SIGN, device=device)    # team swap: the pitch turns round
    sy = torch.as_tensor(blind.FLIP_SIGN, device=device)    # mirror: y flips
    loss_w = torch.tensor([1.0, cfg["change_weight"]], device=device)
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
            w = d["w"][start:start + length].copy()
            if start > first:
                w[:edge] = 0
            if start + length < stop:
                w[-edge:] = 0
            parts["p"].append(np.asarray(d["p"][start:start + length])[..., BLIND_PLAYER])
            parts["m"].append(np.asarray(d["m"][start:start + length]))
            parts["g"].append(np.asarray(d["g"][start:start + length])[:, cols])
            parts["y"].append(d["y"][start:start + length])
            parts["w"].append(w)
        if len({len(x) for x in parts["w"]}) > 1:
            continue
        p = torch.from_numpy(np.stack(parts["p"])).to(device).float()
        m = torch.from_numpy(np.stack(parts["m"])).to(device)
        g = torch.from_numpy(np.stack(parts["g"])).to(device).float() * t_scale
        y = torch.from_numpy(np.stack(parts["y"])).to(device)
        w = torch.from_numpy(np.stack(parts["w"])).to(device)
        swap = (torch.rand(batch, device=device) < 0.5).view(-1, 1, 1, 1, 1)
        mirror = (torch.rand(batch, device=device) < 0.5).view(-1, 1, 1, 1, 1)
        p = torch.where(swap, p.flip(2) * sx, p)
        m = torch.where(swap.squeeze(-1), m.flip(2), m)
        p = torch.where(mirror, p * sy, p)
        with torch.autocast(device, dtype=torch.float16, enabled=amp):
            out = net(p, m, g)
        each = F.binary_cross_entropy_with_logits(out.float(), y, reduction="none") * w
        loss = ((each.sum(dim=(0, 1)) / w.sum(dim=(0, 1)).clamp(min=1)) * loss_w).sum()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        running.append(loss.item())
        if (step + 1) % 500 == 0 or step == 0:
            log("    step {:>5}  loss {:.4f}  ({:.0f} s)".format(
                step + 1, float(np.mean(running[-500:])), time.time() - t0))
    net.eval()
    w = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in net.state_dict().items()}
    w.update(names=list(names), dilations=list(cfg["dilations"]), gscale=gscale, cols=cols)
    return w


def fitted(numbers, cfg=CFG, log=print):
    """Weights of the network trained on these games (standard + extra views); cached."""
    text = json.dumps(["blind", sorted(numbers), cfg, BLIND_PLAYER], sort_keys=True)
    path = os.path.join(fit_neural.CACHE, "nets", "blind_{}_{}.pkl".format(
        "-".join(map(str, sorted(numbers))), hashlib.sha256(text.encode()).hexdigest()[:10]))
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    log("  blind net on games {}".format(" ".join(map(str, sorted(numbers)))))
    views = [fit_neural.load_view(n, tag) for n in sorted(numbers)
             for tag, *_ in fit_neural.STANDARD + fit_neural.EXTRA]
    w = train(views, cfg, log)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "wb") as f:
        pickle.dump(w, f)
    os.replace(path + ".tmp", path)
    return w


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("train",))
    parser.add_argument("--games", nargs="+", type=int, default=list(bench.DEV_TUNING))
    args = parser.parse_args(argv)
    config.seed_everything()
    fitted(args.games)
    return 0


if __name__ == "__main__":
    sys.exit(main())
