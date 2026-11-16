"""Ball-blind dead network: dead-or-live from player positions and motion only.

It never reads the ball (position, presence, gaps) nor anything ordered or measured relative to
it, so it cannot learn "no ball = stoppage". Inputs per frame: every tracked player's velocity
and position on the pitch (per-player encoder, pooled per team by max and mean), plus the
player-motion clues of evidence.motion_design. Dilated temporal convolutions, ~10 s each side.

    python blind.py fit --folds g5        # train fold networks, write dead evidence (GPU)
"""
import argparse, hashlib, json, os, pickle, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nexp                                         # noqa: E402
from nexp import neural, fit_neural, config         # noqa: E402

# indices into neural.PLAYER_FEATURES that do not depend on the ball
BLIND_PLAYER = [neural.PLAYER_FEATURES.index(n) for n in ("s_vx", "y_vy", "s_px", "y_py")]
CFG = {"player_hidden": 48, "channels": 64, "dilations": [1, 2, 4, 8, 16, 32, 64, 128],
       "dropout": 0.25, "steps": 2000, "batch": 24, "window": 2048, "edge": 256,
       "lr": 2e-3, "weight_decay": 1e-2, "seed": 0, "change_weight": 0.5, "change_sigma": 5.0}


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
            """p (B, T, 2, K, 4), mask (B, T, 2, K), g (B, T, G) -> (B, T, 2)."""
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
    dead = gt == "D"
    change = np.zeros(len(gt))
    change[1:] = (dead[1:] != dead[:-1]) & graded[1:] & graded[:-1]
    radius = int(4 * sigma)
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    return np.minimum(np.convolve(change, kernel, "same"), 1.0).astype(np.float32)


def train(views, cfg, log=print):
    import torch
    import torch.nn.functional as F
    device = "cuda"
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    names = views[0]["names"]
    cols = motion_cols(names)
    gscale = nexp.global_scale(views, names)[cols]
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
                            fused=True)
    scaler = torch.amp.GradScaler("cuda")
    steps, warm = cfg["steps"], 100
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * s / steps)))
    t_scale = torch.as_tensor(gscale, device=device)
    # team swap: the pitch turns round (x flips); mirror: y flips. s_vx, y_vy, s_px, y_py
    sx = torch.tensor([-1.0, 1.0, -1.0, 1.0], device=device)
    sy = torch.tensor([1.0, -1.0, 1.0, -1.0], device=device)
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
        with torch.autocast("cuda", dtype=torch.float16):
            out = net(p, m, g)
        each = F.binary_cross_entropy_with_logits(out.float(), y, reduction="none") * w
        loss = ((each.sum(dim=(0, 1)) / w.sum(dim=(0, 1)).clamp(min=1)) * loss_w).sum()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        running.append(loss.item())
        if (step + 1) % 500 == 0 or step == 0:
            log("    step {:>5}  loss {:.4f}  ({:.0f} s)".format(
                step + 1, float(np.mean(running[-500:])), time.time() - t0), flush=True)
    net.eval()
    w = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in net.state_dict().items()}
    w.update(names=list(names), dilations=list(cfg["dilations"]), gscale=gscale, cols=cols)
    return w


def fitted(train_games, cfg, tags=nexp.TRAIN_TAGS):
    text = json.dumps(["blind", sorted(train_games), cfg, tags, BLIND_PLAYER], sort_keys=True)
    path = os.path.join(nexp.CACHE, "nets2", "blind_{}_{}.pkl".format(
        "-".join(map(str, sorted(train_games))), hashlib.sha256(text.encode()).hexdigest()[:10]))
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    print("  blind net on", train_games, flush=True)
    w = train([nexp.mm_view(n, t) for n in sorted(train_games) for t in tags], cfg)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "wb") as f:
        pickle.dump(w, f)
    os.replace(path + ".tmp", path)
    return w


def apply_gpu(w, view, cfg):
    """Dead log-odds per frame; the same for a match and its team-swapped / mirrored copies."""
    import torch
    cols = w["cols"]
    net = build(len(cols), cfg).cuda().eval()
    net.load_state_dict({k: torch.from_numpy(np.asarray(v)) for k, v in w.items()
                         if k not in ("names", "dilations", "gscale", "cols")})
    out = np.zeros(len(view["gt"]))
    t_scale = torch.as_tensor(w["gscale"]).cuda()
    sx = torch.tensor([-1.0, 1.0, -1.0, 1.0]).cuda()
    sy = torch.tensor([1.0, -1.0, 1.0, -1.0]).cuda()
    with torch.no_grad():
        for _, first, last in view["half_blocks"]:
            s = slice(first, last + 1)
            p = torch.from_numpy(np.asarray(view["players"][s])[..., BLIND_PLAYER]).cuda().float()[None]
            m = torch.from_numpy(np.asarray(view["mask"][s])).cuda()[None]
            g = torch.from_numpy(np.asarray(view["globals"][s])[:, cols]).cuda().float()[None] * t_scale
            total = 0
            for swap in (False, True):
                for mirror in (False, True):
                    q, mm = (p.flip(2) * sx, m.flip(2)) if swap else (p, m)
                    q = q * sy if mirror else q
                    total = total + net(q, mm, g)[0, :, 0]
            out[s] = (total / 4).double().cpu().numpy()
    return np.round(out, 3)


def ev_path(name, fold, n, tag):
    return os.path.join(nexp.WORK, "evidence", "blind_" + name, "{}_g{}_{}.npz".format(fold, n, tag))


def cmd_fit(name, fold_name, tags, cfg, seeds=(0,)):
    for train_games, test in nexp.FOLDS[fold_name]:
        todo = [(n, t) for n in test for t in tags if not os.path.exists(ev_path(name, fold_name, n, t))]
        if not todo:
            continue
        t0 = time.time()
        nets = [fitted(train_games, dict(cfg, seed=s)) for s in seeds]
        for n, t in todo:
            view = nexp.mm_view(n, t)
            dead = np.mean([apply_gpu(w, view, cfg) for w in nets], axis=0)
            gt = view["gt"]; ok = view["graded"] & np.isin(gt, ["H", "A", "D"])
            ll = float(np.mean(np.logaddexp(0, np.where(gt[ok] == "D", -dead[ok], dead[ok]))))
            acc = float(np.mean((dead[ok] > 0) == (gt[ok] == "D")))
            p = ev_path(name, fold_name, n, t)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            np.savez(p + ".tmp.npz", dead=dead, ll=ll, acc=acc)
            os.replace(p + ".tmp.npz", p)
            print("   g{} {} blind dead log-loss {:.4f} acc {:.4f}".format(n, t, ll, acc), flush=True)
        print("fold test {} ready ({:.0f} s)".format(test, time.time() - t0), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("fit",))
    ap.add_argument("--name", default="b0")
    ap.add_argument("--folds", default="g5")
    ap.add_argument("--tags", default="std")
    a = ap.parse_args()
    config.seed_everything()
    tags = {"std": nexp.STD, "h15": nexp.ROB[0.15], "h35": nexp.ROB[0.35]}[a.tags]
    cmd_fit(a.name, a.folds, tags, CFG)
