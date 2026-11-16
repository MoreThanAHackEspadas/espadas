"""Experiments on top of the neural team evidence (PR 7), all ten external games.

    python nexp.py fit VARIANT --folds g5        # train the fold networks, write evidence (GPU)
    python nexp.py score VARIANT --folds g5      # decode + score with fixed settings (CPU)

Evidence for a game always comes from networks that did not see it.
"""
import argparse, hashlib, json, multiprocessing, os, pickle, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
from espadas import config, neural, v4            # noqa: E402
from espadas.external import bench                # noqa: E402
import fit_neural, tune                           # noqa: E402

CACHE = fit_neural.CACHE
WORK = os.environ.get("ESPADAS_EXP_WORK") or os.path.join(CACHE, "exp")   # evidence, results
METRICA, IDSSE = (1, 2, 3), (11, 12, 13, 14, 15, 16, 17)
ALL = METRICA + IDSSE
DEV = (1, 3, 12, 13, 15)
STD = [t for t, *_ in fit_neural.STANDARD]
TRAIN_TAGS = [t for t, *_ in fit_neural.STANDARD + fit_neural.EXTRA]
ROB = {h: [t for t, _, hh, _ in fit_neural.ROBUST if hh == h] for h in (0.15, 0.35)}
FOLDS = {
    "g5": [(tuple(g for g in ALL if g not in out), out)
           for out in ((1, 11), (2, 12), (3, 13), (14, 15), (16, 17))],
    "lpo": [(IDSSE, METRICA), (METRICA, IDSSE)],
    "dev5": [(tuple(g for g in DEV if g != out), (out,)) for out in DEV],
    "logo10": [(tuple(g for g in ALL if g != out), (out,)) for out in ALL],
}

VARIANTS = {
    "base": {},                                             # PR 7's configuration
    "ens3": {"seeds": [0, 1, 2]},
    "wide": {"cfg": {"channels": 96, "player_hidden": 64, "steps": 3000}},
    "long": {"cfg": {"steps": 4000}},
    "posb": {"cfg": {"pos_balance": 0.5}},
    "posb1": {"cfg": {"pos_balance": 1.0}},
    "tta": {"tta": True},
    "ens3tta": {"seeds": [0, 1, 2], "tta": True},
}


# --------------------------------------------------------------------------- memory-mapped views

def mm_dir(n, tag):
    return os.path.join(CACHE, "mm", "g{}_{}".format(n, tag))


def mm_view(n, tag):
    """A view with players / mask / globals as float16 memory maps (made on first use)."""
    d = mm_dir(n, tag)
    z = np.load(fit_neural.view_path(n, tag), allow_pickle=False)
    if not os.path.exists(os.path.join(d, "done")):
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, "players.npy"), z["players"].astype(np.float16))
        np.save(os.path.join(d, "mask.npy"), z["mask"])
        np.save(os.path.join(d, "globals.npy"), z["globals"].astype(np.float16))
        open(os.path.join(d, "done"), "w").close()
    names = list(neural.OWN_GLOBALS) + [str(x) for x in z["names"][len(neural.OWN_GLOBALS):]]
    return {"players": np.load(os.path.join(d, "players.npy"), mmap_mode="r"),
            "mask": np.load(os.path.join(d, "mask.npy"), mmap_mode="r"),
            "globals": np.load(os.path.join(d, "globals.npy"), mmap_mode="r"),
            "names": names, "half_blocks": [tuple(int(v) for v in b) for b in z["half_blocks"]],
            "gt": z["gt"], "gt_codes": z["gt_codes"], "graded": z["graded"],
            "name": str(z["name"]), "number": n, "tag": tag}


def global_scale(views, names):
    total, count = np.zeros(len(names)), 0
    for v in views:
        ok = v["graded"]
        g = np.asarray(v["globals"][::4], dtype=np.float64)[ok[::4]]
        total += np.square(g).sum(axis=0); count += len(g)
    ms = total / max(count, 1)
    perm, _ = neural.global_swap(names)
    ms = (ms + ms[perm]) / 2.0
    return (1.0 / np.sqrt(ms + 1e-6)).astype(np.float32)


def possession_weights(gt, live, alpha):
    """Per-frame weight L ** -alpha for the frames of each reference possession (a maximal run of
    one label H / A inside `live`, L frames), divided by the mean over the live frames and
    clipped to [0.25, 4]; 0 elsewhere."""
    gt, live = np.asarray(gt), np.asarray(live, dtype=bool)
    w = np.zeros(len(gt))
    if not live.any():
        return w
    idx = np.flatnonzero(live)
    brk = np.flatnonzero((np.diff(idx) != 1) | (gt[idx][1:] != gt[idx][:-1])) + 1
    starts, stops = np.r_[0, brk], np.r_[brk, len(idx)]
    for a, b in zip(starts, stops):
        w[idx[a:b]] = float(b - a) ** (-alpha)
    w[idx] /= w[idx].mean()
    return np.where(live, np.clip(w, 0.25, 4.0), 0.0)


def train(views, cfg, log=print):
    """fit_neural.train, reading windows from memory maps (same sampling and loss)."""
    import torch
    import torch.nn.functional as F
    device = "cuda"
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    names = views[0]["names"]
    gscale = global_scale(views, names)
    perm, sign = neural.global_swap(names)
    flip = neural.global_flip(names)
    data = []
    for v in views:
        gt, graded = v["gt"], v["graded"]
        ok = graded & np.isin(gt, ["H", "A", "D"])
        team_w = ok & (gt != "D")
        if cfg.get("pos_balance", 0.0) > 0:
            team_w = possession_weights(gt, team_w, cfg["pos_balance"])
        data.append({"p": v["players"], "m": v["mask"], "g": v["globals"],
                     "y": np.stack([gt == "D", gt == "H",
                                    fit_neural.change_target(gt, graded, cfg["change_sigma"])],
                                   axis=1).astype(np.float32),
                     "w": np.stack([ok, team_w, ok], axis=1).astype(np.float32),
                     "blocks": v["half_blocks"]})
    net = fit_neural.build(len(names), cfg).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                            fused=True)
    scaler = torch.amp.GradScaler("cuda")      # mixed precision: same model, ~1.4x faster here
    steps, warm = cfg["steps"], 100
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * s / steps)))
    t_perm = torch.as_tensor(perm, device=device)
    t_sign = torch.as_tensor(sign, device=device)
    t_flip = torch.as_tensor(flip, device=device)
    t_scale = torch.as_tensor(gscale, device=device)
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
            w = d["w"][start:start + length].copy()
            if start > first:
                w[:edge] = 0
            if start + length < stop:
                w[-edge:] = 0
            for k in "pmgy":
                parts[k].append(np.asarray(d[k][start:start + length]))
            parts["w"].append(w)
        if len({len(x) for x in parts["w"]}) > 1:      # a short half: pad is not worth it, redraw
            continue
        p = torch.from_numpy(np.stack(parts["p"])).to(device).float()
        m = torch.from_numpy(np.stack(parts["m"])).to(device)
        g = torch.from_numpy(np.stack(parts["g"])).to(device).float() * t_scale
        y = torch.from_numpy(np.stack(parts["y"])).to(device)
        w = torch.from_numpy(np.stack(parts["w"])).to(device)
        swap = torch.rand(batch, device=device) < 0.5
        mirror = torch.rand(batch, device=device) < 0.5
        s4, s3 = swap.view(-1, 1, 1, 1, 1), swap.view(-1, 1, 1)
        p = torch.where(s4, p.flip(2) * p_swap, p)
        m = torch.where(s4.squeeze(-1), m.flip(2), m)
        g = torch.where(s3, g[..., t_perm] * t_sign, g)
        y = torch.where(s3, torch.stack([y[..., 0], 1 - y[..., 1], y[..., 2]], dim=-1), y)
        p = torch.where(mirror.view(-1, 1, 1, 1, 1), p * p_flip, p)
        g = torch.where(mirror.view(-1, 1, 1), g * t_flip, g)
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
    w.update(names=list(names), dilations=list(cfg["dilations"]), gscale=gscale)
    return w


def net_key(train_games, cfg, tags):
    text = json.dumps([sorted(train_games), cfg, tags], sort_keys=True)
    return "n2_{}_{}".format("-".join(map(str, sorted(train_games))),
                             hashlib.sha256(text.encode()).hexdigest()[:10])


def fitted(train_games, cfg, tags=TRAIN_TAGS):
    path = os.path.join(CACHE, "nets2", net_key(train_games, cfg, tags) + ".pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    print("  training on", train_games, {k: cfg[k] for k in ("seed", "steps", "channels", "dropout")}, flush=True)
    views = [mm_view(n, t) for n in sorted(train_games) for t in tags]
    w = train(views, cfg)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "wb") as f:
        pickle.dump(w, f)
    os.replace(path + ".tmp", path)
    return w


def apply_gpu(w, view, cfg, tta=False):
    """(dead, team, change) raw log-odds per frame from the PyTorch network (as neural.apply)."""
    import torch
    names = view["names"]
    net = fit_neural.build(len(names), cfg).cuda().eval()
    net.load_state_dict({k: torch.from_numpy(np.asarray(v)) for k, v in w.items()
                         if k not in ("names", "dilations", "gscale")})
    perm, sign = neural.global_swap(names)
    n = len(view["gt"])
    out = np.zeros((n, 3))
    t_scale = torch.as_tensor(w["gscale"]).cuda()
    with torch.no_grad():
        for _, first, last in view["half_blocks"]:
            s = slice(first, last + 1)
            p = torch.from_numpy(np.asarray(view["players"][s])).cuda().float()[None]
            m = torch.from_numpy(np.asarray(view["mask"][s])).cuda()[None]
            g = torch.from_numpy(np.asarray(view["globals"][s])).cuda().float()[None] * t_scale
            ps_sign = torch.as_tensor(neural.PLAYER_SWAP_SIGN).cuda()
            t_perm, t_sign = torch.as_tensor(perm).cuda(), torch.as_tensor(sign).cuda()

            def sym(p, g):
                a = net(p, m, g)[0]
                b = net(p.flip(2) * ps_sign, m.flip(2), g[..., t_perm] * t_sign)[0]
                return torch.stack([(a[:, 0] + b[:, 0]) / 2, (a[:, 1] - b[:, 1]) / 2,
                                    (a[:, 2] + b[:, 2]) / 2], dim=1)
            r = sym(p, g)
            if tta:        # pitch mirror keeps the teams, so the outputs keep their meaning
                r = (r + sym(p * torch.as_tensor(neural.PLAYER_FLIP_SIGN).cuda(),
                             g * torch.as_tensor(neural.global_flip(names)).cuda())) / 2
            out[s] = r.double().cpu().numpy()
    return out


def ev_path(variant, fold, n, tag):
    return os.path.join(WORK, "evidence", variant, "{}_g{}_{}.npz".format(fold, n, tag))


def cfg_of(variant, seed):
    return dict(fit_neural.CFG, **VARIANTS[variant].get("cfg", {}), seed=seed)


def cmd_fit(variant, fold_name, tags):
    spec = VARIANTS[variant]
    seeds = spec.get("seeds", [0])
    for train_games, test in FOLDS[fold_name]:
        todo = [(n, t) for n in test for t in tags if not os.path.exists(ev_path(variant, fold_name, n, t))]
        if not todo:
            continue
        t0 = time.time()
        nets = [(fitted(train_games, cfg_of(variant, s)), cfg_of(variant, s)) for s in seeds]
        dead_model = fit_neural.dead_models(train_games)
        for n, t in todo:
            view = mm_view(n, t)
            outs = np.mean([apply_gpu(w, view, c, tta=spec.get("tta", False)) for w, c in nets], axis=0)
            full = fit_neural.load_view(n, t)
            dead = fit_neural.dead_log_odds(dead_model, full)
            p = ev_path(variant, fold_name, n, t)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            np.savez(p + ".tmp.npz", dead=dead, net_dead=np.round(outs[:, 0], 3),
                     team=np.round(outs[:, 1], 3), change=np.round(outs[:, 2], 3))
            os.replace(p + ".tmp.npz", p)
        print("fold test {} ready ({:.0f} s)".format(test, time.time() - t0), flush=True)


_G = {}


def _row(job):
    variant, fold, n, tag, dead_key = job
    view = fit_neural.load_view(n, tag, features=True)
    z = np.load(ev_path(variant, fold, n, tag))
    f = dict(view["features"], dead_lo=z[dead_key], team_lo=z["team"], change_lo=z["change"])
    gt = view["gt"]; ok = view["graded"] & np.isin(gt, ["H", "A", "D"]); live = ok & (gt != "D")
    ll = float(np.mean(np.logaddexp(0, np.where(gt[live] == "H", -z["team"][live], z["team"][live]))))
    acc = float(np.mean((z["team"][live] > 0) == (gt[live] == "H")))
    return (view["name"], view["gt_codes"], f, view["graded"]), (n, tag, ll, acc)


def rows_for(variant, fold_name, games, tags, dead_key="dead", workers=6):
    jobs = [(variant, fold_name, n, t, dead_key) for n in games for t in tags]
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        out = pool.map(_row, jobs)
    return [r for r, _ in out], [m for _, m in out]


def cmd_score(variant, fold_name, tags, params=None, dead_key="dead", workers=14, quiet=False):
    games = sorted(g for _, test in FOLDS[fold_name] for g in test)
    rows, meta = rows_for(variant, fold_name, games, tags, dead_key)
    params = dict(v4.PARAMS, **(params or {}))
    with tune.parallel(v4, rows, workers):
        _, scores = tune.objective(v4, rows, params)
    per = {}
    for (n, tag, ll, acc), s in zip(meta, scores):
        per.setdefault(n, []).append((s, ll, acc))
    pg = {n: tuple(float(np.mean([x[i] for x in v])) for i in range(3)) for n, v in per.items()}
    m = lambda gs, i=0: float(np.mean([pg[g][i] for g in gs if g in pg])) if any(g in pg for g in gs) else float("nan")
    if not quiet:
        print("{} {} {}: ".format(variant, fold_name, dead_key) + "  ".join("g{} {:.4f}".format(n, pg[n][0]) for n in sorted(pg)))
        print("  MEAN all {:.4f} | dev5 {:.4f} | Metrica {:.4f} | IDSSE {:.4f} | team ll {:.4f} acc {:.4f}".format(
            m(ALL), m(DEV), m(METRICA), m(IDSSE), m(ALL, 1), m(ALL, 2)), flush=True)
    return pg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("prep", "fit", "score", "mm"))
    ap.add_argument("variant", nargs="?", default="base")
    ap.add_argument("--folds", default="g5")
    ap.add_argument("--tags", default="std")
    ap.add_argument("--dead", default="dead")
    a = ap.parse_args()
    config.seed_everything()
    tags = {"std": STD, "h15": ROB[0.15], "h35": ROB[0.35]}[a.tags]
    if a.command == "prep":      # views of all ten games (the five spent held-out ones included)
        fit_neural.prep(list(ALL), list(fit_neural.VIEWS), workers=5, spent=True)
    if a.command in ("prep", "mm"):
        for n in ALL:
            for t in TRAIN_TAGS + ROB[0.15] + ROB[0.35]:
                mm_view(n, t)
            print("mm", n, flush=True)
    elif a.command == "fit":
        cmd_fit(a.variant, a.folds, tags)
    else:
        cmd_score(a.variant, a.folds, tags, dead_key=a.dead)
