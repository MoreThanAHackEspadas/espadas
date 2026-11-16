"""Decoder experiments on saved evidence (CPU only).

    python dexp.py ceilings VARIANT --folds g5 [--dead tree|net|blind|split|avg] [--blind NAME]
    python dexp.py score    VARIANT --folds g5 --dead split --blind NAME [--change] [--w-change X]
    python dexp.py cvtune   VARIANT --folds g5 --dead split --blind NAME [--change] [--keys dead|all]

Evidence per game comes from networks that did not see it (nexp.py fit / blind.py fit).
`--games 1 2` replaces the fold list by leave-one-out folds over those games (smoke tests).
"""
import argparse, json, multiprocessing, os, sys, time, types
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nexp, blind, v4c                              # noqa: E402
from nexp import v4, bench, fit_neural, tune, config  # noqa: E402

OUT = os.path.join(nexp.WORK, "out")
DEAD_KEYS = ["dead_smooth", "dead_bias", "w_dead", "c_stop", "c_restart", "min_dead", "min_alive"]


def get_folds(name, games=None):
    if games:
        return [(tuple(g for g in games if g != o), (o,)) for o in games]
    return nexp.FOLDS[name]


def _load(job):
    variant, fold, n, tag, blind_name, mode = job
    view = fit_neural.load_view(n, tag, features=True)
    z = np.load(nexp.ev_path(variant, fold, n, tag))
    f = dict(view["features"], team_lo=z["team"], change_lo=z["change"], _gt=view["gt"])
    present = np.asarray(f["present"], dtype=bool)
    if mode in ("blind", "split", "avg"):
        b = np.load(blind.ev_path(blind_name, fold, n, tag))["dead"]
    if mode == "tree":
        dead = z["dead"]
    elif mode == "net":
        dead = z["net_dead"]
    elif mode == "blind":
        dead = b
    elif mode == "split":
        dead = np.where(present, z["dead"], b)
    elif mode == "avg":
        dead = np.where(present, 0.5 * (b + z["dead"]), b)
    else:
        raise ValueError(mode)
    f["dead_lo"] = np.asarray(dead, dtype=float)
    return (view["name"], view["gt_codes"], f, view["graded"])


def load_rows(variant, fold_name, tags, blind=None, dead_mode="tree", games=None, workers=6):
    """dict game -> list of rows (one per tag); f has dead_lo, team_lo, change_lo and _gt."""
    if dead_mode in ("blind", "split", "avg") and not blind:
        raise SystemExit("--dead {} needs --blind NAME".format(dead_mode))
    test = sorted(g for _, t in get_folds(fold_name, games) for g in t)
    jobs = [(variant, fold_name, n, t, blind, dead_mode) for n in test for t in tags]
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        out = pool.map(_load, jobs)
    res = {}
    for (_, _, n, _, _, _), r in zip(jobs, out):
        res.setdefault(n, []).append(r)
    return res


def flat(rows_by_game):
    games = sorted(rows_by_game)
    return [n for n in games for _ in rows_by_game[n]], [r for n in games for r in rows_by_game[n]]


def score_rows(mod, rows, params, workers):
    with tune.parallel(mod, rows, workers):
        return tune.objective(mod, rows, params)[1]


def per_game(ids, scores):
    d = {}
    for n, s in zip(ids, scores):
        d.setdefault(n, []).append(float(s))
    return {n: float(np.mean(v)) for n, v in d.items()}


def summary(pg):
    m = lambda gs: float(np.mean([pg[g] for g in gs if g in pg])) if any(g in pg for g in gs) else float("nan")
    return {"all": float(np.mean(list(pg.values()))), "dev": m(nexp.DEV),
            "metrica": m(nexp.METRICA), "idsse": m(nexp.IDSSE)}


def fmt(pg):
    return "  ".join("g{} {:.4f}".format(n, pg[n]) for n in sorted(pg))


def fmt_sum(s):
    return "all {all:.4f} | dev5 {dev:.4f} | Metrica {metrica:.4f} | IDSSE {idsse:.4f}".format(**s)


def oracle(f, dead=False, team=False):
    g = f["_gt"]
    f = dict(f)
    if dead:
        f["dead_lo"] = np.where(g == "D", 20.0, -20.0)
    if team:
        f["team_lo"] = np.where(g == "H", 20.0, np.where(g == "A", -20.0, 0.0))
    return f


def save(a, result):
    os.makedirs(OUT, exist_ok=True)
    name = "dexp_{}_{}_{}_{}{}{}".format(a.command, a.variant, a.folds, a.dead,
                                        "_change" if a.change else "",
                                        "_" + a.keys if a.command == "cvtune" else "")
    path = os.path.join(OUT, name + ".json")
    with open(path, "w") as fh:
        json.dump(result, fh, indent=1, default=float)
    print("saved", path)


def cmd_ceilings(a, mod, rows_by_game):
    ids, rows = flat(rows_by_game)
    cases = [("evidence", False, False), ("oracle dead", True, False),
             ("oracle team", False, True), ("oracle both", True, True)]
    allrows = [(nm, gt, oracle(f, d, t), gr) for _, d, t in cases for nm, gt, f, gr in rows]
    scores = score_rows(mod, allrows, dict(mod.PARAMS), a.workers)
    res = {}
    for k, (label, _, _) in enumerate(cases):
        pg = per_game(ids, scores[k * len(rows):(k + 1) * len(rows)])
        res[label] = {"per_game": pg, "mean": summary(pg)}
        print("{:<12} {}\n{:<12} MEAN {}".format(label, fmt(pg), "", fmt_sum(res[label]["mean"])))
    return res


def dead_quality(rows_by_game):
    out = {}
    for n, rows in rows_by_game.items():
        ll, acc = [], []
        for _, _, f, gr in rows:
            g = f["_gt"]; ok = gr & np.isin(g, ["H", "A", "D"])
            z = f["dead_lo"][ok]; y = g[ok] == "D"
            ll.append(np.mean(np.logaddexp(0, np.where(y, -z, z)))); acc.append(np.mean((z > 0) == y))
        out[n] = (float(np.mean(ll)), float(np.mean(acc)))
    return out


def cmd_score(a, mod, rows_by_game):
    ids, rows = flat(rows_by_game)
    params = dict(mod.PARAMS)
    if a.change:
        params["w_change"] = a.w_change
    pg = per_game(ids, score_rows(mod, rows, params, a.workers))
    dq = dead_quality(rows_by_game)
    s = summary(pg)
    ll, acc = (float(np.mean([v[i] for v in dq.values()])) for i in (0, 1))
    print("{} {} dead={}: {}\n  MEAN {} | dead ll {:.4f} acc {:.4f}".format(
        a.variant, a.folds, a.dead, fmt(pg), fmt_sum(s), ll, acc))
    return {"per_game": pg, "mean": s, "dead_logloss": ll, "dead_acc": acc, "dead_per_game": dq,
            "params": params}


def cmd_cvtune(a, mod, rows_by_game):
    tune.MAX_SWEEPS = a.sweeps
    grid = mod.GRID if a.keys == "all" else {k: mod.GRID[k] for k in DEAD_KEYS + (["w_change"] if a.change else [])}
    tmod = types.SimpleNamespace(GRID=grid, decode=mod.decode, PARAMS=mod.PARAMS)
    start = dict(mod.PARAMS)
    folds, res = get_folds(a.folds, a.games), []
    tuned_pg, start_pg = {}, {}
    for train_games, test in folds:
        t0 = time.time()
        ids, rows = flat({n: r for n, r in rows_by_game.items() if n not in test})
        with tune.parallel(tmod, rows, a.workers):
            tr_score, params = tune.descend(tmod, rows, start, log=lambda *x: None)
        tids, trows = flat({n: rows_by_game[n] for n in test})
        s_t = score_rows(tmod, trows, params, a.workers)
        s_s = score_rows(tmod, trows, start, a.workers)
        tp, sp = per_game(tids, s_t), per_game(tids, s_s)
        tuned_pg.update(tp); start_pg.update(sp)
        moved = {k: (start[k], params[k]) for k in params if params[k] != start[k]}
        res.append({"test": list(test), "train_score": tr_score, "test_tuned": tp,
                    "test_start": sp, "moved": moved, "params": params})
        print("fold test {}: train {:.4f} | test tuned {:.4f} | test start {:.4f} | {:.0f} s\n  moved {}".format(
            test, tr_score, np.mean(list(tp.values())), np.mean(list(sp.values())),
            time.time() - t0, moved), flush=True)
    st, ss = summary(tuned_pg), summary(start_pg)
    print("TUNED  {}\n  MEAN {}\nSTART  {}\n  MEAN {}".format(fmt(tuned_pg), fmt_sum(st), fmt(start_pg), fmt_sum(ss)))
    return {"folds": res, "tuned": {"per_game": tuned_pg, "mean": st},
            "start": {"per_game": start_pg, "mean": ss}, "keys": a.keys, "sweeps": a.sweeps}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("ceilings", "score", "cvtune"))
    ap.add_argument("variant")
    ap.add_argument("--folds", default="g5")
    ap.add_argument("--games", nargs="+", type=int, default=None)
    ap.add_argument("--tags", default="std")
    ap.add_argument("--blind", default=None)
    ap.add_argument("--dead", default="tree", choices=("tree", "net", "blind", "split", "avg"))
    ap.add_argument("--change", action="store_true")
    ap.add_argument("--w-change", type=float, default=0.0)
    ap.add_argument("--keys", default="all", choices=("dead", "all"))
    ap.add_argument("--sweeps", type=int, default=2)
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()
    config.seed_everything()
    tags = {"std": nexp.STD, "h15": nexp.ROB[0.15], "h35": nexp.ROB[0.35]}[a.tags]
    mod = v4c if a.change else v4
    rows_by_game = load_rows(a.variant, a.folds, tags, a.blind, a.dead, a.games)
    result = {"ceilings": cmd_ceilings, "score": cmd_score, "cvtune": cmd_cvtune}[a.command](a, mod, rows_by_game)
    save(a, dict(result, args=vars(a)))
