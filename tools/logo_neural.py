"""Leave-one-game-out for the neural evidence model, with cross-fitted evidence.

    python tools/logo_neural.py --workers 16            # five folds, as tools/logo.py
    python tools/logo_neural.py --final --workers 16    # settings for the shipped build

As tools/logo.py: each fold leaves one game out, fits and tunes on the others and scores the one
left out, at every tracker level, with the ball hidden on 25% of live frames (and re-scored at
15% / 35%). One thing is done differently, on purpose. A network fits its training games far
better than a new game, so decoder settings tuned on its in-sample evidence would be tuned for
evidence that is too confident. Here the evidence the settings are tuned on is itself
out-of-sample:

- test game b, training games T (the other four);
- the evidence for b comes from the network trained on T;
- the evidence for each training game a, used to tune the decoder, comes from the network
  trained on T without a (three games), so that network has not seen a either.

That is 5 networks on four games and 10 on three; each is trained once and cached
(tools/fit_neural.py). The left-out game is never used to fit or tune anything in its fold.

--final: tune the decoder on all the games, each with the evidence of the network that did not
see it (the five four-game networks). The shipped network itself is fitted on all the games
(tools/fit_neural.py export).
"""

import argparse
import multiprocessing
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from espadas import config, neural, v4  # noqa: E402
from espadas.external import bench  # noqa: E402
import fit_neural  # noqa: E402
import tune  # noqa: E402

STD = [tag for tag, *_ in fit_neural.STANDARD]
DEAD = "dead"       # "dead": the tree models (the build); "net_dead": the network's own output
ROB = {hide: [tag for tag, _, h, _ in fit_neural.ROBUST if h == hide] for hide in (0.15, 0.35)}


def evidence_path(train, number, tag):
    return os.path.join(fit_neural.CACHE, "evidence",
                        "{}_g{}_{}.npz".format(fit_neural.fit_key(train), number, tag))


def _apply(job):
    train, number, tag = job
    path = evidence_path(train, number, tag)
    if not os.path.exists(path):
        w = fit_neural.fitted(train)
        view = fit_neural.load_view(number, tag)
        net_dead, team, _ = neural.apply(w, view)
        dead = fit_neural.dead_log_odds(fit_neural.dead_models(train), view)
        report = fit_neural.frame_report(w, view) if tag in STD else {}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path + ".tmp.npz", dead=dead, net_dead=net_dead, team=team,
                 **{k: np.array(v) for k, v in report.items()})
        os.replace(path + ".tmp.npz", path)
    return job


def rows(train, number, tags):
    """tune.prepare()-style rows for one game, evidence from the network trained on `train`."""
    out = []
    for tag in tags:
        view = fit_neural.load_view(number, tag, features=True)
        z = np.load(evidence_path(train, number, tag))
        f = dict(view["features"], dead_lo=z[DEAD], team_lo=z["team"])
        out.append((view["name"], view["gt_codes"], f, view["graded"]))
    return out


def frame_quality(train, number):
    z = [np.load(evidence_path(train, number, tag)) for tag in STD]
    return {k: float(np.mean([float(x[k]) for x in z]))
            for k in ("dead_loss", "dead_acc", "team_loss", "team_acc")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--games", nargs="+", type=int, default=list(bench.DEV_TUNING))
    parser.add_argument("--leave-out", nargs="+", type=int, default=None)
    parser.add_argument("--sweeps", type=int, default=tune.MAX_SWEEPS)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--final", action="store_true",
                        help="tune on all the games with out-of-sample evidence; print PARAMS")
    parser.add_argument("--train-only", action="store_true", help="fit the networks and stop")
    parser.add_argument("--net-dead", action="store_true",
                        help="diagnostic only: take the dead evidence from the network too "
                             "(it can read whether the ball is tracked; neural.py, DEAD)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    global DEAD
    DEAD = "net_dead" if args.net_dead else "dead"
    config.seed_everything()
    games = sorted(args.games)
    left_out = [] if args.final else (args.leave_out or games)
    log = print if args.verbose else (lambda *a, **k: None)

    outer = {b: tuple(g for g in games if g != b) for b in games}
    jobs, subsets = [], []
    for b in (games if args.final else left_out):
        subsets.append(outer[b])
        jobs += [(outer[b], b, tag) for tag in STD + ([] if args.final else ROB[0.15] + ROB[0.35])]
    for b in left_out:
        for a in outer[b]:
            inner = tuple(g for g in outer[b] if g != a)
            subsets.append(inner)
            jobs += [(inner, a, tag) for tag in STD]
    for subset in dict.fromkeys(subsets):       # GPU work first, in this process
        t0 = time.time()
        fit_neural.fitted(subset, log=print)
        fit_neural.dead_models(subset)
        print("network on games {} ready ({:.0f} s)".format(" ".join(map(str, subset)),
                                                           time.time() - t0), flush=True)
    if args.train_only:
        return 0
    jobs = list(dict.fromkeys(jobs))
    with multiprocessing.get_context("spawn").Pool(min(args.workers, 6)) as pool:
        for i, _ in enumerate(pool.imap_unordered(_apply, jobs), 1):
            if i % 20 == 0 or i == len(jobs):
                print("evidence {}/{}".format(i, len(jobs)), flush=True)

    old_sweeps, tune.MAX_SWEEPS = tune.MAX_SWEEPS, args.sweeps
    if args.final:
        train_rows = [r for a in games for r in rows(outer[a], a, STD)]
        with tune.parallel(v4, train_rows, args.workers):
            score, params = tune.descend(v4, train_rows, v4.PARAMS, log=log)
        print("out-of-sample evidence, settings tuned on all games: {:.4f}".format(score))
        print("PARAMS = {}".format(params))
        return 0

    results = []
    for b in left_out:
        t0 = time.time()
        train_rows = [r for a in outer[b]
                      for r in rows(tuple(g for g in outer[b] if g != a), a, STD)]
        test_rows = rows(outer[b], b, STD)
        with tune.parallel(v4, train_rows, args.workers):
            train_score, params = tune.descend(v4, train_rows, v4.PARAMS, log=log)
        test_score, per_level = tune.objective(v4, test_rows, params)
        shipped_score, _ = tune.objective(v4, test_rows, v4.PARAMS)
        robust = {h: tune.objective(v4, rows(outer[b], b, ROB[h]), params)[0] for h in ROB}
        q = frame_quality(outer[b], b)
        changed = {k: v for k, v in params.items() if v4.PARAMS.get(k) != v}
        results.append((train_score, test_score, shipped_score, robust, q))
        name = test_rows[0][0]
        print("\nv4  test {}   ({:.0f} s)".format(name, time.time() - t0))
        print("  train (out-of-sample evidence) {:.4f}".format(train_score))
        print("  test, re-tuned settings        {:.4f}   per level {}".format(
            test_score, " ".join("{:.3f}".format(s) for s in per_level)))
        print("  test, shipped settings         {:.4f}".format(shipped_score))
        print("  test at 15% / 35% hidden       {:.4f} / {:.4f}".format(robust[0.15],
                                                                        robust[0.35]))
        print("  per frame: team log-loss {team_loss:.4f} acc {team_acc:.4f}   "
              "network dead (unused) log-loss {dead_loss:.4f} acc {dead_acc:.4f}".format(**q))
        print("  settings moved from shipped    {}".format(changed or "none"), flush=True)
    tune.MAX_SWEEPS = old_sweeps
    mean = lambda xs: float(np.mean(xs))  # noqa: E731
    print("\nv4  LOGO score (mean test, re-tuned) {:.4f}   mean train-test gap {:+.4f}   "
          "shipped settings on test {:.4f}".format(
              mean([r[1] for r in results]), mean([r[0] - r[1] for r in results]),
              mean([r[2] for r in results])))
    print("v4  LOGO at 15% / 35% hidden: {:.4f} / {:.4f}".format(
        mean([r[3][0.15] for r in results]), mean([r[3][0.35] for r in results])))
    print("v4  per frame, left-out games: team log-loss {:.4f} acc {:.4f}   network dead (unused) log-loss "
          "{:.4f} acc {:.4f}".format(*(mean([r[4][k] for r in results]) for k in
                                       ("team_loss", "team_acc", "dead_loss", "dead_acc"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
