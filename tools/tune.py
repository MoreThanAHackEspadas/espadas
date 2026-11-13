"""Tune a labeler's decoding settings on the external TUNING games only, by coordinate descent.

    python tools/tune.py --labeler v2          # prints the search and the best PARAMS

Works for any labeler module exposing features(tracking), decode(features, params), PARAMS and
GRID (v2 onwards; v1 has its own tools/tune_v1.py). Same protocol as tune_v1.py: objective is the
mean spellF1 over the tuning games (default bench.DEV_TUNING: Metrica 1, 3 and IDSSE 12, 13, 15;
--games to change) *and* over the four tracker levels (degrade.TRACKER_LEVELS:
light, medium, heavy and smooth ball blur plus player position error; checklist 6.6, 6.7: a
method must work for any plausible tracker, not one level), with the ball hidden on
degrade.LIVE_MISSING of live frames; the chosen settings are re-scored at 15% / 35% hidden.
The held-out game is not scored: that happens once, at selection, in tools/selection.py.
--heldout scores it here for the chosen settings (never used to choose); --skip-heldout is
still accepted and changes nothing. Our match is never touched. Deterministic.

Copy the printed PARAMS into the labeler module.
"""

import argparse
import contextlib
import importlib
import multiprocessing
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from espadas import config  # noqa: E402
from espadas.external import bench, degrade  # noqa: E402

HIDE_BALL = degrade.LIVE_MISSING
ROBUSTNESS = (0.15, 0.35)
OBJECTIVE_BLURS = degrade.TRACKER_LEVELS   # light, medium, heavy, smooth (6.6, 6.7)
MAX_SWEEPS = 4


def prepare(module, numbers, hide=HIDE_BALL, blur=degrade.BLUR_DEFAULT, heldout=False):
    out = []
    for _, game in bench.available(numbers, heldout=heldout):
        gt, tracking, graded = bench.setup(game, hide, blur)
        out.append((game["name"], gt, module.features(tracking), graded))
    return out


_SHARED = {}


def _score_view(job):
    i, params = job
    _, gt, f, graded = _SHARED["games"][i]
    return bench.evaluate(gt, _SHARED["module"].decode(f, params), graded)["spell_f1_mean"]


@contextlib.contextmanager
def parallel(module, games, workers):
    """Inside the block, objective(module, games, ...) scores the views on `workers` forked
    processes (they inherit the prepared features; nothing is pickled but the params). Same
    numbers as the serial path, in the same order."""
    if workers <= 1:
        yield
        return
    _SHARED.update(module=module, games=games)
    try:
        with multiprocessing.get_context("fork").Pool(workers) as pool:
            _SHARED["pool"] = pool
            yield
    finally:
        _SHARED.clear()


def objective(module, games, params):
    pool = _SHARED.get("pool")
    if pool is not None and _SHARED["games"] is games and _SHARED["module"] is module:
        scores = pool.map(_score_view, [(i, params) for i in range(len(games))])
    else:
        scores = [bench.evaluate(gt, module.decode(f, params), graded)["spell_f1_mean"]
                  for _, gt, f, graded in games]
    return float(np.mean(scores)), scores


def descend(module, games, start, log=print):
    """(best mean spellF1, params): coordinate descent over module.GRID from `start` on `games`
    (prepare() rows). Shared with tools/logo.py."""
    params = dict(start)
    best, _ = objective(module, games, params)
    log("start   {:.4f}  {}".format(best, params))
    for sweep in range(1, MAX_SWEEPS + 1):
        improved = False
        for name, values in module.GRID.items():
            for value in values:
                trial = dict(params, **{name: value})
                score, _ = objective(module, games, trial)
                if score > best + 1e-9:
                    best, params, improved = score, trial, True
            log("sweep {} {:<15} -> {:<6} mean spellF1 {:.4f}".format(
                sweep, name, params[name], best))
        if not improved:
            break
    return best, params


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labeler", required=True, help="module in espadas/, e.g. v2")
    parser.add_argument("--heldout", action="store_true",
                        help="also score the held-out game for the chosen settings (off by "
                             "default: tools/selection.py scores it once, at selection)")
    parser.add_argument("--skip-heldout", action="store_true",
                        help="do not score the held-out game (the default now; kept so old "
                             "commands still run; wins over --heldout)")
    parser.add_argument("--games", nargs="+", type=int, default=list(bench.DEV_TUNING),
                        help="tuning games (default: the development pool {})".format(
                            " ".join(map(str, bench.DEV_TUNING))))
    parser.add_argument("--workers", type=int, default=1,
                        help="processes scoring the views in parallel (same result)")
    parser.add_argument("--pool", choices=sorted(bench.POOLS), default=None,
                        help="named pool instead of --games: dev (the default games) or full "
                             "(all ten external matches, incl. the spent held-out ones; "
                             "docs/refinement2_rule.md)")
    args = parser.parse_args(argv)
    module = importlib.import_module("espadas." + args.labeler)
    config.seed_everything()
    spent = False
    if args.pool:
        args.games, spent = bench.pool(args.pool)
    print("tuning games: {}".format(" ".join(map(str, args.games))))

    tuning = [g for blur in OBJECTIVE_BLURS
              for g in prepare(module, args.games, blur=blur, heldout=spent)]
    with parallel(module, tuning, args.workers):
        best, params = descend(module, tuning, module.PARAMS)
        _, per_game = objective(module, tuning, params)
    print("\nPARAMS = {}".format(params))
    n_games = len(tuning) // len(OBJECTIVE_BLURS)
    for i, ((name, *_), s) in enumerate(zip(tuning, per_game)):
        print("  tune     {:<10} blur {:<6} spellF1 {:.4f}".format(
            name, OBJECTIVE_BLURS[i // n_games], s))
    checks = [(h, b) for h in ROBUSTNESS for b in OBJECTIVE_BLURS]
    for hide in ROBUSTNESS:
        score = np.mean([objective(module, prepare(module, args.games, hide, b, heldout=spent),
                                   params)[0]
                         for b in OBJECTIVE_BLURS])
        print("  tune @ {:.0%} live missing, all blurs: mean spellF1 {:.4f}".format(hide, score))
    if args.skip_heldout or not args.heldout:
        return 0
    for hide, blur in [(HIDE_BALL, b) for b in OBJECTIVE_BLURS] + checks:
        for name, gt, f, graded in prepare(module, bench.HELD_OUT, hide, blur, heldout=True):
            r = bench.evaluate(gt, module.decode(f, params), graded)
            print("  HELDOUT  {:<10} @ {:.0%} live missing, blur {:<6} spellF1 {:.4f}  "
                  "macroF1 {:.4f}  n_pred {}  n_gt {}".format(
                      name, hide, blur, r["spell_f1_mean"], r["macro_f1"],
                      r["n_pred_spells"], r["n_gt_spells"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
