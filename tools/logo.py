"""Leave-one-game-out check on the tuning games: how well does tuning on one game carry to another?

    python tools/logo.py                         # v3 and v4, both folds, all tracker levels
    python tools/logo.py --labeler v4 --levels medium --sweeps 1    # quick look

Each fold uses one tuning game (Metrica 1 or 3) for everything that is fitted or tuned, and
scores the other one:
1. the two evidence models are refitted on the training game only (tools/fit_evidence.py's
   fit_models, same rows: every tracker level, ball hidden on degrade.LIVE_MISSING of live frames);
2. the labeler's settings are re-tuned on the training game only (tools/tune.py's descend, same
   grid and objective: mean spellF1 over the tracker levels), starting from the shipped PARAMS;
3. the test game is scored with those models and settings, at every tracker level.

Printed per fold: the training score (in-sample), the test score with the re-tuned settings
(out-of-sample: the number that matters) and the test score with the shipped settings (fold
evidence models). The LOGO score is the mean test score over the two folds.

Caveats, read the numbers with them: (a) the start point is the shipped PARAMS, which were tuned
on both games, so a little of the test game leaks in through it (coordinate descent from there
mostly stays nearby); (b) the two games are annotated with different conventions
(docs/events_rules_review.md, issue 2), so part of any gap is the reference, not the model; (c)
two folds is very little: use it to catch a change that only fits one game, not to rank
changes that differ by a few hundredths.

Tuning games only. The held-out game and our match are never read. Nothing is written: the
shipped espadas/evidence_models.py and PARAMS are untouched. Deterministic.
"""

import argparse
import contextlib
import importlib
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from espadas import config, evidence  # noqa: E402
from espadas.external import bench, degrade  # noqa: E402
import fit_evidence  # noqa: E402
import tune  # noqa: E402

LEVELS = degrade.TRACKER_LEVELS
HIDE_BALL = degrade.LIVE_MISSING


@contextlib.contextmanager
def using_models(dead_model, team_model):
    """Make evidence.log_odds() (and so every labeler's features) use these models instead of
    the shipped evidence_models.py, inside the block only."""
    original = evidence._fitted
    evidence._fitted = lambda: (dead_model, team_model)
    try:
        yield
    finally:
        evidence._fitted = original


def views(game, levels, hide=HIDE_BALL):
    """[(level, reference labels, gt codes, tracking, graded)]: the degraded views of one game."""
    gt = bench.reference(game)[0]
    out = []
    for level in levels:
        gt_codes, tracking, graded = bench.setup(game, hide, level)
        out.append((level, gt, gt_codes, tracking, graded))
    return out


def fit_on(game_views, name):
    rows = [(name + "/" + level, gt, evidence.clues(tracking), graded)
            for level, gt, _, tracking, graded in game_views]
    dead_model, team_model, _ = fit_evidence.fit_models(rows)
    return dead_model, team_model


def features(module, game_views, name):
    """tune.prepare()-style rows for one game, with whatever evidence models are active."""
    return [(name, gt_codes, module.features(tracking), graded)
            for _, _, gt_codes, tracking, graded in game_views]


def run_fold(module, train, test, levels, sweeps, log):
    (train_name, train_views), (test_name, test_views) = train, test
    t0 = time.time()
    dead_model, team_model = fit_on(train_views, train_name)
    with using_models(dead_model, team_model):
        train_rows = features(module, train_views, train_name)
        test_rows = features(module, test_views, test_name)
    old_sweeps, tune.MAX_SWEEPS = tune.MAX_SWEEPS, sweeps
    try:
        train_score, params = tune.descend(module, train_rows, module.PARAMS, log=log)
    finally:
        tune.MAX_SWEEPS = old_sweeps
    test_score, per_level = tune.objective(module, test_rows, params)
    shipped_score, shipped_levels = tune.objective(module, test_rows, module.PARAMS)
    changed = {k: v for k, v in params.items() if module.PARAMS.get(k) != v}
    return {"train": train_name, "test": test_name, "train_score": train_score,
            "test_score": test_score, "test_levels": per_level, "shipped_score": shipped_score,
            "shipped_levels": shipped_levels, "changed": changed, "seconds": time.time() - t0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labeler", nargs="+", default=["v3", "v4"],
                        help="labeler modules with features/decode/PARAMS/GRID (v2 onwards)")
    parser.add_argument("--levels", nargs="+", default=list(LEVELS), choices=list(LEVELS))
    parser.add_argument("--sweeps", type=int, default=tune.MAX_SWEEPS,
                        help="coordinate-descent sweeps per fold (default {})".format(
                            tune.MAX_SWEEPS))
    parser.add_argument("--verbose", action="store_true", help="print the descent")
    args = parser.parse_args(argv)
    config.seed_everything()
    log = print if args.verbose else (lambda *a, **k: None)

    games = {n: game for n, game in bench.available(bench.TUNING)}
    if len(games) < 2:
        print("need both tuning games downloaded (tools/fetch_metrica.py)")
        return 1
    loaded = {n: (games[n]["name"], views(games[n], args.levels)) for n in games}
    folds = [(a, b) for a in sorted(loaded) for b in sorted(loaded) if a != b]

    print("leave-one-game-out, levels {}, ball hidden on {:.0%} of live frames".format(
        " ".join(args.levels), HIDE_BALL))
    for name in args.labeler:
        module = importlib.import_module("espadas." + name)
        results = []
        for a, b in folds:
            r = run_fold(module, loaded[a], loaded[b], args.levels, args.sweeps, log)
            results.append(r)
            print("\n{}  train {} -> test {}   ({:.0f} s)".format(name, r["train"], r["test"],
                                                                r["seconds"]))
            print("  train (in-sample)            {:.4f}".format(r["train_score"]))
            print("  test, re-tuned settings      {:.4f}   per level {}".format(
                r["test_score"], " ".join("{:.3f}".format(s) for s in r["test_levels"])))
            print("  test, shipped settings       {:.4f}   per level {}".format(
                r["shipped_score"], " ".join("{:.3f}".format(s) for s in r["shipped_levels"])))
            print("  settings moved from shipped  {}".format(r["changed"] or "none"))
        logo = float(np.mean([r["test_score"] for r in results]))
        gap = float(np.mean([r["train_score"] - r["test_score"] for r in results]))
        print("\n{}  LOGO score (mean test, re-tuned) {:.4f}   mean train-test gap {:+.4f}"
              "   shipped settings on test {:.4f}".format(
                  name, logo, gap, float(np.mean([r["shipped_score"] for r in results]))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
