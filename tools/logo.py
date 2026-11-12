"""Leave-one-game-out check on the tuning games: how well does tuning on one game carry to another?

    python tools/logo.py                         # v3 and v4, both folds, all tracker levels
    python tools/logo.py --labeler v4 --levels medium --sweeps 1    # quick look

Each fold leaves one game of --games out (default bench.DEV_TUNING: Metrica 1, 3 and IDSSE 12,
13, 15), fits and tunes on the others, and scores the one left out:
1. the two evidence models are refitted on the training games only (tools/fit_evidence.py's
   fit_models, same rows: every tracker level, ball hidden on degrade.LIVE_MISSING of live frames);
2. the labeler's settings are re-tuned on the training games only (tools/tune.py's descend, same
   grid and objective: mean spellF1 over games and tracker levels), from the shipped PARAMS;
3. the left-out game is scored with those models and settings, at every tracker level.

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
ROBUSTNESS = (0.15, 0.35)      # the fold's models and settings re-scored on the left-out game


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


EVIDENCE = "logistic"          # --evidence: the kind of evidence model fitted in every fold


def fit_on(train):
    """Evidence models fitted on [(name, views)] training games."""
    rows = [(name + "/" + level, gt, evidence.clues(tracking), graded)
            for name, game_views in train for level, gt, _, tracking, graded in game_views]
    dead_model, team_model, _ = fit_evidence.fit_models(rows, EVIDENCE)
    return dead_model, team_model


def features(module, game_views, name):
    """tune.prepare()-style rows for one game, with whatever evidence models are active."""
    return [(name, gt_codes, module.features(tracking), graded)
            for _, _, gt_codes, tracking, graded in game_views]


def run_fold(module, train, test, levels, sweeps, log, workers=1, test_game=None):
    """train: [(name, views)] games to fit and tune on; test: (name, views) the game left out.
    test_game (the loaded game) adds the left-out score at ROBUSTNESS hide levels, with the
    fold's models and settings as they are (nothing is re-fitted or re-tuned for them)."""
    test_name, test_views = test
    t0 = time.time()
    dead_model, team_model = fit_on(train)
    with using_models(dead_model, team_model):
        train_rows = [row for name, views in train for row in features(module, views, name)]
        test_rows = features(module, test_views, test_name)
    old_sweeps, tune.MAX_SWEEPS = tune.MAX_SWEEPS, sweeps
    try:
        with tune.parallel(module, train_rows, workers):
            train_score, params = tune.descend(module, train_rows, module.PARAMS, log=log)
    finally:
        tune.MAX_SWEEPS = old_sweeps
    test_score, per_level = tune.objective(module, test_rows, params)
    shipped_score, shipped_levels = tune.objective(module, test_rows, module.PARAMS)
    changed = {k: v for k, v in params.items() if module.PARAMS.get(k) != v}
    robust = {}
    for hide in ROBUSTNESS if test_game is not None else ():
        with using_models(dead_model, team_model):
            rows = features(module, views(test_game, levels, hide), test_name)
        robust[hide] = tune.objective(module, rows, params)[0]
        del rows
    return {"robust": robust, "train": "+".join(name for name, _ in train), "test": test_name,
            "train_score": train_score,
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
    parser.add_argument("--games", nargs="+", type=int, default=list(bench.DEV_TUNING),
                        help="games to rotate through (default {})".format(
                            " ".join(map(str, bench.DEV_TUNING))))
    parser.add_argument("--leave-out", nargs="+", type=int, default=None,
                        help="only these folds (games left out); default every game")
    parser.add_argument("--workers", type=int, default=1,
                        help="processes scoring the training views in parallel (same result)")
    parser.add_argument("--pool", choices=sorted(bench.POOLS), default=None,
                        help="named pool instead of --games: dev (the default games) or full "
                             "(all ten external matches, incl. the spent held-out ones; "
                             "docs/refinement2_rule.md)")
    parser.add_argument("--evidence", choices=fit_evidence.KINDS, default="logistic",
                        help="evidence models fitted in every fold: logistic (the shipped kind) "
                             "or boosted trees")
    parser.add_argument("--context", nargs="+", type=int, default=None,
                        help="experiment: frames each side for the smoothed clue copies, "
                             "instead of evidence.CONTEXT (refinement 2, step 4)")
    parser.add_argument("--verbose", action="store_true", help="print the descent")
    args = parser.parse_args(argv)
    global EVIDENCE
    EVIDENCE = args.evidence
    if args.context:
        evidence.CONTEXT = tuple(args.context)
    spent = False
    if args.pool:
        args.games, spent = bench.pool(args.pool)
    config.seed_everything()
    log = print if args.verbose else (lambda *a, **k: None)

    games = {n: game for n, game in bench.available(args.games, heldout=spent)}
    if len(games) < 2:
        print("need at least two tuning games downloaded (tools/fetch_metrica.py, fetch_idsse.py)")
        return 1
    loaded = {n: (games[n]["name"], views(games[n], args.levels)) for n in games}
    left_out = args.leave_out or sorted(loaded)
    folds = [([loaded[a] for a in sorted(loaded) if a != b], loaded[b], games[b])
             for b in left_out]

    print("leave-one-game-out, levels {}, ball hidden on {:.0%} of live frames, {} evidence"
          .format(" ".join(args.levels), HIDE_BALL, EVIDENCE)
          + ("" if not args.context else ", context {}".format(evidence.CONTEXT)))
    for name in args.labeler:
        module = importlib.import_module("espadas." + name)
        results = []
        for train, test, test_game in folds:
            r = run_fold(module, train, test, args.levels, args.sweeps, log, args.workers,
                         test_game)
            results.append(r)
            print("\n{}  train {} -> test {}   ({:.0f} s)".format(name, r["train"], r["test"],
                                                                r["seconds"]))
            print("  train (in-sample)            {:.4f}".format(r["train_score"]))
            print("  test, re-tuned settings      {:.4f}   per level {}".format(
                r["test_score"], " ".join("{:.3f}".format(s) for s in r["test_levels"])))
            print("  test, shipped settings       {:.4f}   per level {}".format(
                r["shipped_score"], " ".join("{:.3f}".format(s) for s in r["shipped_levels"])))
            print("  test at {} hidden    {}".format(
                " / ".join("{:.0%}".format(h) for h in ROBUSTNESS),
                " / ".join("{:.4f}".format(r["robust"][h]) for h in ROBUSTNESS)))
            print("  settings moved from shipped  {}".format(r["changed"] or "none"))
        logo = float(np.mean([r["test_score"] for r in results]))
        gap = float(np.mean([r["train_score"] - r["test_score"] for r in results]))
        print("\n{}  LOGO score (mean test, re-tuned) {:.4f}   mean train-test gap {:+.4f}"
              "   shipped settings on test {:.4f}".format(
                  name, logo, gap, float(np.mean([r["shipped_score"] for r in results]))))
        print("{}  LOGO at {} hidden: {}".format(
            name, " / ".join("{:.0%}".format(h) for h in ROBUSTNESS),
            " / ".join("{:.4f}".format(float(np.mean([r["robust"][h] for r in results])))
                       for h in ROBUSTNESS)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
