"""Tune a labeler's decoding settings on the external TUNING games only, by coordinate descent.

    python tools/tune.py --labeler v2          # prints the search, the best PARAMS, then held-out

Works for any labeler module exposing features(tracking), decode(features, params), PARAMS and
GRID (v2 onwards; v1 has its own tools/tune_v1.py). Same protocol as tune_v1.py: objective is the
mean spellF1 over Metrica 1 and 3 with the ball hidden on degrade.LIVE_MISSING of live frames;
the chosen settings are re-scored at 15% / 35%; the held-out game is scored once, for the chosen
settings only, and never used to choose. Our match is never touched. Deterministic.

Copy the printed PARAMS into the labeler module.
"""

import argparse
import importlib
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from espadas import config  # noqa: E402
from espadas.external import bench, degrade  # noqa: E402

HIDE_BALL = degrade.LIVE_MISSING
ROBUSTNESS = (0.15, 0.35)
MAX_SWEEPS = 4


def prepare(module, numbers, hide=HIDE_BALL):
    out = []
    for _, game in bench.available(numbers):
        gt, tracking, graded = bench.setup(game, hide)
        out.append((game["name"], gt, module.features(tracking), graded))
    return out


def objective(module, games, params):
    scores = [bench.evaluate(gt, module.decode(f, params), graded)["spell_f1_mean"]
              for _, gt, f, graded in games]
    return float(np.mean(scores)), scores


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labeler", required=True, help="module in espadas/, e.g. v2")
    args = parser.parse_args(argv)
    module = importlib.import_module("espadas." + args.labeler)
    config.seed_everything()

    tuning = prepare(module, bench.TUNING)
    params = dict(module.PARAMS)
    best, _ = objective(module, tuning, params)
    print("start   {:.4f}  {}".format(best, params))
    for sweep in range(1, MAX_SWEEPS + 1):
        improved = False
        for name, values in module.GRID.items():
            for value in values:
                trial = dict(params, **{name: value})
                score, _ = objective(module, tuning, trial)
                if score > best + 1e-9:
                    best, params, improved = score, trial, True
            print("sweep {} {:<15} -> {:<6} mean spellF1 {:.4f}".format(
                sweep, name, params[name], best))
        if not improved:
            break

    _, per_game = objective(module, tuning, params)
    print("\nPARAMS = {}".format(params))
    for (name, *_), s in zip(tuning, per_game):
        print("  tune     {:<10} spellF1 {:.4f}".format(name, s))
    for hide in ROBUSTNESS:
        score, _ = objective(module, prepare(module, bench.TUNING, hide), params)
        print("  tune @ {:.0%} live missing: mean spellF1 {:.4f}".format(hide, score))
    for hide in (HIDE_BALL,) + ROBUSTNESS:
        for name, gt, f, graded in prepare(module, bench.HELD_OUT, hide):
            r = bench.evaluate(gt, module.decode(f, params), graded)
            print("  HELDOUT  {:<10} @ {:.0%} live missing  spellF1 {:.4f}  macroF1 {:.4f}  "
                  "n_pred {}  n_gt {}".format(name, hide, r["spell_f1_mean"], r["macro_f1"],
                                              r["n_pred_spells"], r["n_gt_spells"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
