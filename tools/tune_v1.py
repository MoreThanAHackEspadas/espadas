"""Tune v1's settings on the external TUNING games only (Metrica 1 and 3), by coordinate descent.

    python tools/tune_v1.py                   # prints the search and the best PARAMS

Objective: mean spellF1 (the ranking metric) over the tuning games, with the ball hidden on
degrade.LIVE_MISSING of live frames (see degrade.py for why live, not overall). The chosen
settings are also scored at 0.15 and 0.35 live missing, as a robustness check. Each setting is swept over a small physically sensible grid while the others are
held, repeated until nothing improves. The held-out game (Metrica 2) is not scored: that happens
once, at selection, in tools/selection.py. --heldout scores it here for the chosen settings
(never used to choose). Our match is never touched. Deterministic.

Note: this tunes on the default tracker level only (degrade.BLUR_DEFAULT), not on the four
levels tools/tune.py uses; v1's PARAMS date from 3 Nov, before the bench gained ball blur and
player error, and were not re-tuned on the final bench.

Copy the printed PARAMS into espadas/v1.py.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from espadas import config, v1  # noqa: E402
from espadas.external import bench, degrade  # noqa: E402

GRID = {
    "out_margin": [0.0, 0.5, 1.0, 2.0, 3.0],
    "still_speed": [0.5, 1.0, 2.0, 3.0, 4.0],
    "walk_speed": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
    "min_dead": [0, 12, 25, 50, 75, 100, 150],
    "min_alive": [0, 25, 50, 100, 150, 250],
    "reach": [0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    "window": [6, 12, 25, 50, 75, 100, 150],
    "switch_share": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
}
HIDE_BALL = degrade.LIVE_MISSING
ROBUSTNESS = (0.15, 0.35)
MAX_SWEEPS = 4


def prepare(numbers, hide=HIDE_BALL, heldout=False):
    out = []
    for n, game in bench.available(numbers, heldout=heldout):
        gt, tracking, graded = bench.setup(game, hide)
        out.append((game["name"], gt, v1.features(tracking), graded))
    return out


def objective(games, params):
    scores = [bench.evaluate(gt, v1.decode(f, params), graded)["spell_f1_mean"]
              for _, gt, f, graded in games]
    return float(np.mean(scores)), scores


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--heldout", action="store_true",
                        help="also score the held-out game for the chosen settings (off by "
                             "default: tools/selection.py scores it once, at selection)")
    args = parser.parse_args(argv)
    config.seed_everything()
    tuning = prepare(bench.TUNING)
    params = dict(v1.PARAMS)
    best, _ = objective(tuning, params)
    print("start   {:.4f}  {}".format(best, params))

    for sweep in range(1, MAX_SWEEPS + 1):
        improved = False
        for name, values in GRID.items():
            for value in values:
                trial = dict(params, **{name: value})
                score, _ = objective(tuning, trial)
                if score > best + 1e-9:
                    best, params, improved = score, trial, True
            print("sweep {} {:<13} -> {:<6} mean spellF1 {:.4f}".format(
                sweep, name, params[name], best))
        if not improved:
            break

    _, per_game = objective(tuning, params)
    print("\nPARAMS = {}".format(params))
    for (name, *_), s in zip(tuning, per_game):
        print("  tune     {:<10} spellF1 {:.4f}".format(name, s))
    for hide in ROBUSTNESS:
        score, per = objective(prepare(bench.TUNING, hide), params)
        print("  tune @ {:.0%} live missing: mean spellF1 {:.4f}".format(hide, score))
    if not args.heldout:
        return 0
    for hide in (HIDE_BALL,) + ROBUSTNESS:
        for name, gt, f, graded in prepare(bench.HELD_OUT, hide, heldout=True):
            r = bench.evaluate(gt, v1.decode(f, params), graded)
            print("  HELDOUT  {:<10} @ {:.0%} live missing  spellF1 {:.4f}  macroF1 {:.4f}  "
                  "n_pred {}  n_gt {}".format(name, hide, r["spell_f1_mean"], r["macro_f1"],
                                              r["n_pred_spells"], r["n_gt_spells"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
