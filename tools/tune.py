"""Tune a labeler's decoding settings on the external TUNING games only, by coordinate descent.

    python tools/tune.py --labeler v2          # prints the search, the best PARAMS, then held-out

Works for any labeler module exposing features(tracking), decode(features, params), PARAMS and
GRID (v2 onwards; v1 has its own tools/tune_v1.py). Same protocol as tune_v1.py: objective is the
mean spellF1 over Metrica 1 and 3 *and* over the light, medium and heavy ball blur
(degrade.BLUR; checklist 6.6: a method must work for any plausible tracker, not one blur level),
with the ball hidden on degrade.LIVE_MISSING of live frames; the chosen settings are re-scored at
15% / 35% hidden. The held-out game is scored once, for the chosen settings only, and never
used to choose (--skip-heldout while developing). Our match is never touched. Deterministic.

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
OBJECTIVE_BLURS = degrade.TRACKER_LEVELS   # light, medium, heavy, smooth (6.6, 6.7)
MAX_SWEEPS = 4


def prepare(module, numbers, hide=HIDE_BALL, blur=degrade.BLUR_DEFAULT):
    out = []
    for _, game in bench.available(numbers):
        gt, tracking, graded = bench.setup(game, hide, blur)
        out.append((game["name"], gt, module.features(tracking), graded))
    return out


def objective(module, games, params):
    scores = [bench.evaluate(gt, module.decode(f, params), graded)["spell_f1_mean"]
              for _, gt, f, graded in games]
    return float(np.mean(scores)), scores


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labeler", required=True, help="module in espadas/, e.g. v2")
    parser.add_argument("--skip-heldout", action="store_true",
                        help="do not score the held-out game (use while developing)")
    args = parser.parse_args(argv)
    module = importlib.import_module("espadas." + args.labeler)
    config.seed_everything()

    tuning = [g for blur in OBJECTIVE_BLURS for g in prepare(module, bench.TUNING, blur=blur)]
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
    for i, ((name, *_), s) in enumerate(zip(tuning, per_game)):
        print("  tune     {:<10} blur {:<6} spellF1 {:.4f}".format(
            name, OBJECTIVE_BLURS[i // len(bench.TUNING)], s))
    checks = [(h, b) for h in ROBUSTNESS for b in OBJECTIVE_BLURS]
    for hide in ROBUSTNESS:
        score = np.mean([objective(module, prepare(module, bench.TUNING, hide, b), params)[0]
                         for b in OBJECTIVE_BLURS])
        print("  tune @ {:.0%} live missing, all blurs: mean spellF1 {:.4f}".format(hide, score))
    if args.skip_heldout:
        return 0
    for hide, blur in [(HIDE_BALL, b) for b in OBJECTIVE_BLURS] + checks:
        for name, gt, f, graded in prepare(module, bench.HELD_OUT, hide, blur):
            r = bench.evaluate(gt, module.decode(f, params), graded)
            print("  HELDOUT  {:<10} @ {:.0%} live missing, blur {:<6} spellF1 {:.4f}  "
                  "macroF1 {:.4f}  n_pred {}  n_gt {}".format(
                      name, hide, blur, r["spell_f1_mean"], r["macro_f1"],
                      r["n_pred_spells"], r["n_gt_spells"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
