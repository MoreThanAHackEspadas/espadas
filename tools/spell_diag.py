"""Diagnostic (refinement 2, step 5): which reference possessions does a labeler miss?

    python tools/spell_diag.py                    # v4 on bench.DEV_TUNING, shipped models

Reference and predicted possessions are matched as the official metric does (one to one, same
team, IoU >= the headline 0.5), then split by length. Per length bin, pooled over the games and
tracker levels: how many reference possessions were found (recall), how many predicted ones are
real (precision). The models and settings are the shipped ones, which were fitted on these
games: a description of the errors, not a test score. External tuning games only. Nothing is
written. Deterministic.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from espadas import config, labelers  # noqa: E402
from espadas.external import bench, degrade  # noqa: E402

pm = config.pm
BINS = ((0, 2, "< 2 s"), (2, 5, "2-5 s"), (5, 15, "5-15 s"), (15, 1e9, "> 15 s"))


def spells(game, labeler, hide, level):
    """(reference spells, predicted spells, matched-reference flags, matched-predicted flags)."""
    gt_codes, tracking, graded = bench.setup(game, hide, level)
    pred = np.asarray(labelers.LABELERS[labeler](tracking), dtype="<U1").copy()
    pred[~graded] = "X"
    runs = pm.valid_runs(gt_codes)
    filled = pm.fill_abstentions(pm.label_codes(pred), runs)
    gt_spells = pm.possession_spells(gt_codes, runs)
    pred_spells = pm.possession_spells(filled, runs)
    gt_hit = np.zeros(len(gt_spells), dtype=bool)
    pred_hit = np.zeros(len(pred_spells), dtype=bool)
    for iou, i, j in pm.spell_overlaps(gt_spells, pred_spells):
        if iou < pm.SPELL_IOU_HEADLINE:
            break
        if not gt_hit[i] and not pred_hit[j]:
            gt_hit[i] = pred_hit[j] = True
    return gt_spells, pred_spells, gt_hit, pred_hit, tracking.get("fps", config.FPS)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labeler", default="v4", choices=sorted(labelers.LABELERS))
    parser.add_argument("--games", nargs="+", type=int, default=list(bench.DEV_TUNING))
    parser.add_argument("--levels", nargs="+", default=list(degrade.TRACKER_LEVELS))
    args = parser.parse_args(argv)
    config.seed_everything()
    counts = {level: np.zeros((len(BINS), 4)) for level in list(args.levels) + ["all"]}
    for _, game in bench.available(args.games):
        for level in args.levels:
            gt_spells, pred_spells, gt_hit, pred_hit, fps = spells(
                game, args.labeler, degrade.LIVE_MISSING, level)
            for which, (group, hit) in enumerate(((gt_spells, gt_hit), (pred_spells, pred_hit))):
                seconds = np.array([(b - a) / float(fps) for a, b, _ in group])
                for k, (low, high, _) in enumerate(BINS):
                    inside = (seconds >= low) & (seconds < high)
                    for key in (level, "all"):
                        counts[key][k, 2 * which] += inside.sum()
                        counts[key][k, 2 * which + 1] += hit[inside].sum()
    for key, c in counts.items():
        print("\n{} at IoU {}, level {}".format(args.labeler, pm.SPELL_IOU_HEADLINE, key))
        print("{:<8}{:>10}{:>9}{:>9}{:>11}{:>9}{:>11}".format(
            "length", "reference", "found", "recall", "predicted", "real", "precision"))
        for (_, _, label), row in zip(BINS + ((0, 0, "all"),), list(c) + [c.sum(axis=0)]):
            print("{:<8}{:>10.0f}{:>9.0f}{:>9.3f}{:>11.0f}{:>9.0f}{:>11.3f}".format(
                label, row[0], row[1], row[1] / max(row[0], 1), row[2], row[3],
                row[3] / max(row[2], 1)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
