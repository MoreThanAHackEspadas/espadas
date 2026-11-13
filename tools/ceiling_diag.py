"""Diagnostic (architecture review): where does the shipped labeler lose spellF1?

    python tools/ceiling_diag.py --workers 16     # v4 on bench.DEV_TUNING, shipped models

For every game and tracker level, the shipped prediction is scored as it is and with one part
replaced by the reference ("oracle"), and the reference itself is scored after simple damage.
The gap to each oracle is an upper bound on what fixing that part alone could gain:

  shipped        the labeler's own output
  dead oracle    reference dead / live; the labeler's team on live frames
  team oracle    the labeler's dead / live; reference team where both say live
  cuts oracle    reference possessions; each one's team by the labeler's majority vote
  no < 2 s       the reference with possessions under 2 s merged into what came before
  no < 5 s       same, under 5 s
  lag 0.5 s      the reference delayed by 0.5 s
  lag 1 s        the reference delayed by 1 s

The models and settings were fitted on these games: a description, not a test score. External
tuning games only. Nothing is written. Deterministic.
"""

import argparse
import multiprocessing
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from espadas import config, labelers  # noqa: E402
from espadas.external import bench, degrade  # noqa: E402

pm = config.pm
NAMES = ("shipped", "dead oracle", "team oracle", "cuts oracle", "no < 2 s", "no < 5 s",
         "lag 0.5 s", "lag 1 s")


def _runs(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def _carry(labels, known):
    """labels with the frames where `known` is False taking the last known label (first known
    one at the head)."""
    index = np.maximum.accumulate(np.where(known, np.arange(len(labels)), -1))
    first = np.flatnonzero(known)
    index[index < 0] = first[0] if len(first) else 0
    return labels[index]


def variants(gt, pred, graded, fps):
    live_gt, live_pred = np.isin(gt, ["H", "A"]), np.isin(pred, ["H", "A"])
    out = [pred]
    dead = np.where(live_gt, _carry(pred, live_pred), gt)
    out.append(dead)
    out.append(np.where(live_gt & live_pred, gt, pred))
    cuts = gt.copy()
    for code in "HA":
        for a, b in _runs(gt == code):
            home, away = np.sum(pred[a:b] == "H"), np.sum(pred[a:b] == "A")
            cuts[a:b] = "H" if home > away else "A" if away > home else code
    out.append(cuts)
    for seconds in (2, 5):
        merged = gt.copy()
        for code in "HA":
            for a, b in _runs(gt == code):
                if b - a < seconds * fps and a > 0 and graded[a - 1]:
                    merged[a:b] = merged[a - 1]
        out.append(merged)
    for seconds in (0.5, 1.0):
        k = int(round(seconds * fps))
        out.append(np.where(graded, np.concatenate([gt[:k], gt[:-k]]), gt))
    return out


def one(job):
    number, level = job
    (_, game), = bench.available([number])
    codes, tracking, graded = bench.setup(game, degrade.LIVE_MISSING, level)
    gt = np.array(pm.LABELS, dtype="<U1")[codes]
    pred = np.asarray(labelers.LABELERS["v4"](tracking), dtype="<U1").copy()
    pred[~graded] = "X"
    rows = [bench.evaluate(codes, v, graded) for v in
            variants(gt, pred, graded, tracking.get("fps", config.FPS))]
    return (game["name"], level, [r["spell_f1_mean"] for r in rows],
            rows[0]["n_pred_spells"], rows[0]["n_gt_spells"],
            [rows[0]["spell_f1"][t][k] for t in pm.SPELL_IOUS for k in ("precision", "recall")])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--games", nargs="+", type=int, default=list(bench.DEV_TUNING))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    config.seed_everything()
    jobs = [(n, level) for n in args.games for level in degrade.TRACKER_LEVELS]
    if args.workers > 1:
        with multiprocessing.get_context("fork").Pool(args.workers) as pool:
            results = pool.map(one, jobs)
    else:
        results = [one(j) for j in jobs]
    print("{:<10}{:<8}".format("game", "level") + "".join("{:>13}".format(n) for n in NAMES)
          + "{:>12}".format("pred / ref"))
    for name, level, scores, n_pred, n_gt, _ in results:
        print("{:<10}{:<8}".format(name, level) + "".join("{:>13.4f}".format(s) for s in scores)
              + "{:>7}/{:<5}".format(n_pred, n_gt))
    print("{:<18}".format("MEAN") + "".join(
        "{:>13.4f}".format(s) for s in np.mean([r[2] for r in results], axis=0)))
    pr = np.mean([r[5] for r in results], axis=0).reshape(-1, 2)
    print("\nshipped, mean precision / recall by IoU threshold:")
    for t, (p, r) in zip(pm.SPELL_IOUS, pr):
        print("  IoU {:.1f}   precision {:.3f}   recall {:.3f}".format(t, p, r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
