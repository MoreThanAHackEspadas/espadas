"""Tune the Phase 8.4 X policy (espadas/abstain.py) on the external TUNING games only.

    python tools/tune_abstain.py --labeler v1 v3 v4

Same bench as tools/tune.py (Metrica 1 + 3, all tracker levels, ball hidden on 25% of live
frames). For each labeler: decode once with its own PARAMS, then search abstain.GRID (full grid)
for the best mean macroF1, checking on every trial that the scorer's abstention fill gives
exactly the file without X (so spellF1 cannot change). The best setting is re-scored at
15% / 35% hidden. Never touches the held-out game or our match. Deterministic.

Accept a setting only if it gains more than ~0.01 macroF1 (the official noise level);
otherwise keep abstain.PARAMS[labeler] = None.
"""

import argparse
import itertools
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from espadas import abstain, config, labelers, v2  # noqa: E402
from espadas.external import bench, degrade  # noqa: E402

pm = config.pm
HIDE_BALL = degrade.LIVE_MISSING
ROBUSTNESS = (0.15, 0.35)
NOISE = 0.01


def prepare(name, hide):
    """[(gt codes, labels with ungraded X, present, team_lo, runs)] over tuning games x levels."""
    out = []
    for blur in degrade.TRACKER_LEVELS:
        for _, game in bench.available(bench.TUNING):
            gt, tracking, graded = bench.setup(game, hide, blur)
            lab = np.asarray(labelers.LABELERS[name](tracking), dtype="<U1").copy()
            lab[~graded] = "X"
            f = v2.features(tracking)
            out.append((gt, lab, f["present"], f["team_lo"], pm.valid_runs(gt)))
    return out


def macro(gt, lab):
    return pm.macro_f1(pm.scored_confusion(pm.confusion(gt, pm.label_codes(lab))))


def score(games, p):
    """(mean macroF1, mean share of graded frames set to X); asserts spellF1 is unchanged."""
    m, share = [], []
    for gt, lab, present, team_lo, runs in games:
        out = abstain.apply(lab, present, team_lo, p)
        before = pm.fill_abstentions(pm.label_codes(lab), runs)
        after = pm.fill_abstentions(pm.label_codes(out), runs)
        assert np.array_equal(before, after), "X policy changed what spellF1 sees"
        m.append(macro(gt, out))
        share.append(np.mean((out == "X") & (lab != "X")) / np.mean(gt != pm.X))
    return float(np.mean(m)), float(np.mean(share))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labeler", nargs="+", default=["v1", "v3", "v4"])
    args = parser.parse_args(argv)
    config.seed_everything()
    keys = list(abstain.GRID)
    for name in args.labeler:
        games = prepare(name, HIDE_BALL)
        base, _ = score(games, None)
        best, best_p = base, None
        for values in itertools.product(*abstain.GRID.values()):
            p = dict(zip(keys, values))
            s, _ = score(games, p)
            if s > best + 1e-9:
                best, best_p = s, p
        share = score(games, best_p)[1]
        verdict = "ACCEPT" if best - base > NOISE else "reject (gain below noise)"
        print("{}: macroF1 without X {:.4f}, best {:.4f} (+{:.4f}), X on {:.1%} of graded "
              "frames, {}  {}".format(name, base, best, best - base, share, best_p, verdict))
        for hide in ROBUSTNESS:
            g = prepare(name, hide)
            b0, b1 = score(g, None)[0], score(g, best_p)[0]
            print("    @ {:.0%} live missing: {:.4f} -> {:.4f} (+{:.4f})".format(
                hide, b0, b1, b1 - b0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
