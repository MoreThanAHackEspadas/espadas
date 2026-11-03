"""Diagnostic only (checklist 4.7): what a labeler claims about OUR match, without any truth.

    python tools/self_consistency.py --labeler v0

Prints possession count, mean duration, home share and label mix on graded frames, next to the
same numbers for the external reference labels, so implausible output (900 possessions of 5 s)
is visible early. Never use these numbers to choose settings: tuning happens on external games
only (organisers' clarification). Prints aggregates only, no per-frame data.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from espadas import atd, config, labelers, write  # noqa: E402
from espadas.external import bench  # noqa: E402

pm = config.pm


def stats(labels):
    codes = pm.label_codes(labels)
    runs = pm.valid_runs(codes)
    s = pm.possession_stats(codes, runs)
    graded = codes != pm.X
    mix = {k: float(np.mean(codes[graded] == getattr(pm, k))) for k in "HAD"}
    return s["n_spells"], s["mean_duration_s"], s["home_share"], mix, int(graded.sum())


def line(name, labels):
    n, mean, share, mix, graded = stats(labels)
    per90 = n / (graded / config.FPS / 60) * 90
    print("{:<22}{:>7}{:>9.0f}{:>9.1f}{:>8.2f}{:>7.0%}{:>7.0%}{:>7.0%}".format(
        name, n, per90, mean, share, mix["H"], mix["A"], mix["D"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labeler", default="v0", choices=sorted(labelers.LABELERS))
    parser.add_argument("--data-dir")
    args = parser.parse_args(argv)
    config.seed_everything()

    print("{:<22}{:>7}{:>9}{:>9}{:>8}{:>7}{:>7}{:>7}".format(
        "", "spells", "per 90'", "mean s", "home", "H", "A", "D"))
    for n, game in bench.available([1, 2, 3]):
        line("ref " + game["name"], bench.reference(game)[0])
    tracking = atd.load_tracking(args.data_dir)
    line("{} on BAR-ATH".format(args.labeler),
         write.finalise(labelers.LABELERS[args.labeler](tracking)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
