"""One command from the data bundle to a validated submission.

    export MTAH_DATA_DIR="/path/to/More Than A Hack 2027 - Initial Challenge - Dataset"
    python -m espadas.run                       # -> out/espadas.csv

Output is deliberately limited to pass/fail, the file path, the SHA-256 and label counts: no
per-frame data is printed.
"""

import argparse
import os
import sys

import numpy as np

from . import abstain, atd, config, labelers, v2, write


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the Team Espadas submission CSV.")
    parser.add_argument("--data-dir", help="data bundle root (default: $MTAH_DATA_DIR)")
    parser.add_argument("--out", default=os.path.join(config.REPO_ROOT, "out",
                                                      config.TEAM_NAME + ".csv"))
    parser.add_argument("--labeler", default="v1", choices=sorted(labelers.LABELERS))
    parser.add_argument("--raw", action="store_true",
                        help="skip finalise() (no forced D window, no X on ungraded frames); "
                             "useful for comparing against the organisers' baseline files")
    parser.add_argument("--no-cache", action="store_true", help="re-parse tracking from scratch")
    parser.add_argument("--abstain", action="store_true",
                        help="X policy on graded frames (abstain.py, Phase 8.4; off by default)")
    args = parser.parse_args(argv)

    config.seed_everything()
    tracking = atd.load_tracking(args.data_dir, use_cache=not args.no_cache)
    labels = labelers.LABELERS[args.labeler](tracking)
    if not args.raw:
        labels = write.finalise(labels)
    if args.abstain:
        f = v2.features(tracking)
        labels = abstain.apply(labels, f["present"], f["team_lo"],
                               abstain.PARAMS.get(args.labeler))

    path = write.write_csv(args.out, labels)
    reason = write.validate(path)
    if reason:
        print("FAIL  {}\n      {}".format(path, reason))
        return 1

    counts = "  ".join("{} {}".format(k, int(np.sum(labels == k))) for k in "HADX")
    print("OK    {}\n      sha256 {}\n      {}".format(path, write.sha256(path), counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
