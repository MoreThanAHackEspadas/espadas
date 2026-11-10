"""One command from the data bundle to a validated submission.

    export MTAH_DATA_DIR="/path/to/More Than A Hack 2027 - Initial Challenge - Dataset"
    python -m espadas.run                       # -> out/espadas.csv

The submitted file is the no-flag build. Whether it has X on graded frames is decided by
abstain.PARAMS[labeler] alone (abstain.resolve); --abstain / --no-abstain force the X policy on
or off for experiments.

Output is deliberately limited to pass/fail, the file path, the SHA-256 and label counts: no
per-frame data is printed.
"""

import argparse
import os
import sys

import numpy as np

from . import abstain, atd, config, labelers, v2, write

# The labeler of the no-flag build, i.e. the submission: v4, chosen on 10 Nov by the
# pre-registered rule (docs/decision_rule.md, docs/selection.md; checklist 9.3 / 11.1). Changed
# only together with tests/test_run.py::test_default_labeler_is_v4 and the README.
DEFAULT_LABELER = "v4"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the Team Espadas submission CSV.")
    parser.add_argument("--data-dir", help="data bundle root (default: $MTAH_DATA_DIR)")
    parser.add_argument("--out", default=os.path.join(config.REPO_ROOT, "out",
                                                      config.TEAM_NAME + ".csv"))
    parser.add_argument("--labeler", default=DEFAULT_LABELER, choices=sorted(labelers.LABELERS))
    parser.add_argument("--raw", action="store_true",
                        help="skip finalise() (no forced D window, no X on ungraded frames); "
                             "useful for comparing against the organisers' baseline files")
    parser.add_argument("--no-cache", action="store_true", help="re-parse tracking from scratch")
    parser.add_argument("--abstain", action="store_true",
                        help="force the X policy on graded frames on (abstain.py, Phase 8.4): "
                             "abstain.PARAMS[labeler], or abstain.SETTING if that is None. "
                             "Without a flag, abstain.PARAMS[labeler] alone decides")
    parser.add_argument("--no-abstain", action="store_true",
                        help="force the X policy off, whatever abstain.PARAMS says "
                             "(wins over --abstain)")
    args = parser.parse_args(argv)

    config.seed_everything()
    tracking = atd.load_tracking(args.data_dir, use_cache=not args.no_cache)
    labels = labelers.LABELERS[args.labeler](tracking)
    if not args.raw:
        labels = write.finalise(labels)
    force = False if args.no_abstain else True if args.abstain else None
    setting = abstain.resolve(args.labeler, force)
    if setting is not None:
        f = v2.features(tracking)
        labels = abstain.apply(labels, f["present"], f["team_lo"], setting)

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
