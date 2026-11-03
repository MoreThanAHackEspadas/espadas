"""How many frames differ between two submission CSVs. Prints counts only.

    python tools/compare_csv.py out/raw.csv "$MTAH_OFFICIAL_DIR/evaluation/results/baseline_smooth50.csv"

MTAH_OFFICIAL_DIR is your clone of the official challenge repository (see README, "Local layout").
"""

import sys

import pandas as pd


def main(a, b):
    la = pd.read_csv(a)["label"].astype(str).str.strip().str.upper().to_numpy()
    lb = pd.read_csv(b)["label"].astype(str).str.strip().str.upper().to_numpy()
    if len(la) != len(lb):
        print("different lengths: {} vs {}".format(len(la), len(lb)))
        return 1
    n = int((la != lb).sum())
    print("{} of {} frames differ".format(n, len(la)))
    return 0 if n == 0 else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
