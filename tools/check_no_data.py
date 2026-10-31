"""Refuse to let match data into git. The repo is public; the data is not redistributable.

    python tools/check_no_data.py            # staged files (used by .githooks/pre-commit)
    python tools/check_no_data.py --all      # every tracked file (used by CI)

Blocks: files over MAX_BYTES, anything named like a match asset, data/media/archive extensions,
and anything under data/ except data/.gitkeep. Exit 1 with the reasons if anything is blocked.
"""

import argparse
import fnmatch
import os
import subprocess
import sys

MAX_BYTES = 500 * 1024

BLOCKED_NAMES = ["BAR-ATH*", "*_atd.*", "*_homography.*", "*_smart_tagging.*",
                 "*possession_gt*", "*Dataset*"]
BLOCKED_EXT = {".csv", ".xml", ".txt", ".mp4", ".mov", ".avi", ".mkv", ".npy", ".npz",
               ".parquet", ".feather", ".pkl", ".pickle", ".h5", ".hdf5", ".zip", ".tar",
               ".gz", ".7z", ".json"}
ALLOWED = ["requirements*.txt", "data/.gitkeep", "third_party/mtah2027/*.py"]


def git_files(all_files):
    cmd = (["git", "ls-files"] if all_files
           else ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"])
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    return [line for line in out.splitlines() if line]


def problems(path):
    name = os.path.basename(path)
    if any(fnmatch.fnmatch(path, pat) for pat in ALLOWED):
        return []
    found = []
    if path.startswith("data/"):
        found.append("inside data/")
    if any(fnmatch.fnmatch(name, pat) for pat in BLOCKED_NAMES):
        found.append("named like a match asset")
    if os.path.splitext(name)[1].lower() in BLOCKED_EXT:
        found.append("data/media extension")
    if os.path.isfile(path) and os.path.getsize(path) > MAX_BYTES:
        found.append("larger than {} KB".format(MAX_BYTES // 1024))
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--all", action="store_true", help="check every tracked file")
    args = parser.parse_args(argv)

    blocked = [(p, why) for p in git_files(args.all) for why in [problems(p)] if why]
    if not blocked:
        return 0
    print("BLOCKED: these files look like match data and must not be committed:")
    for path, why in blocked:
        print("  {}  ({})".format(path, ", ".join(why)))
    print("\nUnstage with: git restore --staged <file>. If a file is genuinely code, "
          "add it to ALLOWED in tools/check_no_data.py.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
