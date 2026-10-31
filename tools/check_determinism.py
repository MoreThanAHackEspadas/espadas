"""Build the submission twice from scratch and check the bytes are identical.

    python tools/check_determinism.py

Both runs bypass the tracking cache, so this is the slow, honest version of "a clean rebuild
gives the same file". Prints only hashes.
"""

import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from espadas import write  # noqa: E402


def build(out, cache_dir):
    env = dict(os.environ, ESPADAS_CACHE_DIR=cache_dir)
    subprocess.run([sys.executable, "-m", "espadas.run", "--no-cache", "--out", out],
                   cwd=ROOT, env=env, check=True)
    return write.sha256(out)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        a = build(os.path.join(tmp, "a.csv"), os.path.join(tmp, "cache_a"))
        b = build(os.path.join(tmp, "b.csv"), os.path.join(tmp, "cache_b"))
    print("run 1 sha256 {}\nrun 2 sha256 {}".format(a, b))
    if a != b:
        print("NOT DETERMINISTIC")
        return 1
    print("deterministic")
    return 0


if __name__ == "__main__":
    sys.exit(main())
