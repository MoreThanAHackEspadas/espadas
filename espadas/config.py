"""Paths, constants and seeds. The one place that knows where things live.

Match constants (frame count, fps, pitch, half boundaries, kickoff anchors) are taken from the
organisers' games.py, vendored unmodified in third_party/mtah2027, so nothing here re-types a
number the organisers already publish.
"""

import os
import random
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFFICIAL_DIR = os.path.join(REPO_ROOT, "third_party", "mtah2027")
if OFFICIAL_DIR not in sys.path:
    sys.path.insert(0, OFFICIAL_DIR)

import games  # noqa: E402  (vendored, needs the path above)
import possession_metrics as pm  # noqa: E402

GAME = games.GAME
_G = games.GAMES[GAME]

N_FRAMES = _G["n_video_frames"]          # 199,319 rows in the submission
FPS = _G["fps"]                          # 25
PITCH = _G["pitch"]                      # (105.3, 68.0) metres

# Graded video-frame blocks per half, inclusive: [(1, 11185, 90651), (2, 110808, 190197)].
HALF_BLOCKS = games.half_video_blocks(GAME)

# Second half is graded from its first frame but only kicks off at the documented anchor
# (FAQ.md: "Those 2,934 frames are graded and all of them are D"). Derived, not typed.
SECOND_HALF_START = HALF_BLOCKS[1][1]
SECOND_HALF_KICKOFF = games.anchor_video_frame(_G["kickoff_2nd"])
PRE_KICKOFF_DEAD = (SECOND_HALF_START, SECOND_HALF_KICKOFF)  # inclusive range, always D

# Kickoff video frame of each half. The first half is graded from its kickoff (DATA.md), so its
# first graded frame is the kickoff; the second is the documented anchor above. (The organisers'
# starter/load_smart_tagging.py uses 113742 for the second, one frame later; that is irrelevant to
# anything read from a window around the kickoff.)
KICKOFFS = (HALF_BLOCKS[0][1], SECOND_HALF_KICKOFF)

TEAM_NAME = "espadas"
SEED = 0

# Where intermediate arrays are cached. Outside version control (.gitignore: cache/).
CACHE_DIR = os.environ.get("ESPADAS_CACHE_DIR") or os.path.join(REPO_ROOT, "cache")


def seed_everything(seed=SEED):
    """Fix every RNG we use. Call once at the start of any entry point.

    PYTHONHASHSEED is deliberately not set here: it only takes effect when the interpreter
    starts, so setting it in-process does nothing. Nothing in the pipeline depends on hash
    order; if that ever changes, set it in the shell before running.
    """
    random.seed(seed)
    np.random.seed(seed)


def half_of_frame(n_frames=N_FRAMES, half_blocks=None):
    """int8 per video frame: 1 or 2 on the graded frames of that half, 0 on ungraded frames.

    half_blocks defaults to this match's HALF_BLOCKS; pass another match's to reuse it there.
    """
    half = np.zeros(n_frames, dtype=np.int8)
    for number, first, last in (HALF_BLOCKS if half_blocks is None else half_blocks):
        half[first:last + 1] = number
    return half


def data_dir(override=None):
    """Root of the organisers' data bundle (contains tactical/ and broadcast/).

    Order: explicit override, then MTAH_DATA_DIR, then ./data in this repo (gitignored).
    """
    return override or os.environ.get("MTAH_DATA_DIR") or os.path.join(REPO_ROOT, "data")


def require(names, override=None, source="tactical"):
    """Paths for the named assets, with the organisers' readable error if any are missing."""
    root = data_dir(override)
    try:
        return games.require(names, GAME, root, source)
    except IOError:
        hint = _video_rename_hint(root, source) if "video" in names else ""
        if hint:
            raise IOError(hint)
        raise


def _video_rename_hint(root, source):
    """The bundle ships videos as e.g. BAR-ATH_tactical-002.mp4; the code expects no suffix."""
    folder = os.path.join(root, source)
    if not os.path.isdir(folder):
        return ""
    expected = "BAR-ATH_{}.mp4".format(source)
    found = [f for f in os.listdir(folder) if f.startswith("BAR-ATH_{}-".format(source))
             and f.endswith(".mp4")]
    if found and not os.path.exists(os.path.join(folder, expected)):
        return ("video is named {} but code expects {} -- rename it:\n  mv '{}' '{}'".format(
            found[0], expected, os.path.join(folder, found[0]), os.path.join(folder, expected)))
    return ""
