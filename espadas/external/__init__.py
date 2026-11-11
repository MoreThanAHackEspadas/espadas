"""External (non-BAR-ATH) games used to tune and validate settings.

Per the organisers' clarification, every parameter is tuned on external data only. The data
lives outside this repo under MTAH_EXTERNAL_DIR (default: ../external_data next to the repo).

The agreed split (checklist 4.4) lives here so the game readers and the bench share one rule:
tune on Games 1 and 3, hold out Game 2 for the final comparison only; IDSSE has its own split
(IDSSE_TUNING / IDSSE_HELD_OUT), guarded the same way. The readers refuse a
held-out game unless the caller passes heldout=True, before any file is opened.
"""

import os

from .. import config

TUNING, HELD_OUT = (1, 3), (2,)

# DFL / Sportec IDSSE matches (espadas/external/idsse.py, game numbers 11-17), split on 11 Nov 2026
# (branch `refinement`) before any of them was scored: one Bundesliga match on each side, the
# rest 2. Bundesliga. Held-out ones are scored only for a pre-registered comparison, like Game 2.
IDSSE_TUNING, IDSSE_HELD_OUT = (12, 13, 15), (11, 14, 16, 17)
ALL_HELD_OUT = HELD_OUT + IDSSE_HELD_OUT
# Tuning pool: every game that may be fitted or tuned on (tools/tune.py, fit_evidence.py,
# logo.py and scoreboard.py default to it). The submitted evidence models and v4 settings were
# fitted on it on 11 Nov.
DEV_TUNING = TUNING + IDSSE_TUNING


class HeldOutError(RuntimeError):
    """A held-out game was asked for without the explicit opt-in (heldout=True)."""


def check_split(numbers, heldout=False):
    """Raise HeldOutError if `numbers` names a held-out game and `heldout` is not set."""
    asked = sorted(set(numbers) & set(ALL_HELD_OUT))
    if asked and not heldout:
        raise HeldOutError(
            "game {} is held out (checklist 4.4): it is scored once, for the final candidates, "
            "by tools/selection.py. Loading it anywhere else needs an explicit opt-in "
            "(heldout=True; on a command line, --heldout).".format(
                " ".join(str(n) for n in asked)))


def external_dir():
    return os.environ.get("MTAH_EXTERNAL_DIR") or os.path.join(
        os.path.dirname(config.REPO_ROOT), "external_data")


def metrica_dir():
    return os.path.join(external_dir(), "metrica")


def idsse_dir():
    """DFL / Sportec IDSSE open data (tools/fetch_idsse.py), outside the repo."""
    return os.path.join(external_dir(), "idsse")
