"""External (non-BAR-ATH) games used to tune and validate settings.

Per the organisers' clarification, every parameter is tuned on external data only. The data
lives outside this repo under MTAH_EXTERNAL_DIR (default: ../external_data next to the repo).

The agreed split (checklist 4.4) lives here so the game readers and the bench share one rule:
tune on Games 1 and 3, hold out Game 2 for the final comparison only. The readers refuse a
held-out game unless the caller passes heldout=True, before any file is opened.
"""

import os

from .. import config

TUNING, HELD_OUT = (1, 3), (2,)


class HeldOutError(RuntimeError):
    """A held-out game was asked for without the explicit opt-in (heldout=True)."""


def check_split(numbers, heldout=False):
    """Raise HeldOutError if `numbers` names a held-out game and `heldout` is not set."""
    asked = sorted(set(numbers) & set(HELD_OUT))
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
