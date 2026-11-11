"""tools/logo.py: the per-fold evidence models are used inside the block and only there."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from espadas import evidence  # noqa: E402
import logo  # noqa: E402


def test_using_models_swaps_and_restores():
    original = evidence._fitted
    dead, team = {"fold": "dead"}, {"fold": "team"}
    with logo.using_models(dead, team):
        assert evidence._fitted() == (dead, team)
    assert evidence._fitted is original


def test_using_models_restores_after_an_error():
    original = evidence._fitted
    try:
        with logo.using_models({}, {}):
            raise ValueError
    except ValueError:
        pass
    assert evidence._fitted is original


def test_logo_never_reads_the_held_out_game(monkeypatch):
    seen = []
    monkeypatch.setattr(logo.bench, "available", lambda numbers, heldout=False:
                        seen.append((list(numbers), heldout)) or [])
    assert logo.main(["--labeler", "v4"]) == 1        # nothing downloaded -> clean exit
    assert seen == [(list(logo.bench.DEV_TUNING), False)]
