"""v1 on synthetic inputs only."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from espadas import v1  # noqa: E402

H, A, N = v1.HOME, v1.AWAY, v1.NONE


def test_hysteresis_ignores_short_flicker_and_backdates_switch():
    ev = np.array([H] * 20 + [A] * 3 + [H] * 20 + [N] * 5 + [A] * 30, dtype=np.int8)
    out, owner = v1.hysteresis(ev, window=10, share=0.7, start_owner=N)
    assert owner == A
    # A reaches 7 of 10 at frame 54; the switch is dated back to after H's last evidence (42),
    # but by at most one window, so it lands at 45.
    assert set(out[:45]) == {H}                     # the 3-frame A blip is not a possession
    assert set(out[45:]) == {A}


def test_hysteresis_carries_owner_when_no_evidence():
    out, owner = v1.hysteresis(np.full(30, N, dtype=np.int8), 10, 0.7, start_owner=A)
    assert owner == A and set(out) == {A}


def _features(n=400, slow=(100, 250)):
    """Two players, home at the ball until frame 200, then away. Players stop in `slow`."""
    t = np.arange(n)
    ball = np.stack([10 + 0.2 * t, np.full(n, 30.0)], axis=1)
    players = np.zeros((n, 2, 2))
    players[:, 0] = ball + [0.5, 0]
    players[:, 1] = ball + [5.0, 0]
    players[200:, 0] = ball[200:] + [5.0, 0]
    players[200:, 1] = ball[200:] + [0.5, 0]
    a, b = slow
    players[a:b] = players[a]                       # everyone standing still
    ball[a:b] = ball[a]
    tracking = {"players": players, "ball": ball, "team_of_slot": np.array([0, 1], np.int8),
                "half_blocks": [(1, 0, 199), (2, 200, n - 1)], "kickoffs": (0, 200),
                "fps": 25, "pitch": (105.0, 68.0)}
    return tracking


def test_decode_marks_stoppage_dead_and_teams_alive():
    labels = v1.v1(_features(), {"min_dead": 25})
    assert labels[50] == "H" and labels[350] == "A"
    assert set(labels[115:185]) == {"D"}            # players and ball still -> dead
    assert labels[300] == "A"


def test_decode_is_deterministic():
    tr = _features()
    assert np.array_equal(v1.v1(tr), v1.v1(tr))
