"""Phase 6 touches/control and v3, on synthetic inputs only."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from espadas import touches, v3  # noqa: E402

TEAMS = np.array([0, 0, 1], dtype=np.int8)


def _pass_play(n=200):
    """Home slot 0 passes to home slot 1 at frame 50; away slot 2 deflects it at frame 120."""
    ball = np.zeros((n, 2))
    ball[:50] = [10.0, 30.0]
    ball[50:120] = [[10.0 + 0.4 * (t - 50), 30.0] for t in range(50, 120)]
    ball[120:] = [[38.0, 30.0 + 0.4 * (t - 120)] for t in range(120, n)]
    players = np.zeros((n, 3, 2))
    players[:, 0] = [10.5, 30.0]                     # passer
    players[:, 1] = ball + [0.5, 0.0]                # stays with the ball after 50 (receiver)
    players[:50, 1] = [30.0, 40.0]
    players[:, 2] = [38.3, 30.0]                     # defender at the deflection point
    return ball, players


def test_detect_finds_kinks_near_players_with_team():
    ball, players = _pass_play()
    near, slot, _ = touches.nearest_slot(players, ball)
    t = touches.detect(touches.kink(ball, 25, touches.KINK_HALF), near, slot, TEAMS,
                       kink_threshold=3.0, touch_reach=1.0)
    frames = list(t["frame"])
    assert any(abs(f - 50) <= 2 for f in frames)
    assert any(abs(f - 120) <= 2 for f in frames)
    for f, team in zip(t["frame"], t["team"]):
        assert team == (0 if f < 100 else 1)


def test_detect_ignores_kink_with_nobody_near():
    ball, players = _pass_play()
    players[:, 2] = [80.0, 10.0]                     # defender far away
    players[100:, 1] = [20.0, 50.0]                  # receiver no longer with the ball
    near, slot, _ = touches.nearest_slot(players, ball)
    t = touches.detect(touches.kink(ball, 25, touches.KINK_HALF), near, slot, TEAMS, 3.0, 1.0)
    assert not any(abs(f - 120) <= 2 for f in t["frame"])


def test_confirmed_by_chain_and_by_hold():
    t = {"frame": np.array([10, 30, 200]), "slot": np.array([0, 1, 2]),
         "team": np.array([0, 0, 1], np.int8), "strength": np.ones(3)}
    dist = np.full((300, 3), 9.0)
    dist[201:240, 2] = 0.5                           # slot 2 keeps the ball after its touch
    ok = touches.confirmed(t, dist, chain_gap=25, hold_reach=1.0, hold_frames=25)
    assert list(ok) == [True, False, True]           # chain / nothing follows / hold


def test_pulses_signed_by_team():
    t = {"frame": np.array([5, 9]), "slot": np.array([0, 2]),
         "team": np.array([0, 1], np.int8), "strength": np.ones(2)}
    p = touches.pulses(12, t, np.array([True, False]), 40.0, 5.0)
    assert p[5] == 40.0 and p[9] == -5.0 and p.sum() == 35.0


def test_v3_confirmed_touch_switches_and_is_deterministic():
    n = 400
    f = {"half_blocks": [(1, 0, 199), (2, 200, n - 1)], "present": np.ones(n, bool),
         "dead_lo": np.full(n, -9.0), "team_lo": np.full(n, 0.2),
         "kink": np.zeros(n), "near": np.full(n, 0.5), "slot": np.zeros(n, np.int64),
         "dist": np.full((n, 3), 9.0), "team_of_slot": TEAMS}
    f["kink"][[300, 310]] = 10.0                     # two away touches in a row = control
    f["slot"][[300, 310]] = 2
    labels = v3.decode(f)
    assert labels[250] == "H" and labels[350] == "A"
    assert np.array_equal(labels, v3.decode(f))
