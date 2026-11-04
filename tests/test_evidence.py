"""Phase 5 evidence and v2, on synthetic inputs only."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from espadas import evidence, v2  # noqa: E402

H, A, N = v2.HOME, v2.AWAY, v2.NONE


def _match(n=400, seed=0):
    """Six players (3 a side) moving about a ball that drifts; home spread wider than away."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    ball = np.stack([20 + 0.1 * t, 30 + 5 * np.sin(t / 40)], axis=1)
    players = np.zeros((n, 6, 2))
    offsets = np.array([[1, 0], [-8, 10], [-8, -10], [3, 0], [5, 3], [5, -3]], dtype=float)
    for k in range(6):
        players[:, k] = ball + offsets[k] + rng.normal(0, 0.1, (n, 2))
    # home defends x = 0 at kickoff: put home behind away around frame 0
    players[:10, :3, 0] -= 15
    return {"players": players, "ball": ball, "team_of_slot": np.array([0, 0, 0, 1, 1, 1], np.int8),
            "half_blocks": [(1, 0, 199), (2, 200, n - 1)], "kickoffs": (5, 205),
            "fps": 25, "pitch": (105.0, 68.0)}


def test_smooth_ignores_nan():
    x = np.array([1.0, np.nan, 3.0, np.nan, np.nan])
    out = evidence.smooth(x, 1)
    assert out[0] == 1.0 and out[1] == 2.0 and out[2] == 3.0 and out[3] == 3.0
    assert np.isnan(out[4])


def test_fit_logistic_recovers_sign_and_is_deterministic():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2000, 2))
    y = (x[:, 0] - 0.5 * x[:, 1] + rng.normal(0, 0.5, 2000)) > 0
    bias, w = evidence.fit_logistic(x, y, intercept=True)
    assert w[0] > 0 > w[1]
    assert np.array_equal(w, evidence.fit_logistic(x, y, intercept=True)[1])
    _, w0 = evidence.fit_logistic(x, y, intercept=False)
    assert w0[0] > 0 > w0[1]


def test_team_clues_flip_sign_when_teams_swap():
    tr = _match()
    swapped = dict(tr, team_of_slot=1 - tr["team_of_slot"])
    names, x = evidence.team_design(evidence.clues(tr))
    _, xs = evidence.team_design(evidence.clues(swapped))
    assert np.allclose(x, -xs)
    assert "spread" in names


def test_dead_model_never_reads_whether_the_ball_is_tracked():
    tr = _match()
    names, _ = evidence.ball_design(evidence.clues(tr))
    motion, _ = evidence.motion_design(evidence.clues(tr))
    assert not any("unseen" in n or "gap" in n for n in names + motion)


def test_log_odds_refuses_mismatched_model():
    c = evidence.clues(_match())
    bad = {"names": ["nope"], "bias": 0.0, "weights": [1.0]}
    with pytest.raises(RuntimeError):
        evidence.log_odds(c, {"motion": bad, "ball": bad}, bad)


def test_log_odds_with_fitted_models_runs_on_synthetic_match():
    pytest.importorskip("espadas.evidence_models")
    dead, team = evidence.log_odds(evidence.clues(_match()))
    assert dead.shape == team.shape == (400,)
    assert np.all(np.isfinite(dead)) and np.all(np.isfinite(team))


def test_cusum_ignores_weak_blip_and_backdates_switch():
    lo = np.concatenate([np.full(50, 1.0), np.full(10, -0.5), np.full(50, 1.0),
                         np.full(40, -1.0)])
    out, owner = v2.cusum(lo, cap=1.0, cost=10.0, start_owner=N)
    assert set(out[:110]) == {H}                    # 10 weak frames never reach the cost
    assert owner == A and set(out[110:]) == {A}     # switch dated to where the run began


def test_cusum_carries_owner_without_evidence():
    out, owner = v2.cusum(np.zeros(30), 1.0, 10.0, start_owner=A)
    assert owner == A and set(out) == {A}


def test_v2_decode_is_deterministic():
    f = {"half_blocks": [(1, 0, 199), (2, 200, 399)], "present": np.ones(400, bool),
         "dead_lo": np.where((np.arange(400) > 40) & (np.arange(400) < 180), 5.0, -5.0),
         "team_lo": np.where(np.arange(400) < 300, 1.0, -1.0)}
    labels = v2.decode(f)
    assert set(labels[50:170]) == {"D"}
    assert labels[20] == "H" and labels[250] == "H" and labels[380] == "A"
    assert np.array_equal(labels, v2.decode(f))


def test_ball_less_frames_use_fixed_threshold_not_the_tuned_one():
    n = 400
    present = np.ones(n, bool)
    present[100:300] = False                         # long ball gap
    f = {"half_blocks": [(1, 0, n - 1)], "present": present,
         "dead_lo": np.full(n, -1.0), "team_lo": np.full(n, 1.0)}
    p = {"dead_smooth": 0, "dead_threshold": -2.0, "min_dead": 25, "min_alive": 0}
    assert set(v2.decode(f, p)[100:300]) == {"H"}    # -1 < 0: motion says live, so live
    assert set(v2.decode(f, dict(p, unseen_threshold=None))[100:300]) == {"D"}


CONTROL_NAMES = ["near_any", "near_any~6", "near_any~25", "near_any~75",
                 "ball_speed", "ball_speed~6", "ball_speed~25", "ball_speed~75",
                 "kink", "kink~6", "kink~25", "kink~75"]


def test_control_design_names_and_finite():
    names, x = evidence.control_design(evidence.clues(_match()))
    assert names == CONTROL_NAMES and x.shape == (400, 12) and np.isfinite(x).all()


def test_control_clues_are_nan_without_a_ball_and_log_odds_follow():
    tracking = _match()
    tracking["ball"][100:160] = np.nan
    c = evidence.clues(tracking)
    assert np.isnan(c["near_any"][100:160]).all() and np.isnan(c["kink"][100:160]).all()
    model = {"names": CONTROL_NAMES, "bias": 0.0, "weights": [0.0] * 12}
    lo = evidence.control_log_odds(c, model)
    assert np.isnan(lo[100:160]).all() and np.isfinite(lo[:100]).all()


def test_control_log_odds_refuses_mismatched_model():
    with pytest.raises(RuntimeError):
        evidence.control_log_odds(evidence.clues(_match()),
                                  {"names": ["wrong"], "bias": 0.0, "weights": [0.0]})
