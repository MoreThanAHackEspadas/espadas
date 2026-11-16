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


def test_boosted_export_matches_sklearn_and_is_symmetric():
    pytest.importorskip("sklearn")
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools"))
    import fit_evidence
    from sklearn.ensemble import HistGradientBoostingClassifier

    rng = np.random.default_rng(0)
    x = rng.normal(size=(6000, 4))
    y = (x[:, 0] * x[:, 1] + 0.5 * x[:, 2] + rng.normal(scale=0.3, size=6000)) > 0
    params = dict(fit_evidence.BOOST, max_iter=20, min_samples_leaf=20)
    model = HistGradientBoostingClassifier(**params).fit(x, y)
    bias, trees = fit_evidence.export_trees(model)
    exported = {"names": list("abcd"), "bias": bias, "symmetric": False, "trees": trees}
    q = rng.normal(size=(500, 4))
    assert np.allclose(evidence.apply_model(q, exported), model.decision_function(q), atol=1e-6)

    old_step, fit_evidence.BOOST_STEP = fit_evidence.BOOST_STEP, 1
    try:
        team = fit_evidence.fit_boosted(list("abcd"), x, y, symmetric=True)
    finally:
        fit_evidence.BOOST_STEP = old_step
    assert team["symmetric"] and len(team["trees"]) == fit_evidence.BOOST["max_iter"]
    assert np.array_equal(evidence.apply_model(q, team), -evidence.apply_model(-q, team))


def test_player_design_swaps_with_the_teams():
    tracking = _match()
    tracking["ball"] = tracking["ball"].copy()
    tracking["ball"][120:150] = np.nan                 # a ball gap: last-seen clues take over
    names, x = evidence.player_design(evidence.clues(tracking))
    swapped = dict(tracking, team_of_slot=1 - tracking["team_of_slot"])
    names2, x2 = evidence.player_design(evidence.clues(swapped))
    perm, sign = evidence.player_swap(names)
    assert names == names2 and len(set(names)) == len(names)
    assert np.isfinite(x).all()
    assert np.allclose(x2, x[:, perm] * np.asarray(sign))
    assert sorted(perm) == list(range(len(names)))


def test_since_and_shift():
    mask = np.array([0, 1, 0, 0, 1, 0], dtype=bool)
    assert list(evidence._since(mask, clip=2)) == [2, 0, 1, 2, 0, 1]
    x = np.array([1.0, 2.0, np.nan, 4.0])
    assert list(evidence._shift(x, 1, 9.0)) == [2.0, 9.0, 4.0, 9.0]
    assert list(evidence._shift(x, -1, 9.0)) == [9.0, 1.0, 2.0, 9.0]


def test_player_model_is_antisymmetric_under_a_team_swap():
    pytest.importorskip("sklearn")
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools"))
    import fit_evidence

    names = ["s_a", "h_d", "a_d", "speed"]
    swap = evidence.player_swap(names)
    assert swap == ([0, 2, 1, 3], [-1.0, 1.0, 1.0, 1.0])
    rng = np.random.default_rng(0)
    x = rng.normal(size=(6000, 4))
    y = (x[:, 0] + x[:, 2] - x[:, 1] + rng.normal(scale=0.3, size=6000)) > 0
    old_step, fit_evidence.BOOST_STEP = fit_evidence.BOOST_STEP, 1
    try:
        model = fit_evidence.fit_boosted(names, x, y, True, swap,
                                         dict(fit_evidence.BOOST, max_iter=20,
                                              min_samples_leaf=20))
    finally:
        fit_evidence.BOOST_STEP = old_step
    assert model["design"] == "players"
    q = rng.normal(size=(500, 4))
    mirrored = q[:, swap[0]] * np.asarray(swap[1])
    z = evidence.apply_model(q, model)
    assert np.array_equal(z, -evidence.apply_model(mirrored, model))
    assert np.mean((z > 0) == ((q[:, 0] + q[:, 2] - q[:, 1]) > 0)) > 0.9
