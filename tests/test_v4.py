"""Phase 7 5-state HMM (v4), on synthetic inputs only."""

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from espadas import touches, v4  # noqa: E402

TEAMS = np.array([0, 0, 1], dtype=np.int8)
QUIET = {"w_touch": 0.0, "w_confirm": 0.0, "w_ctrl": 0.0, "w_loose": 0.0}


def _brute(emis, trans):
    n, s = emis.shape
    best, arg = -np.inf, None
    for path in itertools.product(range(s), repeat=n):
        v = emis[0, path[0]] + sum(trans[a, b] + emis[t + 1, b]
                                   for t, (a, b) in enumerate(zip(path, path[1:])))
        if v > best:
            best, arg = v, path
    return np.array(arg), best


def _score(emis, trans, path):
    return emis[0, path[0]] + sum(trans[a, b] + emis[t + 1, b]
                                  for t, (a, b) in enumerate(zip(path, path[1:])))


def test_viterbi_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(30):
        s, n = int(rng.integers(2, 5)), int(rng.integers(1, 7))
        emis = rng.normal(size=(n, s))
        trans = rng.normal(size=(s, s))
        trans[rng.random((s, s)) < 0.2] = -np.inf
        np.fill_diagonal(trans, 0.0)
        want, best = _brute(emis, trans)
        got = v4.viterbi(emis, trans)
        assert np.isclose(_score(emis, trans, got), best)
        assert np.array_equal(got, want)


def test_viterbi_ties_go_to_lowest_state_and_repeat():
    emis = np.zeros((6, 5))
    trans = np.zeros((5, 5))
    path = v4.viterbi(emis, trans)
    assert set(path) == {0}
    assert np.array_equal(path, v4.viterbi(emis, trans))


def _f(n=400, team=0.0, dead=-9.0, halves=None):
    return {"half_blocks": halves or [(1, 0, n - 1)], "present": np.ones(n, bool),
            "dead_lo": np.full(n, dead), "team_lo": np.full(n, team),
            "kinks": {h: np.zeros(n) for h in touches.KINK_HALVES}, "near": np.full(n, 0.5),
            "slot": np.zeros(n, np.int64), "dist": np.full((n, 3), 9.0), "team_of_slot": TEAMS}


def test_flicker_ignored_and_sustained_switch_dated_back():
    f = _f(team=1.0)
    f["team_lo"][100:110] = -1.0                     # 10 frames of away evidence: a blip
    f["team_lo"][250:] = -1.0                        # away from 250 on
    # The settings this arithmetic assumes are pinned, so a re-tune cannot change the test.
    labels = v4.decode(f, dict(QUIET, c_turnover=20.0, w_team=2.0, team_cap=5.0,
                               loose_scale=0.0, c_loose=2.0, min_dead=50, min_alive=25))
    # The switch goes through one loose frame (250, still home's: a loose ball belongs to the
    # last team in control), so away has it from 251, one frame after its evidence starts.
    assert set(labels[:251]) == {"H"}
    assert set(labels[251:]) == {"A"}


def test_dead_stretch_and_teams():
    f = _f(team=1.0)
    f["dead_lo"][150:250] = 9.0
    f["team_lo"][250:] = -1.0
    labels = v4.decode(f, dict(QUIET, dead_smooth=0))
    assert set(labels[:150]) == {"H"}
    assert set(labels[150:250]) == {"D"}
    assert set(labels[250:]) == {"A"}


def test_missing_ball_is_no_information_and_carries_owner():
    f = _f(team=1.0)
    f["present"][100:300] = False
    f["near"][100:300] = np.inf
    f["team_lo"][100:300] = 0.0
    labels = v4.decode(f, dict(QUIET, w_ctrl=0.5, w_loose=0.5))
    assert set(labels) == {"H"}


def _loose_play(chain=True):
    """Home dribbles to frame 50 and loses it (loose ball); away deflects at 150; optionally an
    away pass chain at 300/310 (control)."""
    f = _f()
    f["near"][:] = 9.0                               # ball far from everyone: loose ...
    f["near"][:50] = 0.5                             # ... after home dribbles it to frame 50
    f["team_lo"][:50] = 1.0
    play = [(50, 0), (150, 2)] + ([(300, 2), (310, 2)] if chain else [])
    for frame, slot in play:
        f["kinks"][2][frame] = 10.0
        f["near"][frame] = 0.5
        f["slot"][frame] = slot
    return f


LOOSE_P = dict(QUIET, kink_half=2, kink_threshold=5.0, touch_reach=1.0, chain_gap=25,
               hold_frames=0, w_confirm=60.0, w_touch=10.0, w_touch_own=5.0, c_turnover=20.0,
               c_loose=1.0, w_loose=0.5, min_dead=0)


def test_lone_deflection_never_switches_possession():
    f = _loose_play(chain=False)
    states = v4.decode_states(f, LOOSE_P)
    assert set(v4.decode(f, LOOSE_P)) == {"H"}       # loose ball stays with home throughout
    assert states[100] == v4.HL


def test_confirmed_control_switches_to_the_other_team():
    labels = v4.decode(_loose_play(chain=True), LOOSE_P)
    assert labels[100] == "H"
    assert set(labels[300:]) == {"A"}


def test_halves_independent_outside_is_dead_and_labels_valid():
    n = 400
    f = _f(n, halves=[(1, 10, 189), (2, 210, 389)])
    f["team_lo"][:200] = 1.0
    f["team_lo"][200:] = -1.0
    labels = v4.decode(f, QUIET)
    assert set(labels) <= {"H", "A", "D"}
    assert set(labels[:10]) == {"D"} and set(labels[190:210]) == {"D"} and set(labels[390:]) == {"D"}
    assert set(labels[10:190]) == {"H"} and set(labels[210:390]) == {"A"}
    second = v4.decode(dict(f, half_blocks=[(2, 210, 389)]), QUIET)
    assert np.array_equal(labels[210:390], second[210:390])


def test_dead_bias_does_not_touch_ball_less_frames():
    f = _f(team=1.0, dead=0.3)
    f["present"][:] = False
    f["near"][:] = np.inf
    for bias in (-5.0, 0.0, 5.0):
        e = v4.emissions(f, dict(v4.PARAMS, **QUIET, dead_bias=bias), 0, 400)
        assert np.allclose(e, v4.emissions(f, dict(v4.PARAMS, **QUIET), 0, 400))
    seen = _f(team=1.0, dead=0.3)
    a = v4.emissions(seen, dict(v4.PARAMS, dead_bias=-5.0), 0, 400)
    b = v4.emissions(seen, dict(v4.PARAMS, dead_bias=5.0), 0, 400)
    assert not np.allclose(a, b)


def test_structure_possession_changes_only_through_loose_and_restarts_are_controlled():
    t = v4.transitions(v4.PARAMS)
    assert t[v4.HC, v4.AC] == -np.inf and t[v4.AC, v4.HC] == -np.inf   # no direct handover
    assert np.isfinite(t[v4.HL, v4.AC]) and np.isfinite(t[v4.AL, v4.HC])
    assert t[v4.D, v4.HL] == -np.inf and t[v4.D, v4.AL] == -np.inf     # restart in control
    assert np.isfinite(t[v4.D, v4.HC]) and np.isfinite(t[v4.D, v4.AC])
    emis = np.zeros((3, 5))
    emis[:, [v4.HL, v4.AL]] = 5.0                     # loose evidence from the first frame
    assert v4.viterbi(emis, t, v4.START)[0] not in (v4.HL, v4.AL)


def test_viterbi_start_scores_shift_the_first_state():
    emis = np.zeros((4, 2))
    trans = np.zeros((2, 2))
    assert v4.viterbi(emis, trans, np.array([-np.inf, 0.0]))[0] == 1
    assert np.array_equal(v4.viterbi(emis, trans), v4.viterbi(emis, trans, np.zeros(2)))


def test_transitions_forbid_loose_to_other_loose():
    t = v4.transitions(v4.PARAMS)
    assert t[v4.HL, v4.AL] == -np.inf and t[v4.AL, v4.HL] == -np.inf
    assert t[v4.AC, v4.HL] == -np.inf and t[v4.HC, v4.AL] == -np.inf
    assert np.all(np.diag(t) == 0.0)


def test_v4_end_to_end_on_tracking_is_deterministic():
    from tests.test_v1 import _features
    tr = _features()
    a = v4.v4(tr)
    assert len(a) == 400 and set(a) <= {"H", "A", "D"}
    assert np.array_equal(a, v4.v4(tr))


def test_clean_up_fills_short_dead_and_short_live():
    s = np.array([v4.HC] * 10 + [v4.D] * 3 + [v4.HC] * 10 + [v4.D] * 20 + [v4.AC] * 2
                 + [v4.D] * 20 + [v4.AC] * 5, dtype=np.int8)
    out = v4.clean_up(s, min_dead=5, min_alive=4)
    assert set(out[10:13]) == {v4.HC}               # 3-frame dead blip -> team before it
    assert set(out[43:45]) == {v4.D}                # 2-frame live blip between stoppages -> dead
    assert set(out[-5:]) == {v4.AC}                 # last live run is not between stoppages
    assert np.array_equal(v4.clean_up(s, 0, 0), s)


def test_touch_modes_kink_carrier_and_both():
    f = _f(team=1.0)
    f["kinks"][2][50] = 10.0                          # one kink touch by slot 0 (home)
    f["slot"][200:260] = 2                            # away slot 2 carries the ball 200-259
    p = dict(v4.PARAMS, kink_half=2, kink_threshold=5.0, touch_reach=1.0,
             carrier_reach=1.0, carrier_frames=12)
    kink, _ = v4.touch_list(f, dict(p, touch_mode="kink"))
    carry, _ = v4.touch_list(f, dict(p, touch_mode="carrier"))
    both, _ = v4.touch_list(f, dict(p, touch_mode="both"))
    assert list(kink["frame"]) == [50]
    assert list(carry["frame"]) == [0, 200, 260]      # carrier changes: 0 -> 2 -> 0
    assert list(both["frame"]) == [0, 50, 200, 260]
