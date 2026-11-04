"""v4 five-state decoder, on synthetic inputs only."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from espadas import labelers, v4  # noqa: E402
from test_evidence import _match  # noqa: E402

RAW = {"min_dead": 0, "min_alive": 0}
NO_TOUCH = {"frame": np.zeros(0, np.int64), "slot": np.zeros(0, np.int64),
            "team": np.zeros(0, np.int8), "strength": np.zeros(0)}


def _f(n=300, team_lo=None, dead_lo=None, present=None, touches=NO_TOUCH, touch_ok=None):
    return {"half_blocks": [(1, 0, n - 1)],
            "present": np.ones(n, bool) if present is None else present,
            "dead_lo": np.full(n, -5.0) if dead_lo is None else dead_lo,
            "team_lo": np.zeros(n) if team_lo is None else team_lo,
            "ctrl_lo": np.zeros(n), "touches": touches,
            "touch_ok": np.zeros(len(touches["frame"]), bool) if touch_ok is None else touch_ok}


def _team(n, before, after, at=150):
    return np.where(np.arange(n) < at, before, after).astype(float)


def test_transitions_are_stochastic_symmetric_and_forbid_direct_switches():
    lt, ls = v4.transitions(v4.PARAMS)
    assert np.allclose(np.exp(lt).sum(axis=1), 1.0)
    for a, b in [(v4.HC, v4.AC), (v4.HC, v4.AL), (v4.HL, v4.AL), (v4.D, v4.HL), (v4.D, v4.AL),
                 (v4.AC, v4.HC), (v4.AC, v4.HL), (v4.AL, v4.HL)]:
        assert lt[a, b] == -np.inf
    assert lt[v4.HC, v4.HL] == lt[v4.AC, v4.AL] and lt[v4.HL, v4.AC] == lt[v4.AL, v4.HC]
    assert lt[v4.D, v4.HC] == lt[v4.D, v4.AC]
    assert ls[v4.HL] == -np.inf and ls[v4.AL] == -np.inf and ls[v4.D] == ls[v4.HC] == ls[v4.AC]


def test_transitions_refuse_impossible_lengths():
    with pytest.raises(ValueError):
        v4.transitions(dict(v4.PARAMS, loose_len=1))


def test_dead_offset_is_fixed_at_zero_without_a_ball():
    f = _f(2, dead_lo=np.array([0.3, 0.3]), present=np.array([True, False]))
    e = v4.emissions(f, dict(v4.PARAMS, dead_offset=0.5, w_dead=1.0), 0, 2)
    assert e[0, v4.D] == pytest.approx(-0.2) and e[1, v4.D] == pytest.approx(0.3)
    assert e[1, v4.HC] == 0.0 and e[1, v4.HL] == 0.0


def test_clear_evidence_switches_weak_or_absent_evidence_does_not():
    # ctrl_lo is 0 in these fixtures, which is only "no information" if the control score is
    # off (a tuned ctrl_offset turns 0 into "loose"); these tests are about the team evidence.
    p = dict(v4.PARAMS, w_ctrl=0.0, **RAW)
    clear = v4.decode(_f(team_lo=_team(300, 2.0, -2.0)), p)
    assert set(clear[:140]) == {"H"} and set(clear[160:]) == {"A"}
    silent = v4.decode(_f(team_lo=_team(300, 2.0, 0.0)), p)
    assert set(silent) == {"H"}
    blip = _team(300, 2.0, 2.0)
    blip[150:170] = -0.3
    assert set(v4.decode(_f(team_lo=blip), p)) == {"H"}


def test_confirmed_touch_switches_and_plain_touch_does_not():
    touch = {"frame": np.array([150]), "slot": np.array([0]), "team": np.array([1], np.int8),
             "strength": np.array([8.0])}
    p = dict(v4.PARAMS, confirm_pulse=12.0, touch_pulse=1.0, w_ctrl=0.0, **RAW)
    f = _f(team_lo=_team(300, 1.0, 0.0), touches=touch, touch_ok=np.array([True]))
    out = v4.decode(f, p)
    assert set(out[:140]) == {"H"} and set(out[160:]) == {"A"}
    f = _f(team_lo=_team(300, 1.0, 0.0), touches=touch, touch_ok=np.array([False]))
    assert set(v4.decode(f, p)) == {"H"}


def test_stoppage_in_the_middle_and_restart_to_the_other_team():
    dead = np.full(300, -5.0)
    dead[100:200] = 5.0
    team = _team(300, 2.0, 0.0, at=100)
    team[200:] = -2.0
    out = v4.decode(_f(dead_lo=dead, team_lo=team), dict(v4.PARAMS, **RAW))
    assert set(out[:98]) == {"H"} and set(out[102:198]) == {"D"} and set(out[202:]) == {"A"}


def test_cleanup_drops_short_dead_runs_and_fills_short_live_runs():
    seg = np.array(list("H" * 10 + "D" * 3 + "H" * 10))
    assert "".join(v4.cleanup(seg, {"min_dead": 5, "min_alive": 0})) == "H" * 23
    seg = np.array(list("H" * 10 + "D" * 10 + "H" * 3 + "D" * 10 + "A" * 10))
    assert "".join(v4.cleanup(seg, {"min_dead": 5, "min_alive": 5})) == \
        "H" * 10 + "D" * 23 + "A" * 10


def test_decode_with_ball_never_tracked():
    f = _f(team_lo=_team(300, 1.0, -1.0), present=np.zeros(300, bool))
    f["ctrl_lo"] = np.full(300, np.nan)
    out = v4.decode(f, dict(v4.PARAMS, **RAW))
    assert len(out) == 300 and set(out) <= {"H", "A", "D"}


def test_touch_on_array_edges():
    touch = {"frame": np.array([0, 299]), "slot": np.array([0, 0]),
             "team": np.array([0, 1], np.int8), "strength": np.array([8.0, 8.0])}
    out = v4.decode(_f(touches=touch, touch_ok=np.array([True, True])), dict(v4.PARAMS, **RAW))
    assert len(out) == 300


def test_v4_runs_on_synthetic_tracking_and_is_registered_and_deterministic():
    tracking = _match()
    first = v4.v4(tracking)
    assert len(first) == 400 and set(first) <= {"H", "A", "D"}
    assert np.array_equal(first, labelers.LABELERS["v4"](tracking))
