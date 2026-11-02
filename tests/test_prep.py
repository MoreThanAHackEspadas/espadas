"""Stage 0 preprocessing and Smart Tagging, on synthetic inputs only. No match data is used."""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from espadas import prep, tagging  # noqa: E402

NAN = np.nan


def track(xs):
    """Ball track along the x axis, y = 0; None marks a missing frame."""
    return np.array([[NAN, NAN] if x is None else [x, 0.0] for x in xs], dtype=float)


def test_reject_isolated_jump():
    # 1 m/frame = 25 m/s is fine; the 60 m spike and back is not.
    ball = track([0, 1, 60, 3, 4])
    cleaned, rejected = prep.reject_ball_jumps(ball)
    assert rejected.tolist() == [False, False, True, False, False]
    assert np.isnan(cleaned[2]).all() and cleaned[3, 0] == 3
    assert not np.isnan(ball[2]).any()   # input untouched


def test_fast_but_real_ball_is_kept():
    ball = track([0, 1.5, 3.0, 4.5])     # 37.5 m/s, under the cap
    assert not prep.reject_ball_jumps(ball)[1].any()


def test_jump_after_long_gap_is_not_a_jump():
    ball = track([0] + [None] * 50 + [30])   # 30 m in 2 s
    assert not prep.reject_ball_jumps(ball)[1].any()


def test_fill_short_gap_linearly():
    filled, mask = prep.fill_ball_gaps(track([0, None, None, 3]))
    assert filled[:, 0].tolist() == [0, 1, 2, 3]
    assert mask.tolist() == [False, True, True, False]


def test_fill_leaves_long_gap_and_open_ends():
    ball = track([None, 0] + [None] * 30 + [3, None])
    filled, mask = prep.fill_ball_gaps(ball)
    assert not mask.any()
    assert np.isnan(filled[0]).all() and np.isnan(filled[-1]).all()


def test_fill_skips_impossible_line():
    filled, mask = prep.fill_ball_gaps(track([0, None, 10]))   # 10 m in 2 frames = 125 m/s
    assert not mask.any() and np.isnan(filled[1]).all()


def test_gap_lengths():
    present = np.array([1, 0, 0, 1, 0, 1, 1, 0], dtype=bool)
    since, run = prep.gap_lengths(present)
    assert since.tolist() == [0, 1, 2, 0, 1, 0, 0, 1]
    assert run.tolist() == [0, 2, 2, 0, 1, 0, 0, 1]


def test_ball_speed_over_window():
    ball = track([float(i) for i in range(20)])   # 1 m/frame = 25 m/s
    speed = prep.ball_speed(ball, half_window=2)
    assert np.isnan(speed[:2]).all() and np.isnan(speed[-2:]).all()
    assert np.allclose(speed[2:-2], 25.0)
    ball[10] = NAN
    speed = prep.ball_speed(ball, half_window=2)
    assert np.isnan(speed[8]) and np.isnan(speed[12]) and not np.isnan(speed[7])


def test_attacking_direction_and_swap():
    n = 40
    players = np.full((n, 2, 2), NAN)
    team = np.array([0, 1], dtype=np.int8)
    players[:20, 0, 0], players[:20, 1, 0] = 30.0, 70.0   # half 1: home on the left
    players[20:, 0, 0], players[20:, 1, 0] = 80.0, 20.0   # half 2: ends swapped
    blocks, kickoffs = [(1, 2, 18), (2, 22, 38)], (5, 25)
    direction = prep.attacking_direction(players, team, blocks, kickoffs, window=2)
    assert direction[10].tolist() == [1, -1] and direction[30].tolist() == [-1, 1]
    assert direction[0].tolist() == [0, 0] and direction[20].tolist() == [0, 0]
    assert prep.directions_swap(direction, blocks)


def test_attacking_direction_needs_players():
    players = np.full((10, 2, 2), NAN)
    with pytest.raises(ValueError):
        prep.attacking_direction(players, np.array([0, 1], dtype=np.int8),
                                 [(1, 0, 9)], (5,), window=2)


SAMPLE_XML = """<file><ALL_INSTANCES>
<instance><ID>1</ID><start>2.0</start><end>4.02</end><code>Set Pieces</code>
  <label><group>Team</group><text>N/A</text></label>
  <label><group>Type</group><text>Throw-in</text></label></instance>
<instance><ID>2</ID><start>4.0</start><end>9.0</end><code>BUILD UP</code>
  <label><group>Team</group><text>Barcelona</text></label></instance>
<instance><ID>3</ID><start>9.0</start><end>10.0</end><code>PROGRESSION</code>
  <label><group>team</group><text>Atletic Club</text></label></instance>
</ALL_INSTANCES></file>"""


def test_tagging_parse_and_mask(tmp_path):
    path = tmp_path / "t.xml"
    path.write_text(SAMPLE_XML)
    events = tagging.parse(str(path), fps=25)
    assert [e["code"] for e in events] == ["SET PIECES", "BUILD UP", "PROGRESSION"]
    assert (events[0]["start_frame"], events[0]["end_frame"]) == (50, 100)
    assert events[0]["tags"] == {"team": "n/a", "type": "throw-in"}
    assert [tagging.team_of(e) for e in events] == [None, tagging.HOME, tagging.AWAY]
    assert tagging.unknown_team_names(events) == []

    mask = tagging.set_piece_mask(events, n_frames=300)
    assert not mask[49] and mask[50] and mask[99] and not mask[100]
    assert tagging.interval_mask(events, lambda e: True, n_frames=120).sum() == 70


def test_tagging_flags_unknown_team(tmp_path):
    path = tmp_path / "t.xml"
    path.write_text(SAMPLE_XML.replace("Barcelona", "Real Madrid"))
    assert tagging.unknown_team_names(tagging.parse(str(path))) == ["real madrid"]


def test_prepare_on_another_match():
    """A short synthetic 'other match' with its own halves and kickoffs."""
    n = 60
    players = np.full((n, 2, 2), NAN)
    players[:30, 0, 0], players[:30, 1, 0] = 20.0, 80.0
    players[30:, 0, 0], players[30:, 1, 0] = 80.0, 20.0
    ball = track([50.0 + 0.1 * i if i % 7 else None for i in range(n)])
    tracking = {"players": players, "ball": ball, "team_of_slot": np.array([0, 1], np.int8)}
    blocks, kickoffs = [(1, 5, 28), (2, 32, 58)], (5, 32)
    base = prep.prepare(tracking, half_blocks=blocks, kickoffs=kickoffs)
    assert all(v.shape[0] == n for v in base.values())
    assert base["half"][4] == 0 and base["half"][5] == 1 and base["half"][32] == 2
    assert prep.directions_swap(base["attack"], blocks)
    assert base["ball_filled"][7] and base["ball_present"][7]


def test_homography_parse_and_roundtrip(tmp_path):
    from espadas import homography
    cols = ",".join(homography.MATRIX_COLUMNS)
    rows = ["frame,{},reprojection_error,valid".format(cols),
            "0,1000,0,10,0,500,20,0,0,1,0.2,1",
            "2,{},,0".format(",".join(["NaN"] * 9))]
    path = tmp_path / "h.csv"
    path.write_text("\n".join(rows) + "\n")
    matrices, error, valid = homography.parse(str(path), n_frames=3)
    assert valid.tolist() == [True, False, False]
    assert np.isnan(matrices[1]).all() and np.isnan(matrices[2]).all()
    px, py = homography.project(matrices[0], 0.5, 0.5)
    assert (float(px), float(py)) == (510.0, 270.0)
    x, y = homography.unproject(matrices[0], px, py)
    assert np.allclose([x, y], [0.5, 0.5])
    assert np.allclose(homography.project_metres(matrices[0], 52.65, 34.0), (510.0, 270.0))
