"""Tests on synthetic inputs only. No match data is used or needed."""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from espadas import atd, config, labelers, write  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "tools"))
import check_no_data  # noqa: E402


def test_constants_match_official_docs():
    assert config.N_FRAMES == 199319
    assert config.HALF_BLOCKS == [(1, 11185, 90651), (2, 110808, 190197)]
    start, end = config.PRE_KICKOFF_DEAD
    assert (start, end) == (110808, 113741)
    assert end - start + 1 == 2934  # FAQ.md: "Those 2,934 frames are graded and all ... D"


def test_finalise_rules():
    out = write.finalise(np.full(config.N_FRAMES, "H"))
    start, end = config.PRE_KICKOFF_DEAD
    assert set(out[start:end + 1]) == {"D"}
    assert out[0] == "X" and out[100000] == "X" and out[-1] == "X"   # ungraded
    assert out[11185] == "H" and out[90651] == "H" and out[end + 1] == "H"
    assert np.sum(out != "X") == 158857                               # DATA.md graded count


def test_write_validate_roundtrip(tmp_path):
    path = str(tmp_path / "s.csv")
    write.write_csv(path, write.finalise(np.full(config.N_FRAMES, "A")))
    assert write.validate(path) is None
    with open(path, "rb") as fh:
        head = fh.read(20)
    assert head.startswith(b"frame,label\n0,X\n") and b"\r" not in head
    assert write.sha256(path) == write.sha256(path)


def test_validate_rejects_short_file(tmp_path):
    path = str(tmp_path / "s.csv")
    write.write_csv(path, ["H"] * 10)
    assert write.validate(path) is not None


def test_nearest_team_and_smoothing():
    players = np.array([[[0, 0], [10, 0]],
                        [[0, 0], [10, 0]],
                        [[np.nan, np.nan], [10, 0]],
                        [[0, 0], [10, 0]]], dtype=float)
    ball = np.array([[1, 0], [9, 0], [0, 0], [np.nan, np.nan]], dtype=float)
    team = np.array([0, 1], dtype=np.int8)
    codes = labelers.nearest_team(players, ball, team)
    assert codes.tolist() == [0, 1, 1, -1]
    assert labelers.majority_smooth(codes, 0).tolist() == ["H", "A", "A", "D"]
    assert labelers.majority_smooth(codes, 1).tolist() == ["H", "A", "A", "A"]


def test_atd_parse_synthetic(tmp_path):
    raw = tmp_path / "x_atd.txt"
    raw.write_text("1:0.5,0.5,1.0;NaN,NaN,NaN:0.5,0.5,0.0\n"
                   "3:0.0,1.0,0.0;1.0,0.0,0.0:NaN,NaN,NaN\n")
    players, ball = atd.parse_raw(str(raw), n_slots=2, n_frames=4, pitch=(100.0, 50.0))
    assert players[0, 0].tolist() == [50.0, 25.0]
    assert np.isnan(players[0, 1]).all()
    assert ball[0].tolist() == [50.0, 25.0]
    assert np.isnan(ball[1]).all() and np.isnan(ball[2]).all()
    assert players[2, 1].tolist() == [100.0, 0.0]


def test_atd_teams_synthetic(tmp_path):
    meta = tmp_path / "m.xml"
    meta.write_text('<Root><Teams><Team id="h"><Name>Home</Name></Team>'
                    '<Team id="a"><Name>Away</Name></Team></Teams>'
                    '<Players><Player teamId="a"/><Player teamId="h"/></Players></Root>')
    assert atd.team_of_each_track(str(meta)).tolist() == [1, 0]


@pytest.mark.parametrize("path,blocked", [
    ("data/tracking.txt", True),
    ("data/.gitkeep", False),
    ("BAR-ATH_tactical-002.mp4", True),
    ("out/espadas.csv", True),
    ("notes/x_smart_tagging.xml", True),
    ("requirements.txt", False),
    ("espadas/run.py", False),
    ("third_party/mtah2027/games.py", False),
])
def test_no_data_guard(path, blocked):
    assert bool(check_no_data.problems(path)) is blocked
