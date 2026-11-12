"""espadas/external/idsse.py on a tiny synthetic DFL file (the real data stays outside the repo)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from espadas import external  # noqa: E402
from espadas.external import bench, idsse  # noqa: E402

INFO = """<?xml version="1.0" encoding="UTF-8"?>
<PutDataRequest><MatchInformation>
<General MatchTitle="Home FC:Away FC" />
<Environment PitchX="105.00" PitchY="68.00" />
<Teams>
<Team TeamId="T-AWAY" Role="guest"><Players /></Team>
<Team TeamId="T-HOME" Role="home"><Players /></Team>
</Teams></MatchInformation></PutDataRequest>
"""


def frameset(section, team, person, frames):
    rows = "\n".join('<Frame N="{}" X="{}" Y="{}"{} />'.format(
        n, x, y, "".join(' {}="{}"'.format(k, v) for k, v in extra.items()))
        for n, x, y, extra in frames)
    return '<FrameSet GameSection="{}" TeamId="{}" PersonId="{}">\n{}\n</FrameSet>'.format(
        section, team, person, rows)


def positions():
    # first half: frames 10000-10003; second half: 100000-100002
    ball1 = [(10000, 0, 0, {"BallStatus": 1, "BallPossession": 1}),
             (10001, 1, 0, {"BallStatus": 1, "BallPossession": 1}),
             (10002, 2, 0, {"BallStatus": 1, "BallPossession": 2}),
             (10003, 60, 0, {"BallStatus": 0, "BallPossession": 2})]
    ball2 = [(100000, 0, 0, {"BallStatus": 1, "BallPossession": 2}),
             (100001, 0, 1, {"BallStatus": 1, "BallPossession": 2}),
             (100002, 0, 2, {"BallStatus": 1, "BallPossession": 1})]
    parts = [
        frameset("firstHalf", "T-HOME", "P-H1", [(n, -10, 5, {}) for n in range(10000, 10004)]),
        frameset("firstHalf", "T-AWAY", "P-A1", [(n, 10, -5, {}) for n in range(10000, 10004)]),
        frameset("firstHalf", "referee", "R-1", [(n, 0, 30, {}) for n in range(10000, 10004)]),
        frameset("firstHalf", "BALL", "B", ball1),
        frameset("secondHalf", "T-HOME", "P-H2", [(n, 10, 5, {}) for n in range(100000, 100003)]),
        frameset("secondHalf", "T-AWAY", "P-A1", [(n, -10, -5, {}) for n in range(100000, 100003)]),
        frameset("secondHalf", "BALL", "B", ball2),
    ]
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<PutDataRequest><Positions>\n'
            '<MetaData><PitchSize X="105.00" Y="68.00" /></MetaData>\n{}\n'
            '</Positions></PutDataRequest>\n'.format("\n".join(parts)))


@pytest.fixture
def match(tmp_path, monkeypatch):
    monkeypatch.setenv("MTAH_EXTERNAL_DIR", str(tmp_path))
    info_path, pos_path = idsse.paths(12)
    os.makedirs(os.path.dirname(info_path))
    with open(info_path, "w") as f:
        f.write(INFO)
    with open(pos_path, "w") as f:
        f.write(positions())
    monkeypatch.setattr(idsse, "GAP", 2)
    return 12


def test_reference_layout_and_coordinates(match):
    game = idsse.load_game(match)
    labels, blocks, kickoffs = bench.reference(game)
    # half 1 (4 frames), 2 ungraded, half 2 (3 frames); provider flags as they are:
    assert "".join(game["raw_labels"]) == "HHAD" + "XX" + "AAH"
    # rule 2: A's 1-frame spell before the ball goes dead is a touch -> H keeps it; H's last
    # frame of half 2 is a spell at the end of the half, not held -> A keeps it
    assert "".join(labels) == "HHHD" + "XX" + "AAA"
    assert blocks == [(1, 0, 3), (2, 6, 8)] and kickoffs == (0, 6)
    t = game["tracking"]
    assert t["players"].shape == (9, 3, 2) and len(t["ball"]) == 9
    # home slots first: P-H1, P-H2 home, P-A1 away; referee skipped
    assert list(t["team_of_slot"]) == [0, 0, 1]
    np.testing.assert_allclose(t["ball"][1], [53.5, 34.0])          # centre origin -> corner
    np.testing.assert_allclose(t["players"][0, 0], [42.5, 39.0])
    assert np.isnan(t["players"][0, 1]).all()                       # P-H2 not on in half 1
    assert np.isnan(t["players"][6, 0]).all()                       # P-H1 off in half 2
    np.testing.assert_allclose(t["players"][6, 2], [42.5, 29.0])    # same player, same slot
    assert np.isnan(t["ball"][4:6]).all()


def test_cache_gives_the_same_arrays(match):
    _, fresh = idsse.load_arrays(match, use_cache=False)
    idsse.load_arrays(match)                  # writes the cache
    _, cached = idsse.load_arrays(match)      # reads it
    for key in ("players", "ball", "team_of_slot", "labels"):
        np.testing.assert_array_equal(fresh[key], cached[key])
    assert fresh["half_blocks"] == cached["half_blocks"]
    assert fresh["kickoffs"] == cached["kickoffs"]


def test_held_out_match_refused_before_any_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MTAH_EXTERNAL_DIR", str(tmp_path))     # nothing there
    for n in external.IDSSE_HELD_OUT:
        with pytest.raises(external.HeldOutError):
            idsse.load_game(n)
        with pytest.raises(external.HeldOutError):
            bench.available([n])


def test_split_is_disjoint_and_covers_all_matches():
    tune, held = set(external.IDSSE_TUNING), set(external.IDSSE_HELD_OUT)
    assert not tune & held and tune | held == set(idsse.MATCHES)
    assert external.HELD_OUT == (2,)          # Metrica split unchanged


def run(raw, touch=3):
    return "".join(idsse.official_labels(np.array(list(raw)), touch))


def test_touch_that_goes_back_is_not_a_change():
    assert run("HHHHAAHHHH") == "HHHHHHHHHH"          # A touches, H keeps it
    assert run("HHHHAAADDD") == "HHHHAAADDD"          # 3 frames = touch_frames: a possession


def test_touch_before_the_ball_goes_dead_is_not_a_change():
    assert run("HHHHAADDAAAA") == "HHHHHHDDAAAA"      # deflected out; A takes the throw-in


def test_restart_and_one_touch_pass_are_kept():
    assert run("DDAAHHHH") == "DDAAHHHH"              # A's short restart spell stays A
    assert run("HHHHAAHHAAAA") == "HHHHHHHHAAAA"      # A touch, H touch-and-lose: one change


def test_flicker_resolves_to_one_change():
    # H A(touch) H(short) A(long): the H spell is H's own, so one change, at the A possession
    assert run("HHHHAHAAAA") == "HHHHHHAAAA"


def test_long_spells_unchanged():
    assert run("HHHHAAAAHHHHDDDAAAA") == "HHHHAAAAHHHHDDDAAAA"


def test_pools():
    assert external.pool("dev") == (external.DEV_TUNING, False)
    games, spent = external.pool("full")
    assert spent and len(games) == 10 and set(games) == {1, 2, 3} | set(idsse.MATCHES)
    # the guard itself is unchanged: the spent games still need the explicit opt-in
    with pytest.raises(external.HeldOutError):
        external.check_split(games)
