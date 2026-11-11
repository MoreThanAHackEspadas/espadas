"""Events -> reference labels, on synthetic events only."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from espadas.external import events as ev  # noqa: E402

H, A = 0, 1


def e(order, team, typ, start, end=None, sub="", period=1):
    return {"order": order, "team": team, "type": typ, "subtype": sub, "period": period,
            "start": start, "end": end}


def run(events, n=40):
    return "".join(ev.labels_from_events(events, n)[0])


def test_kickoff_pass_and_interception():
    events = [e(0, A, "SET PIECE", 2, sub="KICK OFF"), e(1, A, "PASS", 2, 5),
              e(2, H, "RECOVERY", 10, 10, "INTERCEPTION"), e(3, H, "PASS", 12, 15),
              e(4, H, "PASS", 20, 25)]
    assert run(events, 30) == "XX" + "A" * 8 + "H" * 16 + "XXXX"


def test_touch_without_control_keeps_possession():
    # H recovers then the same player puts it out: never controlled, stays A until dead.
    events = [e(0, A, "SET PIECE", 0, sub="KICK OFF"), e(1, A, "PASS", 0, 4),
              e(2, H, "RECOVERY", 6, 6), e(3, H, "BALL OUT", 6, 9),
              e(4, A, "SET PIECE", 14, sub="THROW IN"), e(5, A, "PASS", 14, 18)]
    assert run(events, 20) == "A" * 9 + "D" * 5 + "A" * 5 + "X"


def test_header_alone_is_not_control_but_followed_by_pass_is():
    events = [e(0, A, "SET PIECE", 0, sub="KICK OFF"), e(1, A, "PASS", 0, 4),
              e(2, H, "PASS", 6, 7, "HEAD"),           # aerial won, header
              e(3, A, "PASS", 8, 9),                   # A gets it back: no change
              e(4, H, "RECOVERY", 12, 12), e(5, H, "PASS", 14, 16)]
    assert run(events, 18) == "A" * 12 + "H" * 5 + "X"


def test_foul_and_goal_are_dead_until_restart():
    events = [e(0, H, "SET PIECE", 0, sub="KICK OFF"), e(1, H, "PASS", 0, 2),
              e(2, H, "FAULT RECEIVED", 4, 4), e(3, A, "CHALLENGE", 4, 4, "GROUND-FAULT-LOST"),
              e(4, H, "SET PIECE", 8, sub="FREE KICK"), e(5, H, "SHOT", 9, 11, "ON TARGET-GOAL"),
              e(6, A, "SET PIECE", 15, sub="KICK OFF"), e(7, A, "PASS", 15, 17)]
    assert run(events, 18) == "HHHH" + "DDDD" + "HHH" + "DDDD" + "AAA"


def test_periods_and_ungraded_gap():
    events = [e(0, H, "SET PIECE", 0, sub="KICK OFF"), e(1, H, "PASS", 0, 3),
              e(2, A, "SET PIECE", 8, sub="KICK OFF", period=2), e(3, A, "PASS", 8, 10, period=2)]
    labels, blocks, kickoffs = ev.labels_from_events(events, 12)
    assert "".join(labels) == "HHHH" + "XXXX" + "AAA" + "X"
    assert blocks == [(1, 0, 3), (2, 8, 10)] and kickoffs == (0, 8)


def test_offside_is_dead_until_free_kick():
    events = [e(0, H, "SET PIECE", 0, sub="KICK OFF"), e(1, H, "PASS", 0, 2),
              e(2, H, "BALL LOST", 3, 5, "OFFSIDE"),
              e(3, A, "SET PIECE", 9, sub="FREE KICK-INDIRECT"), e(4, A, "PASS", 9, 11)]
    assert run(events, 12) == "HHHHH" + "DDDD" + "AAA"


def test_referee_stoppages_are_dead_until_the_restart():
    # BALL LOST-FORCED (play halted by the referee), as substring (HEAD-FORCED): dead from its
    # end frame until the free kick, not left live with the losing team (review issue 1).
    for sub in ("FORCED", "HEAD-FORCED", "VOLUNTARY", "END HALF"):
        events = [e(0, H, "SET PIECE", 0, sub="KICK OFF"), e(1, H, "PASS", 0, 3),
                  e(2, H, "BALL LOST", 5, 6, sub), e(3, A, "SET PIECE", 10, sub="FREE KICK"),
                  e(4, A, "PASS", 10, 13)]
        assert run(events, 14) == "H" * 6 + "D" * 4 + "A" * 4, sub


def test_offside_on_a_pass_is_dead_until_the_free_kick():
    events = [e(0, H, "SET PIECE", 0, sub="KICK OFF"), e(1, H, "PASS", 0, 3),
              e(2, H, "PASS", 4, 6, "DEEP BALL-OFFSIDE"), e(3, A, "SET PIECE", 10, sub="FREE KICK"),
              e(4, A, "PASS", 10, 13)]
    assert run(events, 14) == "H" * 6 + "D" * 4 + "A" * 4


def test_other_ball_lost_is_still_a_change_not_a_stoppage():
    events = [e(0, H, "SET PIECE", 0, sub="KICK OFF"), e(1, H, "PASS", 0, 3),
              e(2, H, "BALL LOST", 5, 6, "INTERCEPTION"), e(3, A, "RECOVERY", 7, 7),
              e(4, A, "PASS", 8, 11)]
    assert run(events, 12) == "H" * 7 + "A" * 5
