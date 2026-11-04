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


def ctrl(events, n):
    return "".join("C" if c else "." for c in ev.control_mask(events, n))


def test_pass_flight_is_loose_carry_and_kick_and_reception_are_controlled():
    events = [e(0, H, "SET PIECE", 0, sub="KICK OFF"), e(1, H, "PASS", 0, 4),
              e(2, H, "CARRY", 4, 8), e(3, H, "PASS", 8, 12)]
    assert ctrl(events, 14) == "C...CCCCC...C."


def test_won_ball_is_controlled_from_the_touch_and_lost_ball_is_loose():
    events = [e(0, A, "SET PIECE", 0, sub="KICK OFF"), e(1, A, "PASS", 0, 6),
              e(2, H, "RECOVERY", 6, 6, "INTERCEPTION"), e(3, H, "PASS", 8, 10),
              e(4, H, "BALL LOST", 12, 14, "INTERCEPTION"),
              e(5, A, "RECOVERY", 16, 16), e(6, A, "PASS", 18, 20)]
    assert ctrl(events, 22) == "C.....CCC.CCC...CCC.C."
    assert run(events, 22) == "A" * 6 + "H" * 10 + "A" * 5 + "X"


def test_touch_only_followed_by_the_other_team_is_loose():
    events = [e(0, A, "SET PIECE", 0, sub="KICK OFF"), e(1, A, "PASS", 0, 4),
              e(2, H, "CHALLENGE", 6, 6, "GROUND-LOST"), e(3, A, "PASS", 8, 10)]
    assert ctrl(events, 12) == "C...CC..C.C."


def test_shot_is_loose_after_the_kick_and_recovery_before_a_pass_is_controlled():
    events = [e(0, H, "SET PIECE", 0, sub="KICK OFF"), e(1, H, "PASS", 0, 2),
              e(2, H, "SHOT", 4, 6, "ON TARGET-SAVED"), e(3, A, "RECOVERY", 7, 7),
              e(4, A, "PASS", 8, 9)]
    assert ctrl(events, 11) == "C.CCC..CCC."


def test_shot_saved_and_recovered_on_its_end_frame_is_controlled_there():
    events = [e(0, H, "SET PIECE", 0, sub="KICK OFF"), e(1, H, "PASS", 0, 2),
              e(2, H, "SHOT", 4, 6, "ON TARGET-SAVED"), e(3, A, "RECOVERY", 6, 6),
              e(4, A, "PASS", 8, 9)]
    assert ctrl(events, 11) == "C.CCC.CCCC."
    assert run(events, 11) == "HHHHHH" + "AAAA" + "X"


def test_ball_lost_with_nothing_after_is_loose_to_the_period_end():
    events = [e(0, H, "SET PIECE", 0, sub="KICK OFF"), e(1, H, "PASS", 0, 2),
              e(2, H, "BALL LOST", 4, 6, "INTERCEPTION")]
    assert ctrl(events, 8) == "C.CCC..."


def test_control_mask_is_true_wherever_the_reference_starts_a_new_spell():
    events = [e(0, A, "SET PIECE", 0, sub="KICK OFF"), e(1, A, "PASS", 0, 6),
              e(2, H, "RECOVERY", 6, 6, "INTERCEPTION"), e(3, H, "PASS", 8, 10),
              e(4, H, "BALL LOST", 12, 14, "INTERCEPTION"),
              e(5, A, "RECOVERY", 16, 16), e(6, A, "PASS", 18, 20)]
    labels = ev.labels_from_events(events, 22)[0]
    mask = ev.control_mask(events, 22)
    starts = [t for t in range(1, 22) if labels[t] in "HA" and labels[t] != labels[t - 1]]
    assert starts == [6, 16] and all(mask[t] for t in starts)
