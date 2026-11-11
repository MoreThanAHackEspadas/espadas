"""External games: Metrica event data -> per-frame H/A/D/X reference labels, by code.

These labels are the *reference* our settings are tuned against on external games, so the rules
below follow the official possession definition (CHALLENGE.md, "Possession rules") as written:

- While the ball is in play exactly one team is in possession. A loose or contested ball stays
  with the last team in possession.
- Possession changes only when the other team gains **controlled** possession. Not a change: a
  touch without control, a deflection, an aerial duel won without subsequent control.
- Every restart has a clear possessor: the team taking it.
- Dead ball (ball out, foul, goal) is D until the restart. D is never a possession.

How that maps onto Metrica events (definitions: documentation/events-definitions.pdf):

- **Control events**: SET PIECE, PASS, CARRY, SHOT and BALL LOST, unless the subtype says it was
  a header or a clearance (a single touch). BALL LOST implies the team had the ball first
  (a failed pass, a dispossession).
- **Touch-only events**: RECOVERY and headers/clearances. They never change possession on their
  own, but if the same team follows with a control event before the other team does anything in
  play, the change is dated back to the first of those events (the moment the ball was won).
  CHALLENGE events are ignored altogether (not in IN_PLAY): they neither start such a back-dated
  change nor interrupt one.
- **Dead ball** starts at: BALL OUT (its end frame: when the ball left the pitch), a SHOT whose
  subtype ends in OUT or GOAL (its end frame), FAULT RECEIVED (its start frame: the whistle), a
  BALL LOST tagged OFFSIDE, HAND BALL, FORCED ("play is halted by the referee"), VOLUNTARY (ball
  given up, e.g. for an injury) or END HALF (its end frame), and a PASS tagged OFFSIDE (its end
  frame). It ends at the start of the next control event (the restart: a set piece, a
  goal-kick pass, ...). FORCED / VOLUNTARY / END HALF and PASS-OFFSIDE were added on 11 Nov
  (docs/events_rules_review.md, issues 1 and 4): before that those stoppages were labelled live.
- Graded frames per period run from its first event (the kickoff) to its last event's end.
  Everything else is X (ungraded), like halftime in our match.

Frames are converted from Metrica's 1-based numbering to 0-based array indices here.
These are fixed definitions, not tuned parameters. Fully deterministic.
"""

import numpy as np

HOME, AWAY = 0, 1

IN_PLAY = {"PASS", "SHOT", "CARRY", "RECOVERY", "BALL LOST", "SET PIECE"}
CONTROL = {"PASS", "SHOT", "CARRY", "BALL LOST", "SET PIECE"}
TOUCH_SUBTYPES = ("HEAD", "CLEARANCE")
# BALL LOST subtypes that mean play stopped (Game 3 tags OFFSIDE / HAND BALL instead of a foul;
# FORCED is Metrica's "play is halted by the referee"; VOLUNTARY and END HALF stop it too).
# Matched as substrings, so HEAD-FORCED and GOAL KICK-END HALF count.
WHISTLE_SUBTYPES = ("OFFSIDE", "HAND BALL", "FORCED", "VOLUNTARY", "END HALF")
# PASS subtypes that stop play (an offside flagged on the pass itself, Game 3).
PASS_WHISTLE_SUBTYPES = ("OFFSIDE",)


def is_control(event):
    if event["type"] not in CONTROL:
        return False
    if event["type"] == "SET PIECE":
        return True
    return not any(word in event["subtype"] for word in TOUCH_SUBTYPES)


def dead_start(event):
    """Frame (0-based) at which this event makes the ball dead, or None."""
    end = event["end"] if event["end"] is not None and event["end"] >= event["start"] \
        else event["start"]
    if event["type"] == "BALL OUT":
        return end
    if event["type"] == "SHOT" and (event["subtype"].endswith("OUT")
                                    or event["subtype"].endswith("GOAL")):
        return end
    if event["type"] == "FAULT RECEIVED":
        return event["start"]
    if event["type"] == "BALL LOST" and any(w in event["subtype"] for w in WHISTLE_SUBTYPES):
        return end
    if event["type"] == "PASS" and any(w in event["subtype"] for w in PASS_WHISTLE_SUBTYPES):
        return end
    return None


def periods(events):
    """{period: (first_frame, last_frame)} inclusive, 0-based, from the events themselves."""
    spans = {}
    for e in events:
        last = max(e["start"], e["end"] if e["end"] is not None else e["start"])
        first0, last0 = spans.get(e["period"], (e["start"], last))
        spans[e["period"]] = (min(first0, e["start"]), max(last0, last))
    return dict(sorted(spans.items()))


def ownership_changes(events):
    """[(frame, team)] in time order: the frame from which `team` is in possession."""
    changes, owner, candidate_start = [], None, None
    for e in events:
        if e["type"] not in IN_PLAY or e["team"] is None:
            continue
        if e["team"] == owner:
            candidate_start = None
            continue
        if candidate_start is None:
            candidate_start = e["start"]
        if is_control(e):
            owner = e["team"]
            changes.append((candidate_start, owner))
            candidate_start = None
    return changes


def dead_windows(events):
    """[(start, stop)] half-open frame spans where the ball is dead."""
    windows = []
    for k, e in enumerate(events):
        start = dead_start(e)
        if start is None:
            continue
        stop = None
        for later in events[k + 1:]:
            if later["start"] > start and is_control(later):
                stop = later["start"]
                break
        if stop is None:  # nothing restarts it: dead to the end of the period
            stop = np.iinfo(np.int64).max
        windows.append((start, stop))
    return windows


def labels_from_events(events, n_frames):
    """(labels <U1 array of H/A/D/X, half_blocks [(period, first, last)], kickoffs (f1, f2))."""
    events = sorted(events, key=lambda e: (e["start"], e["order"]))
    labels = np.full(n_frames, "X", dtype="<U1")
    spans = periods(events)

    for period, (first, last) in spans.items():
        in_period = [e for e in events if e["period"] == period]
        segment = np.full(last - first + 1, "D", dtype="<U1")  # D until the first possessor
        for frame, team in ownership_changes(in_period):
            segment[max(frame, first) - first:] = "H" if team == HOME else "A"
        for start, stop in dead_windows(in_period):
            segment[max(start, first) - first:max(0, min(stop, last + 1) - first)] = "D"
        labels[first:last + 1] = segment

    half_blocks = [(p, first, last) for p, (first, last) in spans.items()]
    kickoffs = tuple(first for _, first, _ in half_blocks)
    return labels, half_blocks, kickoffs
