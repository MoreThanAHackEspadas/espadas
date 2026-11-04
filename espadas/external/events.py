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
- **Touch-only events**: RECOVERY, CHALLENGE, and headers/clearances. They never change
  possession on their own, but if the same team follows with a control event before the other
  team does anything in play, the change is dated back to the first of those events (the moment
  the ball was won).
- **Dead ball** starts at: BALL OUT (its end frame: when the ball left the pitch), a SHOT whose
  subtype ends in OUT or GOAL (its end frame), FAULT RECEIVED (its start frame: the whistle), a
  BALL LOST tagged OFFSIDE or HAND BALL (its end frame). It
  ends at the start of the next control event (the restart: a set piece, a goal-kick pass, ...).
- Graded frames per period run from its first event (the kickoff) to its last event's end.
  Everything else is X (ungraded), like halftime in our match.

Control vs loose (Phase 7): `control_mask` marks frames where a player has the ball at his feet,
the complement being loose (in flight or contested). Windows are half-open, per period:

- PASS / SET PIECE that is_control() with end > start: loose [start + 1, min(end, next)). The kick
  frame and the reception frame are at a player's feet. `next` is the start of the next in-play
  event with a team, so no window reaches past the frame where a new spell can start.
- SHOT that is_control(): loose [start + 1, min(end + 1, next)) (end is start when missing). Same
  shape as a pass: the kick frame is controlled, the end frame is not a reception.
- BALL LOST: loose [start + 1, next in-play event's start), to the period's end if none.
- Touch-only in-play events (RECOVERY, CHALLENGE, headers, clearances): scan forward over in-play
  events that have a team. First by the other team: loose [start, that event's start). First by
  the same team that is_control(): no window (the "won ball" rule). Same team, touch-only: keep
  scanning. None found: loose to the period's end.
- CARRY and everything else: no window.

Consequence: the mask is True wherever the reference labels start a new possession spell.

Frames are converted from Metrica's 1-based numbering to 0-based array indices here.
These are fixed definitions, not tuned parameters. Fully deterministic.
"""

import numpy as np

HOME, AWAY = 0, 1

IN_PLAY = {"PASS", "SHOT", "CARRY", "RECOVERY", "BALL LOST", "SET PIECE"}
LOOSE_SOURCES = IN_PLAY | {"CHALLENGE"}  # CHALLENGE is touch-only but not in IN_PLAY
CONTROL = {"PASS", "SHOT", "CARRY", "BALL LOST", "SET PIECE"}
TOUCH_SUBTYPES = ("HEAD", "CLEARANCE")
# BALL LOST subtypes that mean the referee stopped play (Game 3 tags these instead of a foul).
WHISTLE_SUBTYPES = ("OFFSIDE", "HAND BALL")


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


def loose_windows(events):
    """[(start, stop)] half-open 0-based frame spans where the ball is in flight or contested.

    `events` are the (already sorted) events of one period. Rules in the module docstring.
    """
    forever = np.iinfo(np.int64).max
    windows = []
    for k, e in enumerate(events):
        later = [x for x in events[k + 1:] if x["type"] in IN_PLAY]
        next_start = next((x["start"] for x in later if x["team"] is not None), forever)
        end = e["end"] if e["end"] is not None else e["start"]
        if e["type"] in ("PASS", "SET PIECE") and is_control(e):
            if end > e["start"]:
                windows.append((e["start"] + 1, min(end, next_start)))
        elif e["type"] == "SHOT" and is_control(e):
            windows.append((e["start"] + 1, min(max(end, e["start"]) + 1, next_start)))
        elif e["type"] == "BALL LOST" and is_control(e):
            windows.append((e["start"] + 1, later[0]["start"] if later else forever))
        elif e["type"] in LOOSE_SOURCES and not is_control(e):
            stop = forever
            for x in later:
                if x["team"] is None:
                    continue
                if x["team"] != e["team"]:
                    stop = x["start"]
                    break
                if is_control(x):  # the same team won the ball: no loose window
                    stop = None
                    break
            if stop is not None:
                windows.append((e["start"], stop))
    return windows


def control_mask(events, n_frames):
    """bool array: True where a player has the ball at his feet, False where it is loose.

    False outside the periods. Only meaningful on live frames; mask with the H/A/D labels.
    """
    events = sorted(events, key=lambda e: (e["start"], e["order"]))
    mask = np.zeros(n_frames, dtype=bool)
    for period, (first, last) in periods(events).items():
        in_period = [e for e in events if e["period"] == period]
        mask[first:last + 1] = True
        for start, stop in loose_windows(in_period):
            lo, hi = max(start, first), min(stop, last + 1)
            if hi > lo:
                mask[lo:hi] = False
    return mask


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
