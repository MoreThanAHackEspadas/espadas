"""Phase 8.4: the `X` policy ("we decline to answer") on graded frames.

How the organisers score an X (official/EVALUATION.md, possession_metrics.py):
- spellF1 (ranking) never sees it: `fill_abstentions` forward-fills it with our last label in the
  half (back-filled at the start), then possessions are matched. So an X can never raise spellF1.
- macroF1 (first tie-break) counts it as a miss for the true class but as a false positive for
  no class. A wrong guess costs a miss *and* a false positive, so an X beats a guess that is
  probably wrong.

The policy, applied to a finished label array (after write.finalise / the bench's X on ungraded
frames), so it works for any labeler:
1. **Where X is allowed (hard rule, never tuned):** only strictly inside a decoded H or A spell,
   at least `edge` (>= 1) frames from both ends. The frames either side of every X stretch then
   carry that spell's label, so the scorer's fill restores exactly the file without X:
   spellF1 provably cannot change (tests/test_abstain.py checks the fill identity).
2. **Where X is wanted (tuned on Metrica 1 + 3, objective macroF1):** the ball is not tracked
   and the Phase 5 team score is weak (|team_lo| < `weak`) for at least `min_gap` frames in a
   row, i.e. stretches where we are guessing who has it.

Settings are per labeler (PARAMS[labeler]); None = no X on graded frames. Tuned by
tools/tune_abstain.py on the tuning games only; never on our match.

Which setting a build uses is decided in one place, resolve(), shared by espadas.run, the bench
and tools/selection.py: PARAMS[labeler] alone decides what `python -m espadas.run` does with no
flag; --abstain forces the policy on (SETTING when PARAMS[labeler] is None), --no-abstain forces
it off.
"""

import numpy as np

# None = policy off for that labeler. Tuned 5 Nov (tools/tune_abstain.py, Metrica 1 + 3, all
# tracker levels, 25% live ball hidden; best setting the same for every labeler:
# weak = inf, min_gap = 200, edge = 1, i.e. "ball missing for 8 s inside an H/A spell"):
#   v1  0.8310 -> 0.8327 (+0.002)           below noise, rejected
#   v3  0.8108 -> 0.8389 (+0.028), X on 7.8% of graded frames
#   v4  0.8082 -> 0.8361 (+0.028), X on 7.8%; +0.028 / +0.027 at 15% / 35% hidden
# Not shipped anyway: on v4, 89% of those X frames are truly dead (Metrica's ball-less
# stoppages); on the live ones our guess is right as often as wrong (6% / 6%). The gain is the
# Metrica "no ball = stoppage" pattern again, which decision (b) (checklist 6.6) refused to rely
# on because our ball is often missing in live play. Off in the submission by the selection rule
# (docs/decision_rule.md rule 5, 10 Nov). Kept in code (spellF1-neutral, tested): setting
# PARAMS[<labeler>] = SETTING here makes the plain `python -m espadas.run` (no flag) apply it,
# exactly the "X on" variant tools/selection.py scored (both go through resolve()).
SETTING = {"weak": np.inf, "min_gap": 200, "edge": 1}
PARAMS = {"v1": None, "v2": None, "v3": None, "v4": None}

GRID = {
    "weak": [0.25, 0.5, 1.0, 2.0, 4.0, np.inf],
    "min_gap": [1, 12, 25, 50, 100, 200, 300, 500, 750, 1000, 1500],
    "edge": [1, 6, 12, 25, 50],
}


def resolve(labeler, force=None):
    """The X setting to apply for `labeler`, or None for no X on graded frames.

    force None (no flag): PARAMS[labeler] decides.
    force True (--abstain): PARAMS[labeler], or SETTING if that is None.
    force False (--no-abstain): None.
    """
    if force is False:
        return None
    p = PARAMS.get(labeler)
    if p is None and force:
        return SETTING
    return p


def runs(mask):
    """[(start, stop)] of the True runs of a bool array."""
    edges = np.diff(np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def allowed(labels, edge):
    """bool per frame: strictly inside an H or A spell, >= edge frames from both of its ends."""
    edge = max(1, int(edge))
    lab = np.asarray(labels)
    out = np.zeros(len(lab), dtype=bool)
    for team in "HA":
        for start, stop in runs(lab == team):
            if stop - start > 2 * edge:
                out[start + edge:stop - edge] = True
    return out


def wanted(present, team_lo, weak, min_gap):
    """bool per frame: ball not tracked and team score weak, for >= min_gap frames in a row."""
    unsure = ~np.asarray(present, dtype=bool) & (np.abs(np.nan_to_num(team_lo)) < weak)
    out = np.zeros(len(unsure), dtype=bool)
    for start, stop in runs(unsure):
        if stop - start >= min_gap:
            out[start:stop] = True
    return out


def apply(labels, present, team_lo, p):
    """Copy of labels with X where the policy allows and wants it. p None = unchanged."""
    out = np.asarray(labels, dtype="<U1").copy()
    if p is None:
        return out
    out[allowed(out, p["edge"]) & wanted(present, team_lo, p["weak"], int(p["min_gap"]))] = "X"
    return out
