"""v2 labeler: v1's decoding, driven by the Phase 5 evidence instead of v1's hand-made rules.

Same two steps as v1 (see v1.py):

1. **Dead ball.** The dead log-odds from evidence.py (ball and player motion when the ball is
   tracked, player motion only when it is not), smoothed over +/- `dead_smooth` frames; dead
   where it is above `dead_threshold`. Then v1's clean-up: dead runs under `min_dead` dropped,
   live runs under `min_alive` between stoppages filled.
2. **Who has it.** CUSUM on the team log-odds (`cusum()`): evidence against the current owner,
   capped at `team_cap` per frame, piles up; evidence for the owner drains it; at zero it stops.
   Once the pile passes `switch_cost` the other team takes over, dated back to when the pile last
   started from zero (the ball was lost then, not when we became sure). Unlike v1's yes/no vote
   this uses how strong each frame's evidence is: a clear 0.5 s in the other team's feet
   counts as much as a vague 2 s. A loose or unseen ball (log-odds near 0) changes nothing.

v2 keeps v1's dead-ball clean-up and swaps in the Phase 5 evidence plus a switching rule that can
use evidence strength; the 5-state model (Phase 7) will replace the decoding. Tracking only, no Smart Tagging. Decoding
settings tuned by tools/tune.py --labeler v2 on Metrica 1 and 3; never on our match.
"""

import numpy as np

from . import evidence, v1

HOME, AWAY, NONE = v1.HOME, v1.AWAY, v1.NONE

# Tuned with tools/tune.py --labeler v2 on Metrica 1 and 3, ball hidden on 25% of live frames.
# Mean tuning spellF1 0.610 (v1: 0.591); held-out Game 2 0.634 / 0.636 / 0.620 at 25% / 15% / 35%
# live missing (v1: 0.640 / 0.656 / 0.608). Not a clear win on held-out, so v1 stays the default.
# dead_threshold < 0 means the decoder calls dead more readily than the fitted model's 50/50 point;
# why is not yet understood (check with Phase 6 before trusting it).
PARAMS = {
    "dead_smooth": 6,        # frames each side
    "dead_threshold": -2.0,  # log-odds
    "min_dead": 100,         # frames
    "min_alive": 25,         # frames
    "team_cap": 1.0,         # log-odds per frame, cap on any one frame's evidence
    "switch_cost": 30.0,     # summed log-odds against the owner needed to switch
}

GRID = {
    "dead_smooth": [0, 6, 12, 25, 50, 75],
    "dead_threshold": [-4.0, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0],
    "min_dead": [0, 25, 50, 75, 100, 150],
    "min_alive": [0, 25, 50, 100, 150, 250],
    "team_cap": [0.25, 0.5, 1.0, 2.0, 3.0, 5.0],
    "switch_cost": [2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0],
}


def features(tracking):
    c = evidence.clues(tracking)
    dead, team = evidence.log_odds(c)
    return {"half_blocks": c["half_blocks"], "present": c["present"],
            "dead_lo": dead, "team_lo": team}


def dead_mask(f, p, first, stop):
    lo = evidence.smooth(f["dead_lo"][first:stop], int(p["dead_smooth"]))
    dead = lo > p["dead_threshold"]
    for a, b in v1._runs(dead):
        if b - a < p["min_dead"]:
            dead[a:b] = False
    alive = ~dead
    for a, b in v1._runs(alive):
        if a > 0 and b < len(alive) and b - a < p["min_alive"]:
            dead[a:b] = True
    return dead


def cusum(lo, cap, cost, start_owner):
    """Owner per frame of one live segment from home log-odds `lo` (home +).

    The first owner is the sign of the summed evidence over the segment's opening second (or
    carried over from before the stoppage if that is a tie)."""
    n = len(lo)
    lo = np.clip(lo, -cap, cap)
    head = float(np.sum(lo[:25]))
    owner = HOME if head > 0 else AWAY if head < 0 else start_owner
    if owner == NONE:
        owner = HOME if float(np.sum(lo)) >= 0 else AWAY
    out = np.empty(n, dtype=np.int8)
    pile, since, done = 0.0, 0, 0
    for t in range(n):
        against = -lo[t] if owner == HOME else lo[t]
        pile += against
        if pile <= 0.0:
            pile, since = 0.0, t + 1
        elif pile >= cost:
            out[done:since] = owner
            owner = AWAY if owner == HOME else HOME
            done, pile, since = since, 0.0, t + 1
    out[done:] = owner
    return out, owner


def decode(f, params=None):
    """'H'/'A'/'D' per frame; frames outside the halves are D (finalise() / the bench make X)."""
    p = dict(PARAMS, **(params or {}))
    labels = np.full(len(f["present"]), "D", dtype="<U1")
    for _, first, last in f["half_blocks"]:
        stop = last + 1
        dead = dead_mask(f, p, first, stop)
        lo = f["team_lo"][first:stop]
        owner = NONE
        for a, b in v1._runs(~dead):
            seg, owner = cusum(lo[a:b], p["team_cap"], p["switch_cost"], owner)
            labels[first + a:first + b] = np.where(seg == HOME, "H",
                                                   np.where(seg == AWAY, "A", "D"))
    return labels


def v2(tracking, params=None):
    return decode(features(tracking), params)
