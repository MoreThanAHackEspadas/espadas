"""Make external games' ball tracking as patchy as ours *during live play* (checklist 4.8).

Our match is missing the ball on ~41% of graded frames, and the organisers' DATA.md says it is
often missing in live play: tackles, crowded boxes, balls in the air. The Metrica games are
missing about as much overall (~40%), but almost all of it is during stoppages (84-100% of dead
frames, only 6-8% of live frames). Tuned on that as-is, a labeler learns "no ball = dead", which
would not hold on our match.

So the bench hides ball bursts in *live* play (reference H/A frames) until LIVE_MISSING of them
have no ball, preferring crowded moments. LIVE_MISSING is an assumption, not a measurement: if
our dead share is 25-35% and dead frames mostly lack a ball, 41% overall implies roughly 15-30%
of live frames without a ball. 0.25 is the middle; tools/tune_v1.py also reports 0.15 and 0.35
so a setting that only works at one level is visible.

Deterministic: fixed seed, so every bench run sees the same degraded ball.
"""

import numpy as np

CROWD_RADIUS = 5.0       # metres; players within this of the ball make a moment "crowded"
MEAN_BURST = 25          # frames; mean length of an added gap (geometric)
SEED = 0
LIVE_MISSING = 0.25     # fraction of live frames with no ball, after hiding


def hide_ball(ball, players, graded, target_missing, seed=SEED, mean_burst=MEAN_BURST,
              crowd_radius=CROWD_RADIUS):
    """(ball copy with extra NaN bursts, bool mask of hidden frames).

    graded is the mask of frames the target applies to (the bench passes live frames).
    target_missing is the wanted fraction of those frames with no ball. If the game is already at
    or above it, nothing is hidden.
    """
    rng = np.random.default_rng(seed)
    ball = ball.copy()
    hidden = np.zeros(len(ball), dtype=bool)
    present = ~np.isnan(ball[:, 0]) & graded
    n_graded = int(graded.sum())
    to_hide = int(round(target_missing * n_graded)) - int(n_graded - present.sum())
    if to_hide <= 0:
        return ball, hidden

    # Crowding weight per frame: 1 + number of players within crowd_radius of the ball.
    diff = players - ball[:, None, :]
    near = np.nansum(np.hypot(diff[..., 0], diff[..., 1]) < crowd_radius, axis=1)
    weight = np.where(present, 1.0 + near, 0.0)

    candidates = np.flatnonzero(weight > 0)
    order = rng.choice(candidates, size=len(candidates), replace=False,
                       p=weight[candidates] / weight[candidates].sum())
    lengths = rng.geometric(1.0 / mean_burst, size=len(order))
    done = 0
    for start, length in zip(order, lengths):
        if done >= to_hide:
            break
        stop = min(len(ball), start + int(length))
        span = np.arange(start, stop)
        span = span[present[span] & ~hidden[span]][:to_hide - done]
        hidden[span] = True
        done += len(span)
    ball[hidden] = np.nan
    return ball, hidden
