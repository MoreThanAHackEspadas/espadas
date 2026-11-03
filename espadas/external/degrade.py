"""Make external games' ball tracking as patchy as ours (checklist 4.8).

Our match is missing the ball on ~41% of graded frames, mostly in bursts and in crowded
moments (tackles, boxes, balls in the air). External tracking is cleaner, so settings tuned on it
would be over-optimistic. hide_ball() removes extra ball observations in bursts until a target
missing fraction is reached, preferring frames where many players are near the ball.

Deterministic: fixed seed, so every bench run sees the same degraded ball.
"""

import numpy as np

CROWD_RADIUS = 5.0       # metres; players within this of the ball make a moment "crowded"
MEAN_BURST = 25          # frames; mean length of an added gap (geometric)
SEED = 0


def hide_ball(ball, players, graded, target_missing, seed=SEED, mean_burst=MEAN_BURST,
              crowd_radius=CROWD_RADIUS):
    """(ball copy with extra NaN bursts, bool mask of hidden frames).

    target_missing is the wanted fraction of graded frames with no ball (e.g. 0.41). If the game
    is already at or above it, nothing is hidden.
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
