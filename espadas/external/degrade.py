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

**Ball blur (checklist 6.6).** The Metrica ball is drawn as straight lines between key points: it
moves in exactly even steps (0.400 m per frame for stretches), and on ~90% of frames its path bends
by less than 1 cm. So every kick is a razor-sharp corner. Any automatic tracker smooths the ball's
path instead (a kick becomes a curve over several frames) and adds small position noise. A touch
detector tuned on the sharp corners found 1,300-1,600 touches per 90' on Metrica but 11 on our
match. blur_ball() smooths the Metrica ball with a Gaussian of `sigma` frames and adds `noise`
metres of position noise, before the ball is hidden. The levels in BLUR are generic tracker-like
values (0.08-0.32 s of smoothing, 2-10 cm noise), not fitted to our match; like LIVE_MISSING,
methods are tuned at the middle level and must also hold at the others.

**Smooth tracker and player error (checklist 6.7).** Even after 6.6 the touch detector found far
fewer touches on our match than on the blurred Metrica games. Checked on 5 Nov, with only yes/no
answers on our match: at the moments the ball's path bends, a player is close by as often as on
Metrica, but our ball bends far less often than Metrica's at *any* blur level. The BLUR levels pair
more smoothing with more noise, and the noise itself makes kinks (heavy: ~6,600 per 90'), so no
level looked like a tracker that smooths strongly with little noise, which is what a tracker with
a smoothing filter (Kalman, spline) does. So:
- a fourth level, "smooth": 12 frames (0.48 s) of smoothing, 2 cm noise. Set by reasoning (the
  smoothing of a filtered tracker), not fitted to our match.
- player position error on every level (PLAYER_ERROR: 0.3 / 0.6 / 1.0 m per axis, smooth
  offsets over ~1 s, as an automatic tracker drifts). Metrica's players are near-perfect; with
  ~1 m of error the median ball-to-nearest-player distance on Metrica goes from 0.7-1.4 m to
  1.7-2.1 m, the size of the gap the team measured in 6.6. Generic levels, not fitted.
TRACKER_LEVELS lists the levels every tuning and fitting tool must cover at once.
"""

import numpy as np

CROWD_RADIUS = 5.0       # metres; players within this of the ball make a moment "crowded"
MEAN_BURST = 25          # frames; mean length of an added gap (geometric)
SEED = 0
LIVE_MISSING = 0.25     # fraction of live frames with no ball, after hiding
BLUR = {"none": None, "light": (2.0, 0.02), "medium": (4.0, 0.05), "heavy": (8.0, 0.10),
        "smooth": (12.0, 0.02)}
PLAYER_ERROR = {"none": 0.0, "light": 0.3, "medium": 0.6, "heavy": 1.0, "smooth": 0.6}  # m/axis
PLAYER_ERROR_FRAMES = 25   # correlation time of the player offsets (~1 s)
TRACKER_LEVELS = ("light", "medium", "heavy", "smooth")
BLUR_DEFAULT = "medium"
BLUR_SEED = 1
PLAYER_SEED = 2


def blur_ball(ball, sigma, noise, seed=BLUR_SEED):
    """Copy of ball (n, 2) smoothed by a Gaussian of `sigma` frames within each run of tracked
    frames (never across a gap), plus N(0, noise) metres of position noise per axis."""
    out = ball.copy()
    present = ~np.isnan(ball[:, 0])
    if sigma > 0:
        radius = int(np.ceil(3 * sigma))
        kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
        edges = np.diff(np.concatenate(([0], present.astype(np.int8), [0])))
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            n = b - a
            weight = np.convolve(np.ones(n), kernel)[radius:radius + n]
            for axis in (0, 1):
                out[a:b, axis] = np.convolve(ball[a:b, axis], kernel)[radius:radius + n] / weight
    if noise > 0:
        rng = np.random.default_rng(seed)
        out[present] += rng.normal(0.0, noise, (int(present.sum()), 2))
    return out


def jitter_players(players, rms, corr=PLAYER_ERROR_FRAMES, seed=PLAYER_SEED):
    """Copy of players (n, slots, 2) plus a smooth random offset per slot and axis: white noise
    filtered by a Gaussian of `corr` frames, scaled to `rms` metres. NaN (untracked) stays NaN.
    Deterministic (fixed seed)."""
    if rms <= 0:
        return players.copy()
    rng = np.random.default_rng(seed)
    n, slots, _ = players.shape
    radius = int(3 * corr)
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / corr) ** 2)
    white = rng.normal(size=(n + 2 * radius, slots, 2))
    offset = np.empty((n, slots, 2))
    for s in range(slots):
        for axis in (0, 1):
            offset[:, s, axis] = np.convolve(white[:, s, axis], kernel, "valid")
    offset *= rms / offset.std()
    return players + offset


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
