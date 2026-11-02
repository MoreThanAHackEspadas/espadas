"""Step 1: clean the tracking and derive the per-frame base arrays every later step reads.

    tracking = atd.load_tracking()
    base = prep.prepare(tracking, tagging.load_events())

Everything here is algorithmic and applies the same way to any match, so it can run unchanged on
the external validation games. Nothing is corrected by hand and the raw files are never modified.

The thresholds below are physical starting points, not values fitted to this match. They are to
be checked on other games (plan, Phase 4), never on the public five minutes. The frame-count
thresholds assume 25 fps.
"""

import numpy as np

from . import config, tagging

HOME, AWAY = 0, 1

MAX_BALL_SPEED = 40.0     # m/s. The hardest shots are ~35 m/s; faster implied motion is a jump.
MAX_FILL_GAP = 25         # frames (1 s). Longer ball gaps stay missing and are measured instead.
SPEED_HALF_WINDOW = 6     # frames each side, so ball speed is measured over ~0.5 s.
KICKOFF_WINDOW = 25       # frames each side of a kickoff used to see which half a team defends.


def _speeds_between(points, frames, fps):
    """m/s between consecutive observations, over however many frames separate them."""
    step = np.diff(points, axis=0)
    return np.hypot(step[:, 0], step[:, 1]) / (np.diff(frames) / fps)


def reject_ball_jumps(ball, fps=config.FPS, max_speed=MAX_BALL_SPEED):
    """(ball with isolated jumps set to NaN, bool mask of the rejected frames).

    An observation is a jump when the implied speed both from the previous observation and to the
    next one exceeds max_speed: the tracker briefly put the ball somewhere it could not have been.
    The first and last observations are never rejected. Two consecutive bad points are not caught.
    """
    seen = np.flatnonzero(~np.isnan(ball[:, 0]))
    rejected = np.zeros(len(ball), dtype=bool)
    if len(seen) >= 3:
        speed = _speeds_between(ball[seen], seen, fps)
        jump = (speed[:-1] > max_speed) & (speed[1:] > max_speed)
        rejected[seen[1:-1][jump]] = True
    cleaned = ball.copy()
    cleaned[rejected] = np.nan
    return cleaned, rejected


def fill_ball_gaps(ball, fps=config.FPS, max_gap=MAX_FILL_GAP, max_speed=MAX_BALL_SPEED):
    """(ball with short gaps filled by straight lines, bool mask of the filled frames).

    A gap is filled only when it is at most max_gap frames, has an observation on both sides, and
    the straight line between them is no faster than max_speed.
    """
    seen = np.flatnonzero(~np.isnan(ball[:, 0]))
    filled = ball.copy()
    mask = np.zeros(len(ball), dtype=bool)
    if len(seen) < 2:
        return filled, mask

    gap = np.diff(seen) - 1
    speed = _speeds_between(ball[seen], seen, fps)
    for k in np.flatnonzero((gap > 0) & (gap <= max_gap) & (speed <= max_speed)):
        start, end = seen[k], seen[k + 1]
        t = (np.arange(start + 1, end) - start)[:, None] / (end - start)
        filled[start + 1:end] = ball[start] + t * (ball[end] - ball[start])
        mask[start + 1:end] = True
    return filled, mask


def gap_lengths(present):
    """(since, run), int32 per frame, both 0 where the ball is present.

    since: frames since the ball was last present, counting 1 on the first missing frame.
    run:   total length of the missing stretch the frame sits in.
    """
    missing = np.concatenate(([0], (~np.asarray(present)).astype(np.int8), [0]))
    edges = np.diff(missing)
    since = np.zeros(len(present), dtype=np.int32)
    run = np.zeros(len(present), dtype=np.int32)
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        since[start:end] = np.arange(1, end - start + 1)
        run[start:end] = end - start
    return since, run


def ball_speed(ball, fps=config.FPS, half_window=SPEED_HALF_WINDOW):
    """m/s per frame from the displacement between frames i - w and i + w. NaN if either is missing.

    Measured over a window rather than frame to frame, so tracker jitter does not read as motion.
    """
    n, w = len(ball), half_window
    speed = np.full(n, np.nan)
    if n > 2 * w:
        step = ball[2 * w:] - ball[:n - 2 * w]
        speed[w:n - w] = np.hypot(step[:, 0], step[:, 1]) / (2 * w / fps)
    return speed


def attacking_direction(players, team_of_slot, half_blocks=config.HALF_BLOCKS,
                        kickoffs=config.KICKOFFS, window=KICKOFF_WINDOW):
    """int8 (n_frames, 2), column per team: +1 attacking towards x = pitch length, -1 towards
    x = 0, 0 on ungraded frames.

    Read from where each team stands around its half's kickoff, when every player is in their own
    half: the team with the smaller median x defends the x = 0 goal.
    """
    direction = np.zeros((len(players), 2), dtype=np.int8)
    for (number, first, last), kickoff in zip(half_blocks, kickoffs):
        around = players[max(0, kickoff - window):kickoff + window + 1]
        xs = []
        for team in (HOME, AWAY):
            x = around[:, team_of_slot == team, 0]
            x = x[~np.isnan(x)]
            if not x.size:
                raise ValueError("half {}: no {} players tracked around the kickoff".format(
                    number, "home" if team == HOME else "away"))
            xs.append(np.median(x))
        if xs[HOME] == xs[AWAY]:
            raise ValueError("half {}: cannot tell which half each team defends".format(number))
        home = 1 if xs[HOME] < xs[AWAY] else -1
        direction[first:last + 1, HOME] = home
        direction[first:last + 1, AWAY] = -home
    return direction


def directions_swap(direction, half_blocks=config.HALF_BLOCKS):
    """True if the teams change ends at halftime, as they must."""
    (_, first1, _), (_, first2, _) = half_blocks
    return bool(direction[first1, HOME] == -direction[first2, HOME] != 0)


def prepare(tracking, events=None, half_blocks=config.HALF_BLOCKS, kickoffs=config.KICKOFFS,
            fps=config.FPS):
    """Every stage-0 per-frame array for one match, in a dict. Raw tracking is left untouched.

    Defaults describe Barcelona v Athletic. For another match (e.g. an external validation game)
    pass its own half_blocks [(1, first, last), (2, first, last)] in frames, inclusive, its two
    kickoff frames and its fps; tracking needs players, ball (metres) and team_of_slot.
    """
    raw_ball = tracking["ball"]
    ball, rejected = reject_ball_jumps(raw_ball, fps)
    ball, filled = fill_ball_gaps(ball, fps)
    present = ~np.isnan(ball[:, 0])
    since, run = gap_lengths(present)

    base = {
        "ball": ball,                              # metres, cleaned and short gaps filled
        "ball_observed": ~np.isnan(raw_ball[:, 0]) & ~rejected,
        "ball_rejected": rejected,
        "ball_filled": filled,
        "ball_present": present,
        "ball_gap_since": since,
        "ball_gap_run": run,
        "ball_speed": ball_speed(ball, fps),       # from positions, over ~0.5 s
        "attack": attacking_direction(tracking["players"], tracking["team_of_slot"],
                                      half_blocks, kickoffs),
        "half": config.half_of_frame(len(raw_ball), half_blocks),
    }
    if events is not None:
        base["set_piece"] = tagging.set_piece_mask(events, len(raw_ball))
    return base
