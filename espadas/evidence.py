"""Stage 1: per-frame evidence (checklist Phase 5). Tracking -> clues -> two log-odds per frame.

    clues = evidence.clues(tracking)          # dict of per-frame arrays, any match
    dead_lo, team_lo = evidence.log_odds(clues)

`dead_lo` > 0 means the ball is more likely dead than live; `team_lo` > 0 means home is more
likely to have it than away (only meaningful in live play). Later stages (hysteresis, the HMM)
read these two numbers, not the raw clues.

Clues (all from tracking, so they exist on any match, ours and the external games alike):

Dead vs live
- 5.1 ball speed (only when the ball is tracked)
- 5.2 how far the ball is outside the pitch (same)
- 5.3 player motion: mean speed, share of players running (> RUN_SPEED), fastest player. When the
  ball is missing in live play somebody is usually still running; at a stoppage almost nobody is.
  This is what lets us tell a live gap from a stoppage without seeing the ball (part of 5.8).
- Whether the ball is tracked is deliberately NOT a clue: see DEAD_SPLIT.

Home vs away (every team clue is home-minus-away, so swapping the teams flips its sign; the team
model has no intercept, so it has no built-in home bias)
- 5.6 nearest player: which team is closer to the ball, and to the last place it was seen
- 5.7 ball direction: the ball moving towards a team's attacking goal points to that team
- 5.8 convergence on the last-seen ball. On Metrica 1 and 3 the team closing in on the ball is
  usually the *defending* team (pressing); the fitted weight carries that sign.
- team shape: the team with the ball spreads out, the defending team stays compact. Not in the
  original plan; it was the strongest single team clue on the tuning games, and it does not need
  the ball, so it carries through ball gaps.

Not used, and why (checklist):
- 5.4 number of tracked objects: always 22 in the external games, so its effect cannot be
  measured or tuned there.
- 5.5 / 5.9 Smart Tagging: the external games have none, same problem. Team decision (5 Nov): not used.

5.10 Combining: two logistic models over the clues, fitted by tools/fit_evidence.py on the
external tuning pool only (bench.DEV_TUNING: Metrica 1, 3 and IDSSE 12, 13, 15 since 11 Nov;
ball hidden on part of live play, degrade.py) and
written to evidence_models.py as constants, so building our file needs no external data. Never
fitted on our match.
"""

import numpy as np

from . import config, prep

HOME, AWAY = 0, 1

MOTION_HALF_WINDOW = 6      # frames each side (~0.5 s), as prep.ball_speed
RUN_SPEED = 4.0             # m/s; a player faster than this is running
NEAR_CLIP = 10.0            # m; distance differences are clipped to +/- this
BALL_SPEED_CLIP = 30.0      # m/s
BALL_VX_CLIP = 20.0         # m/s
OUT_CLIP = 10.0             # m
CONTEXT = (25, 75)          # frames each side for the smoothed copies of each clue (1 s, 3 s)

# --------------------------------------------------------------------------- clue extraction


def _velocity(x, fps, w=MOTION_HALF_WINDOW):
    """d x / d t over frames i - w .. i + w, along axis 0; NaN at the edges or if either missing."""
    out = np.full(x.shape, np.nan)
    if len(x) > 2 * w:
        out[w:len(x) - w] = (x[2 * w:] - x[:len(x) - 2 * w]) / (2 * w / fps)
    return out


def smooth(x, half_window):
    """Centred moving mean over +/- half_window frames, ignoring NaN; NaN if the window is empty."""
    if half_window <= 0:
        return x.astype(float)
    finite = np.isfinite(x)
    value = np.concatenate(([0.0], np.cumsum(np.where(finite, x, 0.0))))
    count = np.concatenate(([0], np.cumsum(finite)))
    index = np.arange(len(x))
    low = np.maximum(0, index - half_window)
    high = np.minimum(len(x), index + half_window + 1)
    n = count[high] - count[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (value[high] - value[low]) / np.maximum(n, 1), np.nan)


def _nanmean_rows(x):
    count = np.sum(~np.isnan(x), axis=1)
    with np.errstate(invalid="ignore"):
        return np.where(count > 0, np.nansum(x, axis=1) / np.maximum(count, 1), np.nan)


def _spread(team_players):
    """Std of x plus std of y over a team's tracked players, per frame (m)."""
    out = np.full(len(team_players), np.nan)
    count = np.sum(~np.isnan(team_players[..., 0]), axis=1)
    ok = count >= 2
    if ok.any():
        out[ok] = (np.nanstd(team_players[ok, :, 0], axis=1)
                   + np.nanstd(team_players[ok, :, 1], axis=1))
    return out


def _last_seen(ball, present):
    """Ball position held from the last frame it was present; NaN before the first."""
    index = np.where(present, np.arange(len(ball)), -1)
    index = np.maximum.accumulate(index)
    out = ball[np.maximum(index, 0)].copy()
    out[index < 0] = np.nan
    return out


def _nearest(players, point, team_of_slot):
    """(nearest home distance, nearest away distance, mean of the 3 nearest home, same away)."""
    diff = players - point[:, None, :]
    dist = np.hypot(diff[..., 0], diff[..., 1])
    dist = np.where(np.isnan(dist), np.inf, dist)
    out = []
    for team in (HOME, AWAY):
        d = np.sort(dist[:, team_of_slot == team], axis=1)
        near1 = d[:, 0] if d.shape[1] else np.full(len(d), np.inf)
        near3 = d[:, :3].mean(axis=1) if d.shape[1] >= 3 else near1
        out += [near1, near3]
    return out[0], out[2], out[1], out[3]


def _diff(away, home, clip=NEAR_CLIP):
    """away - home, clipped, NaN where either is unknown (inf)."""
    with np.errstate(invalid="ignore"):
        d = away - home
    d[~(np.isfinite(home) & np.isfinite(away))] = np.nan
    return np.clip(d, -clip, clip)


def clues(tracking):
    """Per-frame clue arrays for one match. tracking may carry half_blocks, kickoffs, fps, pitch
    (external games); otherwise our match's values from config are used."""
    half_blocks = tracking.get("half_blocks", config.HALF_BLOCKS)
    kickoffs = tracking.get("kickoffs", config.KICKOFFS)
    fps = tracking.get("fps", config.FPS)
    length, width = tracking.get("pitch", config.PITCH)
    players, team_of_slot = tracking["players"], tracking["team_of_slot"]

    base = prep.prepare(tracking, half_blocks=half_blocks, kickoffs=kickoffs, fps=fps)
    ball, present = base["ball"], base["ball_present"]

    # Dead vs live
    with np.errstate(invalid="ignore"):
        out_dist = np.fmax(np.fmax(-ball[:, 0], ball[:, 0] - length),
                           np.fmax(-ball[:, 1], ball[:, 1] - width))
    step = _velocity(players, fps)
    speed = np.hypot(step[..., 0], step[..., 1])
    tracked = np.sum(~np.isnan(speed), axis=1)
    with np.errstate(invalid="ignore"):
        running = np.where(tracked > 0, np.sum(speed > RUN_SPEED, axis=1) / np.maximum(tracked, 1),
                           np.nan)
        fastest = np.where(tracked > 0, np.nanmax(np.where(np.isnan(speed), -np.inf, speed),
                                                  axis=1), np.nan)

    # Home vs away
    d_home, d_away, _, _ = _nearest(players, ball, team_of_slot)
    last = _last_seen(ball, present)
    l_home, l_away, l3_home, l3_away = _nearest(players, last, team_of_slot)
    attack = base["attack"]
    ball_vx = _velocity(ball[:, 0], fps)
    home_x = _nanmean_rows(players[:, team_of_slot == HOME, 0]) * attack[:, HOME]
    away_x = _nanmean_rows(players[:, team_of_slot == AWAY, 0]) * attack[:, AWAY]

    return {
        "half_blocks": half_blocks,
        "present": present,
        "ball_speed": base["ball_speed"],              # m/s, NaN unseen
        "out_dist": out_dist,                          # m outside the pitch, <= 0 inside, NaN
        "gap_since": base["ball_gap_since"],           # frames since the ball was last present
        "player_speed": _nanmean_rows(speed),          # m/s, mean over tracked players
        "running": running,                            # share of players faster than RUN_SPEED
        "fastest": fastest,                            # m/s
        "d_home": d_home,                              # m to the nearest home player, inf unknown
        "d_away": d_away,
        "near": _diff(d_away, d_home),                 # > 0: home closer to the ball
        "near_last": _diff(l_away, l_home),            # > 0: home closer to the last-seen ball
        # > 0: home's 3 nearest closing on the last-seen ball faster than away's (5.8)
        "converge": np.clip(_velocity(l3_away, fps) - _velocity(l3_home, fps), -10, 10),
        # > 0: ball moving towards the goal home attacks (5.7)
        "ball_dir": np.clip(ball_vx * attack[:, HOME], -BALL_VX_CLIP, BALL_VX_CLIP),
        # > 0: home more spread out than away
        "spread": _spread(players[:, team_of_slot == HOME])
        - _spread(players[:, team_of_slot == AWAY]),
        # > 0: home moving up the pitch faster than away
        "advance": np.clip(_velocity(home_x, fps) - _velocity(away_x, fps), -10, 10),
    }


# --------------------------------------------------------------------------- design matrices

def _fill(x, value=0.0):
    x = np.asarray(x, dtype=float)
    return np.where(np.isfinite(x), x, value)


def _with_context(columns):
    """Each column, plus its smoothed copy at every CONTEXT width."""
    out = []
    for name, x in columns:
        out.append((name, _fill(x)))
        for w in CONTEXT:
            out.append(("{}~{}".format(name, w), _fill(smooth(x, w))))
    return out


def motion_design(c):
    """(names, matrix) from player motion only: the dead model for frames without a ball."""
    columns = _with_context([
        ("player_speed", c["player_speed"]),
        ("running", c["running"]),
        ("fastest", c["fastest"]),
    ])
    names = [n for n, _ in columns]
    return names, np.stack([x for _, x in columns], axis=1)


def ball_design(c):
    """(names, matrix) from player motion plus the ball: the dead model for frames with a ball.

    Ball columns are smoothed over the frames where the ball is present only, so whether the ball
    is tracked is never itself a clue (see DEAD_SPLIT below).
    """
    with np.errstate(invalid="ignore"):
        out = np.clip(c["out_dist"], 0.0, OUT_CLIP)
    names, motion = motion_design(c)
    columns = _with_context([
        ("ball_speed", np.clip(c["ball_speed"], 0, BALL_SPEED_CLIP)),
        ("out", out),
    ])
    names = names + [n for n, _ in columns]
    return names, np.hstack([motion, np.stack([x for _, x in columns], axis=1)])


# DEAD_SPLIT: the dead model is two models. With the ball tracked, it reads the ball and the
# players; without it, the players only. Whether the ball is tracked is deliberately not a clue:
# in the Metrica games it goes missing almost only at stoppages, so a model allowed to see that
# learns "no ball = dead", which our organisers say does not hold on our match (degrade.py). The
# motion-only model is fitted on *all* frames, so its prior is the overall dead share, not the
# (artificially high) dead share among ball-less Metrica frames.


def team_design(c):
    """(names, matrix) for the home-vs-away model. Every column flips sign if the teams swap."""
    columns = _with_context([
        ("near", c["near"]),
        ("near_last", c["near_last"]),
        ("converge", c["converge"]),
        ("ball_dir", c["ball_dir"]),
        ("spread", c["spread"]),
        ("advance", c["advance"]),
    ])
    names = [n for n, _ in columns]
    return names, np.stack([x for _, x in columns], axis=1)


# --------------------------------------------------------------------------- logistic models

def fit_logistic(x, y, intercept, ridge=1.0, iterations=50):
    """L2-regularised logistic regression by Newton's method. Deterministic.

    Columns are scaled to unit spread for the fit (and centred, when there is an intercept), and
    the weights are mapped back, so the returned (bias, weights) apply to the raw columns.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    scale = x.std(axis=0)
    scale[scale == 0] = 1.0
    centre = x.mean(axis=0) if intercept else np.zeros(x.shape[1])
    z = (x - centre) / scale
    if intercept:
        z = np.hstack([np.ones((len(z), 1)), z])
    w = np.zeros(z.shape[1])
    penalty = np.full(z.shape[1], ridge)
    if intercept:
        penalty[0] = 0.0
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-(z @ w)))
        grad = z.T @ (p - y) + penalty * w
        hess = (z * (p * (1 - p))[:, None]).T @ z + np.diag(penalty)
        delta = np.linalg.solve(hess, grad)
        w -= delta
        if np.max(np.abs(delta)) < 1e-8:
            break
    if intercept:
        bias, w = w[0], w[1:]
    else:
        bias = 0.0
    weights = w / scale
    bias = float(bias - np.sum(weights * centre))
    return bias, weights


def apply_logistic(x, bias, weights):
    return bias + np.asarray(x, dtype=float) @ np.asarray(weights, dtype=float)


def _tree_arrays(model):
    """The exported trees as padded arrays (feature, threshold, left, right, value), cached."""
    key = id(model["trees"])
    if key not in _TREE_CACHE:
        trees = model["trees"]
        width = max(len(t["value"]) for t in trees)

        def pad(field, dtype):
            out = np.zeros((len(trees), width), dtype=dtype)
            for i, t in enumerate(trees):
                out[i, :len(t[field])] = t[field]
            return out
        _TREE_CACHE[key] = (model["trees"], pad("feature", np.intp), pad("threshold", float),
                            pad("left", np.intp), pad("right", np.intp), pad("value", float),
                            max(t["depth"] for t in trees))
    return _TREE_CACHE[key][1:]


_TREE_CACHE = {}


def _sum_trees(x, model):
    feature, threshold, left, right, value, depth = _tree_arrays(model)
    rows = np.arange(len(x))
    total = np.full(len(x), float(model["bias"]))
    for t in range(len(feature)):
        node = np.zeros(len(x), dtype=np.intp)
        for _ in range(depth):          # a leaf has left = right = itself, so it stays put
            go_left = x[rows, feature[t][node]] <= threshold[t][node]
            node = np.where(go_left, left[t][node], right[t][node])
        total += value[t][node]
    return total


def apply_trees(x, model):
    """Log-odds from a boosted-tree model exported by tools/fit_evidence.py --model boosted:
    bias plus one leaf value per tree (a row goes left when its clue <= the node's threshold).
    Plain NumPy, so the build needs no scikit-learn. A "symmetric" model (the team model, whose
    clues are all home-minus-away) is evaluated as (f(x) - f(-x)) / 2: swapping the teams flips
    the sign exactly."""
    x = np.ascontiguousarray(x, dtype=float)
    if model.get("symmetric"):
        return 0.5 * (_sum_trees(x, model) - _sum_trees(-x, model))
    return _sum_trees(x, model)


def apply_model(x, model):
    """Log-odds per row from a fitted model dict: boosted trees if it has them, else logistic."""
    if "trees" in model:
        return apply_trees(x, model)
    return apply_logistic(x, model["bias"], model["weights"])


def _fitted():
    """Models written by tools/fit_evidence.py (fitted on the external tuning games only)."""
    try:
        from . import evidence_models
    except ImportError:
        return None, None
    return evidence_models.DEAD_MODEL, evidence_models.TEAM_MODEL


def log_odds(c, dead_model=None, team_model=None):
    """(dead log-odds, home log-odds) per frame."""
    if dead_model is None or team_model is None:
        fitted = _fitted()
        dead_model, team_model = dead_model or fitted[0], team_model or fitted[1]
    if dead_model is None or team_model is None:
        raise RuntimeError("evidence models not fitted: run tools/fit_evidence.py")
    names, x = motion_design(c)
    _check(names, dead_model["motion"])
    dead = apply_model(x, dead_model["motion"])
    names, x = ball_design(c)
    _check(names, dead_model["ball"])
    seen = c["present"]
    dead[seen] = apply_model(x[seen], dead_model["ball"])
    names, x = team_design(c)
    _check(names, team_model)
    team = apply_model(x, team_model)
    return dead, team


def _check(names, model):
    if list(model["names"]) != names:
        raise RuntimeError("evidence model was fitted on different clues; re-run "
                           "tools/fit_evidence.py")
