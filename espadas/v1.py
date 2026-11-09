"""v1 labeler: dead-ball detection + hysteresis on the nearest-player signal.

Two steps, both deliberately simple (checklist 7.1):

1. **Dead ball.** A frame is dead when the ball is clearly off the pitch, or when the players are
   walking and the ball is either still or not tracked. Short dead blips are dropped and short
   live blips between stoppages are filled, so stoppages come out as solid blocks.
2. **Who has it, without flicker.** Per frame the evidence is the team of the nearest player,
   if that player is within `reach` metres. Possession only passes to the other team once that
   team has been the evidence on at least `switch_share` of the last `window` frames, and the
   change is dated back to just after the old team's last evidence frame (the ball was lost
   then, not when we became sure). In between, a loose or unseen ball stays with the last team,
   as the official rules say. After a stoppage the possessor is whoever dominates the evidence
   in the first `window` frames of play.

Uses tracking only (no Smart Tagging), because the external games have no Smart Tagging and
every setting must be tuned on external games (organisers' clarification). All settings live in
PARAMS; they are tuned by tools/tune_v1.py on Metrica Games 1 and 3, never on our match.

features() is the expensive part and runs once per match; decode() is cheap, so tuning can try
many settings on the same features.
"""

import numpy as np

from . import config, prep

HOME, AWAY, NONE = 0, 1, -1

# Tuned (3 Nov) with tools/tune_v1.py on Metrica Games 1 and 3, ball hidden on 25% of live frames
# (degrade.py). On that earlier bench (no ball blur or player error yet; checklist Phase 5/6
# tables): mean tuning spellF1 0.591; held-out Game 2 0.640 (0.656 / 0.608 at 15% / 35%).
# Not re-tuned since; on the current bench (four tracker levels) its tuning mean is 0.484.
# out_margin sits at the top of its grid: the off-pitch rule rarely decides anything once the
# walking rule is in, so its exact value barely matters.
PARAMS = {
    "out_margin": 3.0,      # m beyond a line before the ball counts as off the pitch
    "still_speed": 0.5,     # m/s; slower ball = still
    "walk_speed": 2.5,      # m/s; mean player speed below this = players not in active play
    "min_dead": 100,        # frames (4 s); shorter dead runs are ignored
    "min_alive": 25,        # frames (1 s); shorter live runs between stoppages become dead
    "reach": 2.0,           # m; nearest player further than this gives no team evidence
    "window": 25,           # frames (1 s) of evidence considered for a switch
    "switch_share": 0.7,    # share of the window the other team needs to take over
}

PLAYER_SPEED_HALF_WINDOW = 6   # frames each side, ~0.5 s, as prep.ball_speed


def _mean_player_speed(players, fps, w=PLAYER_SPEED_HALF_WINDOW):
    """m/s per frame, averaged over tracked players, from positions i - w and i + w."""
    n = len(players)
    out = np.full(n, np.nan)
    if n > 2 * w:
        step = players[2 * w:] - players[:n - 2 * w]
        speed = np.hypot(step[..., 0], step[..., 1]) / (2 * w / fps)
        with np.errstate(invalid="ignore"):
            counts = np.sum(~np.isnan(speed), axis=1)
            total = np.nansum(speed, axis=1)
            out[w:n - w] = np.where(counts > 0, total / np.maximum(counts, 1), np.nan)
    return out


def _team_distances(players, ball, team_of_slot):
    """(nearest home distance, nearest away distance) per frame, inf where unknown."""
    diff = players - ball[:, None, :]
    dist = np.hypot(diff[..., 0], diff[..., 1])
    dist = np.where(np.isnan(dist), np.inf, dist)
    home = dist[:, team_of_slot == HOME].min(axis=1)
    away = dist[:, team_of_slot == AWAY].min(axis=1)
    return home, away


def features(tracking):
    """Per-frame arrays decode() needs. tracking may carry half_blocks, kickoffs, fps, pitch
    (external games); otherwise our match's values from config are used."""
    half_blocks = tracking.get("half_blocks", config.HALF_BLOCKS)
    kickoffs = tracking.get("kickoffs", config.KICKOFFS)
    fps = tracking.get("fps", config.FPS)
    length, width = tracking.get("pitch", config.PITCH)

    base = prep.prepare(tracking, half_blocks=half_blocks, kickoffs=kickoffs, fps=fps)
    ball = base["ball"]
    with np.errstate(invalid="ignore"):
        out_dist = np.fmax(np.fmax(-ball[:, 0], ball[:, 0] - length),
                           np.fmax(-ball[:, 1], ball[:, 1] - width))
    d_home, d_away = _team_distances(tracking["players"], ball, tracking["team_of_slot"])
    return {
        "half_blocks": half_blocks,
        "present": base["ball_present"],
        "ball_speed": base["ball_speed"],
        "out_dist": out_dist,                       # m outside the pitch, <= 0 inside, NaN unseen
        "player_speed": _mean_player_speed(tracking["players"], fps),
        "d_home": d_home,
        "d_away": d_away,
    }


def _runs(mask):
    """[(start, stop)] of True runs, half-open."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def dead_mask(f, p, first, stop):
    """bool over [first, stop): the ball is dead."""
    sl = slice(first, stop)
    with np.errstate(invalid="ignore"):
        out = f["out_dist"][sl] > p["out_margin"]
        walking = f["player_speed"][sl] < p["walk_speed"]
        still = f["ball_speed"][sl] < p["still_speed"]
    unseen = ~f["present"][sl]
    dead = out | (walking & (still | unseen))

    for a, b in _runs(dead):
        if b - a < p["min_dead"]:
            dead[a:b] = False
    alive = ~dead
    for a, b in _runs(alive):
        if a > 0 and b < len(alive) and b - a < p["min_alive"]:
            dead[a:b] = True
    return dead


def evidence(f, p, first, stop):
    """int8 over [first, stop): HOME, AWAY or NONE."""
    home, away = f["d_home"][first:stop], f["d_away"][first:stop]
    nearest = np.minimum(home, away)
    ev = np.where(home <= away, HOME, AWAY).astype(np.int8)
    ev[~(nearest <= p["reach"])] = NONE
    return ev


def hysteresis(ev, window, share, start_owner):
    """Owner per frame of one live segment. ev: HOME/AWAY/NONE per frame."""
    n = len(ev)
    counts = [np.concatenate(([0], np.cumsum(ev == team))) for team in (HOME, AWAY)]
    need = share * window

    def count(team, t):
        return counts[team][t + 1] - counts[team][max(0, t - window + 1)]

    # Possessor at the start: whoever dominates the first window, else carried over.
    head = ev[:window]
    h, a = int(np.sum(head == HOME)), int(np.sum(head == AWAY))
    owner = HOME if h > a else AWAY if a > h else start_owner
    if owner == NONE:
        seen = ev[ev != NONE]
        owner = int(seen[0]) if len(seen) else NONE

    out = np.full(n, owner, dtype=np.int8)
    last_owner_ev = -1
    switches = []
    for t in np.flatnonzero(ev != NONE):
        team = int(ev[t])
        if team == owner:
            last_owner_ev = t
            continue
        other = count(team, t)
        if other >= need and other > count(owner, t):
            start = max(last_owner_ev + 1, t - window + 1, 0)
            switches.append((start, team))
            owner, last_owner_ev = team, t
    for start, team in switches:
        out[start:] = team
    return out, owner


def decode(f, params=None):
    """'H'/'A'/'D' per frame. Frames outside the halves are D here; finalise() / the bench
    turn them into X."""
    p = dict(PARAMS, **(params or {}))
    n = len(f["present"])
    labels = np.full(n, "D", dtype="<U1")
    for _, first, last in f["half_blocks"]:
        stop = last + 1
        dead = dead_mask(f, p, first, stop)
        ev = evidence(f, p, first, stop)
        owner = NONE
        for a, b in _runs(~dead):
            seg, owner = hysteresis(ev[a:b], int(p["window"]), p["switch_share"], owner)
            labels[first + a:first + b] = np.where(seg == HOME, "H",
                                                   np.where(seg == AWAY, "A", "D"))
    return labels


def v1(tracking, params=None):
    return decode(features(tracking), params)
