"""v4 labeler: a 5-state hidden Markov model on the Phase 5/6 evidence (checklist Phase 7).

v1-v3 decide frame by frame with a switching rule (hysteresis, CUSUM). v4 instead scores every
possible story of a half at once and keeps the best one (Viterbi), so a switch is placed where
the evidence for it starts, and weak stretches are filled in from what comes before *and* after.

**States** (fixed order; ties go to the lowest index, so decoding is deterministic):

    0 H-ctrl   home has the ball under control       -> 'H'
    1 H-loose  ball loose, home touched it last      -> 'H'
    2 A-ctrl   away has the ball under control       -> 'A'
    3 A-loose  ball loose, away touched it last      -> 'A'
    4 D        dead ball                             -> 'D'

A loose ball stays with the last team in control, as the official rules say (a deflection or a
block is not a change of possession). The loose states exist so the model can be in "nobody has
it right now" without that counting as a turnover, and so a turnover is cheaper through a loose
ball (loose(T) -> ctrl(other)) than as a direct ctrl(T) -> ctrl(other) tackle.

**Evidence per frame** (`emissions()`, log-scores; only differences between states matter):
- Dead: Phase 5 dead log-odds, smoothed over +/- `dead_smooth` frames. On frames *with* a ball
  `dead_bias` is subtracted (tuned). On frames *without* a ball the bias is fixed at 0: checklist
  6.6 decision (b), same as v2.UNSEEN_THRESHOLD, never tuned. D gets +w_dead*d/2, live states
  -w_dead*d/2.
- Team: Phase 5 home log-odds, capped at `team_cap`, weight `w_team` on the ctrl states and
  `loose_scale * w_team` on the loose states.
- Touches (Phase 6, touches.py): a confirmed touch by team T favours T-ctrl by `w_confirm` and
  rules out the other team (both its states lose `w_confirm`). Any other touch by T counts
  against the other team's *ctrl* state (`w_touch`: someone in T touched it, so the other team
  is not in control) and gives T's own states a small `w_touch_own` credit. A lone touch (a
  deflection, a block) cannot switch possession on its own: with the tuned values
  w_touch + w_touch_own (15) stays below the turnover cost (20). When control by T does follow,
  the credit dates the switch back to T's first touch.
- Control vs loose, only when the ball is tracked: nearest player within `ctrl_reach` favours the
  ctrl states by `w_ctrl`; nobody within `loose_reach` favours the loose states by `w_loose`.
- **A missing ball is no information**: no ball-based term is added on those frames, so the
  slower clues (team shape, pressing) and the transition costs decide.

**Transitions** (per-frame costs; staying is free; anything not listed is impossible):
ctrl <-> loose of the same team `c_loose`; loose(T) -> ctrl(other) `c_turnover`;
ctrl(T) -> ctrl(other) `c_direct`; any live -> D `c_stop`; D -> any live `c_restart`.

Each half is decoded on its own with a free start; then an optional duration clean-up
(`min_dead`, `min_alive`, as in v1/v2) removes stoppages and live spells too short to be real. Frames outside the halves are D (finalise()
and the bench turn them into X). Tracking only, no Smart Tagging. Settings tuned by
tools/tune.py --labeler v4 --skip-heldout on Metrica 1 and 3 over light/medium/heavy ball blur;
never on our match. The held-out Game 2 is scored only at selection (Phase 9).
"""

import numpy as np

from . import evidence, touches, v1, v3

HOME, AWAY = v1.HOME, v1.AWAY
HC, HL, AC, AL, D = range(5)
N_STATES = 5
STATE_LABEL = np.array(["H", "H", "A", "A", "D"], dtype="<U1")
UNSEEN_DEAD_BIAS = 0.0     # decision (b): no dead bias on ball-less frames; fixed, never tuned
NEG = -np.inf

# Tuned (5 Nov) with tools/tune.py --labeler v4 --skip-heldout on Metrica 1 and 3 over light/
# medium/heavy ball blur, ball hidden on 25% of live frames, three rounds (widened grids after the
# first; duration clean-up added in the third). Mean tuning spellF1 0.5734 (v3 0.5717, v1 0.558);
# per blur 0.579 / 0.578 / 0.564. At 15% / 35% live missing: 0.584 / 0.550 (v3 0.585 / 0.555),
# so v4 is level with v3 rather than ahead, and drops a little faster as more ball goes missing.
# Held-out Game 2 not scored: done once at selection (Phase 9). run.py keeps v1 as the default.
PARAMS = {
    "dead_smooth": 0,
    "dead_bias": 1.0,        # log-odds, frames with a ball only
    "w_dead": 4.0,
    "team_cap": 2.0,
    "w_team": 1.0,
    "loose_scale": 0.0,
    "w_confirm": 20.0,
    "w_touch": 10.0,
    "w_touch_own": 5.0,      # credit to the toucher's own states for a lone touch
    "w_ctrl": 1.0,
    "w_loose": 0.2,
    "ctrl_reach": 5.0,       # m
    "loose_reach": 2.0,      # m
    "c_loose": 5.0,
    "c_turnover": 20.0,
    "c_direct": 40.0,
    "c_stop": 40.0,
    "c_restart": 40.0,
    "min_dead": 50,          # frames; shorter decoded dead spells go to the team around them
    "min_alive": 0,          # frames; shorter live spells between two stoppages become dead
    "kink_half": 4,
    "kink_threshold": 5.0,
    "touch_reach": 1.5,
    "chain_gap": 100,
    "hold_reach": 1.5,
    "hold_frames": 25,
}

GRID = {
    "dead_smooth": [0, 6, 12, 25, 50],
    "dead_bias": [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
    "w_dead": [0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
    "team_cap": [0.5, 1.0, 2.0, 3.0, 5.0],
    "w_team": [0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
    "loose_scale": [0.0, 0.25, 0.5, 1.0],
    "w_confirm": [0.0, 5.0, 10.0, 20.0, 40.0, 80.0],
    "w_touch": [0.0, 2.0, 5.0, 10.0, 20.0, 40.0],
    "w_touch_own": [0.0, 2.0, 5.0, 10.0, 20.0],
    "w_ctrl": [0.0, 0.1, 0.2, 0.5, 1.0],
    "w_loose": [0.0, 0.1, 0.2, 0.5, 1.0],
    "ctrl_reach": [1.0, 1.5, 2.0, 3.0, 5.0],
    "loose_reach": [2.0, 3.0, 5.0, 8.0],
    "c_loose": [1.0, 2.0, 5.0, 10.0, 20.0],
    "c_turnover": [5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 90.0],
    "c_direct": [10.0, 20.0, 40.0, 60.0, 90.0, 120.0],
    "c_stop": [5.0, 10.0, 20.0, 30.0, 40.0, 60.0, 80.0, 160.0],
    "c_restart": [5.0, 10.0, 20.0, 30.0, 40.0, 60.0, 80.0, 160.0],
    "min_dead": [0, 25, 50, 75, 100, 150, 200],
    "min_alive": [0, 25, 50, 100, 150],
    "kink_half": list(touches.KINK_HALVES),
    "kink_threshold": [1.0, 1.5, 2.0, 3.0, 5.0, 8.0],
    "touch_reach": [0.5, 1.0, 1.5, 2.0, 3.0],
    "chain_gap": [6, 12, 25, 50, 100],
    "hold_frames": [0, 12, 25, 38, 50],
}


def features(tracking):
    """Same inputs as v3: Phase 5 log-odds plus what the touch detector needs."""
    return v3.features(tracking)


def transitions(p):
    """(5, 5) log-scores, row = from, column = to; -inf = impossible."""
    t = np.full((N_STATES, N_STATES), NEG)
    np.fill_diagonal(t, 0.0)
    for ctrl, loose in ((HC, HL), (AC, AL)):
        t[ctrl, loose] = t[loose, ctrl] = -p["c_loose"]
    t[HL, AC] = t[AL, HC] = -p["c_turnover"]
    t[HC, AC] = t[AC, HC] = -p["c_direct"]
    for s in (HC, HL, AC, AL):
        t[s, D] = -p["c_stop"]
        t[D, s] = -p["c_restart"]
    return t


def touch_list(f, p):
    """Detected touches (touches.detect dict) and their confirmed flags."""
    t = touches.detect(f["kinks"][int(p["kink_half"])], f["near"], f["slot"], f["team_of_slot"],
                       p["kink_threshold"], p["touch_reach"])
    ok = touches.confirmed(t, f["dist"], int(p["chain_gap"]), p["hold_reach"],
                           int(p["hold_frames"]))
    return t, ok


def emissions(f, p, first, stop, touch=None):
    """(stop - first, 5) log-scores for the frames [first, stop)."""
    n = stop - first
    e = np.zeros((n, N_STATES))
    present = np.asarray(f["present"][first:stop], dtype=bool)

    d = np.nan_to_num(evidence.smooth(f["dead_lo"][first:stop], int(p["dead_smooth"])))
    d = d - np.where(present, p["dead_bias"], UNSEEN_DEAD_BIAS)
    half = 0.5 * p["w_dead"] * d
    e[:, D] += half
    e[:, :D] -= half[:, None]

    c = 0.5 * p["w_team"] * np.clip(np.nan_to_num(f["team_lo"][first:stop]),
                                    -p["team_cap"], p["team_cap"])
    e[:, HC] += c
    e[:, AC] -= c
    e[:, HL] += p["loose_scale"] * c
    e[:, AL] -= p["loose_scale"] * c

    near = np.asarray(f["near"][first:stop], dtype=float)
    seen = present & np.isfinite(near)
    ctrl = seen & (near <= p["ctrl_reach"])
    loose = seen & (near > p["loose_reach"])
    e[ctrl, HC] += p["w_ctrl"]
    e[ctrl, AC] += p["w_ctrl"]
    e[loose, HL] += p["w_loose"]
    e[loose, AL] += p["w_loose"]

    t, ok = touch if touch is not None else touch_list(f, p)
    inside = (t["frame"] >= first) & (t["frame"] < stop)
    for frame, team, sure in zip(t["frame"][inside] - first, t["team"][inside], ok[inside]):
        own = (HC, HL) if team == HOME else (AC, AL)
        other = (AC, AL) if team == HOME else (HC, HL)
        if sure:
            e[frame, own[0]] += p["w_confirm"]
            e[frame, other[0]] -= p["w_confirm"]
            e[frame, other[1]] -= p["w_confirm"]
        else:
            e[frame, other[0]] -= p["w_touch"]
            e[frame, own[0]] += p["w_touch_own"]
            e[frame, own[1]] += p["w_touch_own"]
    return e


def viterbi(emis, trans):
    """Best state path (int8 per frame) for log-scores emis (n, S) and trans (S, S).

    Free start (equal prior). Ties go to the lowest state index, both between predecessors and
    for the final state, so the answer is deterministic."""
    emis = np.asarray(emis, dtype=float)
    n, s = emis.shape
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    trans = np.asarray(trans, dtype=float)
    preds = [[(i, float(trans[i, j])) for i in range(s) if np.isfinite(trans[i, j])]
             for j in range(s)]
    rows = emis.tolist()
    score = rows[0]
    back = [None] * n
    states = range(s)
    for t in range(1, n):
        row = rows[t]
        new, ptr = [0.0] * s, [0] * s
        for j in states:
            best, arg = NEG, -1
            for i, cost in preds[j]:
                v = score[i] + cost
                if v > best:
                    best, arg = v, i
            new[j] = best + row[j]
            ptr[j] = arg
        score, back[t] = new, ptr
    path = np.empty(n, dtype=np.int8)
    state = max(states, key=lambda j: (score[j], -j))
    for t in range(n - 1, 0, -1):
        path[t] = state
        state = back[t][state]
    path[0] = state
    return path


def clean_up(states, min_dead, min_alive):
    """Duration clean-up after Viterbi (checklist 7.7, light version): dead spells shorter than
    min_dead take the state just before them (just after, at the start of a half); live spells
    shorter than min_alive between two dead spells become dead. 0 = off."""
    out = states.copy()
    for a, b in v1._runs(out == D):
        if b - a < min_dead and (a > 0 or b < len(out)):
            out[a:b] = out[a - 1] if a > 0 else out[b]
    if min_alive > 0:
        for a, b in v1._runs(out != D):
            if a > 0 and b < len(out) and b - a < min_alive:
                out[a:b] = D
    return out


def decode_states(f, params=None):
    """State per frame (int8); -1 outside the halves."""
    p = dict(PARAMS, **(params or {}))
    trans = transitions(p)
    touch = touch_list(f, p)
    out = np.full(len(f["present"]), -1, dtype=np.int8)
    for _, first, last in f["half_blocks"]:
        stop = last + 1
        path = viterbi(emissions(f, p, first, stop, touch), trans)
        out[first:stop] = clean_up(path, int(p["min_dead"]), int(p["min_alive"]))
    return out


def decode(f, params=None):
    """'H'/'A'/'D' per frame; frames outside the halves are D (finalise() / the bench make X)."""
    states = decode_states(f, params)
    labels = np.full(len(states), "D", dtype="<U1")
    inside = states >= 0
    labels[inside] = STATE_LABEL[states[inside]]
    return labels


def v4(tracking, params=None):
    return decode(features(tracking), params)
