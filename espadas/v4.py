"""v4 labeler: a five-state hidden Markov model over the Phase 5 evidence and Phase 6 touches.

States, in tie order (ties go home, like the organisers' baseline):

    D   ball dead                               -> D
    HC  home controls the ball                  -> H
    HL  ball loose, home was last in control    -> H
    AC  away controls the ball                  -> A
    AL  ball loose, away was last in control    -> A

The structure enforces two rules of the official definition, so the decoder cannot break them:

1. **Possession changes only through a loose ball.** `HC` cannot move to `AC` or `AL`; the ball
   first goes loose (`HL`) and the other team then wins it (`HL -> AC`). A loose ball still
   belongs to the last team in control, which is why `HL` is labelled H.
2. **A restart is controlled by the team taking it.** `D` moves only to `HC` or `AC`, never to a
   loose state, and a half never starts loose.

Transitions are set by expected lengths in frames (`ctrl_len`, `loose_len`, `to_dead_len`,
`dead_len`) and by `win_share`, the share of loose spells the other team wins. Per frame the
emission scores are: the dead log-odds pull towards `D`; the home log-odds (capped at
`team_cap`) and the touch pulses pull the two control states apart, and the home log-odds pull
the loose states apart more weakly (`w_loose`); the control log-odds (ball at someone's feet
vs in flight or contested) pull the control states up and the loose states down. Touch pulses
act on control states only: a confirmed touch by away is evidence for `AC`, not `AL`. The pulses
are in per-frame log-odds units (`confirm_pulse`, `touch_pulse`), not v3's CUSUM units.

Checklist 6.6, decision (b), as v2: on frames without a ball the dead offset is fixed at 0 and
the control score is 0 (a missing ball is no information); neither is ever in GRID.

The touch *detection* settings are v3's, fixed (TOUCH), to keep the count of tuned settings down.
Viterbi per half (hmm.py), then v2's run-length clean-up (`min_dead`, `min_alive`). Tracking
only, no Smart Tagging. Tuned by tools/tune.py --labeler v4 on Metrica 1 and 3; never on our
match.
"""

import numpy as np

from . import evidence, hmm, touches, v1, v3

STATES = ("D", "HC", "HL", "AC", "AL")
D, HC, HL, AC, AL = range(5)
LABEL = np.array(["D", "H", "H", "A", "A"])

# Fixed, not tuned (design 4.5). Touches are built once in features(), so re-opening one of
# these would mean building them in decode() (as v3.touch_signal does) before adding it to GRID;
# decided on the tuning games only.
TOUCH = {k: v3.PARAMS[k] for k in ("kink_half", "kink_threshold", "touch_reach", "chain_gap",
                                   "hold_reach", "hold_frames")}

# Tuned 5 Nov 2026 with tools/tune.py --labeler v4 --skip-heldout on Metrica 1 and 3 over
# light/medium/heavy ball blur, ball hidden on 25% of live frames (run 1): mean tuning spellF1
# 0.5697 (light 0.557 / 0.583, medium 0.568 / 0.574, heavy 0.570 / 0.566; v3 on the same bench
# 0.572, v1 ~0.56). At 15% / 35% live ball hidden: 0.584 / 0.549. w_dead, w_ctrl, confirm_pulse
# and touch_pulse ended at the top of their grids and loose_len at the bottom, so those grids
# were widened below for a second run that has not been made yet (checklist 7.8). The tuner
# prefers touches and the control score over the Phase 5 team clues (w_team 0.25).
# Held-out Game 2 not scored (checklist 4.10); scored once at selection.
PARAMS = {
    "ctrl_len": 50,          # frames in control before the ball goes loose
    "loose_len": 6,          # frames a loose ball stays loose
    "win_share": 0.5,        # share of loose spells the other team wins
    "to_dead_len": 1000,     # live frames before a stoppage
    "dead_len": 500,         # frames of a stoppage
    "w_dead": 5.0,           # weight of the dead log-odds
    "dead_offset": 1.0,      # dead log-odds offset, ball-tracked frames only (else 0, fixed)
    "w_team": 0.25,          # weight of the home log-odds in control states
    "team_cap": 1.0,         # clip on the home log-odds
    "w_loose": 0.3,          # weight of the home log-odds in loose states
    "w_ctrl": 5.0,           # weight of the control log-odds
    "ctrl_offset": 1.0,      # control log-odds offset
    "confirm_pulse": 12.0,   # log-odds, one frame, for a confirmed touch
    "touch_pulse": 4.0,     # log-odds, one frame, for any other touch
    "min_dead": 50,          # frames; shorter dead runs are dropped
    "min_alive": 25,         # frames; shorter live runs between stoppages become dead
}

GRID = {
    "ctrl_len": [25, 50, 100, 150, 250, 400, 600, 1000],
    "loose_len": [2, 3, 4, 6, 12, 25, 40, 60, 100, 150],
    "win_share": [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
    "to_dead_len": [250, 500, 1000, 1500, 2500, 5000, 10000],
    "dead_len": [50, 100, 200, 500, 1000, 2000, 3000],
    "w_dead": [0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0],
    "dead_offset": [-4.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
    "w_team": [0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0],
    "team_cap": [0.25, 0.5, 1.0, 2.0, 3.0, 5.0],
    "w_loose": [0.0, 0.1, 0.3, 0.5, 1.0],
    "w_ctrl": [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 8.0, 12.0, 20.0],
    "ctrl_offset": [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
    "confirm_pulse": [0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 20.0, 30.0],
    "touch_pulse": [0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0],
    "min_dead": [0, 25, 50, 100, 150, 250],
    "min_alive": [0, 25, 50, 100, 150, 250],
}


def features(tracking):
    f = v3.features(tracking)
    f["ctrl_lo"] = evidence.control_log_odds(evidence.clues(tracking))
    f["touches"] = touches.detect(f["kinks"][TOUCH["kink_half"]], f["near"], f["slot"],
                                  f["team_of_slot"], TOUCH["kink_threshold"],
                                  TOUCH["touch_reach"])
    f["touch_ok"] = touches.confirmed(f["touches"], f["dist"], int(TOUCH["chain_gap"]),
                                      TOUCH["hold_reach"], int(TOUCH["hold_frames"]))
    return f


def transitions(p):
    """(log_trans (5, 5), log_start (5,)); log_trans[i, j] is from i to j, -inf if forbidden."""
    p_loose = 1.0 / p["ctrl_len"]
    p_dead = 1.0 / p["to_dead_len"]
    p_regain = (1.0 - p["win_share"]) / p["loose_len"]
    p_win = p["win_share"] / p["loose_len"]
    p_restart = 1.0 / p["dead_len"]
    prob = np.zeros((5, 5))
    prob[D, HC] = prob[D, AC] = p_restart / 2      # no home bias at a restart
    prob[HC, HL] = prob[AC, AL] = p_loose
    prob[HL, HC] = prob[AL, AC] = p_regain
    prob[HL, AC] = prob[AL, HC] = p_win
    prob[[HC, HL, AC, AL], D] = p_dead
    stay = 1.0 - prob.sum(axis=1)
    if np.any(stay <= 0):
        raise ValueError("expected lengths too short: {} could not stay put".format(
            ", ".join(s for s, x in zip(STATES, stay) if x <= 0)))
    prob[np.arange(5), np.arange(5)] = stay
    start = np.array([1.0, 1.0, 0.0, 1.0, 0.0]) / 3   # never loose at the start of a half
    with np.errstate(divide="ignore"):
        return np.log(prob), np.log(start)


def emissions(f, p, first, stop):
    """(stop - first, 5) log emission scores over [first, stop), columns in STATES order."""
    sl = slice(first, stop)
    present = f["present"][sl]
    # Decision (b): no dead offset and no control score on frames without a ball.
    e_dead = p["w_dead"] * (f["dead_lo"][sl] - np.where(present, p["dead_offset"], 0.0))
    team = np.clip(f["team_lo"][sl], -p["team_cap"], p["team_cap"])
    e_touch = touches.pulses(len(f["present"]), f["touches"], f["touch_ok"],
                             p["confirm_pulse"], p["touch_pulse"])[sl]
    e_ctrl = np.where(present, p["w_ctrl"] * (np.nan_to_num(f["ctrl_lo"][sl]) - p["ctrl_offset"]),
                      0.0)
    side = (p["w_team"] * team + e_touch) / 2
    loose = p["w_loose"] * team / 2
    e = np.empty((stop - first, 5))
    e[:, D] = e_dead
    e[:, HC] = side + e_ctrl / 2
    e[:, HL] = loose - e_ctrl / 2
    e[:, AC] = -side + e_ctrl / 2
    e[:, AL] = -loose - e_ctrl / 2
    return e


def cleanup(labels, p):
    """v2's run-length clean-up on one half's labels: dead runs under `min_dead` dropped, live
    runs under `min_alive` between stoppages filled. Off at 0."""
    labels = labels.copy()
    n = len(labels)
    for a, b in v1._runs(labels == "D"):
        if b - a < p["min_dead"]:
            # A loose ball stays with the last team, so a dropped stoppage continues the spell
            # before it; only at the very start of a half is there nothing before it.
            if a > 0:
                labels[a:b] = labels[a - 1]
            elif b < n:
                labels[a:b] = labels[b]
    for a, b in v1._runs(labels != "D"):
        if a > 0 and b < n and b - a < p["min_alive"]:
            labels[a:b] = "D"
    return labels


def decode(f, params=None):
    """'H'/'A'/'D' per frame; frames outside the halves are D (finalise() / the bench make X)."""
    p = dict(PARAMS, **(params or {}))
    log_trans, log_start = transitions(p)
    labels = np.full(len(f["present"]), "D", dtype="<U1")
    for _, first, last in f["half_blocks"]:
        stop = last + 1
        path = hmm.viterbi(emissions(f, p, first, stop), log_trans, log_start)
        labels[first:stop] = cleanup(LABEL[path], p)
    return labels


def v4(tracking, params=None):
    return decode(features(tracking), params)
