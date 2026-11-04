"""v3 labeler: v2 plus touches and control (checklist Phase 6).

1. **Dead ball:** exactly as v2 (Phase 5 dead log-odds, smoothed, thresholded, cleaned up).
2. **Who has it:** CUSUM (v2.cusum) on the Phase 5 team log-odds, capped at `team_cap` per frame,
   plus touch pulses from touches.py: `confirm_weight` for a confirmed touch (pass chain or
   hold), `touch_weight` for any other touch. With `confirm_weight` near `switch_cost`, a
   confirmed touch by the other team switches possession on its own; a lone deflection does not.

Tracking only, no Smart Tagging. Settings tuned by tools/tune.py --labeler v3 on Metrica 1 and 3
only; never on our match.
"""

import numpy as np

from . import config, prep, touches, v1, v2

HOME, AWAY, NONE = v1.HOME, v1.AWAY, v1.NONE

# Re-tuned (6.6) with tools/tune.py --labeler v3 on Metrica 1 and 3 over light/medium/heavy ball
# blur (degrade.BLUR), ball hidden on 25% of live frames: mean tuning spellF1 0.613 (heavy blur
# ~0.60). On blurred balls tuning weights confirmed touches *below* plain ones (20 vs 30, switch
# cost 60): in effect "two touches by the other team switch"; the CUSUM does the confirming.
# Before blur (sharp Metrica corners only): tuning 0.681, held-out 0.653 vs v1 0.640, but on our
# match only 11 touches per 90' were found (1,300-1,600 on Metrica), 199 possessions / 90'.
# Held-out not re-scored yet. run.py keeps v1 as the default.
PARAMS = dict(v2.PARAMS, **{
    "dead_smooth": 0,
    "dead_threshold": -2.0,
    "min_dead": 100,
    "min_alive": 25,
    "team_cap": 1.0,
    "switch_cost": 60.0,
    "kink_half": 4,          # frames each side over which the velocity change is measured
    "kink_threshold": 5.0,   # m/s velocity change for a touch
    "touch_reach": 1.5,      # m; nearest player must be this close at a touch
    "chain_gap": 12,         # frames; same-team next touch within this confirms control
    "hold_reach": 1.5,       # m
    "hold_frames": 38,       # frames the same player must stay with the ball (0 = off)
    "confirm_weight": 20.0,  # pulse for a confirmed touch
    "touch_weight": 30.0,    # pulse for any other touch
})

GRID = dict(v2.GRID, min_dead=[0, 25, 50, 100, 150, 200, 250, 350], **{
    "team_cap": [0.5, 1.0, 2.0, 3.0],
    "switch_cost": [20.0, 30.0, 45.0, 60.0, 90.0, 120.0],
    "kink_half": list(touches.KINK_HALVES),
    "kink_threshold": [1.0, 1.5, 2.0, 3.0, 5.0, 8.0],
    "touch_reach": [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0],
    "chain_gap": [6, 12, 25, 50, 100, 150],
    "hold_reach": [1.0, 1.5, 2.0],
    "hold_frames": [0, 12, 25, 38, 50],
    "confirm_weight": [20.0, 30.0, 45.0, 60.0, 90.0, 120.0],
    "touch_weight": [0.0, 3.0, 10.0, 20.0, 30.0],
})


def features(tracking):
    f = v2.features(tracking)
    fps = tracking.get("fps", config.FPS)
    base = prep.prepare(tracking, half_blocks=f["half_blocks"],
                        kickoffs=tracking.get("kickoffs", config.KICKOFFS), fps=fps)
    near, slot, dist = touches.nearest_slot(tracking["players"], base["ball"])
    f.update({"kinks": {h: touches.kink(base["ball"], fps, h) for h in touches.KINK_HALVES},
              "near": near,
              "slot": slot, "dist": dist, "team_of_slot": np.asarray(tracking["team_of_slot"])})
    return f


def touch_signal(f, p):
    t = touches.detect(f["kinks"][int(p["kink_half"])], f["near"], f["slot"], f["team_of_slot"],
                       p["kink_threshold"], p["touch_reach"])
    ok = touches.confirmed(t, f["dist"], int(p["chain_gap"]), p["hold_reach"],
                           int(p["hold_frames"]))
    return touches.pulses(len(f["near"]), t, ok, p["confirm_weight"], p["touch_weight"])


def decode(f, params=None):
    p = dict(PARAMS, **(params or {}))
    signal = np.clip(f["team_lo"], -p["team_cap"], p["team_cap"]) + touch_signal(f, p)
    g = dict(f, team_lo=signal)
    # The cap is already applied to the Phase 5 part; touch pulses must pass through uncapped.
    return v2.decode(g, dict(p, team_cap=np.inf))


def v3(tracking, params=None):
    return decode(features(tracking), params)
