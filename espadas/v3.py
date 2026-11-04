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

# Tuned with tools/tune.py --labeler v3 on Metrica 1 and 3 (25% of live ball hidden): mean
# spellF1 0.681 (v2 0.610, v1 0.591). Held-out Game 2, scored once: 0.653 / 0.662 / 0.626 at
# 25% / 15% / 35% live missing (v1 0.640 / 0.656 / 0.608), so it wins at every level.
# touch_weight == switch_cost: a lone touch by the other team is enough to switch. Forcing a
# confirmation (touch_weight 29) costs only 0.003 on the tuning games, so this is not a cliff.
# On our match (diagnostic only, never used to tune) v3 gives 199 possessions per 90' and 47%
# dead, outside the practice games' 254-377 and 28-39%: it may not transfer to our tracking, so
# run.py keeps v1 as the default until that is understood.
PARAMS = dict(v2.PARAMS, **{
    "dead_smooth": 0,
    "dead_threshold": -2.5,
    "min_dead": 150,
    "team_cap": 0.5,
    "switch_cost": 30.0,
    "kink_threshold": 5.0,   # m/s velocity change for a touch
    "touch_reach": 1.0,      # m; nearest player must be this close at a touch
    "chain_gap": 25,         # frames; same-team next touch within this confirms control
    "hold_reach": 2.0,       # m
    "hold_frames": 12,       # frames the same player must stay with the ball (0 = off)
    "confirm_weight": 60.0,  # pulse for a confirmed touch
    "touch_weight": 30.0,    # pulse for any other touch
})

GRID = dict(v2.GRID, min_dead=[0, 25, 50, 100, 150, 200, 250, 350], **{
    "team_cap": [0.5, 1.0, 2.0, 3.0],
    "switch_cost": [20.0, 30.0, 45.0, 60.0, 90.0, 120.0],
    "kink_threshold": [2.0, 3.0, 5.0, 8.0],
    "touch_reach": [0.5, 0.75, 1.0, 1.5, 2.0, 2.5],
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
    f.update({"kink": touches.kink(base["ball"], fps, touches.KINK_HALF), "near": near,
              "slot": slot, "dist": dist, "team_of_slot": np.asarray(tracking["team_of_slot"])})
    return f


def touch_signal(f, p):
    t = touches.detect(f["kink"], f["near"], f["slot"], f["team_of_slot"],
                       p["kink_threshold"], p["touch_reach"])
    ok = touches.confirmed(t, f["dist"], int(p["chain_gap"]), p["hold_reach"],
                           int(p["hold_frames"]))
    return touches.pulses(len(f["kink"]), t, ok, p["confirm_weight"], p["touch_weight"])


def decode(f, params=None):
    p = dict(PARAMS, **(params or {}))
    signal = np.clip(f["team_lo"], -p["team_cap"], p["team_cap"]) + touch_signal(f, p)
    g = dict(f, team_lo=signal)
    # The cap is already applied to the Phase 5 part; touch pulses must pass through uncapped.
    return v2.decode(g, dict(p, team_cap=np.inf))


def v3(tracking, params=None):
    return decode(features(tracking), params)
