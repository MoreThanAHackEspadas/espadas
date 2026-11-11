"""v3 labeler: v2 plus touches and control (checklist Phase 6).

1. **Dead ball:** exactly as v2 (Phase 5 dead log-odds, smoothed, thresholded, cleaned up).
2. **Who has it:** CUSUM (v2.cusum) on the Phase 5 team log-odds, capped at `team_cap` per frame,
   plus touch pulses from touches.py: `confirm_weight` for a confirmed touch (pass chain or
   hold), `touch_weight` for any other touch. With the tuned PARAMS (confirm 20, other 30,
   `switch_cost` 60) no single touch switches possession on its own: two touches by the other
   team do, or one plus enough team evidence; the CUSUM does the confirming.

Tracking only, no Smart Tagging. Settings tuned by tools/tune.py --labeler v3 on Metrica 1 and 3
only; never on our match.
"""

import numpy as np

from . import config, prep, touches, v1, v2

HOME, AWAY, NONE = v1.HOME, v1.AWAY, v1.NONE

# Re-tuned (5 Nov, checklist 6.7) with tools/tune.py --labeler v3 --skip-heldout on Metrica 1 and 3
# over the four tracker levels (degrade.TRACKER_LEVELS: ball blur incl. the new "smooth" level,
# plus player position error), ball hidden on 25% of live frames, evidence models refitted on
# the same bench: mean tuning spellF1 0.5386 (v1 0.4839); 0.553 / 0.525 at 15% / 35% missing.
# 6.6 history: tuned on light/medium/heavy only, 0.572; before blur 0.681, held-out 0.653.
# Tuning still weights a plain touch (30) above a confirmed one (20): two touches by the other
# team switch; the CUSUM does the confirming. Held-out not re-scored. run.py keeps v1.
# 6.7b (5 Nov): carrier touches tried, each mode fully re-tuned the same way: kink 0.5386,
# both 0.5388, carrier 0.5362. No gain (v4 gained +0.014): the CUSUM adds every touch as a
# pulse, so the noisier carrier team calls cancel out what they add. v3 stays on "kink".
# 11 Nov (branch `refinement`): re-tuned on the 5-game development pool (Metrica 1, 3 with the
# corrected stoppage labels + IDSSE 12, 13, 15), evidence models refitted on the same pool:
# mean spellF1 0.5408 -> 0.5560 in-sample; 0.569 / 0.545 at 15% / 35% missing. Leave-one-game-
# out with re-tuning (tools/logo.py): 0.5408 vs 0.5316 for the previous settings, better on 4 of
# 5 games (all three IDSSE). Moved: dead_threshold 1.0 -> -0.5, min_dead 50 -> 100, switch_cost
# 60 -> 45, kink_half 6 -> 4, kink_threshold 3 -> 5, hold_reach 3 -> 2.
PARAMS = dict(v2.PARAMS, **{
    "dead_smooth": 0,
    "dead_threshold": -0.5,
    "min_dead": 100,
    "min_alive": 25,
    "team_cap": 2.0,
    "switch_cost": 45.0,
    "kink_half": 4,          # frames each side over which the velocity change is measured
    "kink_threshold": 5.0,   # m/s velocity change for a touch
    "touch_reach": 1.5,      # m; nearest player must be this close at a touch
    "chain_gap": 25,         # frames; same-team next touch within this confirms control
    "hold_reach": 2.0,       # m
    "hold_frames": 50,       # frames the same player must stay with the ball (0 = off)
    "confirm_weight": 20.0,  # pulse for a confirmed touch
    "touch_weight": 30.0,    # pulse for any other touch
    "touch_mode": "kink",    # kink / carrier / both (touches.from_features, 6.7)
    "carrier_reach": 2.5,    # m; carrier touches only
    "carrier_frames": 6,     # frames the new carrier must keep the ball
})

GRID = dict(v2.GRID, min_dead=[0, 25, 50, 100, 150, 200, 250, 350], **{
    "team_cap": [0.5, 1.0, 2.0, 3.0],
    "switch_cost": [20.0, 30.0, 45.0, 60.0, 90.0, 120.0],
    "kink_half": list(touches.KINK_HALVES),
    "kink_threshold": [1.0, 1.5, 2.0, 3.0, 5.0, 8.0],
    "touch_reach": [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
    "chain_gap": [6, 12, 25, 50, 100, 150],
    "hold_reach": [1.0, 1.5, 2.0, 3.0],
    "hold_frames": [0, 12, 25, 38, 50],
    "confirm_weight": [20.0, 30.0, 45.0, 60.0, 90.0, 120.0],
    "touch_weight": [0.0, 3.0, 10.0, 20.0, 30.0],
    "carrier_reach": [1.0, 1.5, 2.0, 2.5, 3.5],
    "carrier_frames": [3, 6, 12, 25, 50],
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
    t, ok = touches.from_features(f, p)
    return touches.pulses(len(f["near"]), t, ok, p["confirm_weight"], p["touch_weight"])


def decode(f, params=None):
    p = dict(PARAMS, **(params or {}))
    signal = np.clip(f["team_lo"], -p["team_cap"], p["team_cap"]) + touch_signal(f, p)
    g = dict(f, team_lo=signal)
    # The cap is already applied to the Phase 5 part; touch pulses must pass through uncapped.
    return v2.decode(g, dict(p, team_cap=np.inf))


def v3(tracking, params=None):
    return decode(features(tracking), params)
