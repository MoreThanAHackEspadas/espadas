# Phase 7 design: the 5-state sequence model (`v4`)

Status: draft for team review, 5 Nov 2026. Author: Claude (with the teammate picking up the
repo). Checklist items 7.3–7.8; 7.7 as a second iteration; 7.9 out of scope.

## 1. Intent

Every labeler so far decides frame by frame and then patches the result: a dead threshold,
run-length clean-up, then a switching rule (v1 hysteresis, v2/v3 CUSUM). `v4` instead decodes a
whole half at once: given the per-frame scores we already compute (Phase 5 dead and team
log-odds, Phase 6 touch pulses) plus one new score, it finds the single most probable sequence
of five states under explicit rules about how possession can change. The rules are the official
possession rules written as a state machine, so the model cannot produce a change of possession
without passing through a loose-ball stretch, cannot restart from a stoppage into a loose ball,
and labels a loose ball with the last team in control.

Why now: the Phase 5/6 diagnostic says who-has-it is the bottleneck (perfect team calls lift the
practice score to ~0.86). The CUSUM switch has one number to say "switch"; it cannot tell "the
ball is loose and the other team is near it" from "the other team controls it". The five states
make that distinction explicit and give the decoder one more piece of evidence to make it with.

## 2. Goals, non-goals, anti-goals

Goals
- A labeler `v4` registered in `LABELERS`, built and tuned by the existing tools
  (`tools/tune.py --labeler v4 --skip-heldout`), deterministic, passing the official validator.
- Tuned on Metrica Games 1 and 3 only, over the light/medium/heavy blur, 25% live ball hidden.
- Candidate for selection: it must beat `v1` and `v3` on held-out Game 2, scored **once**, at
  selection (Phase 9), not during this phase.
- Synthetic unit tests for every new piece; CI stays green; 58 existing tests untouched.

Non-goals
- The `X` abstention policy (8.4) and anything about ungraded frames.
- Learned emissions beyond the logistic models (7.9).
- Fixing the touch under-detection on our match (6.7). `v4` inherits `v3`'s touch settings.
- Any use of Smart Tagging (team decision 5.5/5.9 still pending).

Anti-goals (a failure even if the score goes up)
- Any setting chosen by looking at our match, the ground truth or the video.
- Scoring Game 2 while developing.
- Relearning "no ball means stoppage": decision (b) must survive intact.
- A decoder nobody on the team can read. The code must stay as plain as `v2.py`.

## 3. Constraints

- Ground rules in `PLAN_CHECKLIST.md` (organisers' clarification). Tune on external data only.
- Dependencies stay as pinned in `requirements.txt` (numpy, pandas). No numba, no scipy.
- Byte-for-byte reproducible output; `tools/check_determinism.py` must pass.
- `tools/tune.py` protocol unchanged: a labeler module exposes `features(tracking)`,
  `decode(features, params)`, `PARAMS`, `GRID`.
- A decode of one Metrica game should take about 2 s so a tuning run stays under an hour.
- The data guard: no `.md` outside `docs/`, no CSV/TXT anywhere.

## 4. The model

### 4.1 States and output labels

| State | Meaning | Label |
|---|---|---|
| `D` | ball dead | D |
| `HC` | home controls the ball | H |
| `HL` | ball loose, home was last in control | H |
| `AC` | away controls the ball | A |
| `AL` | ball loose, away was last in control | A |

State order for ties is `D, HC, HL, AC, AL` (ties go home, like the organisers' baseline).

### 4.2 Transitions (one matrix for the whole match, log-probabilities)

Parameterised by expected lengths in frames, so every number means something a coach could
sanity-check:

| From | To | Probability per frame |
|---|---|---|
| `HC` | `HL` | `1 / ctrl_len` |
| `HC` | `D` | `1 / to_dead_len` |
| `HC` | `AC`, `AL` | 0 (a change must pass through a loose ball) |
| `HL` | `HC` | `(1 - win_share) / loose_len` |
| `HL` | `AC` | `win_share / loose_len` |
| `HL` | `D` | `1 / to_dead_len` |
| `HL` | `AL` | 0 |
| `D` | `HC`, `AC` | `1 / dead_len` each, split equally (no home bias) |
| `D` | `HL`, `AL` | 0 (a restart is controlled by the team taking it) |

Away rows mirror home. The diagonal takes what is left. Forbidden moves are `-inf`.

Start of a half: `D`, `HC` or `AC` with equal probability; never loose.

### 4.3 Emissions (per frame, added to the path score)

Inputs per frame, all already computed by `v3.features()` plus one new score:
- `dead_lo`: Phase 5 dead log-odds (ball model when the ball is tracked, motion model when not).
- `team_lo`: Phase 5 home log-odds (clipped to `team_cap`).
- `touches`: the Phase 6 touch list (frame, team, confirmed or not) from `touches.detect()` and
  `touches.confirmed()`, found with `v3`'s detection settings.
- `ctrl_lo` (**new**, section 5): log-odds that *some* player has the ball at their feet versus
  the ball being in flight or contested. Only defined when the ball is tracked.

Scores, with `off_dead(t)` = `dead_offset` when the ball is tracked and **0 when it is not**
(decision (b), fixed, never in `GRID`):

```
e_dead(t)  = w_dead * (dead_lo(t) - off_dead(t))
e_team(t)  = w_team * clip(team_lo(t), ±team_cap)
e_touch(t) = sum over touches at frame t of (confirm_pulse if confirmed else touch_pulse),
             signed + for home, - for away
e_ctrl(t)  = w_ctrl * (ctrl_lo(t) - ctrl_offset)      if the ball is tracked, else 0
```

| State | Emission |
|---|---|
| `D`  | `+e_dead` |
| `HC` | `+(e_team + e_touch)/2 + e_ctrl/2` |
| `AC` | `-(e_team + e_touch)/2 + e_ctrl/2` |
| `HL` | `+w_loose * clip(team_lo(t), ±team_cap)/2 - e_ctrl/2` |
| `AL` | `-w_loose * clip(team_lo(t), ±team_cap)/2 - e_ctrl/2` |

Reading it: the dead score pulls towards `D`; the team score pulls the control states apart
and, more weakly (`w_loose`, grid includes 0), the loose states; the control score pulls
control states up and loose states down, and says nothing when the ball is missing. Touch
pulses act on control states only: a confirmed touch by away is evidence for `AC`, not `AL`.

**Pulse units (builder check, call 1).** `v3`'s pulse weights (20 / 30) are in CUSUM units,
sized against `switch_cost = 60`, and must **not** be reused here: in the Viterbi a single
frame worth 15 log-units outweighs the whole `HC -> HL -> AC` detour (about -9 to -15 over the
grids below), so one deflection would flip possession on its own. `v4` has its own two pulse
weights, `confirm_pulse` and `touch_pulse`, in per-frame log-odds units, tuned, and separate
from `w_team` so the tuner can rebalance clues against touches. The sanity bound: a lone
unconfirmed touch must cost less than the cheapest detour, and the grids are chosen so a
confirmed touch can, and a plain one cannot, carry a switch alone.

A missing ball therefore contributes no control evidence and only the motion-based dead
evidence, which is the checklist's "missing ball is no information" requirement.

The logistic models' intercepts carry the training class balance, and the transition matrix
carries the HMM's own prior; `dead_offset` and `ctrl_offset` absorb the double counting and
are tuned like `v2.dead_threshold` is.

### 4.4 Decoding

Viterbi per half (`half_blocks`), log space, float64. Backpointers as `int8`. Ties resolved by
the first index in state order. Frames outside the halves stay `D`; `write.finalise()` and the
bench turn them into `X` as today.

After decoding, the same run-length clean-up as `v2.dead_mask()` (`min_dead`, `min_alive`) is
available as parameters, grid including 0 (off). A dropped short `D` run takes the label of
the spell before it (a loose ball stays with the last team). The second iteration (section 7)
replaces `min_dead`.

Cost: a Python loop over frames with a 5x5 numpy max per step, about 15 µs per frame, so
roughly 2 s per Metrica game. `tools/tune.py` decodes six feature sets per trial (2 games x 3
blurs) and tries a few hundred trials, so a full tuning run is one to two hours. Step 1 of the
build measures the real cost; if it is worse, the fallback is the max-plus parallel scan (same
result, vectorised) or fewer grid values, not a new dependency.

### 4.5 Parameters

Tuned by `tools/tune.py` (coordinate descent over `GRID`):

| Name | Meaning | Start | Grid |
|---|---|---|---|
| `ctrl_len` | expected frames in control before the ball goes loose | 150 | 25 … 1000 |
| `loose_len` | expected frames a loose ball stays loose | 25 | 6 … 150 |
| `win_share` | share of loose spells the other team wins | 0.3 | 0.05 … 0.9 |
| `to_dead_len` | expected live frames before a stoppage | 1500 | 250 … 10000 |
| `dead_len` | expected frames of a stoppage | 500 | 50 … 3000 |
| `w_dead` | weight of the dead score | 1.0 | 0.1 … 5 |
| `dead_offset` | dead log-odds offset, ball-tracked frames only | 0.5 | -4 … 2 |
| `w_team` | weight of the team score in control states | 1.0 | 0.1 … 5 |
| `team_cap` | clip on the Phase 5 team log-odds | 1.0 | 0.25 … 5 |
| `w_loose` | weight of the team score in loose states | 0.3 | 0, 0.1 … 1 |
| `w_ctrl` | weight of the control score | 1.0 | 0, 0.1 … 5 |
| `ctrl_offset` | control log-odds offset | 0.0 | -3 … 3 |
| `confirm_pulse` | log-odds for a confirmed touch, one frame | 6.0 | 0, 1 … 12 |
| `touch_pulse` | log-odds for any other touch, one frame | 1.0 | 0, 0.5 … 4 |
| `min_dead`, `min_alive` | post-hoc clean-up as `v2` | 50, 25 | 0 … 250 |

Fixed, taken from `v3.PARAMS` and **not** in `GRID` (to limit the count of tuned settings,
which 6.5 showed costs held-out score): the touch *detection* settings `kink_half`,
`kink_threshold`, `touch_reach`, `chain_gap`, `hold_reach`, `hold_frames`. Re-opening them is
a one-line change to `GRID`, to be decided on the tuning games only. That makes 14 tuned
settings, one more than `v3`.

## 5. The new control score (`ctrl_lo`)

A third logistic model, fitted like the Phase 5 ones by `tools/fit_evidence.py` on Games 1 and 3
only, across the three blur levels, written to `evidence_models.py` as `CONTROL_MODEL`.

Target, derived by code from the events (`events.py`, new `loose_windows()` / `control_mask()`):
a live frame is **loose** when the ball is in flight or contested, **controlled** otherwise.
Loose spans, half-open, from the events:
- `PASS` and `SET PIECE` with an end after the start: `(start, end)` exclusive, the flight.
- `SHOT`: `[start, end]`.
- `BALL LOST`: from its end (or start when it has no end) to the start of the next control
  event (`is_control()`), the ball is lost and contested.
- `RECOVERY`, `CHALLENGE` and any header/clearance (`TOUCH_SUBTYPES`): from the start to the
  start of the next control event, **unless** that next control event is by the same team and
  no other-team in-play event comes between (the reference's "won ball" rule in
  `ownership_changes()`: the change of possession is dated to this touch, so these frames are
  labelled with the new team). Then the span is **controlled** from the touch-only event's
  start. (Builder check, call 2: without this the control score would be trained to say
  "loose" on exactly the frames where the reference starts the new spell, and every decoded
  possession would start late, at the first control event.)
Every other live frame (inside a `CARRY`, the kick frame of a pass, from a reception to the next
event by the receiver) is controlled. These are fixed definitions, like the H/A/D rules, and
they agree with them by construction: the control target is never "loose" on a frame where the
reference has just handed possession to a new team.

Clues (ball-tracked frames only; team-agnostic, so the score says nothing about who):
- nearest tracked player's distance to the ball, any team (`min(d_home, d_away)`, clipped at
  `NEAR_CLIP`)
- ball speed (clipped as in `ball_design`)
- ball kink at `kink_half` frames (how sharply the ball's velocity just changed; a touch)
- each with the smoothed copies at the existing `CONTEXT` widths plus a 6-frame one, since a
  loose spell is about a second long

Fitted with an intercept on live, ball-tracked frames. Report per-game accuracy on the tuning
games only; `fit_evidence.py` gets a `--skip-heldout` flag, default on, because the current
script reports held-out per-frame accuracy and the team rule now says not to look while
developing.

Refitting must leave `DEAD_MODEL` and `TEAM_MODEL` byte-identical (same data, deterministic);
that is the check that the fit script was not disturbed.

Known transfer risk (6.7): our ball sits further from the nearest player than the Metrica ball
(median 1.9 m vs 0.7–1.4 m), so a distance-based control score will call more of our match
loose. Mitigations: `w_ctrl` grid includes 0; `tools/self_consistency.py --tracking` gains one
aggregate line (share of ball-tracked live frames with `ctrl_lo > 0`) so the gap is visible as a
diagnostic, never as a tuning input; the fair fix stays task 6.7.

## 6. Files

| File | Change |
|---|---|
| `espadas/hmm.py` | **new**: `viterbi(emissions, log_trans, log_start)`; nothing football-specific |
| `espadas/v4.py` | **new**: states, `features()` (reuses `v3.features` + control clues), `emissions()`, `transitions()`, `decode()`, `PARAMS`, `GRID`, `v4()` |
| `espadas/labelers.py` | register `v4` |
| `espadas/evidence.py` | `control_design(c)`, `control_log_odds(c, model)`; `clues()` gains `near_any` and `kink` |
| `espadas/evidence_models.py` | gains `CONTROL_MODEL` (generated) |
| `espadas/external/events.py` | `loose_windows()`, `control_mask()` |
| `tools/fit_evidence.py` | fit and report the control model; `--skip-heldout` |
| `tools/self_consistency.py` | one aggregate line for the control score |
| `tests/test_hmm.py`, `tests/test_v4.py`, additions to `test_external_events.py`, `test_evidence.py` | synthetic tests |
| `README.md`, `PLAN_CHECKLIST.md`, `docs/technical_document.md` | layout row, 7.x ticks and results, a bullet for section 1 and 5 |

`run.py` keeps `v1` as the default. `v4` builds `out/espadas_v4.csv` on demand once someone
with the match data runs it; this machine has no match data.

## 7. Order of work (baby steps, each one tested and benched before the next)

1. `hmm.py` with tests (tiny hand-checked sequences, forbidden moves never taken, ties).
2. `events.py` loose/control mask with tests; `evidence.py` control clues and design.
3. `fit_evidence.py`: fit `CONTROL_MODEL`; confirm the other two models are unchanged.
4. `v4.py` with `w_ctrl = 0` first (loose states are memory only) and tests; bench it on
   Games 1 and 3 next to `v1`–`v3`; then switch the control score on and bench again.
5. `tools/tune.py --labeler v4 --skip-heldout`; copy `PARAMS`; bench at all three blurs and at
   15% / 35% hidden; record in the checklist.
6. Second iteration, kept only if the tuning score improves: minimum durations inside the
   decoder (checklist 7.7) by expanding a state into a chain of sub-states that must be
   traversed before the state can be left; the chain part of the Viterbi update is a shift, so
   the cost stays about the same. Which states (builder check, call 3):
   - `D` gets a chain of `dead_min` frames (grid 1 … 100; 1 = off). Replaces `min_dead`.
   - `HL` and `AL` get a chain of `loose_min` frames (grid 1 … 12). A change of possession then
     has to pass through a loose stretch of real length, which is the point of the loose states.
   - `HC` and `AC` get **no** minimum: short possessions are real, and the checklist already
     says we miss them.
   - `min_alive` stays a post-hoc clean-up (it is a minimum on a whole live stretch between two
     stoppages, which no single state owns). If the tuner sets it to 0, drop it.
   At most about 125 states; with the shift trick the decode cost target still holds.
7. Docs and handoff. No commit without the team member's say-so.

## 8. Left to the builder

- Exact grids (keep each to 6–10 values; the table gives the ranges).
- Whether `kink` for the control model reuses `touches.kink` at `v3.PARAMS["kink_half"]`
  (recommended) or a fixed width.
- Internal layout of `hmm.py` (dense 5x5 loop vs. lists) as long as the cost target holds.
- Test fixtures: reuse `_match()` from `test_evidence.py` and `_pass_play()` from
  `test_touches.py` where they fit.

## 9. Calls made after the builder check (not team decisions; say so if you disagree)

1. `v4` gets its own touch pulse weights in log-odds units (`confirm_pulse`, `touch_pulse`),
   tuned; `v3`'s CUSUM-sized weights are not reused. Section 4.3.
2. The control target follows the reference's "won ball" rule: a touch-only event followed by
   the same team's control is controlled from the touch. Section 5.
3. Second iteration: minimum durations on `D` and on the loose states only. Section 7.
4. Tuning-time estimate corrected to one to two hours; measured before relying on it.

## 10. Open for the team (do not block the build)

- Agree the loose-span definitions in section 5 as fixed rules, next to the H/A/D rules (4.3).
- Whether `v4` should eventually tune the touch settings too (more settings, more overfit).
