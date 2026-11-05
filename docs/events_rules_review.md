# Review of the reference-label rules (`espadas/external/events.py`), checklist 4.3

Reviewed 6 Nov 2026 on the tuning games only (Metrica 1 and 3; Game 2 untouched), against
Metrica's `events-definitions.pdf`, the code, and counts on the data. These rules turn Metrica's
events into the per-frame H/A/D reference that every setting is tuned against, so a wrong rule
is a systematic bias, not noise.

## What checks out

- Every event has a team; no event ends before it starts; one SET PIECE without an end frame.
  Frames are converted 1-based → 0-based consistently in both readers (CSV, EPTS).
- Shot results match the definitions: only `…OUT` and `…GOAL` make the ball dead; `BLOCKED`,
  `SAVED`, `OFF TARGET` (picked up) and `WOODWORK-OFF TARGET` (stays in) do not.
- `BALL OUT` covers own goals (subtype `WOODWORK-GOAL`), dead from its end frame.
- Goal kicks are `PASS`/`BALL LOST` with subtype `GOAL KICK`, not `SET PIECE`; the dead window
  ends at them as intended. 140 of 176 dead windows end at a `SET PIECE`, the rest at a goal
  kick.
- `FAULT RECEIVED` (foul on the ball carrier): 30 of 34 are followed by that team's free kick;
  the 4 others are quick restarts or re-awarded fouls (1–2 per game).
- Dead windows: 88 per game, median 20 s (Game 1) / 12 s (Game 3), none shorter than 1 s.
- Halves start at the kick-off `SET PIECE`, so no frame is "D before the first possessor".
- Reference counts match the practice-game figures used throughout the checklist: Game 1
  **268 possessions / 90'**, dead 39%; Game 3 **377 / 90'**, dead 28%.

## Issue 1: referee stoppages tagged `BALL LOST-FORCED` are labelled live

Metrica's definition of the `FORCED` subtype: *"Play is halted by the referee"* (also when a
free kick is awarded for a foul on a player other than the one in possession). `VOLUNTARY`
(ball given up for an injury) and `END HALF` are stoppages too. `dead_start()` only knows
`OFFSIDE` and `HAND BALL`, so after a `FORCED` the losing team keeps the label until the free
kick: **5 events / 95 s in Game 1, 9 events / 110 s in Game 3**, about 1.5–2% of the graded frames.

Player motion confirms they are stoppages: mean player speed is ~2.2 m/s in live play and
~1.0 m/s in known stoppages (after `FAULT RECEIVED`); in the `FORCED` gaps it is 0.68–0.99 m/s
in 6 cases, 1.4–1.6 m/s in 4 short (5–7 s) ones, and 2.2 m/s in one 5 s case.

Sensitivity: with the fix (`FORCED`, `VOLUNTARY`, `END HALF` added to the whistle subtypes) and
**unchanged `v4` settings**, spellF1 at the medium level, 25% hidden, moves **+0.011 on Game 1
and −0.025 on Game 3** (5 reference spells split by the new stoppages). `v4` was tuned on the
old labels, so it currently calls these stretches live; a re-tune would be needed to judge the
real effect.

## Issue 2: the two tuning games are annotated with different conventions

Game 3 (EPTS format) has 1,395 `CARRY` events; Game 1 has none. 282 of Game 3's 294 recoveries
are followed within one frame by a `CARRY` of the same team, and `CARRY` is a control event, so
**in Game 3 every recovery is immediately a change of possession**, while in Game 1 a recovery
only counts if the same team then passes, shoots or loses the ball first (25% of Game 1's
recoveries never do, and produce no change). 162 of Game 3's 393 changes are confirmed by a
`CARRY` (median carry 1.0 s, half shorter than 1 s).

If `CARRY` did not confirm a change, Game 3 would have **303 possessions / 90' instead of 377**
(spells under 2 s: 15% instead of 26%), close to Game 1's 268 / 12%; Game 1 is unchanged. The
rules as written follow Metrica's events literally, which is defensible, but the two games then
pull the tuning in different directions, and `v4` under-segments Game 3 badly (255 predicted vs
399 reference at the light level) while matching Game 1 better (230 vs 282). Which convention
the hidden ground truth follows is unknown; the official definition's emphasis on *controlled*
possession ("a touch without control is not a change") leans towards the stricter reading.

## Issue 3: `CHALLENGE` is documented as touch-only but ignored by the code

The module docstring lists `CHALLENGE` among the touch-only events that can start a back-dated
change; `ownership_changes()` skips it because it is not in `IN_PLAY`. Effect is small: 5–6
changes per game could be dated earlier (a won challenge followed by the winner's own control
event before the opponent acts), and 16 (Game 1) / 40 (Game 3) changes would move ~1.5 s
earlier. Either the docstring or the code should change; the docstring is the cheap fix.

## Issue 4: offside flagged on a `PASS`

2 events in Game 3 (`PASS-OFFSIDE`, `PASS-DEEP BALL-OFFSIDE`), none in Game 1: the flag stops
play but only `BALL LOST-OFFSIDE` is handled, so up to 40 s is labelled live. Negligible; note it.

## Also seen, no action

- 73 (Game 1) / 59 (Game 3) changes start with the new owner's `BALL LOST`: a recovery followed
  by a failed pass. Those are real short possessions (a pass was attempted), kept.
- A change can be back-dated by up to 53 s through a touch-only event when the ball goes out in
  between; the dead window overrides those frames, so the labels are still right.

## Options for the team

1. **Fix issue 1 (and the docstring of issue 3), then re-tune** `fit_evidence`, `v1`, `v3`,
   `v4` and the X policy on the corrected labels before the freeze (a few hours of compute plus
   review), and note the change in the tech doc. Cleanest reference; costs a day and changes
   every bench number quoted so far.
2. **Change nothing before selection; document issues 1–4** in the tech doc (sections 3 and 5)
   as known limits of the reference labels. Zero risk to the schedule; the reference stays
   ~1.5–2% too "live", and the Game 1 / Game 3 convention gap stays.
3. Option 2 now, and revisit issue 2 only if the selection result is close.

## Decision

6 Nov 2026: **option 2, document only.** The rules stay as they are until after selection;
issues 1–4 are noted in the technical document (sections 3–5). The `CHALLENGE` sentence in the
module docstring is still inaccurate (issue 3); the tech doc describes what the code does.
