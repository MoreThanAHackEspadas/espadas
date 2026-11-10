# Selection on held-out Metrica Game 2

Written by `tools/selection.py`. Each run appends a section; never edit the numbers by hand.

## Selection run: 2026-11-10, commit `2953f01`

Run started 2026-11-10 13:54:10, commit `2953f01`, working tree clean. Held-out scoring of Game 2 begins now. If no results follow below, the run was interrupted after held-out scoring began, and the one shot is spent.

Completed 2026-11-10 13:55:26: all 216 scorings finished.

Candidates scored on held-out Metrica Game 2 and, for reference, on the tuning games (1 + 3). spellF1 = mean over the tracker levels light / medium / heavy / smooth. X on = what `python -m espadas.run --abstain` builds (`abstain.resolve`): `abstain.SETTING` = `{'weak': inf, 'min_gap': 200, 'edge': 1}`. Guidance bands 0.04 spellF1 / 0.01 macroF1: smaller differences are not evidence (see the reading).

### Summary (spellF1, mean over tracker levels)

| candidate | X | tuning @25% | **Game 2 @25%** | Game 2 @15% | Game 2 @35% | macroF1 Game 2 @25% | possessions Game 2 @25% (pred / ref) | X frames |
|---|---|---|---|---|---|---|---|---|
| v4 | off | 0.555 | **0.533** | 0.541 | 0.517 | 0.791 | 231 / 264 | 0 |
| v4 | on | 0.555 | **0.533** | 0.541 | 0.517 | 0.826 | 231 / 264 | 13264 |
| v3 | off | 0.539 | **0.532** | 0.541 | 0.519 | 0.792 | 249 / 264 | 0 |
| v3 | on | 0.539 | **0.532** | 0.541 | 0.519 | 0.828 | 249 / 264 | 13207 |
| v1 | off | 0.484 | **0.470** | 0.491 | 0.465 | 0.834 | 286 / 264 | 0 |
| v1 | on | 0.484 | **0.470** | 0.491 | 0.465 | 0.837 | 286 / 264 | 934 |

### Game 2 by tracker level (spellF1 @25%)

| candidate | X | light | medium | heavy | smooth |
|---|---|---|---|---|---|
| v4 | off | 0.570 | 0.561 | 0.477 | 0.525 |
| v4 | on | 0.570 | 0.561 | 0.477 | 0.525 |
| v3 | off | 0.591 | 0.551 | 0.476 | 0.510 |
| v3 | on | 0.591 | 0.551 | 0.476 | 0.510 |
| v1 | off | 0.568 | 0.498 | 0.386 | 0.430 |
| v1 | on | 0.568 | 0.498 | 0.386 | 0.430 |

### Reading

- Leader on held-out Game 2 at 25% live ball hidden: v4 (X off), spellF1 0.533, ahead of v3 by 0.001, within the organisers' spellF1 guidance band (0.04); ahead of v1 (today's default) by 0.063, above the same band; ahead at 3 of 4 tracker levels (medium, heavy, smooth).
- X leaves spellF1 unchanged for v4 (as designed); macroF1 +0.035 (0.791 -> 0.826), above the organisers' macroF1 guidance band (0.01), X on 13264 graded frames.
- X leaves spellF1 unchanged for v3 (as designed); macroF1 +0.035 (0.792 -> 0.828), above the organisers' macroF1 guidance band (0.01), X on 13207 graded frames.
- X leaves spellF1 unchanged for v1 (as designed); macroF1 +0.003 (0.834 -> 0.837), within the organisers' macroF1 guidance band (0.01), X on 934 graded frames.
- v4: tuning-game mean 0.555 vs held-out 0.533 (-0.022); a large drop may mean the settings are overfitted to the tuning games, but the games also differ in difficulty, so compare the candidates' drops with each other.
- v3: tuning-game mean 0.539 vs held-out 0.532 (-0.007); a large drop may mean the settings are overfitted to the tuning games, but the games also differ in difficulty, so compare the candidates' drops with each other.
- v1: tuning-game mean 0.484 vs held-out 0.470 (-0.013); a large drop may mean the settings are overfitted to the tuning games, but the games also differ in difficulty, so compare the candidates' drops with each other.
- The bands (0.04 spellF1, 0.01 macroF1) are the organisers' guidance for one full match (vendored possession_metrics.py), not noise measured on this bench; the tracker levels are degraded views of the same games, not independent samples, so agreement across levels is not extra evidence.

**Decision (10 Nov 2026, applying `docs/decision_rule.md`, pushed before this run in `2953f01`):** **`v4`, X off.** `v4` was the pre-registered choice (rule 1). No candidate beats it on the main score (Game 2 spellF1 @25%, X off) by 0.04 or more: `v3` is 0.001 behind, `v1` 0.063 behind (rule 3a fails for both), so `v4` stays (rules 3-4). X stays off whatever its macroF1 gain (+0.035 here; rule 5). macroF1, the tuning-vs-held-out drop and the provided match played no part (rules 4, 6).
