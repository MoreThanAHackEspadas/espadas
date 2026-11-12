# Refinement 2: results (branch `refinement-2`)

Judged by `docs/refinement2_rule.md` (pushed in `8b6b7d3` before any of these numbers was read).
All scores: spellF1 on the game left out, mean over the four tracker levels, `X` off; 25% of the
live ball hidden unless stated. Our match is not used anywhere here.

**Steps tried so far: 5 runs (steps 1, 2, 3, 4 and the combination 2 + 3).** Kept: step 3 (five-match pool). Adopted: none; no candidate meets rule 5(a) yet. The frozen submission is still `main@53861c0`.

## Baseline: the frozen method through leave-one-game-out (11 Nov)

`python tools/logo.py --labeler v4 --workers 8` with the code of `main@53861c0`.

| Game left out | Re-tuned in the fold | Shipped settings | 15% hidden | 35% hidden |
|---|---|---|---|---|
| Metrica 1 | 0.5609 | 0.5627 | 0.5645 | 0.5507 |
| Metrica 3 | 0.5358 | 0.5403 | 0.5506 | 0.5183 |
| IDSSE 12 | 0.5257 | 0.5269 | 0.5406 | 0.5087 |
| IDSSE 13 | 0.5569 | 0.5552 | 0.5746 | 0.5491 |
| IDSSE 15 | 0.5817 | 0.5877 | 0.5973 | 0.5716 |
| **Mean** | **0.5522** | 0.5546 | 0.5655 | 0.5397 |

Mean train-test gap +0.016. "Shipped settings" uses the fold's evidence models with the settings
of `main@53861c0` (which were tuned on all five games, so that column is not a fair test score;
it is shown because re-tuning inside the fold does not beat it: on four matches the search
finds nothing that carries over better than the settings we already ship).

The frozen models and settings on the five games they never saw (for rule 6): Metrica 2 0.5589;
IDSSE 11, 14, 16, 17: 0.5541, 0.5523, 0.5455, 0.5657 (at 15% / 35%: 0.5648 / 0.5500, 0.5701 /
0.5292, 0.5681 / 0.5270, 0.5529 / 0.5273, 0.5815 / 0.5502).

## Step 1: wider search ranges — dropped

Nine of `v4`'s 28 settings sat on an edge of their search range that could be extended. The
ranges were widened (`team_cap` to 12, `w_ctrl` to 4, `ctrl_reach` to 12, `c_restart` to 640,
`chain_gap` to 250, kink width to 16; `c_stop` down to 1, `hold_reach` to 0.5, `carrier_frames`
to 1) and LOGO run again, everything else the same.

| Game left out | Baseline | Wider ranges | Difference |
|---|---|---|---|
| Metrica 1 | 0.5609 | 0.5516 | −0.0093 |
| Metrica 3 | 0.5358 | 0.5349 | −0.0009 |
| IDSSE 12 | 0.5257 | 0.5257 | 0 (same settings found) |
| IDSSE 13 | 0.5569 | 0.5569 | 0 (same settings found) |
| IDSSE 15 | 0.5817 | 0.5844 | +0.0027 |
| **Mean** | **0.5522** | **0.5507** | **−0.0015** |

At 15% / 35% hidden: 0.5652 / 0.5410 against 0.5655 / 0.5397. In-sample the wider ranges score
higher (train-test gap +0.019 against +0.016): the extra values are used to fit the training
games, not the game left out. **Not an improvement over the step before, so by rule 7 it is
dropped** and the search ranges of `main@53861c0` are kept for the following steps. The edges
were not hiding a better setting.

## Step 3: boosted evidence models — diagnostic (11 Nov)

`python tools/evidence_diag.py`: for every game left out of the five-match pool, the three
evidence models fitted on the other four, as logistic regressions (shipped) and as small
gradient-boosted models (depth 3, 200 trees, same clue columns; the team model trained on each
frame and its mirror image, so it has no home bias). Per-frame, on the game left out, mean of
the five folds:

| Model | Log-loss, logistic → boosted | Accuracy, logistic → boosted |
|---|---|---|
| Dead, ball not tracked | 0.3377 → 0.3156 | 0.858 → 0.871 |
| Dead, ball tracked | 0.1074 → 0.0677 | 0.968 → 0.977 |
| Team (home / away) | 0.3145 → 0.2920 | 0.868 → 0.873 |
| Team, ball not tracked | | 0.784 → 0.791 |

The log-loss is lower on every game for all three models (15 of 15). Accuracy is up in 14 of 15;
the team accuracy with the ball missing, the number this step was meant to move, is up on three
games and down on two (+0.7 points on average). So the evidence is better calibrated, clearly so
for stoppages, and only slightly better at naming the team. Enough to go on to the real test:
per-frame gains do not have to carry over to spellF1.

Made shippable: `tools/fit_evidence.py --model boosted` exports the trees as plain numbers into
`espadas/evidence_models.py`; `evidence.apply_trees` evaluates them with NumPy (no scikit-learn
in the build; a test checks the export against scikit-learn's own output and the exact team
symmetry). `tools/logo.py --evidence boosted` fits them inside every fold.

## Step 3: boosted evidence models — LOGO (12 Nov): kept, +0.019

`python tools/logo.py --labeler v4 --evidence boosted`: five-match pool, the frozen search
ranges, the boosted models fitted inside every fold.

| Game left out | Baseline | Boosted evidence | Difference | 15% hidden | 35% hidden |
|---|---|---|---|---|---|
| Metrica 1 | 0.5609 | 0.5732 | +0.0123 | 0.5833 | 0.5627 |
| Metrica 3 | 0.5358 | 0.5537 | +0.0179 | 0.5703 | 0.5324 |
| IDSSE 12 | 0.5257 | 0.5545 | +0.0288 | 0.5695 | 0.5373 |
| IDSSE 13 | 0.5569 | 0.5805 | +0.0236 | 0.5944 | 0.5657 |
| IDSSE 15 | 0.5817 | 0.5947 | +0.0130 | 0.6070 | 0.5765 |
| **Mean** | **0.5522** | **0.5713** | **+0.0191** | 0.5849 (+0.0194) | 0.5550 (+0.0153) |

Ahead on 5 of 5 games, at all three hide levels, on both providers. An improvement over the
step before, so by rule 7 it is kept and the next steps build on it.

**On its own it does not meet rule 5(a):** +0.0191 is just under the 0.02 asked for. (b), (c)
hold; (d) not yet checked. Two things to note, neither of which changes that:
- the train-test gap grew from +0.016 to +0.030: the boosted models fit the training games
  much better (in-sample 0.59–0.61) than they carry over;
- re-tuning inside the fold again does worse than the shipped settings with the fold's boosted
  models (0.5713 against 0.5756); that column is not a fair score, as before.

## Step 4: a longer context window — dropped

The team clues already have smoothed copies over ±1 s and ±3 s (`evidence.CONTEXT = (25, 75)`).
Tried on top of step 3: a third copy over ±6 s, for every clue of all three models
(`python tools/logo.py --labeler v4 --evidence boosted --context 25 75 150`).

| Game left out | Step 3 | Step 3 + ±6 s | Difference |
|---|---|---|---|
| Metrica 1 | 0.5732 | 0.5450 | −0.0282 |
| Metrica 3 | 0.5537 | 0.5458 | −0.0079 |
| IDSSE 12 | 0.5545 | 0.5412 | −0.0133 |
| IDSSE 13 | 0.5805 | 0.5825 | +0.0020 |
| IDSSE 15 | 0.5947 | 0.6032 | +0.0085 |
| **Mean** | **0.5713** | **0.5635** | **−0.0078** |

At 15% / 35% hidden: 0.5754 / 0.5510 against 0.5849 / 0.5550. The train-test gap grew again, to
+0.046: the longer window helps the models fit the training games and hurts on three of the
five games left out. **Worse than the step before, dropped by rule 7**; `CONTEXT` stays (25, 75).

## Step 2: all ten external matches — no clear gain (+0.003)

`python tools/logo.py --labeler v4 --pool full`: ten-fold LOGO, logistic evidence, the frozen
search ranges. Compared by rule 6: the baseline's score for a game is its five-fold LOGO score
if the game is in the five-match pool, else the score of the shipped `main@53861c0` on it.

| Game left out | Baseline | Ten-match pool | Difference |
|---|---|---|---|
| Metrica 1 | 0.5609 | 0.5634 | +0.0025 |
| Metrica 2 | 0.5589 | 0.5570 | −0.0019 |
| Metrica 3 | 0.5358 | 0.5352 | −0.0006 |
| IDSSE 11 | 0.5541 | 0.5523 | −0.0018 |
| IDSSE 12 | 0.5257 | 0.5165 | −0.0092 |
| IDSSE 13 | 0.5569 | 0.5607 | +0.0038 |
| IDSSE 14 | 0.5523 | 0.5558 | +0.0035 |
| IDSSE 15 | 0.5817 | 0.5967 | +0.0150 |
| IDSSE 16 | 0.5455 | 0.5573 | +0.0118 |
| IDSSE 17 | 0.5657 | 0.5748 | +0.0091 |
| **Mean** | **0.5538** | **0.5570** | **+0.0032** |

Ahead on 6 of 10 games; at 15% / 35% hidden 0.5699 / 0.5423 against 0.5665 / 0.5382. The
train-test gap is small (+0.007). Going from two to five matches was worth +0.07 on a new
provider; going from five to ten of the same two providers is worth almost nothing with the
logistic models: they are already saturated. Far from 5(a) and 5(b) on its own. Whether the
larger pool helps the boosted models, which overfit more, is the combined run below.

## Steps 2 + 3 together: boosted evidence on ten matches — no better than step 3

`python tools/logo.py --labeler v4 --pool full --evidence boosted`, compared by rule 6.

| Game left out | Baseline | Boosted, ten matches | Difference | 15% hidden | 35% hidden |
|---|---|---|---|---|---|
| Metrica 1 | 0.5609 | 0.5575 | −0.0034 | 0.5741 | 0.5538 |
| Metrica 2 | 0.5589 | 0.5421 | −0.0168 | 0.5470 | 0.5309 |
| Metrica 3 | 0.5358 | 0.5522 | +0.0164 | 0.5696 | 0.5290 |
| IDSSE 11 | 0.5541 | 0.5699 | +0.0158 | 0.5887 | 0.5560 |
| IDSSE 12 | 0.5257 | 0.5555 | +0.0298 | 0.5765 | 0.5377 |
| IDSSE 13 | 0.5569 | 0.5863 | +0.0294 | 0.6001 | 0.5755 |
| IDSSE 14 | 0.5523 | 0.5640 | +0.0117 | 0.5795 | 0.5415 |
| IDSSE 15 | 0.5817 | 0.6039 | +0.0222 | 0.6127 | 0.5904 |
| IDSSE 16 | 0.5455 | 0.6030 | +0.0575 | 0.6151 | 0.5784 |
| IDSSE 17 | 0.5657 | 0.5751 | +0.0094 | 0.5784 | 0.5653 |
| **Mean** | **0.5538** | **0.5710** | **+0.0172** | 0.5842 (+0.0177) | 0.5559 (+0.0177) |

Ahead on 8 of 10 games (all seven IDSSE games; behind on two of the three Metrica games).
Rule 5: (a) **not met**, +0.0172 < 0.02; (b) met, 80%; (c) met.

Against step 3 on the five games both were scored on (Metrica 1, 3; IDSSE 12, 13, 15): 0.5711
with ten matches, 0.5713 with five. The larger pool adds nothing to the boosted models either,
so **step 2 is dropped** and the five-match pool stays.

## Where this stands (12 Nov, morning)

| Step | Change | LOGO difference | Outcome |
|---|---|---|---|
| 1 | Wider search ranges | −0.0015 | dropped |
| 2 | Ten-match pool | +0.0032 (6 of 10 games) | dropped: within noise |
| 3 | Boosted evidence models | **+0.0191** (5 of 5 games) | kept |
| 4 | Extra ±6 s context, on step 3 | −0.0078 | dropped |
| 2 + 3 | Boosted, ten-match pool | +0.0172 (8 of 10 games) | no better than step 3 |

The boosted evidence models are a real and consistent gain of about +0.017 to +0.019 spellF1,
at every hide level and on both providers, measured two ways. That is below the +0.02 written
down beforehand, so **as things stand nothing is adopted and `main@53861c0` ships**. The rule is
not changed after the results. A later step on top of step 3 could still clear it before the
14 Nov cut-off.

One pattern in every run: re-tuning the settings inside a fold scores the same or worse on the
game left out than keeping the settings of `main@53861c0` (0.5522 / 0.5546 logistic, 0.5713 /
0.5756 boosted, 0.5710 / 0.5752 boosted on ten). The settings search has nothing left to give
at this amount of data; the gains come from the evidence.

## Step 5, measurement only: which possessions are missed (12 Nov)

`python tools/spell_diag.py`: the shipped `v4` (`main@53861c0`) on the five-match pool, all four
tracker levels pooled, 25% hidden, matched as the metric does at IoU 0.5. These games were used
to fit the shipped models, so this describes the errors; it is not a test score.

| Possession length | Reference | Found | Recall | Predicted | Real | Precision |
|---|---|---|---|---|---|---|
| < 2 s | 1352 | 113 | 0.08 | 398 | 103 | 0.26 |
| 2–5 s | 2148 | 783 | 0.36 | 1369 | 760 | 0.56 |
| 5–15 s | 2288 | 1570 | 0.69 | 2106 | 1468 | 0.70 |
| > 15 s | 1296 | 1117 | 0.86 | 1556 | 1252 | 0.80 |
| All | 7084 | 3583 | 0.51 | 5429 | 3583 | 0.66 |

- Half of the reference possessions (49%) last under 5 s, and we find 26% of those. Possessions
  over 5 s are found 75% of the time.
- We predict 77% as many possessions as the reference has, and recall (0.51) is the weaker side
  of the score, not precision (0.66): the labeler merges, it does not flicker.
- The largest block of misses is the 2–5 s bin: 1365 missed, more than the < 2 s bin (1239),
  and unlike those they are long enough to be detectable.
- Precision falls with tracker quality (0.71 light, 0.62 heavy); recall much less (0.53 to 0.49).

No decoder change has been made. A change aimed at 2–5 s possessions is the lever with the most
room, and also the one most exposed to the reference conventions differing between providers.

## Step 5: why the short possessions are missed — no decoder change (12 Nov)

Three more measurements on the same setting (shipped `v4`, five-match pool, in-sample), made
before touching the decoder.

**1. What is predicted over a missed possession** (`tools/spell_diag.py`, all levels pooled):

| Possession length | Missed | Mostly the same team | Mostly the other team | Mostly dead |
|---|---|---|---|---|
| < 2 s | 1239 | 448 | 732 | 59 |
| 2–5 s | 1365 | 609 | 687 | 69 |
| 5–15 s | 718 | 507 | 172 | 39 |
| > 15 s | 179 | 152 | 7 | 20 |

Under 5 s, more than half of the misses are labelled the other team: swallowed by the
opponent's possession around them. ("Same team" = right team, but the predicted possession is
too long or too short to match, mostly because a neighbouring short possession was swallowed.)

**2. Stoppages are not the cause.** A reference stoppage between two possessions is predicted
dead over most of its length in 92–98% of cases when it lasts over 5 s (same team either side;
74–95% with different teams). Stoppages under 5 s are rare: 160 of 1940.

**3. The decoder is not what blocks the switch; the evidence points the wrong way.** For the
reference possessions labelled the other team:

| Possession length | Frames where the team evidence favours the right team | Possessions whose summed evidence would pay for the switch (cost 44) |
|---|---|---|
| < 2 s | 10% | 0% |
| 2–5 s | 16% | 1% |
| 5–15 s | 26% | 13% |

The ball is tracked on 72–79% of those frames, so a hidden ball is not the reason. Even the raw
clue is wrong there: with the ball tracked, the nearest player belongs to the right team on
40–44% of the frames of those possessions, worse than a coin (light and heavy levels). Over
*all* reference possessions it is 51% under 2 s, 57% for 2–5 s, 69% for 5–15 s.

**Conclusion.** Short reference possessions are mostly not visible in who is near the ball: the
ball is in flight, contested, or the possession exists by the annotation convention (a pass
received and released, the IDSSE 1 s touch rule). No transition cost or duration model can
recover a possession for which the evidence favours the opponent on 85% of its frames; lowering
the switching cost would add switches where the evidence is noisy, i.e. flicker. The plan's own
stop condition for this step applies, and **the decoder is left unchanged**. The same finding
limits step 7 (carrier-touch attribution), which rests on the same nearest-player geometry.
