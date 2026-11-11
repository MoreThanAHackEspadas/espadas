# Refinement 2: results (branch `refinement-2`)

Judged by `docs/refinement2_rule.md` (pushed in `8b6b7d3` before any of these numbers was read).
All scores: spellF1 on the game left out, mean over the four tracker levels, `X` off; 25% of the
live ball hidden unless stated. Our match is not used anywhere here.

**Steps tried so far: 1.** Steps adopted: 0. The frozen submission is still `main@53861c0`.

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
