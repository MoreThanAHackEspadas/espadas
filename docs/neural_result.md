# Neural team evidence: results (branch `new-arch`, 16 Nov)

Built on the player-level candidate of 13 Nov (branch `architecture-review`, PR #6). All scores:
spellF1 on the match left out, mean over the four tracker levels, `X` off, 25% of the live ball
hidden unless stated. Our match is not used anywhere here.

**Status: not adopted.** This work was done after the 14 Nov cut-off of
`docs/refinement2_rule.md`, which allows one final candidate, and that candidate is PR #6. The
numbers below meet the rule's score conditions by a wide margin, but whether this replaces PR #6
is a decision for the team.

## What changed

One network (`espadas/neural.py`) replaces the tree model for home-or-away. Nothing else in the
pipeline changes: same clues, same dead-ball tree models, same 5-state decoder (`v4`), re-tuned.

- **Input, per frame.** Every tracked player, up to 11 a side: position relative to the
  last-seen ball, velocity, closing speed, position on the pitch. Plus the ball's own state and
  all 68 clues the tree models read.
- **Player encoder.** One small network applied to each player, pooled per team (max, mean, and
  the three nearest in order).
- **Temporal model.** Eight dilated convolutions, 64 channels, looking 255 frames (about 10 s)
  each side.
- **Training targets.** Home or away, dead or live, and a bump at every change of the reference
  label, so the timing of a change is rewarded directly.
- **No home bias.** The network is evaluated on the frames and on their team-swapped copy, and
  the two are combined so that swapping the teams exactly flips the output.
- **Build.** Trained with PyTorch on external matches only (`tools/fit_neural.py`). The weights
  are stored as text (`espadas/neural_weights.py`, 0.9 MB) and evaluated with NumPy, so building
  the submission still needs neither PyTorch nor a GPU.

**Dead evidence is not taken from the network.** Its dead output is right on 95–99% of frames of
a match it never saw, but it can read whether the ball is tracked, and in the external matches a
long gap nearly always means a stoppage. The organisers say that does not hold on our match
(`evidence.py`, DEAD_SPLIT), so the dead evidence stays with the two tree models that cannot use
that shortcut.

## Cross-fitted leave-one-game-out

`tools/logo_neural.py` follows `tools/logo.py`, with one difference. The player-level trees
scored about 0.70 on their training matches against 0.62 on the match left out, so the decoder
was being tuned on evidence that was too confident. Here the evidence the decoder is tuned on is
itself out-of-sample: for test match *b* and training matches *T*, the evidence for *b* comes
from the network trained on *T*, and the evidence for each training match *a* comes from the
network trained on *T* without *a*. That takes 15 networks (5 on four matches, 10 on three), at
2–3 minutes each on a laptop GPU.

| Match left out | Frozen (11 Nov) | Player-level trees (PR #6) | Network, first run | Network, regularised |
|---|---|---|---|---|
| Metrica 1 | 0.5609 | 0.5993 | 0.6235 | 0.6454 |
| Metrica 3 | 0.5358 | 0.5781 | 0.6267 | 0.6291 |
| IDSSE 12 | 0.5257 | 0.6309 | 0.6733 | 0.6796 |
| IDSSE 13 | 0.5569 | 0.6353 | 0.7405 | 0.7390 |
| IDSSE 15 | 0.5817 | 0.6366 | 0.7190 | 0.7299 |
| **Mean** | **0.5522** | **0.6160** | **0.6766** | **0.6846** |
| At 15% / 35% hidden | 0.5655 / 0.5397 | 0.6264 / 0.6050 | 0.6850 / 0.6714 | 0.6946 / 0.6756 |
| Train minus test | +0.016 | +0.089 | −0.001 | −0.004 |

Per frame, on the matches left out (home-or-away, live frames):

| | Log-loss | Accuracy |
|---|---|---|
| Logistic (11 Nov) | 0.2927 | 0.8725 |
| Player-level trees | 0.2624 | 0.8862 |
| Network, first run | 0.2783 | 0.9093 |
| Network, regularised | 0.2328 | 0.9094 |

Against rule 5 of `docs/refinement2_rule.md`, the regularised network: +0.132 over the frozen
method (needs +0.02), ahead on 5 of 5 matches, ahead at 15% and 35% hidden, build valid and
deterministic. It is also ahead of PR #6 on 5 of 5 matches, by +0.069 on average.

## How the regularised network was chosen

The first run (dropout 0.1, 3,000 steps) was a configuration fixed before any result. Its
per-frame log-loss was worse than the trees' although its accuracy was better: it was
overconfident. Two variants were then compared on inner validation only (the ten three-match
networks, each scored on the two matches it did not see, decoder settings fixed):

| Variant | spellF1 | Log-loss | Accuracy |
|---|---|---|---|
| First run | 0.6650 | 0.350 | 0.906 |
| Regularised (dropout 0.25, 2,000 steps) | 0.6721 | 0.264 | 0.905 |
| Two first-run networks averaged | 0.6710 | 0.335 | 0.908 |

The regularised network was ahead on all five matches and is half the size of the ensemble, so
it was kept and given the full leave-one-game-out run above.

## Settings and build

The shipped network is fitted on all five matches. The decoder was tuned on all five with
out-of-sample evidence (`tools/logo_neural.py --final`): 0.6888. Settings that moved from the
13 Nov values: `dead_smooth` 6 → 0, `dead_bias` −1.0 → −0.5, `w_dead` 4 → 16, `team_cap` 2 → 5,
`loose_scale` 1 → 0, `w_touch_own` 20 → 10, `w_loose` 0 → 1, `kink_half` 10 → 6,
`kink_threshold` 1 → 3, `chain_gap` 25 → 100, `hold_reach` 1.5 → 1.0.

Built once on our match, as a validity check only (no number from our match chose anything):
valid, deterministic, SHA-256 (unchanged by the later switch to 64-bit arithmetic)
`40a1012ce46f49ca2c715312524132c16ccdd69e3c1b9605169e5bf30c695c65`; 324 possessions (275 per
90 min), mean 11.5 s, home share 60%; H 56,483, A 36,900, D 65,474, X 40,462. PR #6 gives 302
possessions (257), 12.4 s, 61%.

## The three models side by side

By data provider, leave-one-game-out spellF1 (mean of that provider's matches):

| Provider | Frozen (11 Nov) | Player-level trees (PR #6) | Network (this branch) |
|---|---|---|---|
| Metrica (2 matches) | 0.5484 | 0.5887 | 0.6373 |
| DFL / Sportec IDSSE (3 matches) | 0.5548 | 0.6343 | 0.7162 |

On our match there are no labels to score against, so the three outputs can only be compared
with each other. Aggregates only; none of this chose anything, and none of it says which
output is right.

| Output | H | A | D | Possessions | Per 90 min | Mean length | Under 2 s | 2–5 s | Over 15 s |
|---|---|---|---|---|---|---|---|---|---|
| Frozen | 35.0% | 21.6% | 43.4% | 311 | 264 | 11.6 s | 31 | 82 | 78 |
| PR #6 | 36.1% | 23.1% | 40.8% | 302 | 257 | 12.4 s | 25 | 92 | 82 |
| Network | 35.6% | 23.2% | 41.2% | 324 | 275 | 11.5 s | 44 | 96 | 87 |

| Pair | Same label | Same dead / live call | Same team where both live | Similarity as spellF1 |
|---|---|---|---|---|
| Frozen and PR #6 | 90.5% | 95.0% | 91.9% | 0.626 |
| Frozen and network | 88.2% | 95.3% | 87.2% | 0.532 |
| PR #6 and network | 92.9% | 98.1% | 90.9% | 0.697 |

Shares are of the 158,857 graded frames. "Similarity as spellF1" scores one output against the
other as if it were the reference. The network and PR #6 share the same kind of dead models, so
their stoppages agree on 98% of frames; where both call the ball live they disagree on the team
on 9% of frames, and the network finds more short possessions. The reference matches have 269
and 382 possessions per 90 min (Metrica 1 and 3), so all three outputs are in or just below that
range.

## Weaknesses

- **Selection on the same matches.** This is the seventh and eighth run judged by the same five
  matches. The regularised variant was chosen with all five as inner validation, so 0.6846 is
  slightly optimistic; 0.6766 is the number for a configuration fixed in advance.
- **Start point.** As in `tools/logo.py`, each fold's tuning starts from the shipped settings,
  which were tuned on all five matches.
- **Tracker transfer is untested.** The network reads positions and velocities more directly
  than the trees do, and our match's tracker is not one it was trained on. The gain over PR #6
  is +0.05 on both Metrica matches and +0.05 to +0.10 on the IDSSE ones.
- **Cross-machine determinism.** Two builds on one machine are byte-identical. The network is
  floating-point matrix arithmetic, where another CPU or maths library can round differently.
  The forward pass therefore runs in 64-bit (differences of about 1e-13) and the log-odds are
  rounded to 3 decimals, which makes a different file on another machine very unlikely, but
  unlike the trees it is not guaranteed by construction. A fresh-clone rebuild on a second
  machine should confirm the hash before this is submitted; it has not been done.
- **Training is not bit-reproducible.** GPU training with a fixed seed gives slightly different
  weights each time. The submitted weights are the ones in the repository; re-running
  `fit_neural.py export` would give a different file and hash.
- **The ten-match pool and the network's change output are unused.** Both are possible next
  steps, not tried.
