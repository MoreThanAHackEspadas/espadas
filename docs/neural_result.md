# Neural team evidence: results (branch `new-arch`, 16 Nov)

Built on the player-level candidate of 13 Nov (branch `architecture-review`, PR #6). All scores:
spellF1 on the match left out, mean over the four tracker levels, `X` off, 25% of the live ball
hidden unless stated. Our match is not used anywhere here.

**Status: proposed for submission, pending the team's confirmation on PR #7.** This work was
done after the 14 Nov cut-off of `docs/refinement2_rule.md`, which allows one final candidate,
and that candidate is PR #6. Submitting this instead is a decision for the team.

**The shipped models are those of the last two sections** (ten matches, possession-weighted
loss, ball-blind stoppage network; SHA-256
`3b35ab3bce0b73f20e627f580db79525011dacc5d1ad9e0f594404cfe7fe1fe0`). The sections before them describe
the first version of this branch (five-match pool, SHA-256 `40a1012c…5c65`) and are kept as the
record of how it was measured.

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

## Ten matches, a ball-blind stoppage network, possession-weighted loss (16 Nov, evening)

Code and commands: `experiments/neural_v2/` (README there). All ten external matches are used:
the five-match pool plus Metrica 2 and IDSSE 11, 14, 16, 17, which were spent as held-out tests
on 10 and 11 Nov. Folds: (1, 11), (2, 12), (3, 13), (14, 15), (16, 17); the networks of a fold
are trained on the other eight matches, so every match is scored by networks that did not see
it. Decoder settings are those of the section "Settings and build" above, unchanged; they were
tuned on the five-match pool, so that column is slightly optimistic.

| | All ten | Five-match pool | Metrica (3) | IDSSE (7) |
|---|---|---|---|---|
| Network of this branch, eight training matches, tree dead models | 0.7019 | 0.6952 | 0.6365 | 0.7300 |
| + ball-blind network on frames without a ball | 0.7147 | 0.7131 | 0.6794 | 0.7299 |
| + possession-weighted team loss (**shipped**) | **0.7235** | **0.7198** | **0.6792** | **0.7425** |
| Upper bound: reference dead / live | 0.7494 | 0.7451 | 0.7326 | 0.7567 |

The first and last rows were measured on a teammate's machine, the middle two on the machine
that built the submission (the second row was measured on both: 0.7152 and 0.7147).

- **More training matches.** Same network and configuration, eight matches instead of four:
  0.6952 on the five-match pool against 0.6846.
- **Ball-blind dead network** (`espadas/blind.py`). Reads every player's velocity and pitch
  position and the player-motion clues, and nothing measured from the ball, so it cannot learn
  "no ball = stoppage" (the reason the team network's dead output is not used). It replaces the
  motion-only tree model on frames without a ball; the tree model stays on frames with one. In
  the Metrica matches the ball is tracked on 0 to 16% of dead frames, so stoppages have to be
  read from the players: Metrica 1 / 2 / 3 go from 0.637 / 0.631 / 0.642 to 0.682 / 0.683 /
  0.671. IDSSE is unchanged.
- **Possession-weighted loss.** In the team loss each live frame is weighted by the length of
  its possession to the power -0.5 (mean 1, clipped to 0.25 to 4). Ahead on 8 of 10 matches
  (behind on Metrica 2 by 0.013 and IDSSE 13 by 0.003); level on Metrica as a group, +0.013 on
  IDSSE.

Per match, second row to third: Metrica 1 0.6805 to 0.6870, 2 0.6814 to 0.6681, 3 0.6763 to
0.6823; IDSSE 11 0.7178 to 0.7562, 12 0.7161 to 0.7216, 13 0.7582 to 0.7557, 14 0.7170 to
0.7236, 15 0.7344 to 0.7524, 16 0.7206 to 0.7254, 17 0.7453 to 0.7626.

## The shipped build (16 Nov, evening)

`experiments/neural_v2/final.py export posb --blind-mode split`: one team network and one
ball-blind network fitted on all ten matches, and the tree dead models, written to
`espadas/neural_weights.py`, `blind_weights.py` and `evidence_models.py`. Both networks are
evaluated in 64-bit arithmetic. `final.py check` on Metrica 1: the build's NumPy path and the
training path give the same labels on 100% of frames.

On our match, as a validity check only: valid, deterministic, SHA-256
`3b35ab3bce0b73f20e627f580db79525011dacc5d1ad9e0f594404cfe7fe1fe0`; H 56,329, A 36,956, D 65,572,
X 40,462; 337 possessions, mean 11.1 s (first version of this branch: 324, 11.5 s; PR #6: 302,
12.4 s). It gives the same label as the first version on 91.7% of graded frames.

What to keep in mind, besides the weaknesses above:

- **The decoder settings were not re-tuned** for these models.
- **The possession-weighted variant was chosen after seeing these ten matches**, and these are
  further runs on matches already used; all ten are now spent.
- **The shipped networks have no held-out score**: they are trained on all ten matches, and the
  numbers come from the fold networks.
- **The ball-blind network depends on player tracking only.** Our match has track slots that
  swap identities, which the external matches do not have; it reads velocities over about half
  a second, so a swap shows up as a short burst of false speed.
- **A second-machine rebuild has not been done.**
