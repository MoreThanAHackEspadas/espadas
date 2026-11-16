# Experiments on top of the neural team evidence (16 Nov)

Work in progress, started on 16 Nov on top of `new-arch` (PR #7). Nothing here changes the build:
the scripts read the external matches only and write to a cache outside the repository. Our
match is never read.

## What is being tested

All ten external matches are used (the five-match pool plus the five spent held-out ones). Every
score is spellF1 on a match the networks did not see, mean of the four tracker levels, 25% of
the live ball hidden, `X` off.

Folds (`g5`): five folds that each leave two matches out, (1, 11), (2, 12), (3, 13), (14, 15),
(16, 17). The networks of a fold are trained on the other eight matches.

| Variant | What it is |
|---|---|
| `base` | PR #7's network and configuration, trained on eight matches instead of four |
| `ens3` | three networks with different seeds, log-odds averaged |
| `tta`, `ens3tta` | the network also evaluated on the pitch-mirrored match, averaged |
| `posb`, `posb1` | team loss weighted by possession length to the power -0.5 / -1, so short possessions count more (each possession counts once in spellF1) |
| `wide`, `long` | 96 channels and 3,000 steps; 4,000 steps |
| blind network (`blind.py`) | a second network for dead-or-live that never reads the ball: player velocities and pitch positions, plus the player-motion clues. It cannot learn "no ball = stoppage" |
| change bonus (`v4c.py`) | the network's unused "the label changes here" output lowers the decoder's switching cost at that frame (`w_change`) |

How the dead evidence is combined (`dexp.py --dead`): `tree` (the build of PR #7), `split` (blind
network on frames without a ball, tree model on frames with one), `avg`, `blind`, and `net` (the
team network's own dead output: diagnostic only, it reads whether the ball is tracked).

## How to run

```bash
export MTAH_EXTERNAL_DIR=...            # the ten external matches
export ESPADAS_NEURAL_CACHE=...         # about 45 GB free
pip install -r requirements-fit.txt
cd experiments/neural_v2

python nexp.py prep                                  # views of all ten matches (CPU, ~40 min)
python nexp.py fit base --folds g5                   # 5 networks + evidence (GPU)
python blind.py fit --name b0 --folds g5             # 5 blind networks (GPU, small)
python dexp.py ceilings base --folds g5              # evidence as is, and with oracle dead / team
python dexp.py score base --folds g5 --dead split --blind b0
python dexp.py score base --folds g5 --change --w-change 20
python dexp.py cvtune base --folds g5 --dead split --blind b0 --change --keys all
```

`cvtune` is the number to quote: for each fold the decoder settings are tuned on the other eight
matches (each with evidence from networks that did not see it) and the two left out are scored.
`score` uses the settings of PR #7 unchanged, which were tuned on the five-match pool, so it is a
screening number only.

## Results so far (16 Nov, 17:35 IST)

All ten matches, each scored by networks that did not see it (fold networks trained on eight
matches). Decoder settings: those of PR #7, unchanged (tuned on the five-match pool, so these are
screening numbers; the cross-fitted tune is still running).

| | All ten | Five-match pool | Metrica (3) | IDSSE (7) |
|---|---|---|---|---|
| `base`, tree dead models (PR #7's method, eight training matches) | 0.7019 | 0.6952 | 0.6365 | 0.7300 |
| **`base` + blind network on frames without a ball (`split`)** | **0.7152** | **0.7116** | **0.6783** | **0.7310** |
| `base` + blind network, averaged with the tree on frames with a ball (`avg`) | 0.7135 | 0.7104 | 0.6774 | 0.7289 |
| `base`, change bonus `w_change` 10, tree dead | 0.7038 | 0.6975 | 0.6385 | 0.7317 |
| Upper bound: reference dead / live | 0.7494 | 0.7451 | 0.7326 | 0.7567 |
| Upper bound: reference home / away | 0.9412 | 0.9380 | 0.8557 | 0.9778 |
| Diagnostic: the team network's own dead output (reads ball presence) | 0.7359 | 0.7314 | 0.7017 | 0.7506 |

For comparison, PR #7 reports 0.6846 on the five-match pool with networks trained on four matches
and settings re-tuned in the fold.

Per match, `base` tree -> `split`: Metrica 1 0.637 -> 0.682, Metrica 2 0.631 -> 0.683, Metrica 3
0.642 -> 0.671; IDSSE 11 0.727 -> 0.728, 12 0.720 -> 0.720, 13 0.747 -> 0.751, 14 0.723 -> 0.721,
15 0.730 -> 0.735, 16 0.722 -> 0.720, 17 0.741 -> 0.742.

Why the blind network helps on Metrica and not on IDSSE:

- In the Metrica matches the ball is tracked on 0 to 16% of dead frames, in IDSSE on 100%. So on
  Metrica a stoppage has to be read from the players alone. On Metrica 1 the end of a possession
  into a stoppage was found a median of 54 frames late with the motion-only tree model (16%
  within half a second); with the blind network it is 7 frames late (40% within half a second).
  On IDSSE 11 it was already within half a second 81% of the time.
- On frames without a ball the dead-or-live call is right on 81% of frames with the tree model
  and 90% with the blind network (Metrica 1); 93% and 96% on IDSSE 11.
- Our match is Metrica tracking with the ball missing on 41% of graded frames, so the Metrica
  rows are the ones that resemble it.

Other findings:

- The blind network alone, on every frame, is worse than the trees (0.653 / 0.673 on Metrica 1 /
  IDSSE 11): when the ball is tracked, the ball is the better clue. Hence `split`.
- Possessions under 2 s are found 8% (Metrica 1) and 33% (IDSSE 11) of the time; the network's
  sign is right on 32% / 55% of their frames. `posb` is aimed at that; not measured yet.
- `final.py check` on Metrica 1 (networks that did not see it, `split`): the package path
  (NumPy, from raw tracking) and the experiment path give the same labels on 100% of frames
  (spellF1 0.6924 both, medium level). Team log-odds differ by at most 0.002, the blind network
  by at most 0.035. The tree dead models differ by more than 0.01 on 1.7% of ball-seen frames,
  because the experiment path reads the clues as cached 32-bit numbers and the build as 64-bit.

## Building the candidate

```bash
cd experiments/neural_v2
python final.py export base --blind-mode split     # one team network and one blind network on all
                                                   # ten matches (GPU), tree dead models (CPU),
                                                   # written to espadas/ and read back
python final.py check base --blind-mode split      # package path against experiment path
```

`export` rewrites `espadas/neural_weights.py`, `espadas/evidence_models.py` and writes
`espadas/blind_weights.py`. Training is not bit-reproducible, so the files written are the
submission's models: commit them, do not regenerate them afterwards.
