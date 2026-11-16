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

## Findings so far (first fold only: Metrica 1 and IDSSE 11)

| | Metrica 1 | IDSSE 11 |
|---|---|---|
| `base`, tree dead models, PR #7 settings | 0.637 | 0.727 |
| with reference dead / live (upper bound) | 0.725 | 0.744 |
| with reference home / away (upper bound) | 0.871 | 0.981 |
| team network's own dead output (diagnostic) | 0.698 | 0.737 |
| change bonus, `w_change` 20, nothing re-tuned | 0.642 | 0.735 |

- On Metrica 1 the end of a possession into a stoppage is found a median of 54 frames late (16%
  within half a second); on IDSSE 11 it is within half a second 81% of the time. In the Metrica
  matches the ball is tracked on 0 to 16% of dead frames, in IDSSE on 100%, so on Metrica the
  stoppage has to be read from the players alone and the motion-only tree model is slow at it.
  Our match is Metrica tracking with the ball missing on 41% of graded frames, so this is the
  case that matters. It is what the blind network is for.
- Possessions under 2 s are found 8% (Metrica 1) and 33% (IDSSE 11) of the time; the network's
  sign is right on 32% / 55% of their frames. That is what `posb` is for.
