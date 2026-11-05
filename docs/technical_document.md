# Team Espadas: technical document (MTAH 2027, Phase 1)

> Draft. Exported to `espadas.pdf` for submission (SUBMISSION.md, deliverable 3).

## 1. Approach

### Sequence model (`v4`, Phase 7)

Each half is decoded as one sequence by a 5-state hidden Markov model: home in control, home
loose (ball loose, home had it last), away in control, away loose, dead. A loose ball keeps the
label of the last team in control, as the official rules say, so a deflection or block is not a
change of possession. Per-frame evidence is the Phase 5 dead and home log-odds plus the Phase 6
touches: a confirmed touch (pass chain or hold) favours that team in control, a lone touch only
rules out the other team being in control. When the ball is not tracked, no ball-based evidence
is used at all (no information), and on those frames the dead score keeps a fixed, untuned
threshold (decision (b), section 2). Switching between states has a cost per kind of change
(control to loose and back, turnover via a loose ball, stoppage, restart); a direct change from
one team's control to the other's is ruled out, so every change of possession passes through a
loose ball, and a restart is always taken in control. Viterbi finds the single best
labelling of the half, so a change of possession is placed where its evidence starts. All
weights and costs are tuned on the external tuning games only.

## 2. Assets used (and tried, then dropped)

**Used:** the automatic tracking data (players and ball) and the match-level frame layout
(halves, kick-offs, the forced-`D` window).

**Not used, by team decision (5 Nov):**
- **Smart Tagging.** The automatic phase and set-piece tags look useful (a set piece means the
  ball is dead, then the tagged team restarts), but they are approximate, a defensive phase
  names the defending team, throw-ins have no team, and none of our external games have Smart
  Tagging. Any rule or weight for it could not be tested or tuned on external data, and checking
  it against our own match would mean letting the provided game choose a setting. We use only
  evidence we can measure on external games.
- **Number of tracked objects.** Always 22 in the external games, so it cannot be tested or
  tuned either. Same reasoning.

**Tried, then dropped:**
- **A learned "ball at a player's feet" (control vs loose) score** (reviewed 6 Nov). A logistic
  model fitted on the external games, with control/loose reference frames derived by code from
  the event data (a pass is loose from the kick to the reception, a shot until it ends, a header
  or clearance until the other team acts). Its clues were the distance to the nearest player, the
  ball speed and the bend in the ball's path. The bend clue does not carry over to our match: the
  automatic tracker smooths the ball, so its path rarely bends (the same reason touches are also
  detected from changes of the ball's carrier, section 1). The distance clue is already in `v4`
  as a tuned rule (nearest player within reach favours control, nobody within reach favours
  loose), and on the external bench the learned score tied the rule. The control/loose split is
  internal to the model and never graded, so we kept the simpler rule.

## 3. Training and tuning data

Every external dataset: source, licence or written permission, who produced the labels and how,
how much. None of it comes from the provided game; nothing is tuned on the public five minutes or
the provided video (organisers' clarification, Nov 2026).

## 4. Possession rules implemented

Loose balls, challenges, restarts, dead balls, abstention (`X`).

## 5. Limitations

## 6. How to reproduce

See README: setup, `MTAH_DATA_DIR`, `python -m espadas.run`, expected SHA-256.
