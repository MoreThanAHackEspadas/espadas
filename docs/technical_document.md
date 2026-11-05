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

One external dataset, and nothing from the provided game.

**Metrica Sports sample data, Games 1–3.** Source `github.com/metrica-sports/sample-data`,
pinned to commit `e706dd5`, downloaded by `tools/fetch_metrica.py` (size-checked, kept outside
the repository). Each game is a full match: tracking of all 22 players and the ball at 25 Hz
(about 145,000 / 141,000 / 144,000 frames, roughly 96 minutes each; Games 1–2 in Metrica's CSV
format, Game 3 in the EPTS format) plus the event data as published by Metrica Sports. The events
were produced by the data provider, not by us; we use them only to derive per-frame reference
labels **by code** (section 4), so no human-made label of ours exists anywhere in the pipeline.
**Rights:** written permission to use the sample games for this entry, received from
[who, date, channel — to be filled in; quote the key sentence here], with the source credited
here and in the repository and no redistribution of the data.

**Nothing from the provided game.** Nothing is fitted or tuned on the public five minutes, the
private ground truth or the video (organisers' clarification, Nov 2026). The only things ever run
on the provided game are pass/fail checks (format validator, determinism) and aggregate counts
(possessions per 90 minutes, share of dead frames), which were compared with the practice games
but never used to choose a setting.

**Split.** Games 1 and 3 are the tuning games; Game 2 is held out and scored once, at the final
choice between variants. Everything with a number in it was fitted or tuned on Games 1 and 3
only: the two per-frame logistic models (dead-ball and team log-odds, `tools/fit_evidence.py`,
written to `evidence_models.py` as constants), every labeller's settings (`tools/tune_v1.py`,
`tools/tune.py`) and the abstention policy (`tools/tune_abstain.py`). Building the submission
therefore needs no external data at all.

**Making the external games look like our tracking.** Metrica's tracking is far cleaner than the
automatic tracking of the provided game, so the bench degrades it before any score is taken
(`espadas/external/degrade.py`): (1) the ball is hidden in bursts during **live play** until 25%
of live frames have no ball (Metrica loses the ball almost only at stoppages, the provided game
also in live play; 25% is an assumption, so every result is also checked at 15% and 35%); (2) the
ball's path is smoothed and given position noise at four generic "tracker levels" (0.08–0.48 s of
smoothing, 2–10 cm of noise), because Metrica's ball is drawn as straight lines between key
points while an automatic tracker smooths it; (3) players get 0.3–1.0 m of slowly drifting
position error. Settings are tuned on the mean over the four levels and must hold at each one.
The levels were set by reasoning about trackers, not fitted to the provided game.

## 4. Possession rules implemented

Two places draw the line, and both follow the official definition: while the ball is in play
exactly one team is in possession; a loose or contested ball stays with the last team in
possession; possession changes only when the other team gains controlled possession; every
restart has a clear possessor; a dead ball is `D` and never a possession.

**Reference labels on the external games** (`espadas/external/events.py`; fixed definitions,
not tuned), mapped onto Metrica's event types:
- *Control events:* set piece, pass, carry, shot and ball-lost, unless the subtype says header or
  clearance (a single touch). A ball-lost implies the team had the ball first.
- *Touch-only events:* recovery, challenge, headers and clearances. They never change possession
  on their own; if the same team follows with a control event before the other team does
  anything in play, the change is dated back to the first of those touches (when the ball was won).
- *Dead ball* starts when the ball leaves the pitch, at a shot that goes out or in, at the whistle
  for a foul, or at an offside or handball call, and ends at the next control event (the restart).
- Each half is graded from its kick-off to its last event; everything else is `X`, as in the
  provided game.

**In the labeller (`v4`):**
- *Loose balls.* Two "loose" states per team. A loose ball keeps the label of the team that had it
  last, so a deflection, a block, a clearance picked up by the same team or an aerial duel without
  control is not a change.
- *Changes of possession* are built into the model's structure: control of one team can never pass
  directly to control of the other. The ball goes loose first (still that team's label), and only
  then can the other team win it, at a tuned turnover cost.
- *Challenges and single touches.* A touch is detected from the ball's path (a bend with a player
  within reach) or from a change of the ball's carrier (a different player with the ball at their
  feet). A touch confirmed by a same-team next touch (pass chain) or by holding the ball counts as
  control. A lone touch by the other team mostly counts as a change too: tuning chose weights
  that let one clear touch switch possession, because on the external games a lone detected touch
  is usually a real interception whose next touch was not seen. Forbidding that costs 0.007
  spellF1 on the bench. It is the one place where we lean away from the letter of "a touch
  without control is not a change".
- *Restarts.* A dead ball can only be followed by a team in control, never by a loose ball, and a
  half never starts loose. Which team restarts is decided by the evidence after the whistle
  (nearest players, team shape, the first touches).
- *Dead balls.* A stoppage state with its own entry and exit costs; decoded stoppages shorter than
  3 s are given to the team around them. Where the ball is not tracked only the players' motion
  can say "dead", with a fixed, untuned threshold (decision (b), section 2).
- *Match-level rules.* The organisers' second-half pre-kick-off window (frames 110,808–113,741)
  is always `D`; frames outside the two halves are `X`.
- *Abstention.* `X` on graded frames is implemented (`espadas/abstain.py`: long ball-less gaps
  strictly inside a decoded spell, never near a change) but is switched **off** in the submitted
  file; section 5 says why.

[If `v1` is the submitted variant, replace the `v4` block with: a frame is dead when the ball is
clearly off the pitch, or the players are walking and the ball is still or untracked, with dead
blips shorter than 4 s and live blips shorter than 1 s removed; the possessor is the team of the
nearest player within 2 m; possession passes only once the other team has been that evidence on
70% of the last second, dated back to the old team's last evidence frame; a loose or unseen ball
stays with the last team; after a stoppage, whoever dominates the first second.]

## 5. Limitations

- **The missing-ball assumption.** The provided game loses the ball in live play, the external
  games do not, so how much live ball is missing is an assumption (25%, checked at 15% and 35%),
  and on ball-less frames only the players' motion can call a stoppage, with a threshold set by
  reasoning rather than tuning. Long live stretches without a ball are where the labels are least
  reliable, and a long stoppage where players keep moving can be called live.
- **Tracker transfer.** Settings are tuned on Metrica tracking degraded to look like an automatic
  tracker, and the generic levels do not reproduce ours exactly. Our ball bends even less than at
  any bench level, so touches found from the ball's path are rare on the provided game and most
  touches come from changes of the ball's carrier, whose team is right only about 70% of the time
  at the cleaner bench levels and about 50% at the harsher ones. `v4` scores 0.50–0.59 spellF1
  across the levels, the realistic ones at the low end.
- **Possession count.** On the provided game `v4` gives about 246 possessions per 90 minutes
  (diagnostic only; the practice games have 254–377), so some short possessions are probably
  merged into their neighbours. `v1` gives 301 but collapses on the realistic tracker levels
  (0.43–0.45 spellF1).
- **Little tuning data.** Two tuning games and one held-out game. `v3`'s held-out gain was a
  seventh of its tuning gain, so some settings are overfitted. Who has the ball is the bottleneck,
  not stoppages: on the practice games perfect team calls would lift spellF1 to about 0.86,
  perfect stoppage calls only to about 0.65.
- **Abstention is off.** Abstaining on long ball-less gaps inside a spell raised macroF1 by 0.028
  on the bench, but 89% of those frames were Metrica's ball-less stoppages, the pattern we decided
  not to rely on, so the gain may not transfer. spellF1 is unaffected either way.
- **Kick-off frame.** The organisers' `games.py` anchors the second-half kick-off one frame before
  the frame used in their Smart Tagging starter; we follow `games.py`. At most one frame differs.
- **Not used:** the video (no detector of our own), Smart Tagging and the tracked-object count
  (section 2).
- **With more time:** an own ball detector on the tactical video, trained on external data, to
  shrink the missing-ball problem at its source; more external games from an optical tracker
  (for instance the DFL/Sportec open dataset, CC BY 4.0) to tune on tracking closer to ours and to
  cut overfitting; a better team attribution for carrier touches.

## 6. How to reproduce

Python 3.12; dependencies are pinned in `requirements.txt`. Building the file needs the
organisers' data bundle and nothing else: every fitted model and tuned setting is a constant in
the repository.

```bash
git clone https://github.com/MoreThanAHackEspadas/espadas && cd espadas
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
export MTAH_DATA_DIR="<the unpacked bundle: the folder that contains tactical/>"
python -m espadas.run                      # -> out/espadas.csv, validated, SHA-256 printed
```

The output is deterministic: two clean builds give byte-identical files
(`python tools/check_determinism.py`); the expected SHA-256 of the submitted file is in the README
[filled in at the code freeze]. The first run parses the tracking (a couple of minutes) and caches
arrays under `cache/`; the cache never changes the output.
`python third_party/mtah2027/validate_submission.py out/espadas.csv` runs the organisers' own
checks on the file.

Re-running the tuning is not needed to build the file. `python tools/fetch_metrica.py` downloads
the external games outside the repository, `python -m espadas.external.bench --labeler v1 v3 v4
--games 1 3` scores them, and `tools/fit_evidence.py`, `tools/tune_v1.py`, `tools/tune.py` and
`tools/tune_abstain.py` regenerate the constants; see the README's "External data" section.
