# Team Espadas: technical document (MTAH 2027, Phase 1)

> Draft. Exported to `espadas.pdf` for submission (SUBMISSION.md, deliverable 3).

## 1. Approach

The file is produced by a pipeline that reads only the automatic tracking of the provided game.
No setting in it was fitted or tuned on that game or its ground truth (label-free aggregate
measurements of its tracking did inform some design choices; section 3 says which):

1. **Preprocessing.** Ball positions that jump faster than 40 m/s are dropped, gaps of up to 1 s
   are filled, speeds are computed, and each team's attacking direction is read from the kick-off
   formation of each half.
2. **Per-frame evidence.** Two small logistic models turn tracking clues into a "ball is dead"
   score and a "home has it" score for every frame. Dead-ball clues: ball speed and position (when
   it is tracked), and how the players move (mean speed, share running, fastest player), which is
   what tells a live ball gap from a stoppage when the ball is missing. Team clues, all written as
   home-minus-away so the model has no home bias: nearest player to the ball and to where it was
   last seen, ball direction against each team's attacking direction, which team is closing in on
   the ball (usually the defender, pressing), team shape (the team with the ball spreads out: the
   strongest single clue) and which team is moving up the pitch.
3. **Touches.** Moments a player plays the ball, found in two ways: a bend in the ball's path
   with a player within reach, and a change of who has the ball at their feet (the nearest player
   within reach for a run of frames), which needs no bend. A touch followed by a same-team touch
   within a second is "confirmed" (control). Confirmation by a hold (the same player staying
   with the ball) exists in the code but is switched off in `v4`'s tuned settings; `v3` uses it,
   and finds touches from bends only.
4. **Sequence model.** The scores and touches of a whole half are decoded at once (below).
5. **Output.** The organisers' pre-kick-off window is forced to `D`, ungraded frames are `X`, the
   file is validated with the organisers' own checks and its SHA-256 is printed.

**Why this, rather than something else.** We have no labels on the provided game and may not
make any, the tuning data is two external matches from a different, much cleaner tracker, and
the file must be reproducible. That ruled out a large learned model: with two games it would
memorise the tuning tracker rather than learn possession. Instead each stage is small and
checkable on its own (a clue is right or wrong on an external game), the parts of the official
definition that are structural (a loose ball keeps its team, possession changes only through a
loose ball, a restart is taken in control) are built into the model rather than learned, and
only a few dozen weights and costs are tuned, on an external bench made deliberately harder than
the tuning games. We kept a plain rule-based variant (`v1`: dead-ball rules plus a hysteresis
switch on the nearest player) as a candidate throughout; it is simpler but collapses on
realistic tracker noise (with its settings from the earlier, easier bench; section 5), which is
why the sequence model was built. We did not build our own ball detector on the video: the
provided tracking already gives positions, the detector would have to be trained on external
data only, and the time was better spent on transfer between trackers, which turned out to be
the real problem (sections 3 and 5).

_[Selection pending (10–12 Nov): the variant below is the one leading on the tuning games; the
final choice is made once on the held-out game and recorded in `docs/selection.md`.]_

### Sequence model (`v4`, Phase 7)

Each half is decoded as one sequence by a 5-state hidden Markov model: home in control, home
loose (ball loose, home had it last), away in control, away loose, dead. A loose ball keeps the
label of the last team in control, as the official rules say, so a deflection or block is not a
change of possession. Per-frame evidence is the Phase 5 dead and home log-odds plus the Phase 6
touches: a confirmed touch (a same-team next touch within a second) favours that team in
control; a lone touch counts against the other team being in control and gives the toucher's
team a large credit, tuned larger than a confirmed touch's, so one clear touch can switch
possession (section 4). When the ball is not tracked, no ball-based evidence is used at all (no
information), and on those frames the dead score keeps a fixed, untuned threshold (decision
(b), section 3). Switching between states has a cost per kind of change
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
The rules were reviewed against Metrica's event definitions and the data
(`docs/events_rules_review.md`); their known limits are listed in section 5.
**Rights:** our written permission to use these games is the README of Metrica's public GitHub
repository (`metrica-sports/sample-data`, `README.md`, read at the pinned commit `e706dd5`; the
repository has no separate licence file). Metrica Sports publishes these games there openly, "to
provide some sample data to play around with to the broader football analytics community". The
repository's terms ("Legal stuff") ask users to "be responsible with the use of this data" and,
for anything public, to "acknowledge the source". We acknowledge **Metrica Sports** as the source
here and in the repository README. We do not redistribute the data: it is downloaded from
Metrica's own repository (`metrica-sports/sample-data`) at a pinned commit and kept outside ours,
and it is used only to fit and tune our settings (Games 1 and 3) and to compare the final
candidates (Game 2).

**Nothing fitted or tuned on the provided game.** No setting is fitted or tuned on the provided
game: not on the public five minutes, the private ground truth or the video (organisers'
clarification, Nov 2026). What has been run on it is pass/fail checks (format validator,
determinism) and label-free aggregate measurements of its tracking, set beside the same numbers
on the practice games: possessions per 90 minutes, share of dead frames, share of frames without
a ball, how often the ball's path bends, how far the nearest player is from the ball, and how
many touches the detectors find. No number was tuned against these aggregates, but they did
inform design choices, and we list them:
- *The bench's ball hiding and tracker levels* (below). The provided game has no ball on about
  41% of graded frames, and the organisers say it is often missing in live play; from that we
  reasoned the assumed 25% of live frames without a ball (a middle value, checked at 15% and
  35%). The bend-based touch detector found 11 touches per 90 minutes on the provided game
  against 1,300–1,600 on the practice games, which led to blurring the practice games' ball. Its
  ball still bent less often than at any blur level, which led to the strongly smoothed level.
  Player position error was added at every level, and its largest level (1 m) was checked to
  give about the gap in ball-to-nearest-player distance measured on the provided game (median
  about 1.9 m against 0.7–1.4 m on the practice games). The values of the levels are generic,
  set by reasoning about trackers, not fitted.
- *Decision (b): a fixed, untuned threshold for a stoppage when the ball is missing.* With a
  threshold tuned on the practice games, where the ball goes missing almost only at stoppages,
  the decoders called about 91% of the provided game's ball-less frames dead. On ball-less frames
  a stoppage is therefore called only if the players' motion says dead is more likely than not
  (threshold 0, never tuned), a choice made by reasoning that scores about 0.04–0.05 lower on the
  bench.
- *Carrier touches* (section 1). The bend-based detector found about 290 touches per 90 minutes
  on the provided game against about 1,800 on the bench, so touches are also found from changes
  of the ball's carrier, which need no bend (1,200–1,360 per 90 minutes on the bench, 1,295 on
  the provided game).

**Split.** Games 1 and 3 are the tuning games; Game 2 is held out: nothing was fitted or tuned on
it. It has been scored before, though. During development, on the earlier bench, the baseline
`v0` and the labellers `v1`, `v2` and `v3` were scored on Game 2 (`v2` more than once, while it
was being built), and one comparison there kept `v1` as the submission over `v2`; those scores
are recorded in the code comments and the team checklist. The bench then changed (ball blur on
4 Nov, player position error and the smoothed level on 5 Nov), and Game 2 has not been scored on
the current bench. The final comparison between variants on it is scored once, at selection
(`tools/selection.py`, `docs/selection.md`). Everything with a number in it was fitted or tuned
on Games 1 and 3 only: the two per-frame logistic models (dead-ball and team log-odds,
`tools/fit_evidence.py`, written to `evidence_models.py` as constants), every labeller's settings
(`tools/tune_v1.py`, `tools/tune.py`) and the abstention policy (`tools/tune_abstain.py`).
Building the submission therefore needs no external data at all.

**Making the external games look like our tracking.** Metrica's tracking is far cleaner than the
automatic tracking of the provided game, so the bench degrades it before any score is taken
(`espadas/external/degrade.py`): (1) the ball is hidden in bursts during **live play** until 25%
of live frames have no ball (Metrica loses the ball almost only at stoppages, the provided game
also in live play; 25% is an assumption, so every result is also checked at 15% and 35%); (2) the
ball's path is smoothed and given position noise at four generic "tracker levels" (0.08–0.48 s of
smoothing, 2–10 cm of noise), because Metrica's ball is drawn as straight lines between key
points while an automatic tracker smooths it; (3) players get 0.3–1.0 m of slowly drifting
position error. Settings are tuned on the mean over the four levels and must hold at each one.
The levels' values were set by reasoning about trackers, not fitted to the provided game; the
reasons the bench needed them came from the label-free comparisons listed above.

## 4. Possession rules implemented

Two places draw the line, and both follow the official definition: while the ball is in play
exactly one team is in possession; a loose or contested ball stays with the last team in
possession; possession changes only when the other team gains controlled possession; every
restart has a clear possessor; a dead ball is `D` and never a possession.

**Reference labels on the external games** (`espadas/external/events.py`; fixed definitions,
not tuned), mapped onto Metrica's event types:
- *Control events:* set piece, pass, carry, shot and ball-lost, unless the subtype says header or
  clearance (a single touch). A ball-lost implies the team had the ball first.
- *Touch-only events:* recovery, headers and clearances (challenge events are not used). They
  never change possession on their own; if the same team follows with a control event before the
  other team does anything in play, the change is dated back to the first of those touches (when
  the ball was won).
- *Dead ball* starts when the ball leaves the pitch, at a shot that goes out or in, at the whistle
  for a foul, or at an offside or handball call, and ends at the next control event (the restart).
- Each half is graded from its kick-off to its last event; everything else is `X`, as in the
  provided game.

**In the labeller (`v4`):**
- *Loose balls.* One "loose" state per team, beside its "in control" state. A loose ball keeps
  the label of the team that had it last, so a deflection, a block, a clearance picked up by the
  same team or an aerial duel without control is not a change.
- *Changes of possession* are built into the model's structure: control of one team can never pass
  directly to control of the other. The ball goes loose first (still that team's label), and only
  then can the other team win it, at a tuned turnover cost.
- *Challenges and single touches.* A touch is detected from the ball's path (a bend with a player
  within reach) or from a change of the ball's carrier (a different player with the ball at their
  feet). A touch confirmed by a same-team next touch within a second (pass chain) counts as
  control; confirmation by holding the ball is switched off in the tuned settings. A lone touch
  by the other team mostly counts as a change too: tuning chose weights that let one clear touch
  switch possession, because on the external games a lone detected touch is usually a real
  interception whose next touch was not seen. Forbidding that costs 0.007 spellF1 on the bench.
  It is the one place where we lean away from the letter of "a touch without control is not a
  change".
- *Restarts.* A dead ball can only be followed by a team in control, never by a loose ball, and a
  half never starts loose. Which team restarts is decided by the evidence after the whistle
  (nearest players, team shape, the first touches).
- *Dead balls.* A stoppage state with its own entry and exit costs; decoded stoppages shorter than
  3 s are given to the team around them. Where the ball is not tracked only the players' motion
  can say "dead", with a fixed, untuned threshold (decision (b), section 3).
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

[If `v3` is the submitted variant, replace the `v4` block with: a frame is dead when the Phase 5
dead score is above a tuned threshold (with the ball tracked) or, without a ball, when the
players' motion says dead is more likely than not (decision (b)), with stoppages shorter than 2 s
dropped and live spells shorter than 1 s between stoppages made dead. Who has the ball is a
running sum of the evidence against the current team: the Phase 5 home score, capped per frame,
plus a pulse at every touch, found only from a bend in the ball's path with a player within
1.5 m (20 for a touch confirmed by a same-team next touch within a second or by the same player
staying within 3 m of the ball for the next 2 s, 30 for any other touch). Evidence for the
current team drains the sum, never below zero; possession passes once the sum reaches 60, dated
back to when it last started from zero, so two touches by the other team switch possession on
their own and one does not. Weak or missing evidence changes nothing, so a loose or unseen ball
stays with the last team, while during ball gaps the team-shape and pressing clues still count.
After a stoppage, the sign of the evidence over the first second decides, else the team that had
it before the stoppage keeps it.]

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
  (0.43–0.45 spellF1). `v1`'s settings date from 3 Nov: they were tuned on the earlier bench,
  before the ball blur and player error, and were not re-tuned on the final one, while `v3` and
  `v4` were. So these numbers compare the candidates as they stand, not what each method could
  reach with equal tuning.
- **Little tuning data.** Two tuning games and one held-out game. `v3`'s held-out gain was a
  seventh of its tuning gain, so some settings are overfitted. Who has the ball is the bottleneck,
  not stoppages: on the practice games perfect team calls would lift spellF1 to about 0.86,
  perfect stoppage calls only to about 0.65.
- **Reference labels.** The H/A/D reference on the external games comes from the provider's
  events through fixed rules. A review of those rules (`docs/events_rules_review.md`) found two
  things we chose to document rather than change before selection: referee stoppages that
  Metrica tags `BALL LOST-FORCED` stay labelled live until the free kick (5–9 per game, about
  1.5–2% of frames, so the labelers were tuned to call some such stoppages live), and the two
  tuning games are annotated with different conventions (Game 3's carry events make every
  recovery an immediate change of possession: 377 against 268 possessions per 90 minutes), so
  the tuning averages over two readings of "controlled possession".
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
--games 1 3` scores them, and `tools/fit_evidence.py`, `tools/tune.py` and
`tools/tune_abstain.py` regenerate the evidence models, the `v3` and `v4` settings and the
abstention setting, all fitted on the final bench. `v1` is the exception: its settings date from
3 Nov and were tuned with `tools/tune_v1.py` on the earlier bench, before the ball blur and
player error; it was not re-tuned on the final bench, so re-running `tools/tune_v1.py` now need
not give the shipped values. (`v2`, not a candidate, was last tuned on the 4 Nov bench.) See the
README's "External data" section.
