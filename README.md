# espadas

Team Espadas' submission for **More Than A Hack 2027, Phase 1**: for every frame of the tactical
video of FC Barcelona v Athletic Club, label who has the ball (`H` / `A` / `D` / `X`).

> **No match data is in this repository, and none may be added.** The data is licensed for the
> competition only. `.gitignore`, a pre-commit hook and CI all block data files.

## Setup

Python 3.12.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # pinned; add -dev for tests
git config core.hooksPath .githooks      # once per clone: blocks committing match data
```

### Local layout

Keep three folders side by side. Only `espadas/` is this repository; the other two never go in it.

```
<workdir>/
├── espadas/      this repo
├── official/     clone of https://github.com/metrica-sports/more-than-a-hack-2027 (scorer, baselines, docs)
└── mtah_data/    the organisers' data bundle, unpacked
    ├── tactical/     BAR-ATH_tactical_atd.xml, _atd.txt, _homography.csv, _smart_tagging.xml (+ .mp4)
    └── broadcast/    optional; not aligned to tactical
```

Folder names are up to you; the code only reads two environment variables:

| Variable | Points at | Used by |
|---|---|---|
| `MTAH_DATA_DIR` | the bundle root, the folder that contains `tactical/` | `espadas.run`, all loaders |
| `MTAH_OFFICIAL_DIR` | your clone of the official repo | only the manual comparison in `tools/compare_csv.py` |

```bash
export MTAH_DATA_DIR="<workdir>/mtah_data"
export MTAH_OFFICIAL_DIR="<workdir>/official"
```

If `MTAH_DATA_DIR` is unset, the code falls back to `espadas/data/`, which is gitignored.

The bundle ships videos as `BAR-ATH_tactical-002.mp4` / `BAR-ATH_broadcast-001.mp4`; rename them to
`BAR-ATH_tactical.mp4` / `BAR-ATH_broadcast.mp4` before running any video code.

## Build the submission

```bash
python -m espadas.run                   # -> out/espadas.csv (labeler v4), validated, prints its SHA-256
```

**Expected SHA-256 of the submitted `out/espadas.csv`** (labeler `v4`, `X` off; checked
10 Nov, deterministic):

```
bc0a51d1544a72a0b2f187676667e6f3fc5b204ee9fbc81e0d305808e47d6291
```

A clean build must print exactly this value.

The first run parses the tracking (a couple of minutes) and caches arrays in `cache/`. The cache
never changes the output.

The submitted file is always this no-flag build. Whether it has `X` on graded frames (the
Phase 8.4 abstention policy, `espadas/abstain.py`) is decided by `abstain.PARAMS[<labeler>]`
alone: `None` (the current setting for every labeler) means no graded `X`; to ship the policy,
set `abstain.PARAMS[<labeler>] = abstain.SETTING`. `--abstain` / `--no-abstain` force the policy
on / off for experiments only; `--abstain` gives exactly the build that setting `PARAMS` would,
and that is the "X on" variant `tools/selection.py` scores.

## Checks

```bash
python -m pytest -q                      # unit tests, synthetic data only
python tools/check_no_data.py --all      # nothing tracked looks like match data
python tools/check_determinism.py        # two clean builds -> identical bytes
python third_party/mtah2027/validate_submission.py out/espadas.csv   # official format check
python -m espadas.run --labeler v0 --raw --out out/raw.csv && \
  python tools/compare_csv.py out/raw.csv "$MTAH_OFFICIAL_DIR/evaluation/results/baseline_smooth50.csv"  # v0 == official baseline
```

## Layout

| Path | What |
|---|---|
| `espadas/config.py` | data path, match constants (from the official `games.py`), seeds |
| `espadas/atd.py` | tracking loader: positions and speeds (step 1) |
| `espadas/tagging.py` | Smart Tagging loader and per-frame masks (step 1) |
| `espadas/prep.py` | ball cleaning, gap filling, gap lengths, ball speed, attacking direction (step 1); `prepare()` takes another match's halves/kickoffs too |
| `espadas/homography.py` | optional pitch -> pixel homography, for video work |
| `espadas/labelers.py` | per-frame labelers; `v0` reproduces the organisers' baseline |
| `espadas/v1.py` | `v1`: dead-ball detection + hysteresis on the nearest player; settings tuned by `tools/tune_v1.py` on external games (3 Nov, on the earlier bench; not re-tuned on the final one) |
| `espadas/evidence.py` | Phase 5 clues (ball, player motion, nearest player, ball direction, convergence, team shape) combined into a dead log-odds and a home log-odds per frame; models fitted by `tools/fit_evidence.py` on external games, stored in `evidence_models.py` |
| `espadas/v2.py` | `v2`: the Phase 5 evidence + v1's dead-ball clean-up + CUSUM switching; settings tuned by `tools/tune.py --labeler v2` |
| `espadas/touches.py` | Phase 6: touch detection (ball velocity kink with a player close) and carrier touches (a change of who has the ball at their feet, no kink needed; 6.7b), merging the two, team, control confirmation (pass chain or hold), touch pulses for the decoder; `v3` uses kink touches, `v4` both |
| `espadas/v3.py` | `v3`: `v2` + touch pulses; settings tuned by `tools/tune.py --labeler v3` |
| `espadas/v4.py` | `v4`: Phase 7 5-state hidden Markov model (home/away in control or loose, dead) on the Phase 5 scores and Phase 6 touches, Viterbi per half; a missing ball is no information; settings tuned by `tools/tune.py --labeler v4` |
| `espadas/write.py` | final match rules, CSV writer, official validation (step 5) |
| `espadas/run.py` | the one command |
| `espadas/external/` | external games: readers, events → reference labels, ball hiding, bench |
| `tools/` | data guard, determinism check, CSV diff, Metrica download, tuning, the one-shot selection (`selection.py`) |
| `third_party/mtah2027/` | official `games.py`, `possession_metrics.py`, `validate_submission.py` (MIT, Metrica Sports) |

## Documentation

Team docs and the technical document live in `docs/` as Markdown (`*.md` is gitignored
everywhere else). The tech doc is exported to `espadas.pdf` for submission.

## External data

Per the organisers' clarification, every parameter is tuned on **external data only**, never on
the ground truth or the provided video. External data lives outside this repo under
`MTAH_EXTERNAL_DIR` (default: `external_data/` next to the repo) and is never committed.

| Dataset | Source | Terms | Used for |
|---|---|---|---|
| Metrica Sports sample data, Games 1–3 (tracking + events) | https://github.com/metrica-sports/sample-data, commit `e706dd5` | Written permission: the `README.md` of Metrica's repository. Published openly by Metrica Sports for the analytics community; no formal licence file. That README's terms: "be responsible with the use of this data" and, for public use, "acknowledge the source". **Source: Metrica Sports**, credited here and in the tech doc. Not redistributed (downloaded at a pinned commit, never committed). | Tuning (Games 1, 3) and held-out comparison (Game 2) |

Reference labels on external games are derived **by code** from each game's event data
(`espadas/external/events.py`, rules documented there). No hand labels are used.

```bash
export MTAH_EXTERNAL_DIR="<workdir>/external_data"
python tools/fetch_metrica.py                      # download, pinned to one commit
python -m espadas.external.bench --labeler v1 v3 v4  # score labelers on the tuning games (1 and 3)
python tools/tune_v1.py                            # re-tune v1 on the tuning games (prints PARAMS)
python tools/fit_evidence.py                       # refit the Phase 5 evidence models (writes evidence_models.py)
#   ^ rewrites the tracked espadas/evidence_models.py: never run it between selection and the freeze
python tools/tune.py --labeler v2                  # re-tune v2's decoding (prints PARAMS)
python tools/tune.py --labeler v3                  # re-tune v3 (prints PARAMS)
python tools/tune.py --labeler v4                  # re-tune the 5-state model (prints PARAMS)
python tools/selection.py --dry-run                # Phase 9: the one-shot held-out comparison; --smoke rehearses on the tuning games
```

**Game 2 is held out, in code.** The game readers (`metrica_csv`, `metrica_epts`) and
`bench.available()` refuse the held-out game unless the caller opts in (`heldout=True`), before
any file is opened. Only a real run of `tools/selection.py` does that by default (once, at
selection). `bench`, `tune.py`, `tune_v1.py`, `fit_evidence.py` and
`self_consistency.py` work on Games 1 and 3 and touch Game 2 only with an explicit `--heldout`
(`--skip-heldout` is still accepted and changes nothing).

The Metrica games lose the ball almost only during stoppages, while ours also loses it in live
play. So the bench hides ball bursts in live play (default 25% of live frames; see
`espadas/external/degrade.py`). `--hide-ball -1` turns it off.

The Metrica ball is drawn as straight lines between key points (razor-sharp corners at every
kick), while an automatic tracker like ours smooths the ball's path. So the bench also blurs the
Metrica ball and gives the Metrica players tracker-like position error (`--blur
light|medium|heavy|smooth|none`, default `medium`; "smooth" = strong smoothing with little noise,
like a filtered tracker). `tools/tune.py` and `tools/fit_evidence.py` tune and fit over all four
tracker levels at once (`degrade.TRACKER_LEVELS`); `tools/tune_v1.py` uses the default level only.
`python tools/self_consistency.py --tracking` compares input quality (aggregates only) between
the external games and ours.
