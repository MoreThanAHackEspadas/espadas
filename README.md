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
python -m espadas.run                   # -> out/espadas.csv (labeler v1), validated, prints its SHA-256
```

The first run parses the tracking (a couple of minutes) and caches arrays in `cache/`. The cache
never changes the output.

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
| `espadas/v1.py` | `v1`: dead-ball detection + hysteresis on the nearest player; settings tuned by `tools/tune_v1.py` on external games |
| `espadas/evidence.py` | Phase 5 clues (ball, player motion, nearest player, ball direction, convergence, team shape) combined into a dead log-odds and a home log-odds per frame; models fitted by `tools/fit_evidence.py` on external games, stored in `evidence_models.py` |
| `espadas/v2.py` | `v2`: the Phase 5 evidence + v1's dead-ball clean-up + CUSUM switching; settings tuned by `tools/tune.py --labeler v2` |
| `espadas/touches.py` | Phase 6: touch detection (ball velocity kink with a player close), team, control confirmation (pass chain or hold), touch pulses for the decoder |
| `espadas/v3.py` | `v3`: `v2` + touch pulses; settings tuned by `tools/tune.py --labeler v3` |
| `espadas/v4.py` | `v4`: Phase 7 5-state hidden Markov model (home/away in control or loose, dead) on the Phase 5 scores and Phase 6 touches, Viterbi per half; a missing ball is no information; settings tuned by `tools/tune.py --labeler v4 --skip-heldout` |
| `espadas/write.py` | final match rules, CSV writer, official validation (step 5) |
| `espadas/run.py` | the one command |
| `espadas/external/` | external games: readers, events → reference labels, ball hiding, bench |
| `tools/` | data guard, determinism check, CSV diff, Metrica download |
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
| Metrica Sports sample data, Games 1–3 (tracking + events) | https://github.com/metrica-sports/sample-data, commit `e706dd5` | No formal licence; README: "be responsible with the use of this data", "acknowledge the source". Written confirmation requested from the organisers. | Tuning (Games 1, 3) and held-out comparison (Game 2) |

Reference labels on external games are derived **by code** from each game's event data
(`espadas/external/events.py`, rules documented there). No hand labels are used.

```bash
export MTAH_EXTERNAL_DIR="<workdir>/external_data"
python tools/fetch_metrica.py                      # download, pinned to one commit
python -m espadas.external.bench --labeler v1 v3 v4 --games 1 3  # score labelers on the tuning games
python tools/tune_v1.py                            # re-tune v1 on the tuning games (prints PARAMS)
python tools/fit_evidence.py                       # refit the Phase 5 evidence models (writes evidence_models.py)
python tools/tune.py --labeler v2                  # re-tune v2's decoding (prints PARAMS)
python tools/tune.py --labeler v3                  # re-tune v3 (prints PARAMS)
python tools/tune.py --labeler v4 --skip-heldout   # re-tune the 5-state model, Game 2 unseen
```

The Metrica games lose the ball almost only during stoppages, while ours also loses it in live
play. So the bench hides ball bursts in live play (default 25% of live frames; see
`espadas/external/degrade.py`). `--hide-ball -1` turns it off.

The Metrica ball is drawn as straight lines between key points (razor-sharp corners at every
kick), while an automatic tracker like ours smooths the ball's path. So the bench also blurs the
Metrica ball (`--blur light|medium|heavy|none`, default `medium`), and `tools/tune.py` and
`tools/fit_evidence.py` tune and fit over all three blur levels at once.
`python tools/self_consistency.py --tracking` compares input quality (aggregates only) between
the external games and ours.
