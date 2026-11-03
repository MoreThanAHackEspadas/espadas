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
python -m espadas.run                   # -> out/espadas.csv, validated, prints its SHA-256
```

The first run parses the tracking (a couple of minutes) and caches arrays in `cache/`. The cache
never changes the output.

## Checks

```bash
python -m pytest -q                      # unit tests, synthetic data only
python tools/check_no_data.py --all      # nothing tracked looks like match data
python tools/check_determinism.py        # two clean builds -> identical bytes
python third_party/mtah2027/validate_submission.py out/espadas.csv   # official format check
python -m espadas.run --raw --out out/raw.csv && \
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
| `espadas/write.py` | final match rules, CSV writer, official validation (step 5) |
| `espadas/run.py` | the one command |
| `tools/` | data guard, determinism check, CSV diff |
| `third_party/mtah2027/` | official `games.py`, `possession_metrics.py`, `validate_submission.py` (MIT, Metrica Sports) |

## Documentation

Team docs and the technical document live in `docs/` as Markdown (`*.md` is gitignored
everywhere else). The tech doc is exported to `espadas.pdf` for submission.

## External data

None yet. Every external dataset used for development or tuning will be listed here with its
source and licence. No data about this match from any outside source is used.
