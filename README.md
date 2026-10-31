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

Point `MTAH_DATA_DIR` at the unpacked organisers' bundle (the folder containing `tactical/` and
`broadcast/`):

```bash
export MTAH_DATA_DIR="/path/to/More Than A Hack 2027 - Initial Challenge - Dataset"
```

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
```

## Layout

| Path | What |
|---|---|
| `espadas/config.py` | data path, match constants (from the official `games.py`), seeds |
| `espadas/atd.py` | tracking loader (step 1) |
| `espadas/labelers.py` | per-frame labelers; `v0` reproduces the organisers' baseline |
| `espadas/write.py` | final match rules, CSV writer, official validation (step 5) |
| `espadas/run.py` | the one command |
| `tools/` | data guard, determinism check, CSV diff |
| `third_party/mtah2027/` | official `games.py`, `possession_metrics.py`, `validate_submission.py` (MIT, Metrica Sports) |

## External data

None yet. Every external dataset used for development or tuning will be listed here with its
source and licence. No data about this match from any outside source is used.
