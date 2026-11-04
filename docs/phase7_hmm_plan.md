# Phase 7 (`v4` five-state HMM) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A new labeler `v4` that decodes each half as the most probable sequence of five states (dead, home control, home loose, away control, away loose) over the Phase 5/6 scores plus a new "ball at feet" score, tuned on Metrica Games 1 and 3 only.

**Architecture:** A football-agnostic Viterbi in `espadas/hmm.py`; a new control-vs-loose logistic model fitted by `tools/fit_evidence.py` from event-derived labels in `espadas/external/events.py`; `espadas/v4.py` turns scores into emissions and expected durations into a transition matrix, decodes per half, and plugs into the existing `tools/tune.py` protocol (`features`, `decode`, `PARAMS`, `GRID`).

**Tech Stack:** Python 3.12, numpy 2.5.4, pandas 3.1.0, pytest 9.2.0 (all pinned; nothing new). Run everything with `.venv/bin/python` from the repo root `/home/storage/visca barca/espadas`.

**Spec:** `docs/phase7_hmm_design.md` (same folder; sections are cited as "spec §N").

## Global Constraints

- Tune, fit and bench on Metrica Games 1 and 3 only: `--games 1 3` on the bench, `--skip-heldout` on the tuner, no `--heldout` on the fit script. Game 2 is never scored in this plan.
- No new dependencies (`requirements.txt` unchanged). Deterministic output: same input, same bytes.
- On frames without a ball the dead offset is 0 and the control evidence is 0; neither is in `GRID` (spec §4.3, decision (b)).
- Direct `HC -> AC`, `HC -> AL`, `HL -> AL`, `D -> HL`, `D -> AL` (and mirrors) are `-inf` (spec §4.2). A half starts in `D`, `HC` or `AC`, never loose.
- Tie order of states: `D, HC, HL, AC, AL` (spec §4.1).
- Touch *detection* settings are `v3.PARAMS` values and not tuned; touch *pulse* weights are `v4`'s own, in log-odds units (spec §4.3, §4.5).
- Markdown is only tracked under `docs/*.md`; data files never enter the repo (`tools/check_no_data.py --all` must pass).
- `run.py` keeps `v1` as default. Commits are local only; never push (the team pushes to `main` themselves).
- Existing 58 tests keep passing unchanged.

## Review Focus

1. A half where the ball is never tracked: every control emission must be 0 and decoding must still return valid labels with no NaN (pinned in Task 5, `test_decode_with_ball_never_tracked`).
2. A one-frame half, or a half shorter than any chain minimum: the decoder must return the best single state, not crash (Task 1, `test_viterbi_single_frame`; Task 8, `test_expand_shorter_than_minimum`).
3. A touch on the first or last frame of the array: pulse indexing must stay in bounds (Task 5, `test_touch_on_array_edges`).
4. Grid values that make a transition row exceed probability 1 (e.g. `loose_len = 1`): `transitions()` must refuse with `ValueError` instead of producing `log` of a negative number (Task 5, `test_transitions_refuse_impossible_lengths`).
5. Refitting the evidence models must not change `DEAD_MODEL` or `TEAM_MODEL` by a single byte (Task 4, verified by `git diff`).

---

### Task 1: Generic Viterbi (`espadas/hmm.py`)

**Files:**
- Create: `espadas/hmm.py`
- Test: `tests/test_hmm.py`

**Interfaces:**
- Produces: `viterbi(emissions, log_trans, log_start) -> np.ndarray` where `emissions` is `(T, S)` float64, `log_trans` is `(S, S)` float64 with `-inf` for forbidden moves (`log_trans[i, j]` = from `i` to `j`), `log_start` is `(S,)`; returns an `int64` array of length `T` of state indices. Ties go to the lowest index. `T >= 1`.

- [ ] **Step 1: Write the failing tests**

```python
"""Generic Viterbi, on hand-checked sequences."""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from espadas import hmm  # noqa: E402

NEG = -np.inf


def test_viterbi_follows_emissions_when_transitions_are_free():
    em = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    free = np.log(np.full((2, 2), 0.5))
    assert hmm.viterbi(em, free, np.log([0.5, 0.5])).tolist() == [0, 1, 1, 0]


def test_viterbi_routes_through_the_bridge_state_when_the_direct_move_is_forbidden():
    em = np.array([[5.0, -10.0, -10.0], [5.0, -10.0, -10.0], [0.0, 0.0, 5.0],
                   [0.0, -10.0, 5.0], [0.0, -10.0, 5.0]])
    trans = np.log(np.array([[0.9, 0.1, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]]))
    start = np.array([0.0, NEG, NEG])
    assert hmm.viterbi(em, trans, start).tolist() == [0, 0, 1, 2, 2]


def test_viterbi_ties_go_to_the_lowest_state_index_and_start_is_respected():
    em = np.zeros((4, 2))
    free = np.log(np.full((2, 2), 0.5))
    assert hmm.viterbi(em, free, np.log([0.5, 0.5])).tolist() == [0, 0, 0, 0]
    assert hmm.viterbi(em, free, np.array([NEG, 0.0])).tolist() == [1, 1, 1, 1]


def test_viterbi_single_frame():
    assert hmm.viterbi(np.array([[0.0, 2.0, 1.0]]), np.zeros((3, 3)),
                       np.array([0.0, 0.0, 0.0])).tolist() == [1]


def test_viterbi_matches_brute_force_and_is_deterministic():
    rng = np.random.default_rng(0)
    em = rng.normal(size=(6, 3))
    trans = np.log(rng.dirichlet(np.ones(3), size=3))
    start = np.log(rng.dirichlet(np.ones(3)))
    best, best_score = None, NEG
    for path in itertools.product(range(3), repeat=6):
        score = start[path[0]] + em[0, path[0]]
        for t in range(1, 6):
            score += trans[path[t - 1], path[t]] + em[t, path[t]]
        if score > best_score:
            best, best_score = list(path), score
    first = hmm.viterbi(em, trans, start)
    assert first.tolist() == best
    assert np.array_equal(first, hmm.viterbi(em, trans, start))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_hmm.py -q`
Expected: FAIL with `ImportError: cannot import name 'hmm'` (or `ModuleNotFoundError`).

- [ ] **Step 3: Implement `viterbi(emissions, log_trans, log_start)` in `espadas/hmm.py`**

Module docstring: what it is (max-probability state path in log space), that it knows nothing about football, that ties go to the lowest index because `np.argmax` returns the first maximum. Loop over frames: `score = delta[:, None] + log_trans` (shape `(S, S)`), `back[t] = np.argmax(score, axis=0)`, `delta = score[back[t], np.arange(S)] + emissions[t]`; `delta` starts as `log_start + emissions[0]`. Backpointers in an `int8` array when `S <= 127`, else `int64`. Backtrack from `np.argmax(delta)`. Use `np.errstate(invalid="ignore")` so `-inf + -inf` does not warn.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_hmm.py -q`
Expected: `5 passed`.

- [ ] **Step 5: Measure the cost on a Metrica-sized half**

Run:
```bash
.venv/bin/python - <<'EOF'
import time, numpy as np
from espadas import hmm
rng = np.random.default_rng(0)
em = rng.normal(size=(70000, 5)); tr = np.log(rng.dirichlet(np.ones(5), size=5)); st = np.log(np.full(5, .2))
t = time.perf_counter(); hmm.viterbi(em, tr, st); print("{:.2f} s per 70k frames".format(time.perf_counter() - t))
EOF
```
Expected: under 1.5 s (two halves per game, so under 3 s per game). If it is over 3 s, rewrite the inner loop over Python lists for the `S <= 8` case before continuing; the tests pin the behaviour.

- [ ] **Step 6: Commit (local only)**

```bash
git add espadas/hmm.py tests/test_hmm.py
git commit -m "Phase 7: generic Viterbi decoder"
```

---

### Task 2: Control-vs-loose reference labels (`espadas/external/events.py`)

**Files:**
- Modify: `espadas/external/events.py` (append after `dead_windows()`, before `labels_from_events()`; extend the module docstring)
- Test: `tests/test_external_events.py`

**Interfaces:**
- Consumes: existing `is_control(event)`, `IN_PLAY`, `TOUCH_SUBTYPES`, `periods(events)`.
- Produces: `loose_windows(events) -> list[tuple[int, int]]`, half-open 0-based frame spans where the ball is in flight or contested, for the (already sorted) events of one period; `control_mask(events, n_frames) -> np.ndarray[bool]`, True on every frame inside a period's span that is not in a loose window, False outside the periods. Only meaningful on live frames; callers mask with the H/A/D labels.

Rules (spec §5 with two consistency choices, stated in the docstring):
- `PASS` / `SET PIECE` that `is_control()` with `end > start`: loose `[start + 1, end)` (the kick frame and the reception frame are at a player's feet).
- `SHOT` that `is_control()`: loose `[start + 1, end + 1)` (`end` is `start` when missing). Same shape as a pass: the kick frame is controlled; the end frame is not a reception.
- `BALL LOST`: loose `[start + 1, next in-play event's start)`; to the period's end if there is none.
- Touch-only in-play events (`is_control()` false: `RECOVERY`, `CHALLENGE`, headers, clearances): scan forward over in-play events with a team. First one by the **other** team: loose `[start, that event's start)`. First one by the **same** team that `is_control()`: no window (the "won ball" rule). Same team and touch-only: keep scanning. None: loose to the period's end.
- `CARRY` and everything else: no window.
- "Period's end" is `np.iinfo(np.int64).max`, clipped by `control_mask` like `dead_windows()`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_external_events.py`)

```python
def ctrl(events, n):
    return "".join("C" if c else "." for c in ev.control_mask(events, n))


def test_pass_flight_is_loose_carry_and_kick_and_reception_are_controlled():
    events = [e(0, H, "SET PIECE", 0, sub="KICK OFF"), e(1, H, "PASS", 0, 4),
              e(2, H, "CARRY", 4, 8), e(3, H, "PASS", 8, 12)]
    assert ctrl(events, 14) == "C...CCCCC...C."


def test_won_ball_is_controlled_from_the_touch_and_lost_ball_is_loose():
    events = [e(0, A, "SET PIECE", 0, sub="KICK OFF"), e(1, A, "PASS", 0, 6),
              e(2, H, "RECOVERY", 6, 6, "INTERCEPTION"), e(3, H, "PASS", 8, 10),
              e(4, H, "BALL LOST", 12, 14, "INTERCEPTION"),
              e(5, A, "RECOVERY", 16, 16), e(6, A, "PASS", 18, 20)]
    assert ctrl(events, 22) == "C.....CCC.CCC...CCC.C."
    assert run(events, 22) == "A" * 6 + "H" * 10 + "A" * 5 + "X"


def test_touch_only_followed_by_the_other_team_is_loose():
    events = [e(0, A, "SET PIECE", 0, sub="KICK OFF"), e(1, A, "PASS", 0, 4),
              e(2, H, "CHALLENGE", 6, 6, "GROUND-LOST"), e(3, A, "PASS", 8, 10)]
    assert ctrl(events, 12) == "C...CC..C.C."


def test_shot_is_loose_after_the_kick_and_recovery_before_a_pass_is_controlled():
    events = [e(0, H, "SET PIECE", 0, sub="KICK OFF"), e(1, H, "PASS", 0, 2),
              e(2, H, "SHOT", 4, 6, "ON TARGET-SAVED"), e(3, A, "RECOVERY", 7, 7),
              e(4, A, "PASS", 8, 9)]
    assert ctrl(events, 11) == "C.CCC..CCC."


def test_control_mask_is_true_wherever_the_reference_starts_a_new_spell():
    events = [e(0, A, "SET PIECE", 0, sub="KICK OFF"), e(1, A, "PASS", 0, 6),
              e(2, H, "RECOVERY", 6, 6, "INTERCEPTION"), e(3, H, "PASS", 8, 10),
              e(4, H, "BALL LOST", 12, 14, "INTERCEPTION"),
              e(5, A, "RECOVERY", 16, 16), e(6, A, "PASS", 18, 20)]
    labels = ev.labels_from_events(events, 22)[0]
    mask = ev.control_mask(events, 22)
    starts = [t for t in range(1, 22) if labels[t] in "HA" and labels[t] != labels[t - 1]]
    assert starts == [6, 16] and all(mask[t] for t in starts)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_external_events.py -q`
Expected: 5 new FAIL with `AttributeError: module ... has no attribute 'control_mask'`; the 6 old tests pass.

- [ ] **Step 3: Implement `loose_windows(events)` and `control_mask(events, n_frames)` in `espadas/external/events.py`**

`control_mask` sorts like `labels_from_events` (`(start, order)`), loops over `periods()`, filters `in_period`, sets the span True, then False over each clipped window. Add the rules above to the module docstring under a new heading "Control vs loose (Phase 7)".

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_external_events.py -q`
Expected: `11 passed`.

- [ ] **Step 5: Commit (local only)**

```bash
git add espadas/external/events.py tests/test_external_events.py
git commit -m "Phase 7: control-vs-loose reference labels from events"
```

---

### Task 3: Control clues and design matrix (`espadas/evidence.py`)

**Files:**
- Modify: `espadas/evidence.py` (`clues()`, `_with_context()`, new `control_design()`, `control_log_odds()`, `_fitted_control()`; module docstring gains a "Control (Phase 7)" paragraph)
- Test: `tests/test_evidence.py`

**Interfaces:**
- Consumes: `touches.kink(ball, fps, half)`.
- Produces:
  - new module constants `CONTROL_KINK_HALF = 4` (the `v3.PARAMS["kink_half"]` value, written as a number to avoid a circular import), `KINK_CLIP = 20.0` (m/s), `CONTROL_CONTEXT = (6, 25, 75)`.
  - `clues()` gains `"near_any"` (nearest tracked player of either team to the ball, m, `NaN` where the ball is not present, from `np.minimum(d_home, d_away)` with `inf` turned into `NaN`) and `"kink"` (`touches.kink(base["ball"], fps, CONTROL_KINK_HALF)`, `NaN` where the ball is not present).
  - `_with_context(columns, widths=CONTEXT)`.
  - `control_design(c) -> (names, matrix)` with columns `near_any` clipped to `[0, NEAR_CLIP]`, `ball_speed` clipped to `[0, BALL_SPEED_CLIP]`, `kink` clipped to `[0, KINK_CLIP]`, each with context widths `CONTROL_CONTEXT`; names exactly `["near_any", "near_any~6", "near_any~25", "near_any~75", "ball_speed", "ball_speed~6", "ball_speed~25", "ball_speed~75", "kink", "kink~6", "kink~25", "kink~75"]`.
  - `control_log_odds(c, model=None) -> np.ndarray` float64, `NaN` where `c["present"]` is False; `model=None` uses `evidence_models.CONTROL_MODEL` via `_fitted_control()` (returns `None` if the attribute is missing) and raises `RuntimeError("control model not fitted: run tools/fit_evidence.py")` when there is none; a mismatched model raises through `_check()`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_evidence.py`)

```python
CONTROL_NAMES = ["near_any", "near_any~6", "near_any~25", "near_any~75",
                 "ball_speed", "ball_speed~6", "ball_speed~25", "ball_speed~75",
                 "kink", "kink~6", "kink~25", "kink~75"]


def test_control_design_names_and_finite():
    names, x = evidence.control_design(evidence.clues(_match()))
    assert names == CONTROL_NAMES and x.shape == (400, 12) and np.isfinite(x).all()


def test_control_clues_are_nan_without_a_ball_and_log_odds_follow():
    tracking = _match()
    tracking["ball"][100:160] = np.nan
    c = evidence.clues(tracking)
    assert np.isnan(c["near_any"][100:160]).all() and np.isnan(c["kink"][100:160]).all()
    model = {"names": CONTROL_NAMES, "bias": 0.0, "weights": [0.0] * 12}
    lo = evidence.control_log_odds(c, model)
    assert np.isnan(lo[100:160]).all() and np.isfinite(lo[:100]).all()


def test_control_log_odds_refuses_mismatched_model():
    with pytest.raises(RuntimeError):
        evidence.control_log_odds(evidence.clues(_match()),
                                  {"names": ["wrong"], "bias": 0.0, "weights": [0.0]})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_evidence.py -q`
Expected: 3 new FAIL with `AttributeError` on `control_design`; the 10 old tests pass.

- [ ] **Step 3: Implement the constants, the two new clues, the `widths` parameter, `control_design()`, `_fitted_control()` and `control_log_odds()` in `espadas/evidence.py`**

`control_log_odds` applies `apply_logistic` on all rows, then sets `NaN` where not present. Keep `log_odds()` and its models untouched.

- [ ] **Step 4: Run the whole suite to verify it passes**

Run: `.venv/bin/python -m pytest -q`
Expected: `71 passed` (58 + 5 + 5 + 3).

- [ ] **Step 5: Commit (local only)**

```bash
git add espadas/evidence.py tests/test_evidence.py
git commit -m "Phase 7: control clues and design matrix"
```

---

### Task 4: Fit the control model (`tools/fit_evidence.py`)

**Files:**
- Modify: `tools/fit_evidence.py` (`rows()`, `stack()`, `report()`, `main()`, docstring)
- Generated: `espadas/evidence_models.py` (gains `CONTROL_MODEL`)

**Interfaces:**
- Consumes: `events.control_mask`, `evidence.control_design`, `evidence.control_log_odds`.
- Produces: `evidence_models.CONTROL_MODEL = {"names": [...12 names...], "bias": float, "weights": [12 floats]}`; `rows()` tuples become `(name, gt, clues, graded, ctrl)` with `ctrl = ev.control_mask(game["events"], game["n_frames"])`; `stack(games, design, which)` accepts `which == "control"` (rows: graded, H/A only, ball present; target: `ctrl`); `report()` prints one more number per game, "ctrl acc" = share of live ball-present frames where `control_log_odds > 0` equals `ctrl`; `main()` gets `argparse` with one flag `--heldout` (`store_true`; off by default, the held-out report is printed only when it is given, which is the team rule of checklist 4.10). The generated file's docstring says the control model is included.

- [ ] **Step 1: Make the script changes**

The report line for the control model follows the existing style: `"  {:<8} {:<10} ... ctrl acc {:.3f}"`. Print the control model's weights like the other two (`"control (at feet +)"`).

- [ ] **Step 2: Run the fit**

Run: `.venv/bin/python tools/fit_evidence.py`
Expected: three weight tables, a `tune` report per game and blur with `ctrl acc` around 0.75 or better, no `HELDOUT` lines, and `wrote espadas/evidence_models.py`.

- [ ] **Step 3: Verify the other two models did not move**

Run: `git diff -U0 espadas/evidence_models.py | grep '^[-+]' | grep -v '^+++\|^---'`
Expected: only added lines: the docstring sentence about the control model and `+CONTROL_MODEL = {...}`. No `-DEAD_MODEL` / `-TEAM_MODEL` line. If either model moved, stop: the fit rows changed and the task is wrong.

- [ ] **Step 4: Run the suite**

Run: `.venv/bin/python -m pytest -q`
Expected: `71 passed`.

- [ ] **Step 5: Commit (local only)**

```bash
git add tools/fit_evidence.py espadas/evidence_models.py
git commit -m "Phase 7: fit the control-vs-loose model on the tuning games"
```

---

### Task 5: The `v4` labeler (`espadas/v4.py`)

**Files:**
- Create: `espadas/v4.py`
- Modify: `espadas/labelers.py` (import and register `"v4"`)
- Test: `tests/test_v4.py`

**Interfaces:**
- Consumes: `hmm.viterbi`, `v3.features`, `v3.PARAMS` (touch detection values), `touches.detect / confirmed / pulses`, `evidence.clues / control_log_odds`, `v1._runs`.
- Produces (module `espadas/v4.py`):
  - `STATES = ("D", "HC", "HL", "AC", "AL")`; `D, HC, HL, AC, AL = range(5)`; `LABEL = np.array(["D", "H", "H", "A", "A"])`.
  - `TOUCH = {k: v3.PARAMS[k] for k in ("kink_half", "kink_threshold", "touch_reach", "chain_gap", "hold_reach", "hold_frames")}` (fixed, not tuned).
  - `PARAMS` (start values, spec §4.5): `ctrl_len 150, loose_len 25, win_share 0.3, to_dead_len 1500, dead_len 500, w_dead 1.0, dead_offset 0.5, w_team 1.0, team_cap 1.0, w_loose 0.3, w_ctrl 1.0, ctrl_offset 0.0, confirm_pulse 6.0, touch_pulse 1.0, min_dead 50, min_alive 25`.
  - `GRID`:
    ```
    ctrl_len [25, 50, 100, 150, 250, 400, 600, 1000]; loose_len [6, 12, 25, 40, 60, 100, 150]
    win_share [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]; to_dead_len [250, 500, 1000, 1500, 2500, 5000, 10000]
    dead_len [50, 100, 200, 500, 1000, 2000, 3000]; w_dead [0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0]
    dead_offset [-4.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]; w_team [0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0]
    team_cap [0.25, 0.5, 1.0, 2.0, 3.0, 5.0]; w_loose [0.0, 0.1, 0.3, 0.5, 1.0]
    w_ctrl [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]; ctrl_offset [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    confirm_pulse [0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0]; touch_pulse [0.0, 0.5, 1.0, 2.0, 4.0]
    min_dead [0, 25, 50, 100, 150, 250]; min_alive [0, 25, 50, 100, 150, 250]
    ```
  - `features(tracking) -> dict`: `v3.features(tracking)` plus `"ctrl_lo"` (`evidence.control_log_odds(evidence.clues(tracking))`), `"touches"` (`touches.detect(f["kinks"][TOUCH["kink_half"]], f["near"], f["slot"], f["team_of_slot"], TOUCH["kink_threshold"], TOUCH["touch_reach"])`) and `"touch_ok"` (`touches.confirmed(f["touches"], f["dist"], int(TOUCH["chain_gap"]), TOUCH["hold_reach"], int(TOUCH["hold_frames"]))`).
  - `transitions(p) -> (log_trans (5, 5), log_start (5,))` per spec §4.2: `p_loose = 1/ctrl_len`, `p_dead = 1/to_dead_len`, `p_regain = (1 - win_share)/loose_len`, `p_win = win_share/loose_len`, `p_restart = 1/dead_len` split half to `HC`, half to `AC`; diagonals take the remainder; raises `ValueError` if any diagonal would be `<= 0`. `log_start = log(1/3)` for `D, HC, AC`, `-inf` for `HL, AL`.
  - `emissions(f, p, first, stop) -> (stop - first, 5)` per spec §4.3; `e_touch` from `touches.pulses(len(f["present"]), f["touches"], f["touch_ok"], p["confirm_pulse"], p["touch_pulse"])[first:stop]`; `e_ctrl` is `0.0` where not present (use `np.where(present, w_ctrl * (np.nan_to_num(ctrl_lo) - ctrl_offset), 0.0)`).
  - `cleanup(labels, p) -> labels` on one half's `<U1` array: a `D` run shorter than `min_dead` takes the label of the frame before it (the frame after it when the run starts the half; left alone when the half is all `D`); then a live run shorter than `min_alive` that sits between two `D` runs becomes `D` (exactly `v2.dead_mask`'s second rule). Both rules are off at 0.
  - `decode(f, params=None) -> labels` (`<U1`, length `len(f["present"])`, `D` outside the halves) and `v4(tracking, params=None)`.

- [ ] **Step 1: Write the failing tests**

```python
"""v4 five-state decoder, on synthetic inputs only."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from espadas import labelers, v4  # noqa: E402

RAW = {"min_dead": 0, "min_alive": 0}
NO_TOUCH = {"frame": np.zeros(0, np.int64), "slot": np.zeros(0, np.int64),
            "team": np.zeros(0, np.int8), "strength": np.zeros(0)}


def _f(n=300, team_lo=None, dead_lo=None, present=None, touches=NO_TOUCH, touch_ok=None):
    return {"half_blocks": [(1, 0, n - 1)],
            "present": np.ones(n, bool) if present is None else present,
            "dead_lo": np.full(n, -5.0) if dead_lo is None else dead_lo,
            "team_lo": np.zeros(n) if team_lo is None else team_lo,
            "ctrl_lo": np.zeros(n), "touches": touches,
            "touch_ok": np.zeros(len(touches["frame"]), bool) if touch_ok is None else touch_ok}


def _team(n, before, after, at=150):
    return np.where(np.arange(n) < at, before, after).astype(float)


def test_transitions_are_stochastic_symmetric_and_forbid_direct_switches():
    lt, ls = v4.transitions(v4.PARAMS)
    assert np.allclose(np.exp(lt).sum(axis=1), 1.0)
    for a, b in [(v4.HC, v4.AC), (v4.HC, v4.AL), (v4.HL, v4.AL), (v4.D, v4.HL), (v4.D, v4.AL),
                 (v4.AC, v4.HC), (v4.AC, v4.HL), (v4.AL, v4.HL)]:
        assert lt[a, b] == -np.inf
    assert lt[v4.HC, v4.HL] == lt[v4.AC, v4.AL] and lt[v4.HL, v4.AC] == lt[v4.AL, v4.HC]
    assert lt[v4.D, v4.HC] == lt[v4.D, v4.AC]
    assert ls[v4.HL] == -np.inf and ls[v4.AL] == -np.inf and ls[v4.D] == ls[v4.HC] == ls[v4.AC]


def test_transitions_refuse_impossible_lengths():
    with pytest.raises(ValueError):
        v4.transitions(dict(v4.PARAMS, loose_len=1))


def test_dead_offset_is_fixed_at_zero_without_a_ball():
    f = _f(2, dead_lo=np.array([0.3, 0.3]), present=np.array([True, False]))
    e = v4.emissions(f, dict(v4.PARAMS, dead_offset=0.5, w_dead=1.0), 0, 2)
    assert e[0, v4.D] == pytest.approx(-0.2) and e[1, v4.D] == pytest.approx(0.3)
    assert e[1, v4.HC] == 0.0 and e[1, v4.HL] == 0.0


def test_clear_evidence_switches_weak_or_absent_evidence_does_not():
    p = dict(v4.PARAMS, **RAW)
    clear = v4.decode(_f(team_lo=_team(300, 2.0, -2.0)), p)
    assert set(clear[:140]) == {"H"} and set(clear[160:]) == {"A"}
    silent = v4.decode(_f(team_lo=_team(300, 2.0, 0.0)), p)
    assert set(silent) == {"H"}
    blip = _team(300, 2.0, 2.0)
    blip[150:170] = -0.3
    assert set(v4.decode(_f(team_lo=blip), p)) == {"H"}


def test_confirmed_touch_switches_and_plain_touch_does_not():
    touch = {"frame": np.array([150]), "slot": np.array([0]), "team": np.array([1], np.int8),
             "strength": np.array([8.0])}
    p = dict(v4.PARAMS, confirm_pulse=12.0, touch_pulse=1.0, **RAW)
    f = _f(team_lo=_team(300, 1.0, 0.0), touches=touch, touch_ok=np.array([True]))
    out = v4.decode(f, p)
    assert set(out[:140]) == {"H"} and set(out[160:]) == {"A"}
    f = _f(team_lo=_team(300, 1.0, 0.0), touches=touch, touch_ok=np.array([False]))
    assert set(v4.decode(f, p)) == {"H"}


def test_stoppage_in_the_middle_and_restart_to_the_other_team():
    dead = np.full(300, -5.0)
    dead[100:200] = 5.0
    team = _team(300, 2.0, 0.0, at=100)
    team[200:] = -2.0
    out = v4.decode(_f(dead_lo=dead, team_lo=team), dict(v4.PARAMS, **RAW))
    assert set(out[:98]) == {"H"} and set(out[102:198]) == {"D"} and set(out[202:]) == {"A"}


def test_cleanup_drops_short_dead_runs_and_fills_short_live_runs():
    seg = np.array(list("H" * 10 + "D" * 3 + "H" * 10))
    assert "".join(v4.cleanup(seg, {"min_dead": 5, "min_alive": 0})) == "H" * 23
    seg = np.array(list("H" * 10 + "D" * 10 + "H" * 3 + "D" * 10 + "A" * 10))
    assert "".join(v4.cleanup(seg, {"min_dead": 5, "min_alive": 5})) == \
        "H" * 10 + "D" * 23 + "A" * 10


def test_decode_with_ball_never_tracked():
    f = _f(team_lo=_team(300, 1.0, -1.0), present=np.zeros(300, bool))
    f["ctrl_lo"] = np.full(300, np.nan)
    out = v4.decode(f, dict(v4.PARAMS, **RAW))
    assert len(out) == 300 and set(out) <= {"H", "A", "D"}


def test_touch_on_array_edges():
    touch = {"frame": np.array([0, 299]), "slot": np.array([0, 0]),
             "team": np.array([0, 1], np.int8), "strength": np.array([8.0, 8.0])}
    out = v4.decode(_f(touches=touch, touch_ok=np.array([True, True])), dict(v4.PARAMS, **RAW))
    assert len(out) == 300


def _match(n=400, seed=0):
    """Copy of tests/test_evidence.py::_match (six players about a drifting ball)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    ball = np.stack([20 + 0.1 * t, 30 + 5 * np.sin(t / 40)], axis=1)
    players = np.zeros((n, 6, 2))
    offsets = np.array([[1, 0], [-8, 10], [-8, -10], [3, 0], [5, 3], [5, -3]], dtype=float)
    for k in range(6):
        players[:, k] = ball + offsets[k] + rng.normal(0, 0.1, (n, 2))
    players[:10, :3, 0] -= 15
    return {"players": players, "ball": ball, "team_of_slot": np.array([0, 0, 0, 1, 1, 1], np.int8),
            "half_blocks": [(1, 0, 199), (2, 200, n - 1)], "kickoffs": (5, 205),
            "fps": 25, "pitch": (105.0, 68.0)}


def test_v4_runs_on_synthetic_tracking_and_is_registered_and_deterministic():
    tracking = _match()
    first = v4.v4(tracking)
    assert len(first) == 400 and set(first) <= {"H", "A", "D"}
    assert np.array_equal(first, labelers.LABELERS["v4"](tracking))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_v4.py -q`
Expected: FAIL at import (`cannot import name 'v4'`).

- [ ] **Step 3: Implement `espadas/v4.py` and register it in `espadas/labelers.py`**

Module docstring in the style of `v2.py`: the five states, the two rules the structure enforces, the emission formula in words, decision (b), the fixed touch-detection settings, "tuned by tools/tune.py --labeler v4 on Metrica 1 and 3; never on our match". `decode()` builds `transitions(p)` once, then per half block: `emissions`, `hmm.viterbi`, `LABEL[path]`, `cleanup`, assign. In `labelers.py` add `from .v4 import v4 as v4_hmm` and `"v4": v4_hmm` to `LABELERS`.

- [ ] **Step 4: Run the suite to verify it passes**

Run: `.venv/bin/python -m pytest -q`
Expected: `82 passed` (71 + 11).

- [ ] **Step 5: Bench `v4` at its start values next to `v1` and `v3`, and compare memory-only loose states (`w_ctrl = 0`) with the control score on**

Run:
```bash
.venv/bin/python -m espadas.external.bench --labeler v1 v3 v4 --games 1 3
.venv/bin/python - <<'EOF'
import sys; sys.path.insert(0, "tools")
import tune
from espadas import v4
from espadas.external import bench
games = [g for b in tune.OBJECTIVE_BLURS for g in tune.prepare(v4, bench.TUNING, blur=b)]
for w in (0.0, 1.0):
    print("w_ctrl", w, "mean tuning spellF1 {:.4f}".format(tune.objective(v4, games, dict(v4.PARAMS, w_ctrl=w))[0]))
EOF
```
Expected: six bench rows (no `HELDOUT`), `v4` somewhere near `v2`/`v3` before tuning (0.45 to 0.60 is plausible; below 0.3 means a sign error in the emissions, so stop and check `test_clear_evidence_switches...` against a real game), and two `w_ctrl` lines. Note both numbers in the commit message; they go into the checklist in Task 9.

- [ ] **Step 6: Commit (local only)**

```bash
git add espadas/v4.py espadas/labelers.py tests/test_v4.py
git commit -m "Phase 7: v4 five-state HMM labeler (untuned)"
```

---

### Task 6: Tune `v4` on the tuning games

**Files:**
- Modify: `espadas/v4.py` (`PARAMS` and the comment block above it, as `v2.py`/`v3.py` do)

**Interfaces:**
- Consumes: `tools/tune.py` as is.

- [ ] **Step 1: Start the tuner in the background and time it**

Run (from the repo root; `out/` is gitignored):
```bash
mkdir -p out && nohup .venv/bin/python tools/tune.py --labeler v4 --skip-heldout > out/tune_v4.log 2>&1 &
```
Expected: `out/tune_v4.log` grows with `sweep N <param> -> <value> mean spellF1 ...` lines; a full run takes one to two hours. Check with `tail -3 out/tune_v4.log` every 20 minutes or so, not more often.

- [ ] **Step 2: Copy the result into `v4.py`**

When the log ends with `PARAMS = {...}`, the per-game lines and the two `tune @ 15% / 35%` lines: paste the dict into `PARAMS`, and write the comment block above it: mean tuning spellF1, the 15% / 35% numbers, the date, "held-out not scored (checklist 4.10)".

- [ ] **Step 3: Re-bench the tuned `v4` at every blur**

Run:
```bash
for b in light medium heavy; do .venv/bin/python -m espadas.external.bench --labeler v1 v3 v4 --games 1 3 --blur $b; done
```
Expected: `v4` at or above `v3` on the mean over the six rows. If it is below `v3`, the plan does not fail; record the numbers honestly and continue (Task 8 may recover it, and selection is the team's call).

- [ ] **Step 4: Run the suite and the determinism of the decode**

Run: `.venv/bin/python -m pytest -q`
Expected: `82 passed`.

- [ ] **Step 5: Commit (local only)**

```bash
git add espadas/v4.py
git commit -m "Phase 7: tune v4 on Metrica 1 and 3 (all blur levels, held-out untouched)"
```

---

### Task 7: Control-score diagnostic line (`tools/self_consistency.py`)

**Files:**
- Modify: `tools/self_consistency.py` (`tracking_line()`, `tracking_table()`)

**Interfaces:**
- Consumes: `evidence.clues`, `evidence.control_log_odds`.
- Produces: one more column in the `--tracking` table, header `ctrl>0`, value = share of graded, ball-present frames with `control_log_odds > 0`; `tracking_table()` prints `BAR-ATH: no match data on this machine` instead of raising when `atd.load_tracking` raises `IOError`/`OSError`.

- [ ] **Step 1: Make the change**

Widths follow the existing format strings (add `{:>8}` / `{:>8.0%}`). Update the docstring's list of what `--tracking` compares.

- [ ] **Step 2: Run it on the external games**

Run: `.venv/bin/python tools/self_consistency.py --tracking`
Expected: three Metrica rows with a `ctrl>0` column between 0.4 and 0.8, then `BAR-ATH: no match data on this machine`. (Someone with the match data runs it again later; the `BAR-ATH` number is a diagnostic only, never a tuning input.)

- [ ] **Step 3: Commit (local only)**

```bash
git add tools/self_consistency.py
git commit -m "Phase 7: control-score share in the tracking diagnostic"
```

---

### Task 8: Minimum durations inside the decoder (checklist 7.7, kept only if it helps)

**Files:**
- Modify: `espadas/hmm.py` (new `expand()`), `espadas/v4.py` (`dead_min`, `loose_min` in `PARAMS`/`GRID`; `decode()` uses `expand` when either is above 1)
- Test: `tests/test_hmm.py`, `tests/test_v4.py`

**Interfaces:**
- Produces: `hmm.expand(log_trans, log_start, mins) -> (log_trans_x, log_start_x, macro)` where `mins` is a sequence of `int >= 1` per state; state `s` becomes a chain of `mins[s]` sub-states, entered at its first link, with forced moves (log 0) along the chain and the original row of `s` on the last link (self-loop stays on the last link, exits go to the first link of the target); `log_start_x` puts `log_start[s]` on the first link; `macro` is an `int64` array mapping each expanded state to its original index. `viterbi` output is mapped back with `macro[path]`.
- `v4.PARAMS` gains `dead_min 1, loose_min 1`; `GRID` gains `dead_min [1, 12, 25, 50, 75, 100]`, `loose_min [1, 2, 4, 8, 12]`; `mins = [dead_min, 1, loose_min, 1, loose_min]` in state order.

- [ ] **Step 1: Write the failing tests**

In `tests/test_hmm.py`:
```python
def test_expand_with_all_ones_is_the_identity():
    trans = np.log(np.array([[0.9, 0.1], [0.2, 0.8]]))
    start = np.log([0.5, 0.5])
    tx, sx, macro = hmm.expand(trans, start, [1, 1])
    assert np.array_equal(tx, trans) and np.array_equal(sx, start) and macro.tolist() == [0, 1]


def test_expand_enforces_the_minimum_run_length():
    em = np.zeros((12, 2))
    em[::2, 0], em[1::2, 1] = 10.0, 10.0           # the evidence flips every frame
    free = np.log(np.full((2, 2), 0.5))
    tx, sx, macro = hmm.expand(free, np.log([0.5, 0.5]), [3, 1])
    path = macro[hmm.viterbi(em[:, macro], tx, sx)]
    runs = [len(list(g)) for k, g in itertools.groupby(path) if k == 0]
    assert runs and min(runs) >= 3 and path.tolist()[:3] == [0, 0, 0]


def test_expand_shorter_than_minimum():
    free = np.log(np.full((2, 2), 0.5))
    tx, sx, macro = hmm.expand(free, np.log([0.5, 0.5]), [5, 5])
    assert macro[hmm.viterbi(np.array([[0.0, 1.0], [0.0, 1.0]])[:, macro], tx, sx)].tolist() == [1, 1]
```

In `tests/test_v4.py`:
```python
def test_dead_min_rejects_a_short_stoppage_blip_and_loose_min_still_switches():
    dead = np.full(300, -5.0)
    dead[100:110] = 5.0                             # 10 frames of "dead", then live again
    f = _f(dead_lo=dead, team_lo=_team(300, 2.0, 2.0))
    assert "D" in v4.decode(f, dict(v4.PARAMS, dead_min=1, **RAW))[100:110]
    assert "D" not in v4.decode(f, dict(v4.PARAMS, dead_min=25, **RAW))
    out = v4.decode(_f(team_lo=_team(300, 2.0, -2.0)), dict(v4.PARAMS, loose_min=8, **RAW))
    assert set(out[:140]) == {"H"} and set(out[160:]) == {"A"}
```
(With `dead_min = 25` the decoder would have to pay 15 frames of strong live evidence to
stay dead, so it never enters `D`; with `dead_min = 1` the 10-frame blip is worth it.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_hmm.py tests/test_v4.py -q`
Expected: 4 FAIL: three on `AttributeError: ... no attribute 'expand'`, one because `dead_min` is ignored so the second assertion finds a `D`.

- [ ] **Step 3: Implement `hmm.expand()` and wire `dead_min` / `loose_min` into `v4.decode()`**

Build the expanded matrix with `-inf` everywhere, then fill chain links and the last link's row. `v4.decode()` computes `tx, sx, macro = hmm.expand(lt, ls, mins)` once when `max(mins) > 1`, passes `E[:, macro]` to `viterbi` and maps back with `macro[path]`; otherwise unchanged.

- [ ] **Step 4: Run the suite and measure the cost at the largest chain**

Run: `.venv/bin/python -m pytest -q` then the Task 1 Step 5 timing snippet with `tr, st, macro = hmm.expand(tr, st, [100, 1, 12, 1, 12])` and `em[:, macro]`.
Expected: `86 passed`; timing under 3 s per 70k frames. If it is over, implement the shift trick (chain links copy the previous link's score and emission without a max; only first links and last links take a max over predecessors) behind the same `expand`/`viterbi` interface, and re-run the tests.

- [ ] **Step 5: Re-tune from the Task 6 values with the two new grids**

Run: `nohup .venv/bin/python tools/tune.py --labeler v4 --skip-heldout > out/tune_v4_min.log 2>&1 &`
Expected: coordinate descent starts from the tuned `PARAMS`; the log shows whether `dead_min` / `loose_min` move off 1. Keep the new `PARAMS` only if the mean tuning spellF1 improves by at least 0.005 over Task 6; otherwise leave `PARAMS` as tuned in Task 6 (with `dead_min 1, loose_min 1`) and say so in the comment block.

- [ ] **Step 6: Commit (local only)**

```bash
git add espadas/hmm.py espadas/v4.py tests/test_hmm.py tests/test_v4.py
git commit -m "Phase 7: minimum durations inside the decoder (7.7)"
```

---

### Task 9: Docs, checklist, handoff

**Files:**
- Modify: `README.md` (layout table: rows for `espadas/hmm.py` and `espadas/v4.py`; "External data" section: `fit_evidence.py` also fits the control model; bench example lists `v4`)
- Modify: `docs/technical_document.md` (section 1: one paragraph on the five-state decoder; section 2: the control score and why the touch pulses were re-scaled; section 4: the loose-span rules; section 5: the 6.7 transfer risk of a distance-based control score)
- Modify: `../PLAN_CHECKLIST.md` (outside the repo, one level up): tick 7.3–7.6 and 7.7, 7.8 as `[~]` with "tuned on 1 and 3; Game 2 at selection"; the Phase 6/7 results table gains a `v4` row with the Task 6 (or Task 8) numbers at all three blurs; Snapshot row 7; 9.1 adds `v4`
- Create: `../PHASE7_HANDOFF.md` (outside the repo, next to `PHASE5-6_HANDOFF.md`, same shape: one paragraph, what was built, results table, how to run, what's next, open questions)

- [ ] **Step 1: Write the four documents**

The handoff's "how to run" block is:
```bash
python -m pytest -q                                              # 86 passed
python tools/fit_evidence.py                                     # refits all three models
python -m espadas.external.bench --labeler v1 v3 v4 --games 1 3  # tuning games only
python tools/tune.py --labeler v4 --skip-heldout
python -m espadas.run --labeler v4 --out out/espadas_v4.csv      # needs the match data
```
Its "what's next" names: 9 (selection: score `v1`, `v3`, `v4` on Game 2 once), 8.4 (`X` policy), 6.7 (touch gap; the control score shares the risk), and the open question whether the loose-span rules in `events.py` are agreed (section 10 of the design).

- [ ] **Step 2: Verify nothing data-like is staged and the suite is green**

Run: `.venv/bin/python tools/check_no_data.py --all && .venv/bin/python -m pytest -q && git status --short`
Expected: the guard passes, `86 passed`, and only `README.md`, `docs/technical_document.md`, `docs/phase7_hmm_design.md`, `docs/phase7_hmm_plan.md` show as modified or untracked inside the repo.

- [ ] **Step 3: Commit (local only)**

```bash
git add README.md docs/technical_document.md docs/phase7_hmm_design.md docs/phase7_hmm_plan.md
git commit -m "Phase 7: docs, design and plan"
```

Pushing to `main`, re-scoring Game 2 and changing the `run.py` default are **not** part of this plan; they belong to the team's selection step.
