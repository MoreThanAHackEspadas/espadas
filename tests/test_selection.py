"""tools/selection.py: the one-shot held-out comparison (checklist 9.2 / 9.3).

Every test here works on made-up scores. No external game is ever read.
"""

import importlib.util
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEVELS = ("light", "medium", "heavy", "smooth")


def load():
    spec = importlib.util.spec_from_file_location(
        "selection_tool", os.path.join(ROOT, "tools", "selection.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake(candidate, abstain, game, hide, level, spell, macro=0.8, n_pred=250, n_gt=280, n_x=0):
    return {"candidate": candidate, "abstain": abstain, "game": game, "hide": hide,
            "level": level, "spell_f1": spell, "macro_f1": macro, "n_pred": n_pred,
            "n_gt": n_gt, "n_x": n_x}


def test_plan_lists_every_combination_exactly_once():
    s = load()
    runs = s.plan(candidates=["v1", "v4"], abstain=(False, True), hides=(0.15, 0.25, 0.35),
                  levels=LEVELS, games=(1, 3, 2))
    assert len(runs) == 2 * 2 * 3 * 4 * 3
    assert len(set(runs)) == len(runs)
    assert ("v4", True, 2, 0.35, "smooth") in runs


def test_summary_means_held_out_over_levels_and_tuning_over_games_and_levels():
    s = load()
    results = []
    for i, level in enumerate(LEVELS):
        results.append(fake("v4", False, 2, 0.25, level, 0.5 + 0.1 * i))   # 0.5 .. 0.8 -> 0.65
        results.append(fake("v4", False, 1, 0.25, level, 0.4))
        results.append(fake("v4", False, 3, 0.25, level, 0.6))
    summary = s.summarise(results, tuning=(1, 3), heldout=2)
    row = summary[("v4", False)]
    assert row["heldout"][0.25]["spell_f1"] == pytest.approx(0.65)
    assert row["heldout"][0.25]["by_level"]["smooth"] == pytest.approx(0.8)
    assert row["tuning"][0.25]["spell_f1"] == pytest.approx(0.5)


def test_reading_names_the_leader_its_margin_and_the_levels_it_wins():
    s = load()
    results = []
    for level in LEVELS:
        results.append(fake("v4", False, 2, 0.25, level, 0.60))
        results.append(fake("v3", False, 2, 0.25, level, 0.61 if level == "smooth" else 0.55))
        results.append(fake("v1", False, 2, 0.25, level, 0.40))
    summary = s.summarise(results, tuning=(1, 3), heldout=2)
    text = "\n".join(s.reading(summary, hide=0.25, levels=LEVELS))
    assert "v4" in text.splitlines()[0]
    assert "0.035" in text            # margin over v3 (0.600 - 0.565)
    assert "within" in text           # ... which is inside the 0.04 spellF1 band
    assert "3 of 4" in text           # ahead at 3 of the 4 tracker levels


def two_candidates(lead, second):
    rows = []
    for level in LEVELS:
        rows.append(fake("v4", False, 2, 0.25, level, lead))
        rows.append(fake("v3", False, 2, 0.25, level, second))
    return rows


def test_bands_are_the_organisers_spell_band_from_the_scorer_and_0_01_for_macro():
    s = load()
    assert s.SPELL_BAND == s.config.pm.SPELL_NOISE_BAND == 0.04
    assert s.MACRO_BAND == 0.01


def test_spell_margins_are_judged_against_the_spell_band_not_0_01():
    s = load()
    within = s.summarise(two_candidates(0.600, 0.580), tuning=(1, 3), heldout=2)
    first = s.reading(within, hide=0.25, levels=LEVELS)[0]
    assert "0.020" in first and "within" in first and "0.04" in first
    above = s.summarise(two_candidates(0.650, 0.600), tuning=(1, 3), heldout=2)
    assert "above" in s.reading(above, hide=0.25, levels=LEVELS)[0]


def test_macro_margins_are_judged_against_0_01():
    s = load()
    for gain, verdict in ((0.03, "above"), (0.005, "within")):
        results = []
        for level in LEVELS:
            results.append(fake("v4", False, 2, 0.25, level, 0.60, macro=0.80))
            results.append(fake("v4", True, 2, 0.25, level, 0.60, macro=0.80 + gain, n_x=1000))
        summary = s.summarise(results, tuning=(1, 3), heldout=2)
        x_line = [l for l in s.reading(summary, hide=0.25, levels=LEVELS) if "macroF1" in l
                  and "unchanged" in l][0]
        assert "{:+.3f}".format(gain) in x_line
        assert verdict in x_line and "0.01" in x_line


def test_reading_says_what_the_bands_are_and_are_not():
    s = load()
    summary = s.summarise(two_candidates(0.600, 0.595), tuning=(1, 3), heldout=2)
    text = "\n".join(s.reading(summary, hide=0.25, levels=LEVELS))
    assert "guidance" in text and "full match" in text
    assert "not noise measured on this bench" in text
    assert "same games" in text and "not independent" in text


def test_reading_checks_that_x_leaves_spell_f1_unchanged():
    s = load()
    results = []
    for level in LEVELS:
        results.append(fake("v4", False, 2, 0.25, level, 0.60, macro=0.80))
        results.append(fake("v4", True, 2, 0.25, level, 0.60, macro=0.83, n_x=1000))
    summary = s.summarise(results, tuning=(1, 3), heldout=2)
    text = "\n".join(s.reading(summary, hide=0.25, levels=LEVELS))
    assert "unchanged" in text and "+0.030" in text


def test_render_has_the_summary_and_per_level_tables_and_the_provenance():
    s = load()
    results = []
    for hide in (0.15, 0.25, 0.35):
        for level in LEVELS:
            for game in (1, 3, 2):
                results.append(fake("v4", False, game, hide, level, 0.6))
                results.append(fake("v4", True, game, hide, level, 0.6, macro=0.83))
    summary = s.summarise(results, tuning=(1, 3), heldout=2)
    md = s.render(summary, hides=(0.15, 0.25, 0.35), levels=LEVELS,
                  meta={"date": "2026-11-11", "commit": "abc1234", "dirty": False})
    assert "abc1234" in md and "2026-11-11" in md
    assert "| v4 | off |" in md and "| v4 | on |" in md
    assert "light" in md and "smooth" in md
    assert "not noise measured on this bench" in md


# --------------------------------------------------------------------------- the once-rule

META = {"date": "2026-11-11", "commit": "abc1234", "dirty": False}
SMALL = ["--candidates", "v4", "v3", "--levels", "light", "medium", "--hides", "0.25"]


def test_start_marker_is_the_lock_and_again_appends(tmp_path):
    s = load()
    path = str(tmp_path / "selection.md")
    s.start_report(path, META)
    with pytest.raises(s.AlreadyRun):
        s.start_report(path, META)
    first = open(path).read()
    s.start_report(path, dict(META, commit="def5678"), again=True)
    text = open(path).read()
    assert text.startswith(first)                       # appended, never overwritten
    assert text.count("## Selection run") == 2 and "def5678" in text


@pytest.fixture
def fake_bench(monkeypatch):
    """Fake game loading and scoring. state["fail_on"] = a game number makes scoring that game
    raise state["error"]; state["lock_seen"] records, per scored game, whether the report existed
    at that moment."""
    s = load()
    state = {"fail_on": None, "error": KeyboardInterrupt, "lock_seen": [], "out": None,
             "loaded": 0}

    def fake_available(numbers, heldout=False):
        state["loaded"] += 1
        s.bench.check_split(numbers, heldout)
        return [(n, {"name": "fake{}".format(n), "n": n}) for n in numbers]

    def fake_score(name, game, hide_ball=None, blur=None, use_abstain=None):
        state["lock_seen"].append((game["n"], os.path.exists(state["out"])))
        if game["n"] == state["fail_on"]:
            raise state["error"]()
        return {"spell_f1_mean": 0.6 if name == "v4" else 0.5, "macro_f1": 0.8,
                "n_pred_spells": 250, "n_gt_spells": 280,
                "n_abstain_on_graded": 100 if use_abstain else 0}

    monkeypatch.setattr(s.bench, "available", fake_available)
    monkeypatch.setattr(s.bench, "score", fake_score)
    monkeypatch.setattr(s, "provenance", lambda: dict(META))
    return s, state


def test_interrupted_during_heldout_leaves_a_lock_and_a_plain_rerun_is_refused(fake_bench,
                                                                             tmp_path):
    s, state = fake_bench
    out = state["out"] = str(tmp_path / "selection.md")
    state["fail_on"] = 2
    with pytest.raises(KeyboardInterrupt):
        s.main(SMALL + ["--out", out])
    tuning = [lock for game, lock in state["lock_seen"] if game != 2]
    held = [lock for game, lock in state["lock_seen"] if game == 2]
    assert tuning and not any(tuning)          # no lock while the tuning games were scored
    assert held == [True]                      # the lock was on disk before Game 2 was scored
    text = open(out).read()
    assert "## Selection run: 2026-11-11, commit `abc1234`" in text
    assert "Run started" in text and "the one shot is spent" in text
    assert "### Summary" not in text           # no results: visibly interrupted

    state["fail_on"], state["lock_seen"], state["loaded"] = None, [], 0
    assert s.main(SMALL + ["--out", out]) == 1
    assert state["lock_seen"] == [] and state["loaded"] == 0   # refused before reading anything
    assert open(out).read() == text


def test_interrupted_before_heldout_leaves_no_lock(fake_bench, tmp_path):
    s, state = fake_bench
    out = state["out"] = str(tmp_path / "selection.md")
    state["fail_on"], state["error"] = 1, RuntimeError
    with pytest.raises(RuntimeError):
        s.main(SMALL + ["--out", out])
    assert not os.path.exists(out)
    state["fail_on"] = None
    assert s.main(SMALL + ["--out", out]) == 0         # the one shot is still there
    assert os.path.exists(out)


def test_a_normal_run_writes_one_well_formed_report(fake_bench, tmp_path):
    s, state = fake_bench
    out = state["out"] = str(tmp_path / "selection.md")
    assert s.main(SMALL + ["--out", out]) == 0
    text = open(out).read()
    assert text.startswith("# Selection on held-out Metrica Game 2\n")
    assert text.count("# Selection on held-out") == 1
    assert text.count("## Selection run") == 1
    order = ["## Selection run: 2026-11-11, commit `abc1234`", "Run started",
             "the one shot is spent", "Completed", "### Summary", "### Game 2 by tracker level",
             "### Reading", "**Decision:**"]
    positions = [text.index(marker) for marker in order]
    assert positions == sorted(positions)
    assert "| v4 | off |" in text and "| v3 | on |" in text

    assert s.main(SMALL + ["--out", out, "--again"]) == 0
    again = open(out).read()
    assert again.startswith(text) and again.count("## Selection run") == 2


def test_score_one_forces_x_on_or_off_and_leaves_params_alone(monkeypatch):
    s = load()
    seen = []

    def fake_score(name, game, hide_ball=None, blur=None, use_abstain=None):
        seen.append((use_abstain, dict(s.abstain.PARAMS)))
        return {"spell_f1_mean": 0.5, "macro_f1": 0.8, "n_pred_spells": 1, "n_gt_spells": 1,
                "n_abstain_on_graded": 0}

    monkeypatch.setattr(s.bench, "score", fake_score)
    before = dict(s.abstain.PARAMS)
    s.score_one(game=object(), candidate="v4", abstain_on=True, hide=0.25, level="light")
    s.score_one(game=object(), candidate="v4", abstain_on=False, hide=0.25, level="light")
    assert [force for force, _ in seen] == [True, False]
    assert all(params == before for _, params in seen)       # nothing injected while scoring
    assert s.abstain.PARAMS == before


def test_bench_x_on_is_exactly_what_the_build_gives_once_params_is_set(monkeypatch):
    """What selection scores as "X on" (bench.score(use_abstain=True)) is the no-flag result
    after abstain.PARAMS[candidate] = SETTING; "X off" is the no-flag result with None."""
    s = load()
    bench, abstain = s.bench, s.abstain
    n = 3000
    graded = np.ones(n, dtype=bool)
    graded[:100] = False
    present = np.ones(n, dtype=bool)
    present[500:900] = False                     # 400 frames without a ball inside an H spell
    labels_seen = []
    monkeypatch.setattr(bench, "setup", lambda game, hide, blur: (None, {}, graded))
    monkeypatch.setitem(bench.labelers.LABELERS, "v4", lambda tracking: np.full(n, "H"))
    monkeypatch.setattr(bench.v2, "features",
                        lambda tracking: {"present": present, "team_lo": np.zeros(n)})
    monkeypatch.setattr(bench, "evaluate",
                        lambda gt, labels, graded: labels_seen.append(np.array(labels)) or {})

    monkeypatch.setitem(abstain.PARAMS, "v4", None)
    bench.score("v4", game=None, use_abstain=True)
    bench.score("v4", game=None)
    monkeypatch.setitem(abstain.PARAMS, "v4", abstain.SETTING)
    bench.score("v4", game=None)
    bench.score("v4", game=None, use_abstain=False)
    x_on, default_off, default_params, forced_off = labels_seen
    assert np.any(x_on[graded] == "X")
    assert np.array_equal(x_on, default_params)
    assert not np.any(default_off[graded] == "X")
    assert np.array_equal(default_off, forced_off)


def test_smoke_mode_scores_only_tuning_games_and_writes_no_report(monkeypatch, tmp_path):
    s = load()
    asked = {}

    def fake_available(numbers, heldout=False):
        asked["games"] = tuple(numbers)
        return [(n, {"name": "fake{}".format(n)}) for n in numbers]

    def fake_score(name, game, hide_ball=None, blur=None, use_abstain=False):
        return {"spell_f1_mean": 0.5, "macro_f1": 0.8, "n_pred_spells": 1, "n_gt_spells": 1,
                "n_abstain_on_graded": 0}

    monkeypatch.setattr(s.bench, "available", fake_available)
    monkeypatch.setattr(s.bench, "score", fake_score)
    out = str(tmp_path / "selection.md")
    assert s.main(["--smoke", "--candidates", "v4", "--levels", "light", "--out", out]) == 0
    assert asked["games"] == s.bench.TUNING
    assert not os.path.exists(out)


def test_only_the_real_run_asks_for_the_heldout_game(monkeypatch, tmp_path):
    s = load()
    asked = []

    def fake_available(numbers, heldout=False):
        asked.append((tuple(numbers), heldout))
        s.bench.check_split(numbers, heldout)             # the real guard
        return [(n, {"name": "fake{}".format(n)}) for n in numbers]

    def fake_score(name, game, hide_ball=None, blur=None, use_abstain=None):
        return {"spell_f1_mean": 0.5, "macro_f1": 0.8, "n_pred_spells": 1, "n_gt_spells": 1,
                "n_abstain_on_graded": 0}

    monkeypatch.setattr(s.bench, "available", fake_available)
    monkeypatch.setattr(s.bench, "score", fake_score)
    out = str(tmp_path / "selection.md")
    small = ["--candidates", "v4", "--levels", "light", "--hides", "0.25", "--no-x", "--out", out]
    assert s.main(["--smoke"] + small) == 0
    assert asked == [(s.bench.TUNING, False)]
    asked.clear()
    assert s.main(small) == 0
    assert asked == [(s.bench.TUNING + s.bench.HELD_OUT, True)]


def test_dry_run_prints_the_plan_and_reads_no_game(monkeypatch, capsys, tmp_path):
    s = load()

    def boom(*args, **kwargs):
        raise AssertionError("a dry run must not load any game")

    monkeypatch.setattr(s.bench, "available", boom)
    monkeypatch.setattr(s.bench, "score", boom)
    assert s.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "216" in out                    # 3 candidates x 2 X modes x 3 hides x 4 levels x 3 games
    report = str(tmp_path / "selection.md")
    assert s.main(["--dry-run", "--out", report]) == 0
    assert not os.path.exists(report)
