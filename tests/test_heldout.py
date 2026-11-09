"""The held-out split is enforced in code (checklist 4.4), not by convention.

Only a real run of tools/selection.py may load held-out Game 2; everything else needs an explicit
opt-in. No external game is ever read here: the game readers and bench.available are fakes.
"""

import importlib.util
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from espadas.external import bench  # noqa: E402

FAKE_RESULT = {"spell_f1_mean": 0.5, "spell_f1_at_headline": 0.5, "macro_f1": 0.8,
               "n_pred_spells": 1, "n_gt_spells": 1, "n_abstain_on_graded": 0}


def load_tool(name):
    spec = importlib.util.spec_from_file_location(
        "tool_" + name, os.path.join(ROOT, "tools", name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def readers(monkeypatch):
    """Fake game readers; the list records every game actually read."""
    read = []

    def reader(n):
        read.append(n)
        return {"name": "fake{}".format(n)}

    monkeypatch.setattr(bench, "GAMES", {n: reader for n in (1, 2, 3)})
    return read


@pytest.fixture
def calls(monkeypatch):
    """Fake bench.available that keeps the real guard and records (games, heldout); loads nothing."""
    seen = []

    def fake_available(numbers, heldout=False):
        numbers = tuple(numbers)
        seen.append((numbers, heldout))
        bench.check_split(numbers, heldout)
        return []

    monkeypatch.setattr(bench, "available", fake_available)
    return seen


def touched_heldout(seen):
    return any(set(numbers) & set(bench.HELD_OUT) or heldout for numbers, heldout in seen)


# --------------------------------------------------------------------------- the choke point

def test_split_is_tune_on_1_and_3_hold_out_2():
    assert bench.TUNING == (1, 3) and bench.HELD_OUT == (2,)


def test_available_refuses_the_heldout_game_without_permission(readers):
    with pytest.raises(bench.HeldOutError, match="tools/selection.py"):
        bench.available([1, 2, 3])
    assert readers == []                      # refused before anything was read


def test_available_loads_the_heldout_game_only_with_permission(readers):
    assert [n for n, _ in bench.available([1, 2, 3], heldout=True)] == [1, 2, 3]


def test_tuning_games_need_no_permission(readers):
    assert [n for n, _ in bench.available(bench.TUNING)] == [1, 3]


# --------------------------------------------------------------------------- bench command line

def test_bench_defaults_to_the_tuning_games(readers, monkeypatch):
    monkeypatch.setattr(bench, "score", lambda *a, **k: FAKE_RESULT)
    assert bench.main(["--labeler", "v0"]) == 0
    assert readers == [1, 3]


def test_bench_refuses_a_heldout_game_without_the_flag(readers, capsys):
    with pytest.raises(SystemExit) as stop:
        bench.main(["--labeler", "v0", "--games", "1", "2"])
    assert stop.value.code != 0
    assert "tools/selection.py" in capsys.readouterr().err
    assert readers == []


def test_bench_heldout_flag_allows_it(readers, monkeypatch):
    monkeypatch.setattr(bench, "score", lambda *a, **k: FAKE_RESULT)
    assert bench.main(["--labeler", "v0", "--games", "2", "--heldout"]) == 0
    assert readers == [2]


# --------------------------------------------------------------------------- development tools

@pytest.mark.filterwarnings("ignore::RuntimeWarning")       # means over no game at all
@pytest.mark.parametrize("argv,heldout", [
    (["--labeler", "v2"], False),
    (["--labeler", "v2", "--skip-heldout"], False),           # old commands still run
    (["--labeler", "v2", "--heldout"], True),
    (["--labeler", "v2", "--heldout", "--skip-heldout"], False),
])
def test_tune_scores_heldout_only_when_asked(calls, argv, heldout):
    assert load_tool("tune").main(argv) == 0
    assert calls and touched_heldout(calls) is heldout


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("argv,heldout", [([], False), (["--heldout"], True)])
def test_tune_v1_scores_heldout_only_when_asked(calls, argv, heldout):
    assert load_tool("tune_v1").main(argv) == 0
    assert calls and touched_heldout(calls) is heldout


@pytest.mark.parametrize("argv,heldout", [
    ([], False),
    (["--skip-heldout"], False),
    (["--heldout"], True),
])
def test_fit_evidence_reports_heldout_only_when_asked(calls, monkeypatch, tmp_path, argv,
                                                      heldout):
    tool = load_tool("fit_evidence")
    dummy = {"names": [], "bias": 0.0, "weights": []}
    monkeypatch.setattr(tool, "fit", lambda *a, **k: (dummy, 0))
    monkeypatch.setattr(tool, "report", lambda *a, **k: None)
    monkeypatch.setattr(tool, "OUT", str(tmp_path / "evidence_models.py"))  # never the real one
    assert tool.main(argv) == 0
    assert calls and touched_heldout(calls) is heldout


class Stop(Exception):
    pass


@pytest.mark.parametrize("argv,heldout", [
    (["--labeler", "v0"], False),
    (["--tracking"], False),
    (["--labeler", "v0", "--heldout"], True),
    (["--tracking", "--heldout"], True),
])
def test_self_consistency_reads_heldout_only_when_asked(calls, monkeypatch, argv, heldout):
    tool = load_tool("self_consistency")

    def no_match_data(*args, **kwargs):
        raise Stop()

    monkeypatch.setattr(tool.atd, "load_tracking", no_match_data)
    with pytest.raises(Stop):
        tool.main(argv)
    assert calls and touched_heldout(calls) is heldout
