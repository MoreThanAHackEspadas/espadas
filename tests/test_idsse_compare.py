"""tools/idsse_compare.py: rule 4 logic, challenger parsing, refusals (no data needed)."""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import idsse_compare as c  # noqa: E402

GAMES = [11, 14, 16, 17]


def grid(base, chal, tweak=None):
    s = {}
    for g in GAMES:
        for lv in c.LEVELS:
            for h in c.HIDES:
                s[("base", g, lv, h)] = base
                s[("chal", g, lv, h)] = chal
    for key, value in (tweak or {}).items():
        s[key] = value
    return s


def test_clear_broad_win_qualifies():
    adopt, checks = c.decide(grid(0.40, 0.45), GAMES)
    assert adopt and all(ok for _, ok, _ in checks)


def test_small_win_is_a_tie():
    adopt, checks = c.decide(grid(0.40, 0.43), GAMES)
    assert not adopt and not dict((k, ok) for k, ok, _ in checks)["a"]


def test_win_carried_by_one_match_fails_b():
    s = grid(0.40, 0.39)
    for lv in c.LEVELS:
        for h in c.HIDES:
            s[("chal", 11, lv, h)] = 0.80
    adopt, checks = c.decide(s, GAMES)
    ok = dict((k, v) for k, v, _ in checks)
    assert ok["a"] and not ok["b"] and not adopt


def test_behind_at_35_fails_d():
    s = grid(0.40, 0.46)
    for g in GAMES:
        for lv in c.LEVELS:
            s[("chal", g, lv, 0.35)] = 0.39
    adopt, checks = c.decide(s, GAMES)
    assert not adopt and not dict((k, v) for k, v, _ in checks)["d"]


def test_named_challenger_line():
    assert c.named_challenger("x\nChallenger: `abc1234` `v3`\n") == ("abc1234", "v3")
    assert c.named_challenger("Challenger: abc1234 v4") == ("abc1234", "v4")
    assert c.named_challenger("Challenger: (to be named)") is None


def test_real_run_refuses_without_named_challenger(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "REPORT", str(tmp_path / "idsse_comparison.md"))
    assert c.main([]) == 2


def test_real_run_refuses_a_second_time(tmp_path, monkeypatch):
    report = tmp_path / "idsse_comparison.md"
    report.write_text("Challenger: abc1234 v3\n\n## Run started 2026-11-14 10:00\n")
    monkeypatch.setattr(c, "REPORT", str(report))
    monkeypatch.setattr(c, "dirty", lambda: [])
    assert c.main([]) == 2


def test_frozen_reference_check_passes_on_this_checkout():
    assert c.frozen_ok() == []


def test_challenger_flag_only_for_smoke():
    with pytest.raises(SystemExit):
        c.main(["--challenger", "abc1234", "v3"])
