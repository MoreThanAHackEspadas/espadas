"""Phase 8.4 X policy: synthetic tests. The key one: X never changes what spellF1 sees."""

import numpy as np

from espadas import abstain, config

pm = config.pm

P = {"weak": np.inf, "min_gap": 1, "edge": 5}


def random_labels(rng, n=3000):
    """Spells of random length and label, with ungraded X at both ends and a halftime hole."""
    out = []
    while len(out) < n:
        out += [rng.choice(list("HAD"))] * int(rng.integers(1, 120))
    lab = np.array(out[:n], dtype="<U1")
    lab[:100] = "X"
    lab[1400:1600] = "X"
    lab[-80:] = "X"
    return lab


def graded_runs(lab):
    return pm.valid_runs(pm.label_codes(np.where(lab == "X", "X", "H")))


def test_fill_identity_spellf1_cannot_change():
    rng = np.random.default_rng(0)
    for trial in range(30):
        lab = random_labels(rng)
        present = rng.random(len(lab)) < 0.5
        team_lo = rng.normal(0, 2, len(lab))
        for p in ({"weak": np.inf, "min_gap": 1, "edge": 1},
                  {"weak": 1.0, "min_gap": 12, "edge": 6}):
            out = abstain.apply(lab, present, team_lo, p)
            runs = graded_runs(lab)
            before = pm.fill_abstentions(pm.label_codes(lab), runs)
            after = pm.fill_abstentions(pm.label_codes(out), runs)
            assert np.array_equal(before, after)
            # and the official scorer agrees: same spells against any truth
            gt = pm.label_codes(lab)
            a, b = pm.evaluate(gt, pm.label_codes(lab), bootstrap_n=0), \
                pm.evaluate(gt, pm.label_codes(out), bootstrap_n=0)
            assert a["spell_f1_mean"] == b["spell_f1_mean"]


def test_no_x_near_a_transition_or_on_dead_spells():
    lab = np.array(list("H" * 40 + "A" * 40 + "D" * 40 + "H" * 8), dtype="<U1")
    out = abstain.apply(lab, np.zeros(len(lab), bool), np.zeros(len(lab)), P)
    x = np.flatnonzero(out == "X")
    assert list(x) == list(range(5, 35)) + list(range(45, 75))   # edge 5 each side
    assert not np.any(out[80:] == "X")       # D spell, and an H spell too short for 2*edge+1


def test_only_where_ball_missing_weak_and_long():
    n = 200
    lab = np.array(["H"] * n, dtype="<U1")
    present = np.ones(n, bool)
    present[50:60] = False        # short gap
    present[100:150] = False      # long gap
    team_lo = np.zeros(n)
    team_lo[120:130] = 3.0        # strong evidence inside the long gap
    out = abstain.apply(lab, present, team_lo, {"weak": 1.0, "min_gap": 15, "edge": 1})
    x = np.flatnonzero(out == "X")
    assert list(x) == list(range(100, 120)) + list(range(130, 150))


def test_off_keeps_ungraded_x_and_is_deterministic():
    rng = np.random.default_rng(1)
    lab = random_labels(rng)
    present = rng.random(len(lab)) < 0.3
    team_lo = rng.normal(0, 1, len(lab))
    assert np.array_equal(abstain.apply(lab, present, team_lo, None), lab)
    out1 = abstain.apply(lab, present, team_lo, P)
    out2 = abstain.apply(lab, present, team_lo, P)
    assert np.array_equal(out1, out2)
    assert set(np.unique(out1)) <= set("HADX")
    assert np.all(out1[lab == "X"] == "X")             # ungraded X untouched
    assert np.all((out1 == lab) | (out1 == "X"))       # only ever adds X
