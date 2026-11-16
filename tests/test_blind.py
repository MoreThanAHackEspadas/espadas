"""Ball-blind dead network and the change bonus, on synthetic inputs and random weights only."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "tools"))

from espadas import blind, evidence, neural, v4  # noqa: E402
from test_evidence import _match  # noqa: E402
from test_neural import mirrored  # noqa: E402
from test_v4 import _f  # noqa: E402


def _weights(names, hidden=8, channels=12, dilations=(1, 4), seed=0):
    rng = np.random.default_rng(seed)

    def r(*shape):
        return rng.normal(0, 0.3, shape).astype(np.float32)
    w = {"names": list(names), "dilations": list(dilations),
         "gscale": (1 + r(len(names))).astype(np.float32),
         "p1.weight": r(hidden, len(blind.FEATURES) + 1), "p1.bias": r(hidden),
         "p2.weight": r(hidden, hidden), "p2.bias": r(hidden),
         "frame.weight": r(channels, 4 * hidden + len(names)), "frame.bias": r(channels),
         "out_norm.weight": 1 + r(channels), "out_norm.bias": r(channels),
         "out.weight": r(2, channels), "out.bias": r(2)}
    for i in range(len(dilations)):
        w["norms.%d.weight" % i], w["norms.%d.bias" % i] = 1 + r(channels), r(channels)
        w["convs.%d.weight" % i], w["convs.%d.bias" % i] = r(channels, channels, 3), r(channels)
        w["mix.%d.weight" % i], w["mix.%d.bias" % i] = r(channels, channels), r(channels)
    return w


def _apply(tr, w):
    return blind.apply(w, blind.inputs(tr))


@pytest.fixture
def exact(monkeypatch):
    monkeypatch.setattr(neural, "ROUND", 9)      # compare before the 3-decimal rounding


def test_inputs_are_in_slot_order_and_never_touch_the_ball():
    tr = _match()
    tr["players"] = tr["players"].copy()
    tr["players"][30:60, 1] = np.nan                 # a player lost for a while
    inp = blind.inputs(tr)
    n = len(tr["ball"])
    assert inp["players"].shape == (n, 2, neural.K_PLAYERS, 4)
    assert inp["names"] == evidence.motion_design(evidence.clues(tr))[0]
    assert np.isfinite(inp["players"]).all() and np.isfinite(inp["globals"]).all()
    assert inp["mask"][0].sum() == 6 and inp["mask"][40].sum() == 5
    assert not inp["players"][40][~inp["mask"][40]].any()      # untracked entries are zero
    assert inp["mask"][40, 0, :2].all() and not inp["mask"][40, 0, 2:].any()
    again = blind.inputs(dict(tr, ball=np.full_like(tr["ball"], np.nan)))
    assert all(np.array_equal(inp[k], again[k]) for k in ("players", "mask", "globals"))


def test_packing_a_trained_dict_gives_motion_names_and_survives_a_round_trip():
    names = evidence.motion_design(evidence.clues(_match()))[0]
    every = ["g_present"] + ["dead:" + n for n in names] + ["dead:ball_speed", "dead:out~25"]
    cols = [i for i, n in enumerate(every) if n.startswith("dead:") and "ball" not in n
            and not n.startswith("dead:out")]
    w = _weights(names)
    trained = dict(w, names=every, cols=cols)
    again = neural.unpack(eval(repr(blind.pack(trained))))
    assert again["names"] == names and "cols" not in again
    assert all(np.array_equal(w[k], again[k]) for k in w)
    tr = _match()
    assert np.array_equal(_apply(tr, w), _apply(tr, again))


def test_numpy_forward_equals_the_pytorch_network():
    torch = pytest.importorskip("torch")
    import fit_blind
    tr = _match()
    inp = blind.inputs(tr)
    cfg = {"player_hidden": 8, "channels": 12, "dilations": [1, 4], "dropout": 0.25}
    torch.manual_seed(0)
    net = fit_blind.build(len(inp["names"]), cfg).eval()
    w = {k: v.detach().numpy().astype(np.float32) for k, v in net.state_dict().items()}
    w.update(names=list(inp["names"]), dilations=cfg["dilations"],
             gscale=(1 + 0.2 * np.random.default_rng(0).normal(size=len(inp["names"])))
             .astype(np.float32))
    p, m, g = inp["players"], inp["mask"], inp["globals"] * w["gscale"]
    with torch.no_grad():
        want = net(torch.from_numpy(p)[None], torch.from_numpy(m)[None],
                   torch.from_numpy(g)[None])[0].numpy()
    got = blind.forward(w, p, m, g)
    assert got.shape == want.shape == (len(p), 2)
    assert np.abs(got - want).max() < 1e-3 and np.abs(want).max() > 0.01


def test_the_ball_does_not_matter(exact):
    tr = _match()
    names = evidence.motion_design(evidence.clues(tr))[0]
    w = _weights(names)
    base = _apply(tr, w)
    assert base.std() > 0
    rng = np.random.default_rng(3)
    for ball in (np.full_like(tr["ball"], np.nan), rng.uniform(0, 100, tr["ball"].shape)):
        assert np.array_equal(_apply(dict(tr, ball=ball), w), base)


def test_team_swap_and_pitch_mirror_leave_the_output_unchanged(exact):
    tr = _match()
    w = _weights(evidence.motion_design(evidence.clues(tr))[0])
    base = _apply(tr, w)
    swapped = dict(tr, team_of_slot=1 - tr["team_of_slot"])
    assert np.allclose(_apply(swapped, w), base, atol=1e-4)
    assert np.allclose(_apply(mirrored(tr), w), base, atol=1e-4)


def test_two_copies_of_a_net_equal_the_net_and_names_are_checked():
    tr = _match()
    w = _weights(evidence.motion_design(evidence.clues(tr))[0])
    assert np.array_equal(_apply(tr, [w, dict(w)]), _apply(tr, w))
    with pytest.raises(RuntimeError):
        _apply(tr, dict(w, names=w["names"][::-1]))


def _stump(names, bias):
    return {"names": names, "bias": bias, "weights": [0.0] * len(names)}


def test_log_odds_blind_switch():
    tr = _match()
    tr["ball"] = tr["ball"].copy()
    tr["ball"][100:300] = np.nan                     # a long gap: not filled
    c = evidence.clues(tr)
    seen = c["present"]
    assert seen.any() and (~seen).any()
    w = _weights(evidence.motion_design(c)[0])
    base = {"motion": _stump(evidence.motion_design(c)[0], 0.25),
            "ball": _stump(evidence.ball_design(c)[0], 0.5)}
    team = _stump(evidence.team_design(c)[0], 0.0)
    trees, team_lo = evidence.log_odds(c, base, team)
    assert np.array_equal(trees[seen], np.full(seen.sum(), 0.5)) and trees[~seen][0] == 0.25
    b = blind.apply(w, blind.inputs(tr, c))
    assert (b[~seen] != 0.25).any() and (b[seen] != 0.5).any()
    for mode, want in (("split", np.where(seen, trees, b)),
                       ("avg", np.where(seen, 0.5 * (trees + b), b)), ("blind", b)):
        dead, team2 = evidence.log_odds(c, dict(base, blind=mode, blind_w=w), team)
        assert np.array_equal(dead, want) and np.array_equal(team2, team_lo)
    assert np.array_equal(evidence.log_odds(c, dict(base, blind=None), team)[0], trees)
    with pytest.raises(ValueError):
        evidence.log_odds(c, dict(base, blind="nonsense", blind_w=w), team)


def test_viterbi_with_zero_bonus_is_viterbi_and_a_bonus_moves_the_switch():
    rng = np.random.default_rng(0)
    emis, trans = rng.normal(size=(300, 5)), v4.transitions(v4.PARAMS)
    base = v4.viterbi(emis, trans, v4.START)
    assert np.array_equal(v4.viterbi(emis, trans, v4.START, np.zeros(300)), base)
    assert np.array_equal(v4.viterbi(emis, trans, v4.START, None), base)
    # brute force on tiny problems: a move into frame t gains bonus[t]
    import itertools
    for _ in range(20):
        s, n = 3, 5
        e, tr, b = rng.normal(size=(n, s)), rng.normal(size=(s, s)), rng.uniform(0, 3, n)
        np.fill_diagonal(tr, 0.0)

        def score(path):
            return e[0, path[0]] + sum(tr[a, c] + e[t + 1, c] + (b[t + 1] if a != c else 0.0)
                                       for t, (a, c) in enumerate(zip(path, path[1:])))
        best = max(itertools.product(range(s), repeat=n), key=score)
        assert np.isclose(score(v4.viterbi(e, tr, None, b)), score(best))


def test_decode_with_zero_change_weight_is_the_old_decode():
    f = _f(team=1.0)
    f["team_lo"][250:] = -1.0
    f["dead_lo"][100:140] = 9.0
    old = v4.decode(f)
    rng = np.random.default_rng(1)
    f["change_lo"] = rng.normal(0, 5, len(f["team_lo"]))
    assert v4.PARAMS["w_change"] == 0.0
    assert np.array_equal(v4.decode(f), old)
    assert np.array_equal(v4.decode(f, {"w_change": 0.0}), old)
    g = dict(f, change_lo=None)                      # no network: the bonus is off
    assert np.array_equal(v4.decode(g, {"w_change": 20.0}), old)
    f["change_lo"] = np.full(len(f["team_lo"]), -50.0)   # P(change) = 0: nothing to gain
    assert np.array_equal(v4.decode(f, {"w_change": 20.0}), old)
    assert v4.GRID["w_change"] == [0.0, 2.0, 5.0, 10.0, 20.0, 40.0]
