"""Neural evidence model (NumPy side), on synthetic inputs and random weights only."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from espadas import evidence, neural  # noqa: E402
from test_evidence import _match  # noqa: E402


def _weights(names, hidden=8, channels=12, dilations=(1, 4), seed=0):
    rng = np.random.default_rng(seed)

    def r(*shape):
        return rng.normal(0, 0.3, shape).astype(np.float32)
    w = {"names": list(names), "dilations": list(dilations),
         "gscale": np.ones(len(names), np.float32),
         "p1.weight": r(hidden, len(neural.PLAYER_FEATURES)), "p1.bias": r(hidden),
         "p2.weight": r(hidden, hidden), "p2.bias": r(hidden),
         "frame.weight": r(channels, 2 * (2 + neural.TOP) * hidden + len(names)),
         "frame.bias": r(channels),
         "out_norm.weight": 1 + r(channels), "out_norm.bias": r(channels),
         "out.weight": r(3, channels), "out.bias": r(3)}
    for i in range(len(dilations)):
        w["norms.%d.weight" % i], w["norms.%d.bias" % i] = 1 + r(channels), r(channels)
        w["convs.%d.weight" % i], w["convs.%d.bias" % i] = r(channels, channels, 3), r(channels)
        w["mix.%d.weight" % i], w["mix.%d.bias" % i] = r(channels, channels), r(channels)
    return w


def test_inputs_have_the_documented_shapes_and_no_nan():
    tr = _match()
    tr["ball"] = tr["ball"].copy()
    tr["ball"][50:120] = np.nan                      # a ball gap
    tr["players"] = tr["players"].copy()
    tr["players"][30:60, 1] = np.nan                 # a player lost for a while
    inp = neural.inputs(tr)
    n = len(tr["ball"])
    assert inp["players"].shape == (n, 2, neural.K_PLAYERS, len(neural.PLAYER_FEATURES))
    assert inp["globals"].shape == (n, len(inp["names"]))
    assert np.isfinite(inp["players"]).all() and np.isfinite(inp["globals"]).all()
    assert inp["mask"][0].sum() == 6 and inp["mask"][40].sum() == 5
    dist = inp["players"][100, 0, :3, neural.PLAYER_FEATURES.index("dist")]
    assert np.all(np.diff(dist) >= 0)                # nearest first


def test_swapping_the_teams_flips_home_and_keeps_dead():
    tr = _match()
    swapped = dict(tr, team_of_slot=1 - tr["team_of_slot"])
    a, b = neural.inputs(tr), neural.inputs(swapped)
    perm, sign = neural.global_swap(a["names"])
    p, mask, g = neural.swap_inputs(a["players"], a["mask"], a["globals"], perm, sign)
    assert np.allclose(p, b["players"], atol=1e-5) and np.array_equal(mask, b["mask"])
    assert np.allclose(g, b["globals"], atol=1e-5)
    w = _weights(a["names"])
    for mirror in (False, True):
        dead, team, change = neural.apply(w, a, mirror)
        dead_s, team_s, change_s = neural.apply(w, b, mirror)
        assert np.allclose(dead, dead_s, atol=2e-3) and np.allclose(team, -team_s, atol=2e-3)
        assert np.allclose(change, change_s, atol=2e-3)
        assert np.abs(team).max() > 0


def test_convolution_matches_a_direct_sum():
    rng = np.random.default_rng(1)
    x, weight, bias = rng.normal(size=(20, 3)), rng.normal(size=(4, 3, 3)), rng.normal(size=4)
    got = neural._conv(x, weight, bias, 2)
    pad = np.vstack([np.zeros((2, 3)), x, np.zeros((2, 3))])
    want = np.array([sum(weight[:, :, k] @ pad[t + 2 * k] for k in range(3)) + bias
                     for t in range(20)])
    assert np.allclose(got, want)


def test_weights_survive_packing_and_the_build_is_repeatable():
    tr = _match()
    inp = neural.inputs(tr)
    w = _weights(inp["names"])
    again = neural.unpack(eval(repr(neural.pack(w))))
    assert all(np.array_equal(w[k], again[k]) for k in w)
    assert np.array_equal(neural.apply(w, inp)[1], neural.apply(again, inp)[1])


def test_log_odds_takes_home_from_the_network_and_dead_from_the_trees():
    tr = _match()
    c = evidence.clues(tr)
    w = _weights(neural.global_names(c))
    def stump(names):
        return {"names": names, "bias": 0.5, "weights": [0.0] * len(names)}
    dead_model = {"motion": stump(evidence.motion_design(c)[0]),
                  "ball": stump(evidence.ball_design(c)[0])}
    dead, team = evidence.log_odds(c, dead_model, {"design": "neural", "w": w})
    assert np.array_equal(team, neural.apply(w, neural.inputs(tr, c))[1])
    assert np.allclose(dead, 0.5)
    assert np.array_equal(c["change_lo"], neural.apply(w, neural.inputs(tr, c))[2])


def _old_apply(w, inp):
    """neural.apply as it was before ensembles and mirroring (dead and home only)."""
    perm, sign = neural.global_swap(list(inp["names"]))
    n = len(inp["globals"])
    dead, team = np.zeros(n), np.zeros(n)
    for _, first, last in inp["half_blocks"]:
        s = slice(first, last + 1)
        p, mask = inp["players"][s], inp["mask"][s]
        g = inp["globals"][s] * w["gscale"]
        a = neural.forward(w, p, mask, g)
        b = neural.forward(w, *neural.swap_inputs(p, mask, g, perm, sign))
        dead[s] = (a[:, 0] + b[:, 0]) / 2.0
        team[s] = (a[:, 1] - b[:, 1]) / 2.0
    return np.round(dead, neural.ROUND), np.round(team, neural.ROUND)


def mirrored(tr):
    """The same match seen from the other touchline (y -> width - y)."""
    out = dict(tr, players=tr["players"].copy(), ball=tr["ball"].copy())
    out["players"][..., 1] = tr["pitch"][1] - out["players"][..., 1]
    out["ball"][:, 1] = tr["pitch"][1] - out["ball"][:, 1]
    return out


def test_one_net_without_mirror_is_the_old_formula_exactly():
    inp = neural.inputs(_match())
    w = _weights(inp["names"])
    dead, team, change = neural.apply(w, inp)
    old_dead, old_team = _old_apply(w, inp)
    assert np.array_equal(dead, old_dead) and np.array_equal(team, old_team)
    assert np.abs(change).max() > 0
    assert all(np.array_equal(x, y) for x, y in zip(neural.apply([w], inp, False),
                                                     (dead, team, change)))


def test_an_ensemble_of_two_copies_is_the_net():
    inp = neural.inputs(_match())
    w = _weights(inp["names"])
    for mirror in (False, True):
        one, two = neural.apply(w, inp, mirror), neural.apply([w, dict(w)], inp, mirror)
        assert all(np.array_equal(x, y) for x, y in zip(one, two))
    other = _weights(inp["names"], seed=5)
    mean = np.mean([neural.apply(x, inp, True)[1] for x in (w, other)], axis=0)
    assert np.allclose(neural.apply([w, other], inp, True)[1], mean, atol=2e-3)


def test_mirror_averaging_makes_the_outputs_pitch_symmetric():
    tr = _match()
    a, b = neural.inputs(tr), neural.inputs(mirrored(tr))
    w = _weights(a["names"])
    plain = [neural.apply(w, a)[k] - neural.apply(w, b)[k] for k in range(3)]
    assert max(np.abs(x).max() for x in plain) > 0.005         # the net alone is not symmetric
    for got, want in zip(neural.apply(w, b, True), neural.apply(w, a, True)):
        assert np.allclose(got, want, atol=2e-3)


def test_packed_ensembles_and_the_old_single_net_both_load(monkeypatch):
    import types

    import espadas
    inp = neural.inputs(_match())
    w1, w2 = _weights(inp["names"]), _weights(inp["names"], seed=1)
    packed = eval(repr(neural.pack_many([w1, w2], True)))
    assert packed["mirror"] is True and len(packed["nets"]) == 2
    for value, count, mirror in ((packed, 2, True), (neural.pack(w1), 1, False)):
        fake = types.SimpleNamespace(PACKED=value)
        monkeypatch.setitem(sys.modules, "espadas.neural_weights", fake)
        monkeypatch.setattr(espadas, "neural_weights", fake, raising=False)
        monkeypatch.setattr(neural, "_CACHE", {})
        ws, flag = neural.weights()
        assert len(ws) == count and flag is mirror
        assert np.array_equal(ws[0]["p1.weight"], w1["p1.weight"])
    assert neural.weights({"w": w1}) == ([w1], False)
    assert neural.weights({"w": [w1, w2], "mirror": True}) == ([w1, w2], True)
