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
    dead, team = neural.apply(w, a)
    dead_s, team_s = neural.apply(w, b)
    assert np.allclose(dead, dead_s, atol=2e-3) and np.allclose(team, -team_s, atol=2e-3)
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
