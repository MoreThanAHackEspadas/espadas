"""tools/tune.py: the parallel objective gives exactly the serial numbers, in order."""

import os
import sys
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from espadas import config  # noqa: E402
import tune  # noqa: E402


def fake_module():
    m = types.ModuleType("fake")

    def decode(f, params):
        lab = np.array(list(f["truth"]))
        lab[: params["flip"]] = "A"
        return lab
    m.decode = decode
    return m


def test_parallel_equals_serial():
    module = fake_module()
    truths = ["HHHHHAAAAADDDHHHH", "AAAHHHHHDDDDAAAAA", "HHAAHHAADDHHAAHHA"]
    games = [("g{}".format(i), config.pm.label_codes(np.array(list(t))), {"truth": t},
              np.ones(len(t), dtype=bool)) for i, t in enumerate(truths)]
    for flip in (0, 3, 7):
        serial = tune.objective(module, games, {"flip": flip})
        with tune.parallel(module, games, 2):
            par = tune.objective(module, games, {"flip": flip})
        assert serial == par
    assert tune._SHARED == {}
