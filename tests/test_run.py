"""`python -m espadas.run` end to end on a small made-up bundle (tests/conftest.py).

The X policy contract: abstain.PARAMS[labeler] alone decides what the no-flag build does;
--abstain forces the policy on (abstain.SETTING when PARAMS[labeler] is None), --no-abstain forces
it off. No match data is used.
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from espadas import abstain, atd, config, labelers, run, write  # noqa: E402
from conftest import read_labels  # noqa: E402

GRADED = config.half_of_frame() > 0
DEFAULT = run.DEFAULT_LABELER          # the no-flag build's labeler, whatever it is


def graded_x(path):
    return int(np.sum((read_labels(path) == "X") & GRADED))


def same_bytes(a, b):
    with open(a, "rb") as x, open(b, "rb") as y:
        return x.read() == y.read()


@pytest.fixture
def params_none(monkeypatch):
    """Today's shipped state, made explicit: no labeler has an X setting."""
    for name in list(abstain.PARAMS):
        monkeypatch.setitem(abstain.PARAMS, name, None)


# --------------------------------------------------------------------------- the default build

def test_default_labeler_is_v1(build, monkeypatch):
    # Intentional tripwire: it fails when the default changes. Update it (and the README's
    # "(labeler v1)" line) at the freeze, when the selected variant becomes the default (11.1).
    called = []
    original = labelers.LABELERS["v1"]

    def spy(tracking):
        called.append("v1")
        return original(tracking)

    monkeypatch.setitem(labelers.LABELERS, "v1", spy)
    build("--no-cache")
    assert called == ["v1"]


def test_x_is_off_by_default_and_the_build_is_labeler_plus_finalise(build, bundle, tmp_path,
                                                                    monkeypatch, params_none):
    def no_extra_work(*args, **kwargs):
        raise AssertionError("the default build must not compute the X policy's features")

    monkeypatch.setattr(run.v2, "features", no_extra_work)
    path = build("--no-cache")
    assert graded_x(path) == 0
    # byte for byte what the labeler plus finalise() give, nothing else
    tracking = atd.load_tracking(bundle, use_cache=False)
    expected = write.write_csv(str(tmp_path / "expected.csv"),
                               write.finalise(labelers.LABELERS[DEFAULT](tracking)))
    assert same_bytes(path, expected)


# --------------------------------------------------------------------------- switching X on / off

def test_abstain_flag_alone_gives_x(build, params_none):
    assert graded_x(build("--no-cache", "--abstain")) > 0


def test_params_alone_gives_x_and_matches_the_flag(build, monkeypatch, params_none):
    flag = build("--no-cache", "--abstain", out="flag.csv")
    monkeypatch.setitem(abstain.PARAMS, DEFAULT, abstain.SETTING)
    params = build("--no-cache", out="params.csv")
    assert graded_x(params) > 0
    assert same_bytes(flag, params)          # what selection scores as "X on" is what ships


def test_no_abstain_wins(build, monkeypatch, params_none):
    off = build("--no-cache", out="off.csv")
    monkeypatch.setitem(abstain.PARAMS, DEFAULT, abstain.SETTING)
    assert same_bytes(build("--no-cache", "--no-abstain", out="a.csv"), off)
    assert same_bytes(build("--no-cache", "--abstain", "--no-abstain", out="b.csv"), off)


def test_x_only_inside_the_region_the_policy_allows(build, params_none):
    off = read_labels(build("--no-cache", out="off.csv"))
    on = read_labels(build("--no-cache", "--abstain", out="on.csv"))
    added = (on == "X") & (off != "X")
    assert added.any()
    assert np.array_equal(on[~added], off[~added])           # X is the only change
    assert not np.any(added & ~GRADED)
    assert not np.any(added & ~abstain.allowed(off, abstain.SETTING["edge"]))


# --------------------------------------------------------------------------- the shared rule

@pytest.mark.parametrize("name", sorted(labelers.LABELERS))
def test_resolve_follows_params_and_the_flags(name, monkeypatch):
    own = {"weak": 1.0, "min_gap": 50, "edge": 3}
    monkeypatch.setitem(abstain.PARAMS, name, None)
    assert abstain.resolve(name) is None
    assert abstain.resolve(name, force=True) == abstain.SETTING
    assert abstain.resolve(name, force=False) is None
    monkeypatch.setitem(abstain.PARAMS, name, own)
    assert abstain.resolve(name) == own
    assert abstain.resolve(name, force=True) == own
    assert abstain.resolve(name, force=False) is None
