"""Shared synthetic fixtures. Every number here is made up: no match data of any kind is read."""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from espadas import config  # noqa: E402

LIVE_FRAMES = 3000          # frames written after each kickoff; the rest of the match is absent
BALL_GAP = (1000, 1400)     # frames after each kickoff with no ball while the players keep running


def _group(x, y):
    length, width = config.PITCH
    return "{:.6f},{:.6f},1.0".format(x / length, y / width)


def write_bundle(root, ball_x_offset=0.5):
    """A made-up data bundle in the organisers' layout and ATD text format.

    tactical/BAR-ATH_tactical_atd.xml (2 home, 2 away track slots) and _atd.txt: around each
    kickoff the four players run small circles in their own half (attacking directions swap at
    halftime), the ball `ball_x_offset` m from one of them, except for BALL_GAP. Every other frame
    of the full-length match is absent from the file (all NaN), so it parses in a fraction of a
    second but run.main can still build and validate a full submission from it.
    """
    tactical = os.path.join(root, "tactical")
    os.makedirs(tactical, exist_ok=True)
    with open(os.path.join(tactical, "BAR-ATH_tactical_atd.xml"), "w") as handle:
        handle.write('<Root><Teams><Team id="h"><Name>Home</Name></Team>'
                     '<Team id="a"><Name>Away</Name></Team></Teams><Players>'
                     '<Player teamId="h"/><Player teamId="h"/>'
                     '<Player teamId="a"/><Player teamId="a"/></Players></Root>')
    length, width = config.PITCH
    lines = []
    for kickoff, home_x, carrier in ((config.KICKOFFS[0], 30.0, 0),
                                     (config.KICKOFFS[1], length - 30.0, 2)):
        centres = (home_x, home_x - 10.0, length - home_x, length - home_x + 10.0)
        for frame in range(kickoff - 30, kickoff + LIVE_FRAMES):
            t = (frame - kickoff) / config.FPS
            pos = [(cx + 5.0 * np.cos(t + s), width / 2 + 5.0 * np.sin(t + s))
                   for s, cx in enumerate(centres)]
            if BALL_GAP[0] <= frame - kickoff < BALL_GAP[1]:
                ball = "NaN,NaN,NaN"
            else:
                ball = _group(pos[carrier][0] + ball_x_offset, pos[carrier][1])
            lines.append("{}:{}:{}\n".format(frame + 1, ";".join(_group(*p) for p in pos), ball))
    with open(os.path.join(tactical, "BAR-ATH_tactical_atd.txt"), "w") as handle:
        handle.writelines(lines)
    return root


def read_labels(path):
    """The label column of a submission CSV, as a <U1 array."""
    with open(path, encoding="utf-8") as handle:
        next(handle)
        return np.array([line.rstrip("\n").split(",")[1] for line in handle], dtype="<U1")


@pytest.fixture
def bundle(tmp_path):
    return write_bundle(str(tmp_path / "bundle"))


@pytest.fixture
def build(bundle, tmp_path, monkeypatch):
    """build(*argv, out=name) runs `python -m espadas.run` on the synthetic bundle (cache in
    tmp_path, never the repository's) and returns the path of the CSV it wrote."""
    monkeypatch.setattr(config, "CACHE_DIR", str(tmp_path / "cache"))
    from espadas import run

    def go(*argv, out="espadas.csv"):
        path = str(tmp_path / "out" / out)
        assert run.main(["--data-dir", bundle, "--out", path] + list(argv)) == 0
        return path

    return go
