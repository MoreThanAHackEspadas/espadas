"""Step 5: final rules, write the CSV, validate it with the organisers' own checks."""

import hashlib
import os

import numpy as np

from . import config

pm = config.pm


def finalise(labels, abstain_ungraded=True):
    """Apply match-level rules that hold for any labeler.

    - Second half pre-kickoff window (documented by the organisers) is always D.
    - Ungraded frames (before kickoff, halftime, after the whistle) become X: never scored, and X
      makes it obvious in the file that we are not claiming anything there.
    """
    labels = np.asarray(labels, dtype="<U1").copy()
    if labels.shape != (config.N_FRAMES,):
        raise ValueError("expected {} labels, got {}".format(config.N_FRAMES, labels.shape))

    start, end = config.PRE_KICKOFF_DEAD
    labels[start:end + 1] = "D"

    if abstain_ungraded:
        graded = np.zeros(config.N_FRAMES, dtype=bool)
        for _, first, last in config.HALF_BLOCKS:
            graded[first:last + 1] = True
        labels[~graded] = "X"
    return labels


def write_csv(path, labels):
    """frame,label CSV with '\\n' line endings on every OS, so the bytes are reproducible."""
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    body = "frame,label\n" + "".join("{},{}\n".format(i, l) for i, l in enumerate(labels))
    with open(path, "w", newline="\n", encoding="utf-8") as handle:
        handle.write(body)
    return path


def validate(path, n_expected=config.N_FRAMES):
    """None if the file passes the official checks, else the official one-line reason."""
    return pm.validate_submission(pm.read_label_csv(path), n_expected)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
