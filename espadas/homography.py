"""Step 1 input (optional): per-frame pitch -> pixel homography for the tactical video.

CSV, one row per video frame (see the organisers' starter/load_homography.py, MIT):

    frame,h00,h01,h02,h10,h11,h12,h20,h21,h22,reprojection_error,valid

The matrix maps *normalised* pitch coordinates (0-1, the ATD convention) to image pixels. Our
tracking arrays are in metres, so project_metres() divides by the pitch first. Where valid is 0
the matrix is NaN; reprojection_error is a soft trust signal, lower is better.

Only needed for video work (drawing tracking on frames, or a ball detector on the video).
"""

import numpy as np
import pandas as pd

from . import config

MATRIX_COLUMNS = ["h{}{}".format(row, col) for row in range(3) for col in range(3)]


def parse(path, n_frames=config.N_FRAMES):
    """(matrices (n_frames, 3, 3), reprojection_error (n_frames,), valid (n_frames,) bool).

    Rows are placed by their frame column; frames the file does not mention are NaN / invalid.
    """
    table = pd.read_csv(path)
    missing = [c for c in ["frame", "reprojection_error", "valid"] + MATRIX_COLUMNS
               if c not in table.columns]
    if missing:
        raise ValueError("{}: missing columns {}".format(path, missing))

    frames = table["frame"].to_numpy()
    keep = (frames >= 0) & (frames < n_frames)
    frames = frames[keep]
    matrices = np.full((n_frames, 3, 3), np.nan)
    error = np.full(n_frames, np.nan)
    valid = np.zeros(n_frames, dtype=bool)
    matrices[frames] = table[MATRIX_COLUMNS].to_numpy(np.float64)[keep].reshape(-1, 3, 3)
    error[frames] = table["reprojection_error"].to_numpy(np.float64)[keep]
    valid[frames] = table["valid"].to_numpy()[keep].astype(bool)
    return matrices, error, valid


def load(data_dir=None):
    """Homography for the tactical source."""
    (path,) = config.require(["homography"], data_dir)
    return parse(path)


def project(matrix, x, y):
    """Normalised pitch coordinates (0-1) -> image pixels. NaN matrix in, NaN out."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    w = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
    return ((matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]) / w,
            (matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]) / w)


def unproject(matrix, px, py):
    """Image pixels -> normalised pitch coordinates. The inverse of project()."""
    return project(np.linalg.inv(matrix), px, py)


def project_metres(matrix, x_m, y_m, pitch=config.PITCH):
    """Pitch coordinates in metres (our tracking arrays) -> image pixels."""
    length, width = pitch
    return project(matrix, np.asarray(x_m) / length, np.asarray(y_m) / width)
