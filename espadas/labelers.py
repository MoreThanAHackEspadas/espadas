"""Labelers: tracking arrays -> per-frame labels.

v0 is a re-implementation of the organisers' published baseline (starter/baseline.py, MIT):
nearest tracked player to the ball, then a centred majority vote of +/- 50 frames. It exists so
the one-command pipeline is end-to-end from day one and so we can check our loader against the
published baseline_smooth50.csv. The real pipeline (clues -> touches -> HMM) replaces it.
"""

import numpy as np

HOME, AWAY, NOBODY = 0, 1, -1


def nearest_team(players, ball, team_of_slot):
    """int8 per frame: team of the nearest tracked player to the ball, NOBODY if no ball."""
    n_frames = ball.shape[0]
    out = np.full(n_frames, NOBODY, dtype=np.int8)
    seen = ~np.isnan(ball[:, 0])
    if not seen.any():
        return out

    diff = players[seen] - ball[seen, None, :]                  # (n_seen, n_slots, 2)
    dist = np.einsum("fsk,fsk->fs", diff, diff)                 # squared metres
    dist = np.where(np.isnan(dist), np.inf, dist)
    has_player = np.isfinite(dist).any(axis=1)
    nearest = np.argmin(dist, axis=1)                           # first slot wins ties

    idx = np.flatnonzero(seen)
    out[idx[has_player]] = team_of_slot[nearest[has_player]]
    return out


def majority_smooth(codes, window):
    """'H'/'A'/'D' per frame by centred majority vote of +/- window frames; ties go home."""
    if window <= 0:
        return np.where(codes == HOME, "H", np.where(codes == AWAY, "A", "D"))

    home_c = np.concatenate(([0], np.cumsum(codes == HOME)))
    away_c = np.concatenate(([0], np.cumsum(codes == AWAY)))
    index = np.arange(len(codes))
    low = np.maximum(0, index - window)
    high = np.minimum(len(codes), index + window + 1)
    home = home_c[high] - home_c[low]
    away = away_c[high] - away_c[low]

    labels = np.full(len(codes), "D", dtype="<U1")
    seen = (home + away) > 0
    labels[seen] = np.where(home[seen] >= away[seen], "H", "A")
    return labels


def v0_baseline(tracking, window=50):
    """The published baseline_smooth50, from our own loader."""
    codes = nearest_team(tracking["players"], tracking["ball"], tracking["team_of_slot"])
    return majority_smooth(codes, window)


LABELERS = {"v0": v0_baseline}
