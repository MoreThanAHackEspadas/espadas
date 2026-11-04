"""Generic Viterbi decoder: the maximum-probability state path, in log space.

It knows nothing about football. A caller builds per-frame log emissions and a
log transition matrix (``-inf`` for forbidden moves) and gets back one state
index per frame. Ties go to the lowest state index, because ``np.argmax``
returns the first maximum.
"""
import numpy as np


def viterbi(emissions, log_trans, log_start):
    """Most likely state path.

    emissions: (T, S) log emission scores, T >= 1.
    log_trans: (S, S), ``log_trans[i, j]`` is the score of moving from i to j.
    log_start: (S,) log initial scores.
    Returns an int64 array of length T of state indices.
    """
    emissions = np.asarray(emissions, dtype=np.float64)
    T, S = emissions.shape
    back = np.zeros((T, S), dtype=np.int8 if S <= 127 else np.int64)
    states = np.arange(S)
    with np.errstate(invalid="ignore"):  # -inf + -inf must not warn
        delta = np.asarray(log_start, dtype=np.float64) + emissions[0]
        for t in range(1, T):
            score = delta[:, None] + log_trans
            back[t] = np.argmax(score, axis=0)
            delta = score[back[t], states] + emissions[t]
    path = np.empty(T, dtype=np.int64)
    path[-1] = np.argmax(delta)
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path
