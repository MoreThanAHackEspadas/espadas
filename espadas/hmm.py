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


def expand(log_trans, log_start, mins):
    """Chain each state so it cannot be left before ``mins[s]`` frames.

    State s becomes ``mins[s]`` sub-states. It is entered at its first link,
    moves along the chain by forced steps (log 0), and the last link carries
    the original row of s: its self-loop stays on the last link and its exits
    go to the first link of the target. Everything else is ``-inf``.
    Returns ``(log_trans_x, log_start_x, macro)``; ``macro`` is an int64 array
    mapping each expanded state to its original index, so a caller runs
    ``viterbi(emissions[:, macro], tx, sx)`` and maps back with ``macro[path]``.
    """
    log_trans = np.asarray(log_trans, dtype=np.float64)
    log_start = np.asarray(log_start, dtype=np.float64)
    mins = np.asarray(mins, dtype=np.int64)
    S = len(mins)
    first = np.cumsum(mins) - mins  # index of each state's first link
    last = first + mins - 1
    macro = np.repeat(np.arange(S, dtype=np.int64), mins)
    n = int(mins.sum())
    tx = np.full((n, n), -np.inf)
    sx = np.full(n, -np.inf)
    sx[first] = log_start
    for s in range(S):
        for k in range(first[s], last[s]):
            tx[k, k + 1] = 0.0
        for j in range(S):
            tx[last[s], last[s] if j == s else first[j]] = log_trans[s, j]
    return tx, sx, macro
