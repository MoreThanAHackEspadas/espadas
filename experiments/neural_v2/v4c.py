"""v4 decoder with a change bonus: every switch between two different states arriving at frame t
gets + w_change * sigmoid(change_lo[t]) (the network's log-odds that the reference label changes
there). With w_change == 0, or without f["change_lo"], it is v4.decode exactly."""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from espadas import v4                                # noqa: E402

NEG = v4.NEG
PARAMS = dict(v4.PARAMS, w_change=0.0)
GRID = dict(v4.GRID, w_change=[0.0, 2.0, 5.0, 10.0, 20.0, 40.0])
features = v4.features


def viterbi_bonus(emis, trans, start=None, bonus=None):
    """v4.viterbi, but a transition i -> j (i != j) into frame t adds bonus[t]."""
    emis = np.asarray(emis, dtype=float)
    n, s = emis.shape
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    trans = np.asarray(trans, dtype=float)
    preds = [[(i, float(trans[i, j]), i != j) for i in range(s) if np.isfinite(trans[i, j])]
             for j in range(s)]
    rows = emis.tolist()
    bon = [0.0] * n if bonus is None else [float(b) for b in bonus]
    score = rows[0] if start is None else [a + b for a, b in zip(rows[0], np.asarray(start,
                                                                                   float))]
    back = [None] * n
    states = range(s)
    for t in range(1, n):
        row, bt = rows[t], bon[t]
        new, ptr = [0.0] * s, [0] * s
        for j in states:
            best, arg = NEG, -1
            for i, cost, move in preds[j]:
                v = score[i] + cost + (bt if move else 0.0)
                if v > best:
                    best, arg = v, i
            new[j] = best + row[j]
            ptr[j] = arg
        score, back[t] = new, ptr
    path = np.empty(n, dtype=np.int8)
    state = max(states, key=lambda j: (score[j], -j))
    for t in range(n - 1, 0, -1):
        path[t] = state
        state = back[t][state]
    path[0] = state
    return path


def decode_states(f, params=None):
    p = dict(PARAMS, **(params or {}))
    if not p["w_change"] or f.get("change_lo") is None:
        return v4.decode_states(f, p)
    trans = v4.transitions(p)
    touch = v4.touch_list(f, p)
    bonus = p["w_change"] / (1.0 + np.exp(-np.clip(np.nan_to_num(f["change_lo"]), -50, 50)))
    out = np.full(len(f["present"]), -1, dtype=np.int8)
    for _, first, last in f["half_blocks"]:
        stop = last + 1
        path = viterbi_bonus(v4.emissions(f, p, first, stop, touch), trans, v4.START,
                             bonus[first:stop])
        out[first:stop] = v4.clean_up(path, int(p["min_dead"]), int(p["min_alive"]))
    return out


def decode(f, params=None):
    states = decode_states(f, params)
    labels = np.full(len(states), "D", dtype="<U1")
    inside = states >= 0
    labels[inside] = v4.STATE_LABEL[states[inside]]
    return labels
