"""Generic Viterbi, on hand-checked sequences."""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from espadas import hmm  # noqa: E402

NEG = -np.inf


def test_viterbi_follows_emissions_when_transitions_are_free():
    em = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    free = np.log(np.full((2, 2), 0.5))
    assert hmm.viterbi(em, free, np.log([0.5, 0.5])).tolist() == [0, 1, 1, 0]


def test_viterbi_routes_through_the_bridge_state_when_the_direct_move_is_forbidden():
    em = np.array([[5.0, -10.0, -10.0], [5.0, -10.0, -10.0], [0.0, 0.0, 5.0],
                   [0.0, -10.0, 5.0], [0.0, -10.0, 5.0]])
    trans = np.array([[np.log(0.9), np.log(0.1), NEG],
                      [NEG, np.log(0.5), np.log(0.5)],
                      [NEG, NEG, 0.0]])
    start = np.array([0.0, NEG, NEG])
    assert hmm.viterbi(em, trans, start).tolist() == [0, 0, 1, 2, 2]


def test_viterbi_ties_go_to_the_lowest_state_index_and_start_is_respected():
    em = np.zeros((4, 2))
    free = np.log(np.full((2, 2), 0.5))
    assert hmm.viterbi(em, free, np.log([0.5, 0.5])).tolist() == [0, 0, 0, 0]
    assert hmm.viterbi(em, free, np.array([NEG, 0.0])).tolist() == [1, 0, 0, 0]


def test_viterbi_single_frame():
    assert hmm.viterbi(np.array([[0.0, 2.0, 1.0]]), np.zeros((3, 3)),
                       np.array([0.0, 0.0, 0.0])).tolist() == [1]


def test_viterbi_matches_brute_force_and_is_deterministic():
    rng = np.random.default_rng(0)
    em = rng.normal(size=(6, 3))
    trans = np.log(rng.dirichlet(np.ones(3), size=3))
    start = np.log(rng.dirichlet(np.ones(3)))
    best, best_score = None, NEG
    for path in itertools.product(range(3), repeat=6):
        score = start[path[0]] + em[0, path[0]]
        for t in range(1, 6):
            score += trans[path[t - 1], path[t]] + em[t, path[t]]
        if score > best_score:
            best, best_score = list(path), score
    first = hmm.viterbi(em, trans, start)
    assert first.tolist() == best
    assert np.array_equal(first, hmm.viterbi(em, trans, start))
