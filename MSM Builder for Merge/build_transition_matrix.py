import numpy as np

def build_transition_matrix(all_start, all_final, n_states):
    edges = np.arange(n_states + 1)

    transitions, _, _ = np.histogram2d(
        all_start, all_final, bins=(edges, edges)
    )

    T = transitions / np.maximum(
        transitions.sum(axis=1, keepdims=True),
        1e-10)

    return T


