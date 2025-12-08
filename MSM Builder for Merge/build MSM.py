import numpy as np
from generate_bins import generate_bins_MB
from bin_trajectories import bin_trajectories_2d
from build_transition_matrix import build_transition_matrix

def build_MSM(trajectories, n_bins, lag_time, dt):
    """
    Full MSM builder using the modular steps
    """

    T, S, P, dim = trajectories.shape
    lag_steps = int(lag_time / dt)

    # bin edges (Müller–Brown)
    bin_edges = generate_bins_MB(n_bins, num_particles=P, dim=dim)

    # discretize + extract pairs
    all_start, all_final, D = bin_trajectories_2d(trajectories, bin_edges, lag_steps)

    # total number of states
    n_states = n_bins ** D


    return build_transition_matrix(all_start, all_final, n_states)
