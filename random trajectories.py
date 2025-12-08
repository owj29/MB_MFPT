import numpy as np
import matplotlib.pyplot as plt


def build_MSM(trajectories, n_bins, lag_time, dt):
    """
    Build MSM from tensor containing ALL simulations and ALL particles.

    trajectories[t, s, p, d]:
        t = time
        s = simulation index
        p = particle index
        d = spatial dimension
    """

    T, S, P, dim = trajectories.shape
    D = P * dim # system dimension per frame
    lag_steps = int(lag_time / dt)

    # make bins
    flat = trajectories.reshape(T * S, D)
    bin_edges = []
    for d in range(D):
        low  = flat[:, d].min()
        high = flat[:, d].max()
        edges = np.linspace(low, high, n_bins + 1)
        bin_edges.append(edges)

    # multipliers for flattening the many dimension bins into 1D index
    multipliers = (n_bins ** np.arange(D)[::-1]).astype(int)
    n_states = n_bins ** D

    all_start = [] # start coordinates for a given trajectory
    all_final = [] # final coordinates for a given trajectory

    # iterate over all the simulations and particels to get all the trajectories 
    for s in range(S):

        # trajectory for simulation s: shape (T, P, dim)
        traj = trajectories[:, s, :, :]  # (T, P, dim)
        traj = traj.reshape(T, D)        # (T, P*dim)

        # storage for system bins across time
        bins = np.zeros((T, D), dtype=int)
        valid = np.ones(T, dtype=bool)

        # digitize each coordinate dimension 
        for d in range(D):
            bd = np.digitize(traj[:, d], bin_edges[d]) - 1
            bins[:, d] = bd
            valid &= (bd >= 0) & (bd < n_bins)
            
        pair_valid = valid[:-lag_steps] & valid[lag_steps:]

        start_bins = bins[:-lag_steps][pair_valid]
        final_bins = bins[lag_steps:][pair_valid]

        # Convert d bin tuples into state indices
        start_idx = (start_bins * multipliers).sum(axis=1)
        final_idx = (final_bins * multipliers).sum(axis=1)

        all_start.append(start_idx)
        all_final.append(final_idx)



    # concatenate all the start/final indices 
    all_start = np.concatenate(all_start)
    all_final = np.concatenate(all_final)

    print(f"Collected total pairs: {all_start.size}")

    # construct the actual transition matrix
    edges = np.arange(n_states + 1)
    transitions, _, _ = np.histogram2d(all_start, all_final,
                                       bins=(edges, edges))

    transition_matrix = transitions / np.maximum(
        transitions.sum(axis=1, keepdims=True), 1e-10
    )

    return transition_matrix


def generate_random_trajectories(steps, write_every, num_simulations,
                                 num_particles=1, dimension=2,
                                 seed=None):
    """
    Generate a random trajectory tensor with shape:
    (num_time_steps, num_simulations, num_particles, dimension)

    Parameters
    ----------
    steps : int
        Total number of MD steps (not counting step 0).
    write_every : int
        Interval between recorded frames.
    num_simulations : int
        Number of parallel simulations.
    num_particles : int
        Number of particles per simulation (default: 2)
    dimension : int
        Spatial dimension (default: 2)
    seed : int or None
        Random seed for reproducibility

    Returns
    -------
    positions : np.ndarray
        Array of shape (T, S, P, D)
    """

    if seed is not None:
        np.random.seed(seed)

    # Number of saved time steps (including t=0)
    num_time_steps = steps // write_every + 1

    # Generate random positions
    positions = np.random.randn(num_time_steps,
                                num_simulations,
                                num_particles,
                                dimension)

    return positions


# Example usage
positions = generate_random_trajectories(
    steps=10000,
    write_every=10,
    num_simulations=5,   # for example
    num_particles=1,
    dimension=2,
    seed=1
)

msm = build_MSM(positions, 10, 0.1, 0.05)

plt.figure(figsize=(6, 5))
plt.imshow(msm, cmap='viridis', interpolation='nearest')
plt.colorbar(label="Probability")
plt.title('thing')
plt.xlabel("To state")
plt.ylabel("From state")

# Ensure equal aspect for square matrix
plt.gca().set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.show()
