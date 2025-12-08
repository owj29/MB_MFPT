import numpy as np

def bin_trajectories_2d(trajectories, bin_edges, lag_steps):

    """
    Method to bin the trajectories inside a defined set of bin boundaries
    """

    T, S, P, dim = trajectories.shape
    D = P * dim

    # multipliers for flattening D-D bin tuple into an integer
    n_bins = len(bin_edges[0]) - 1
    multipliers = (n_bins ** np.arange(D)[::-1]).astype(int)

    all_start = []
    all_final = []

    for s in range(S):

        traj = trajectories[:, s, :, :].reshape(T, D)  # (T, D)

        bins = np.zeros((T, D), dtype=int)
        valid = np.ones(T, dtype=bool)

        # digitize each coordinate
        for d in range(D):
            bd = np.digitize(traj[:, d], bin_edges[d]) - 1
            bins[:, d] = bd
            valid &= (bd >= 0) & (bd < n_bins)

        # valid pairs (both endpoints valid)
        pair_valid = valid[:-lag_steps] & valid[lag_steps:]

        start_bins = bins[:-lag_steps][pair_valid]
        final_bins = bins[lag_steps:][pair_valid]

        # flatten bin tuples into state indices
        start_idx = (start_bins * multipliers).sum(axis=1)
        final_idx = (final_bins * multipliers).sum(axis=1)

        all_start.append(start_idx)
        all_final.append(final_idx)

    return (
        np.concatenate(all_start),
        np.concatenate(all_final)
    )



