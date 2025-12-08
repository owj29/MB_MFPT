import numpy as np


def generate_bins_MB(n_bins, dim=2):

    """
    Method to generate the bin boundaries for the MSM generation. Defined for 1 particle, 2 dimensions, on the Muller Brown Potential

    """

    MB_bounds = [
        (-1.5, 1.2),  # x
        (-0.2,  2.0)]  # y
    
    bin_edges = []

    for d_coord in range(dim):
        low, high = MB_bounds[d_coord]
        edges = np.linspace(low, high, n_bins + 1)
        bin_edges.append(edges)

    return bin_edges


