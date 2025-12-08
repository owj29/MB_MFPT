import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt


def V(x, y): # generate muller-brown potential
    # potential parameters from "Predicting rare events using neural networks and
    # short-trajectory data"
    C = [-200, -100, -170, 15]
    a_param = [-1, -1, -6.5, 0.7]
    b_param = [0, 0, 11, 0.6]
    c_param = [-10, -10, -6.5, 0.7]
    x_0 = [1, -0.27, -0.5, -1]
    y_0 = [0, 0.5, 1.5, 1.0]

    V_r = 0
    for Ck, ak, bk, ck, x_0k, y_0k in zip(C, a_param, b_param, c_param, x_0, y_0):
        V_r += 0.05*Ck*jnp.exp(ak*(x-x_0k)**2 + bk*(x-x_0k)*(y-y_0k) + ck*(y-y_0k)**2)
    return(V_r)


n_bins = 10

x_edge = jnp.linspace(-1.5, 1.0, n_bins + 1)
y_edge = jnp.linspace(-0.5, 2.0, n_bins + 1)

x_centers = 0.5 * (x_edge[:-1] + x_edge[1:])
y_centers = 0.5 * (y_edge[:-1] + y_edge[1:])

T = np.loadtxt("/Users/student/Desktop/transition_matrix_0.005_lag.csv", delimiter=",")

zero_rows = np.where(np.isclose(T.sum(axis=1), 0))[0]
print("Zero rows (unvisited states):", zero_rows)
print("Number of zero rows:", len(zero_rows))


def stationary_power(P, tol=1e-12, max_iters=200000):
    """
    computes the stationary distribution for the transition matrix
    using the power method on the left eigenvector:  pi = pi @ T.
    """
    n = P.shape[0]

    # start from a uniform distribution
    pi = np.ones(n) / n

    for k in range(max_iters):
        pi_new = pi @ P
        if np.linalg.norm(pi_new - pi, 1) < tol:
            return pi_new / pi_new.sum()   # normalize just in case
        pi = pi_new

    print("Warning: power method did not converge.")
    return pi / pi.sum()


equilibrium_dist = stationary_power(T)
probs = np.asarray(equilibrium_dist).reshape(-1)


E_grid = np.zeros((n_bins, n_bins))
for i, xc in enumerate(x_centers):
    for j, yc in enumerate(y_centers):
        E_grid[i, j] = float(V(xc, yc))

# Flatten energies in row-major order (same as your labels/MSM indexing)
E_flat = E_grid.flatten()   # shape (100,)

order = np.argsort(E_flat)      # indices of states in increasing E

# Reordered energies and probabilities
E_sorted = E_flat[order]
probs_sorted = probs[order]
labels_sorted = [f"({i},{j})" for i in range(n_bins) for j in range(n_bins)]
labels_sorted = np.array(labels_sorted)[order]

plt.figure(figsize=(10,4))
plt.bar(np.arange(len(probs_sorted)), probs_sorted)
plt.xlabel("State index (sorted by potential energy)")
plt.ylabel("Equilibrium probability")
plt.title("Equilibrium probabilities ordered by potential energy")
plt.tight_layout()
plt.show()


plt.figure(figsize=(8,4))
bin_idx = np.arange(probs.size)
plt.bar(bin_idx, probs)
plt.xlabel('Bin index (flattened, row-major)')
plt.ylabel('Equilibrium probability')
plt.title('Equilibrium probability per bin (flattened)')
plt.tight_layout()
plt.show()



labels = [f"({i},{j})" for i in range(n_bins) for j in range(n_bins)]
plt.figure(figsize=(max(8, 0.5*len(labels)), 4))  # widen if many labels
plt.bar(bin_idx, probs)
plt.xticks(bin_idx, labels, rotation=90, fontsize=8)
plt.xlabel('Bin (i,j)')
plt.ylabel('Equilibrium probability')
plt.title('Equilibrium probability per 2D bin (labeled)')
plt.tight_layout()
plt.show()


# plotting:

#region
# make plot labels for bin indexes
#labels = [f"({i},{j})" for j in range(n_bins) for i in range(n_bins)]

plt.imshow(equilibrium_dist.reshape(n_bins, n_bins),
           origin='lower',
           extent=[x_edge[0], x_edge[-1], y_edge[0], y_edge[-1]],
           cmap='viridis',
           aspect='auto')
plt.colorbar(label='Equilibrium Probability')
plt.title('MB potentil equilibrium distribution')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
