# Calculate Markov State Model for Overdamped langevin dynamics on MB potential

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


def OD_Langevin_sim(r0, beta, m, dt, tf, V, rand_seed):
    grad_x = jax.grad(V, argnums=0)  # x component of grad
    grad_y = jax.grad(V, argnums=1)  # y component of grad

    @jax.jit # speeds calculations up so much!!
    def force(x, y):
        return jnp.array([grad_x(x, y), grad_y(x, y)])

    # loop setup
    steps = int(tf / dt)
    r = r0

    # random noise parameters
    key = jax.random.PRNGKey(rand_seed)
    noise_prev = 0

    for i in range(steps - 1):

        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, (2,))  # generate random noise

        F_potential = force(r[0], r[1])

        r = r - F_potential*dt + jnp.sqrt(dt/(2*beta)) * (noise + noise_prev)

        noise_prev = noise
    return(r)


def Trial(init_r, n, m, beta, dt, tf, V, seed, x_edge, y_edge):

    x_start, y_start = [], []
    x_final, y_final = [], []

    for i, r in enumerate(init_r):
        for run in range(n):
            r_final = OD_Langevin_sim(r, beta, m, dt, tf, V, seed + i * n + run)

            # store initial and final positions
            x_start.append(r[0])
            y_start.append(r[1])
            x_final.append(r_final[0])
            y_final.append(r_final[1])

        print(f"Trajectories from {r} complete.")

    # compute which bin each coordinate falls into
    x_bin_start = np.digitize(x_start, x_edge) - 1
    y_bin_start = np.digitize(y_start, y_edge) - 1
    x_bin_final = np.digitize(x_final, x_edge) - 1
    y_bin_final = np.digitize(y_final, y_edge) - 1

    # only keep points that landed in bins (not outside of the domain!)
    valid = ((x_bin_start >= 0) & (x_bin_start < 10) &
            (y_bin_start >= 0) & (y_bin_start < 10) &
            (x_bin_final >= 0) & (x_bin_final < 10) &
            (y_bin_final >= 0) & (y_bin_final < 10))

    x_bin_start = x_bin_start[valid]
    y_bin_start = y_bin_start[valid]
    x_bin_final = x_bin_final[valid]
    y_bin_final = y_bin_final[valid]

    # map 2D (i,j) bins to single state index from 0–99
    start_index = y_bin_start * 10 + x_bin_start
    final_index = y_bin_final * 10 + x_bin_final

    n_bins = 10

    # count transitions
    transitions, _, _ = np.histogram2d(start_index, final_index, bins=(np.arange(n_bins * n_bins + 1), np.arange(n_bins * n_bins + 1)))

    # turn the counts to the probabilies
    transition_matrix = transitions / np.maximum(transitions.sum(axis=1, keepdims=True), 1e-10)

    return transition_matrix


# simulation parameters
dt = 0.001 # time step
tf = 0.025 #final trajectory time
seed = 1 # rng seed

# initial state / params
m = 1.0 # mass
beta = 1.0 # thermodynamic beta = 1/kBT, Rare events paper uses 1.0–– high values have basically no random transitions
n = 250


x_edge = jnp.linspace(-1.5, 1.0, 11)
y_edge = jnp.linspace(-0.5, 2.0, 11)

x_centers = 0.5 * (x_edge[:-1] + x_edge[1:])
y_centers = 0.5 * (y_edge[:-1] + y_edge[1:])

X, Y = np.meshgrid(x_centers, y_centers)
positions = np.column_stack((X.ravel(), Y.ravel()))

T = Trial(positions, n=n, m=m, beta=beta, dt=dt, tf=tf, V=V, seed=seed, x_edge=x_edge, y_edge=y_edge)

# plotting:

#region
nx, ny = 10, 10

# make plot labels for bin indexes
labels = [f"({i},{j})" for j in range(ny) for i in range(nx)]

plt.figure(figsize=(8,6))
plt.imshow(T, origin='lower', cmap='plasma')
plt.colorbar(label='Transition Probability')
plt.title(f'MSM Transition Matrix for t={tf}, n={n}')

# put labels on axes
plt.xticks(ticks=np.arange(100), labels=labels, rotation=90, fontsize=6)
plt.yticks(ticks=np.arange(100), labels=labels, fontsize=6)

plt.xlabel('End Bin (x, y)')
plt.ylabel('Start Bin (x, y)')
plt.tight_layout()
plt.show()
#endregion
