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
    r = r0.copy()
    positions = []
    positions.append(np.array(r))

    # random noise parameters
    key = jax.random.PRNGKey(rand_seed)
    noise_prev = 0

    for i in range(steps - 1):

        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, (2,))  # generate random noise

        F_potential = force(r[0], r[1])

        r = r - F_potential*dt + jnp.sqrt(dt/(2*beta)) * (noise + noise_prev)
        noise_prev = noise

        positions.append(np.array(r))
    return(np.vstack(positions))


def Trial(init_r, n, m, beta, dt, tf, V, seed, n_bins, lag_time, x_edge, y_edge):

    lag_steps = int(lag_time/dt)
    n_states = n_bins**2

    # we'll gather *all* start and end state indices across every trajectory
    all_start_indices = []
    all_final_indices = []

    for i, r in enumerate(init_r):
        for run in range(n):
            trajectory = OD_Langevin_sim(r, beta, m, dt, tf, V, seed + i * n + run)
            xs = trajectory[:, 0]
            ys = trajectory[:, 1]


            full_x_bin = np.digitize(xs, x_edge) - 1
            full_y_bin = np.digitize(ys, y_edge) - 1
            full_valid = ((full_x_bin >= 0) & (full_x_bin < n_bins) &
                          (full_y_bin >= 0) & (full_y_bin < n_bins))

            pair_valid = full_valid[:-lag_steps] & full_valid[lag_steps:]

            if np.any(pair_valid):
                start_x = full_x_bin[:-lag_steps][pair_valid]
                start_y = full_y_bin[:-lag_steps][pair_valid]
                final_x = full_x_bin[lag_steps:][pair_valid]
                final_y = full_y_bin[lag_steps:][pair_valid]

                start_idx = start_y * n_bins + start_x
                final_idx = final_y * n_bins + final_x

                all_start_indices.append(start_idx)
                all_final_indices.append(final_idx)
        print(f"Trajectories from {r} complete, processed {n} trajectories")

    all_start_indices = np.concatenate(all_start_indices)
    all_final_indices = np.concatenate(all_final_indices)

    # diagnostics
    print(f"Collected total pairs: {all_start_indices.size}")

    # build histogram and normalize
    edges = np.arange(n_states + 1)
    transitions, _, _ = np.histogram2d(all_start_indices, all_final_indices, bins=(edges, edges))
    transition_matrix = transitions / np.maximum(transitions.sum(axis=1, keepdims=True), 1e-10)

    return transition_matrix



# simulation parameters
dt = 0.001 # time step
tf = 0.02 # final trajectory time
lag_time = 0.01
seed = 1 # rng seed
n_bins = 10

# initial state / params
m = 1.0 # mass
beta = 1.0 # thermodynamic beta = 1/kBT, Rare events paper uses 1.0–– high values have basically no random transitions
n = 300


x_edge = jnp.linspace(-1.5, 1.0, n_bins + 1)
y_edge = jnp.linspace(-0.5, 2.0, n_bins + 1)

x_centers = 0.5 * (x_edge[:-1] + x_edge[1:])
y_centers = 0.5 * (y_edge[:-1] + y_edge[1:])

X, Y = np.meshgrid(x_centers, y_centers)
positions = np.column_stack((X.ravel(), Y.ravel()))

T = Trial(positions, n=n, m=m, beta=beta, dt=dt, tf=tf, V=V, seed=seed, n_bins=n_bins, lag_time=lag_time, x_edge=x_edge, y_edge=y_edge)
np.savetxt(f'/Users/student/Desktop/transition_matrix_{lag_time}_lag.csv', T, delimiter=',')

# plotting:

#region
# make plot labels for bin indexes
labels = [f"({i},{j})" for j in range(n_bins) for i in range(n_bins)]

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
