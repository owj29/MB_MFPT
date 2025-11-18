# Implements Overdamped langevin dynamics on the muller brown potential using Jax for JIT compiling

import numpy as np
import jax.numpy as jnp
import jax
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time

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

def in_A(r):
    if 6.5*(r[0]+0.5)**2 - 11*(r[0]+0.5)*(r[1]-1.5) + 6.5*(r[1]-1.5) < 0.3:
        return True

def in_B(r):
    if (r[0]-0.6)**2 + 5*(r[1]-0.02) < 0.2:
        return True


def OD_Langevin_sim(r0, beta, m, dt, tf, V, rand_seed):
    grad_x = jax.grad(V, argnums=0)  # x component of grad
    grad_y = jax.grad(V, argnums=1)  # y component of grad

    @jax.jit
    def force(x, y):
        return jnp.array([grad_x(x, y), grad_y(x, y)])

    # setup parameters
    steps = int(tf / dt)
    r = np.zeros((steps, 2))
    r[0] = np.array(r0)

    # random noise parameters
    key = jax.random.PRNGKey(rand_seed)
    noise_prev = 0

    for i in range(steps - 1):

        key, subkey = jax.random.split(key)
        noise = jax.random.normal(subkey, (2,))  # random noise

        F_potential = force(r[i, 0], r[i, 1]) # jnp.array([grad_x(r[i, 0], r[i, 1]), grad_y(r[i, 0], r[i, 1])])

        r[i+1] = r[i] - F_potential*dt + jnp.sqrt(dt/(2*beta)) * (noise+noise_prev)

        noise_prev = noise

        if i % 1000 == 0:
            print(f"Step {i} of {steps}")

    # plotting:
    # region
    x = jnp.linspace(-1.6, 1.0, 700)
    y = jnp.linspace(-0.5, 2.1, 700)
    X, Y = jnp.meshgrid(x, y)

    Z = V(X, Y)
    V_traj = jnp.array([V(r[i, 0], r[i, 1]) for i in range(len(r))])

    # 3D plot of potential + trajectory (your existing code)
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, Z, levels=100, cmap='turbo')
    plt.colorbar(contour, label='V(x, y)')
    plt.plot(r[:, 0], r[:, 1], color='black', lw=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Overdamped Langevin on MUller–Brown Potential')
    plt.legend()
    plt.show()

    # endregion

# simulation parameters
dt = 0.001 # time step
tf = 10.0 #final time
seed = int(time.time())

# particle initial state / params
r0 = [-0.5, 1.5] # init position [x, y]
m = 1.0 # mass
kB = 1.380649e-23 # J/K
T = 300 # K
beta = 1.25 #(kB*T)**-1 # thermodynamic beta = 1/kBT

OD_Langevin_sim(r0, beta, m, dt, tf, V, seed)

