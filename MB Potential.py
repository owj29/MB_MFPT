# Plots Muller Brown potential

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Parameters
C = [-200, -100, -170, 15]
a_param = [-1, -1, -6.5, 0.7]
b_param = [0, 0, 11, 0.6]
c_param = [-10, -10, -6.5, 0.7]
x_0 = [1, -0.27, -0.5, -1]
y_0 = [0, 0.5, 1.5, 1.0]

def V(x, y):
    U = 0
    for Ck, ak, bk, ck, x_0k, y_0k in zip(C, a_param, b_param, c_param, x_0, y_0):
        U += Ck*jnp.exp(ak*(x-x_0k)**2 + bk*(x-x_0k)*(y-y_0k) + ck*(y-y_0k)**2)
    return(U)

x = np.linspace(-1.5, 1.0, 500)
y = np.linspace(-0.5, 2.0, 500)
X, Y = np.meshgrid(x, y)

Z = V(X, Y)

plt.figure(figsize=(8,6))
contour = plt.contourf(X, Y, Z, levels=150, cmap='rainbow')
plt.colorbar(contour, label='Potential Energy')
plt.xlabel('Grid X index')
plt.ylabel('Grid Y index')
plt.title('Müller–Brown Potential Energy Surface (10×10 Grid)')

# Number of grid divisions
nx, ny = 10, 10
x_edges = np.linspace(x.min(), x.max(), nx+1)
y_edges = np.linspace(y.min(), y.max(), ny+1)

# Draw grid lines
for xe in x_edges:
    plt.axvline(x=xe, color='k', lw=0.8, alpha=0.5)
for ye in y_edges:
    plt.axhline(y=ye, color='k', lw=0.8, alpha=0.5)

# Replace tick labels with box indices (0–9)
x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

plt.xticks(x_centers, np.arange(nx))
plt.yticks(y_centers, np.arange(ny))

plt.show()
