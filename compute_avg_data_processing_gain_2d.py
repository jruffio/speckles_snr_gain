import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Define log-spaced parameters
p_over_b0_vals = np.logspace(-2, 0, 100)   # rp from 1e-2 to 10
delta0_over_b0_vals = np.linspace(0, 1, 100)   # rs from 0.1 to 1
t = np.linspace(0, 1, 10000)      # Uniform sampling of t
sin_term = np.sin(np.pi * t)    # Precompute sine

Z = np.zeros((len(delta0_over_b0_vals), len(p_over_b0_vals)))

# Compute the integral over the (rs, rp) grid
for i, rs in enumerate(delta0_over_b0_vals):
    for j, rp in enumerate(p_over_b0_vals):
        val = (rs * sin_term) / (1 + rp)
        integrand = 1.0 / np.sqrt(1 - val**2)
        Z[i, j] = np.trapezoid(integrand, t)  # Normalize for half-domain

# Plotting the 2D map
plt.figure(figsize=(9, 6))
mesh = plt.pcolormesh(
    p_over_b0_vals, delta0_over_b0_vals, Z,
    shading='auto',
    cmap='viridis'
)
plt.xscale('log')
plt.xlabel('$p/b_0$')
plt.ylabel('$\Delta_0/b_0$')
plt.colorbar(mesh, label='Integral value')

contours = plt.contour(
    p_over_b0_vals, delta0_over_b0_vals, Z,
    levels=[1.001,1.01,1.1, 2],
    colors='white',
    linewidths=1.5
)
plt.clabel(contours, fmt='Z=%.3f', colors='white')

plt.tight_layout()
plt.show()