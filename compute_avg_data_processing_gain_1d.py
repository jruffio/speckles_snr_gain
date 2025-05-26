import numpy as np
import matplotlib.pyplot as plt

fontsize=12
plt.figure(figsize=(6, 5))

# p_over_b0 is the ratio of the planet flux to the average background level
p_over_b0_vals = np.logspace(-2, 1, 100)
t = np.linspace(0, 1, 10000)       # Uniform sampling of theta
# delta0_over_b0 the ratio of the amplitude of the speckle fluctuations compared to the average background
for delta0_over_b0_fixed in [0.25,0.5,0.75,0.9,1]:
    avg_gain = []

    # Precompute sine term for speed
    sin_term = np.sin(np.pi * t)

    # Compute integral for each rp
    for r_over_b0 in p_over_b0_vals:
        val = (delta0_over_b0_fixed * sin_term) / (1 + r_over_b0)
        integrand = 1.0 / np.sqrt(1 - val**2)
        integral = np.trapezoid(integrand, t)
        avg_gain.append(integral)

    avg_gain = np.array(avg_gain)

    plt.plot(p_over_b0_vals, avg_gain)
    plt.text(p_over_b0_vals[0], avg_gain[0],f'$\Delta_0/b_0 = {delta0_over_b0_fixed}$', fontsize=10, color='black', ha='left', va='bottom')
plt.xscale('log')
plt.xlabel('$p/b_0$',fontsize=fontsize)
plt.gca().tick_params(axis='x', labelsize=fontsize)
plt.gca().tick_params(axis='y', labelsize=fontsize)
plt.ylabel('Average gain (<g>)',fontsize=fontsize)
plt.grid(True, which='both', ls='--', alpha=0.6)
plt.tight_layout()
plt.legend()
plt.show()
