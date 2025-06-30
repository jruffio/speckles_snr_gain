import numpy as np
import matplotlib.pyplot as plt
import os

fig_dir = "./figures/"

fontsize=12
plt.figure(figsize=(6, 6))
color_list = ["#006699","#ff9900", "#6600ff", "purple", "grey"]

# p_over_b0 is the ratio of the planet flux to the average background level
p_over_b0_vals = np.logspace(-2, 1, 100)
delta0_over_b0_list = [0.25,0.5,0.75,0.9,1]
# p_over_b0_vals = [0.01]
# delta0_over_b0_list = [1]
N_theta_list = [2,3,10,100]
linestyle_list = ["-","--","-.",":"]

t = np.linspace(0, 2., 10000)       # Uniform sampling of theta

# delta0_over_b0 the ratio of the amplitude of the speckle fluctuations compared to the average background
for id0,(delta0_over_b0_fixed,mycolor) in enumerate(zip(delta0_over_b0_list,color_list[::-1])):
    avg_gain_ref = []

    # Precompute sine term for speed
    t_diff = np.diff(t)

    # sin_term = np.sin(np.pi * t)
    #
    # # Compute integral for each rp
    # for p_over_b0 in p_over_b0_vals:
    #     val = (delta0_over_b0_fixed * sin_term) / (1 + p_over_b0)
    #     integrand = 1.0 / np.sqrt(1 - val**2)
    #     integral = np.sum(integrand[1::]*t_diff)/(t[-1]-t[0])
    #     avg_gain_ref.append(integral)
    #
    # avg_gain_ref = np.array(avg_gain_ref)
    #
    # plt.plot(p_over_b0_vals, avg_gain_ref,linestyle=":",linewidth=3,color=mycolor)
    # plt.text(p_over_b0_vals[0], avg_gain_ref[0],f'$\Delta_0/b_0 = {delta0_over_b0_fixed}$', fontsize=10, color='black', ha='left', va='bottom')

    for id1,(N_theta,ls) in enumerate(zip(N_theta_list,linestyle_list)):
        t_diff = np.diff(t)
        avg_gain = []
        # Compute integral for each rp
        for p_over_b0 in p_over_b0_vals:
            integrand = np.sqrt(np.sum(1/(1+(delta0_over_b0_fixed/(1+p_over_b0))*np.sin(np.pi*(np.arange(N_theta)[:,None]*2/N_theta+t[None,:]))),axis=0)/N_theta)
            # plt.plot(t,(1+1/p_over_b0+delta0_over_b0_fixed*np.sin(np.pi*(0*2/N_theta+t))))
            # plt.plot(t,(1+1/p_over_b0+delta0_over_b0_fixed*np.sin(np.pi*(1*2/N_theta+t))))
            # plt.show()
            integral = np.sum(integrand[1::]*t_diff)/(t[-1]-t[0])
            avg_gain.append(integral)
        avg_gain = np.array(avg_gain)
        if id1 == 0:
            plt.text(p_over_b0_vals[0], avg_gain[0],f'$\Delta_0/b_0 = {delta0_over_b0_fixed}$', fontsize=10, color='black', ha='left', va='bottom')
        if id0 ==0:
            plt.plot(p_over_b0_vals, avg_gain,linestyle=ls,linewidth=1,color=mycolor,label=f"$N_\\theta={N_theta}$")
        else:
            plt.plot(p_over_b0_vals, avg_gain,linestyle=ls,linewidth=1,color=mycolor)

plt.legend(loc="upper right",fontsize=fontsize)
plt.xscale('log')
plt.xlabel('$p/b_0$',fontsize=fontsize)
plt.gca().tick_params(axis='x', labelsize=fontsize)
plt.gca().tick_params(axis='y', labelsize=fontsize)
plt.ylabel('Average gain (<g>)',fontsize=fontsize)
plt.grid(True, which='both', ls='--', alpha=0.75)
plt.tight_layout()

out_filename = os.path.join(fig_dir, "average_ppgain_Nrolls.png")
print("Saving " + out_filename)
plt.savefig(out_filename, dpi=300)
plt.savefig(out_filename.replace(".png", ".pdf"))
plt.show()
