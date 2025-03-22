##getting pre-calculated data from the data directory to get the plot 9 of the paper- linear fit to the square relative change in action

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from analysis.utils import fit_line

#relative change in actions as function of delta_t and t
rel_JR = np.load('stellar-actions-I/data/rel_JR.npz')
rel_Jz = np.load('stellar-actions-I/data/rel_Jz.npz')

dt = rel_JR['dt']
JR=rel_JR['JR']
Jz = rel_Jz['Jz']

#relative change in actions only as function of delta_t
rel_JR_delt = np.load('stellar-actions-I/data/rel_JR_delt.npz')
rel_Jz_delt = np.load('stellar-actions-I/data/rel_Jz_delt.npz')

JR_delt = rel_JR_delt['JR']
Jz_delt = rel_Jz_delt['Jz']

#linear fit for z
#in Myr, extent of the linear fit
t1z=5
t2z=40      
m,b,r_square,y = fit_line(dt[t1z:t2z], ((Jz_delt**2))[t1z:t2z])

#linear fit for R
#in Myr, extent of the linear fit
t1R=5
t2R=50      
m1,b1,r_square1,y1 = fit_line(dt[t1R:t2R], ((JR_delt**2))[t1R:t2R])

##plot
k=100
colormap = cm.get_cmap('viridis', len(range(1, 8, 1)))
colors = colormap(np.linspace(0, 1, len(range(1, 8, 1))))

fig, axs = plt.subplots(2, 1, figsize=(5,7),sharex='all')
delta_t_bins=dt[:k]
final_heatmap=JR[:,:k]
axs[0].plot(delta_t_bins, (final_heatmap**2)[400,:], '-',c=colors[0],alpha=1)#, label="400 Myr age bin")
axs[0].plot(delta_t_bins, (final_heatmap**2)[300,:], '-',c=colors[1],alpha=1)#, label="300 Myr age bin")
axs[0].plot(delta_t_bins, (final_heatmap**2)[200,:], '-',c=colors[2],alpha=1)#, label="200 Myr age bin")
axs[0].plot(delta_t_bins, (final_heatmap**2)[100,:], '-',c=colors[3],alpha=1)#, label="100 Myr age bin")
axs[0].plot(delta_t_bins, (final_heatmap**2)[50,:], '-',c=colors[4],alpha=1)#, label="50 Myr age bin")
axs[0].plot(delta_t_bins, (final_heatmap**2)[10,:], '-',c=colors[5],alpha=1)#, label="10 Myr age bin")
axs[0].plot(delta_t_bins, (final_heatmap**2)[5,:], '-',c=colors[6],alpha=1)#, label="5 Myr age bin")
axs[0].plot(delta_t_bins[t1R:t2R], y1, '--r',linewidth=3, label=rf'$\langle \delta J_R^2 \rangle = 2 ({m1/2:.4f}) \Delta t + ({b1:.2f})$' f'\n $R^2$={r_square1}')

axs[0].set_ylabel(r"$\langle \delta J_{R}^2 \rangle$")


final_heatmap=Jz[:,:k]
axs[1].plot(delta_t_bins, (final_heatmap**2)[400,:], '-',c=colors[0],alpha=1)#, label="400 Myr")
axs[1].plot(delta_t_bins, (final_heatmap**2)[300,:], '-',c=colors[1],alpha=1)#, label="300 Myr")
axs[1].plot(delta_t_bins, (final_heatmap**2)[200,:], '-',c=colors[2],alpha=1)#, label="200 Myr")
axs[1].plot(delta_t_bins, (final_heatmap**2)[100,:], '-',c=colors[3],alpha=1)#, label="100 Myr")
axs[1].plot(delta_t_bins, (final_heatmap**2)[50,:], '-',c=colors[4],alpha=1)#, label="50 Myr")
axs[1].plot(delta_t_bins, (final_heatmap**2)[10,:], '-',c=colors[5],alpha=1)#, label="10 Myr")
axs[1].plot(delta_t_bins, (final_heatmap**2)[5,:], '-',c=colors[6],alpha=1)#, label="5 Myr")
axs[1].plot(delta_t_bins[t1z:t2z], y, '--r',linewidth=3, label=rf'$\langle \delta J_z^2 \rangle = 2 ({m/2:.4f}) \Delta t+ ({b:.2f})$' f'\n $R^2$={r_square}')

axs[1].set_ylabel(r"$\langle \delta J_{z}^2 \rangle$")
axs[1].set_xlabel(r"$\Delta t$ (Myr)")
axs[0].legend()
axs[1].legend()
axs[0].grid(True)
axs[1].grid(True)

plt.tight_layout()
# plt.savefig('/g/data/jh2/ax8338/action/heatmap/paper_plots/R3_fit.pdf')
plt.show()

