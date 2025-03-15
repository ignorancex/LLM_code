##getting pre-calculated data from the data directory to get the plot 8 of the paper- relative change in action

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

rel_JR = np.load('stellar-actions-I/data/rel_JR.npz')
rel_Jz = np.load('stellar-actions-I/data/rel_Jz.npz')

dt = rel_JR['dt']
JR=rel_JR['JR']
Jz = rel_Jz['Jz']

import matplotlib.cm as cm
k=400
colormap = cm.get_cmap('viridis', len(range(1, 8, 1)))
colors = colormap(np.linspace(0, 1, len(range(1, 8, 1))))

fig, axs = plt.subplots(2, 1, figsize=(5,7),sharex='all')
delta_t_bins=dt[:k]
final_heatmap=JR[:,:k]
axs[0].plot(delta_t_bins, (final_heatmap**2)[400,:], '-',c=colors[0],alpha=1)
axs[0].plot(delta_t_bins, (final_heatmap**2)[300,:], '-',c=colors[1],alpha=1)
axs[0].plot(delta_t_bins, (final_heatmap**2)[200,:], '-',c=colors[2],alpha=1)
axs[0].plot(delta_t_bins, (final_heatmap**2)[100,:], '-',c=colors[3],alpha=1)
axs[0].plot(delta_t_bins, (final_heatmap**2)[50,:], '-',c=colors[4],alpha=1)
axs[0].plot(delta_t_bins, (final_heatmap**2)[10,:], '-',c=colors[5],alpha=1)
axs[0].plot(delta_t_bins, (final_heatmap**2)[5,:], '-',c=colors[6],alpha=1)


axs[0].set_ylabel(r"$\langle \delta J_{R}^2 \rangle$")


final_heatmap=Jz[:,:k]
axs[1].plot(delta_t_bins, (final_heatmap**2)[400,:], '-',c=colors[0],alpha=1, label="400 Myr")
axs[1].plot(delta_t_bins, (final_heatmap**2)[300,:], '-',c=colors[1],alpha=1, label="300 Myr")
axs[1].plot(delta_t_bins, (final_heatmap**2)[200,:], '-',c=colors[2],alpha=1, label="200 Myr")
axs[1].plot(delta_t_bins, (final_heatmap**2)[100,:], '-',c=colors[3],alpha=1, label="100 Myr")
axs[1].plot(delta_t_bins, (final_heatmap**2)[50,:], '-',c=colors[4],alpha=1, label="50 Myr")
axs[1].plot(delta_t_bins, (final_heatmap**2)[10,:], '-',c=colors[5],alpha=1, label="10 Myr")
axs[1].plot(delta_t_bins, (final_heatmap**2)[5,:], '-',c=colors[6],alpha=1, label="5 Myr")

axs[1].set_ylabel(r"$\langle \delta J_{z}^2 \rangle$")



axs[1].set_xlabel(r"$\Delta t$ (Myr)")
axs[1].legend(ncol=2)

axs[0].grid(True)
axs[1].grid(True)

plt.tight_layout()
# plt.savefig('/g/data/jh2/ax8338/action/heatmap/paper_plots/all_data_median.pdf')
plt.show()

