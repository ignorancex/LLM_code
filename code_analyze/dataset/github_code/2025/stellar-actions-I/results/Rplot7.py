##getting pre-calculated data from the data directory to get the plot 7 of the paper- absolute change in action

import numpy as np
import matplotlib.pyplot as plt
import matplotlib


abs_JR = np.load('stellar-actions-I/data/abs_JR.npz')
abs_Jz = np.load('stellar-actions-I/data/abs_Jz.npz')
abs_Jphi = np.load('stellar-actions-I/data/abs_Jphi.npz')

delta_t_bins = abs_Jphi['dt']
JR=abs_JR['JR']
Jz = abs_Jz['Jz']
Jphi = abs_Jphi['Jphi']

import matplotlib.cm as cm

colormap = cm.get_cmap('viridis', len(range(1, 8, 1)))
colors = colormap(np.linspace(0, 1, len(range(1, 8, 1))))

fig, axs = plt.subplots(3, 1, figsize=(5,10),sharex='all')

final_heatmap=JR
axs[0].plot(delta_t_bins, (final_heatmap)[400,:], '-',c=colors[0], label="400 Myr age bin")
axs[0].plot(delta_t_bins, (final_heatmap)[300,:], '-',c=colors[1], label="300 Myr age bin")
axs[0].plot(delta_t_bins, (final_heatmap)[200,:], '-',c=colors[2], label="200 Myr age bin")
axs[0].plot(delta_t_bins, (final_heatmap)[100,:], '-',c=colors[3], label="100 Myr age bin")
axs[0].plot(delta_t_bins, (final_heatmap)[50,:], '-',c=colors[4], label="50 Myr age bin")
axs[0].plot(delta_t_bins, (final_heatmap)[10,:], '-',c=colors[5], label="10 Myr age bin")
axs[0].plot(delta_t_bins, (final_heatmap)[5,:], '-',c=colors[6], label="5 Myr age bin")
axs[0].set_ylabel(r"$\langle \Delta J_{R} \rangle$ (kpc km/s)")


final_heatmap=Jz
axs[1].plot(delta_t_bins, (final_heatmap)[400,:], '-',c=colors[0], label="400 Myr age bin")
axs[1].plot(delta_t_bins, (final_heatmap)[300,:], '-',c=colors[1], label="300 Myr age bin")
axs[1].plot(delta_t_bins, (final_heatmap)[200,:], '-',c=colors[2], label="200 Myr age bin")
axs[1].plot(delta_t_bins, (final_heatmap)[100,:], '-',c=colors[3], label="100 Myr age bin")
axs[1].plot(delta_t_bins, (final_heatmap)[50,:], '-',c=colors[4], label="50 Myr age bin")
axs[1].plot(delta_t_bins, (final_heatmap)[10,:], '-',c=colors[5], label="10 Myr age bin")
axs[1].plot(delta_t_bins, (final_heatmap)[5,:], '-',c=colors[6], label="5 Myr age bin")
axs[1].set_ylabel(r"$\langle \Delta J_{z} \rangle$ (kpc km/s)")


final_heatmap=Jphi
axs[2].plot(delta_t_bins, (final_heatmap)[400,:], '-',c=colors[0], label="400 Myr")
axs[2].plot(delta_t_bins, (final_heatmap)[300,:], '-',c=colors[1], label="300 Myr")
axs[2].plot(delta_t_bins, (final_heatmap)[200,:], '-',c=colors[2], label="200 Myr")
axs[2].plot(delta_t_bins, (final_heatmap)[100,:], '-',c=colors[3], label="100 Myr")
axs[2].plot(delta_t_bins, (final_heatmap)[50,:], '-',c=colors[4], label="50 Myr")
axs[2].plot(delta_t_bins, (final_heatmap)[10,:], '-',c=colors[5], label="10 Myr")
axs[2].plot(delta_t_bins, (final_heatmap)[5,:], '-',c=colors[6], label="5 Myr")
axs[2].set_ylabel(r"$\langle \Delta J_{\phi} \rangle$ (kpc km/s)")


axs[2].set_xlabel(r"$\Delta t$ (Myr)")
axs[2].legend()
axs[0].grid(True)
axs[1].grid(True)
axs[2].grid(True)

plt.tight_layout()
# plt.savefig('/g/data/jh2/ax8338/action/heatmap/paper_plots/R2.pdf')
plt.show()

