##getting pre-calculated data from the data directory to get the plot 6 of the paper- heatmap

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

#heatmap plot
plt.close('all')
fig, axs = plt.subplots(3, 1, figsize=(5,10),sharex='all',sharey='all')
final_heatmap = JR
final_heatmap[final_heatmap <= 0] = np.nan
norm = matplotlib.colors.LogNorm(vmin=np.nanmin(final_heatmap), vmax=np.nanmax(final_heatmap))
axs[0].vlines(x=[5,10,50,100,200,300,400],ymin=0,ymax=465,ls='--',color='grey')
cax=axs[0].imshow(final_heatmap.T, cmap='viridis',norm=norm, aspect='auto', origin='lower', interpolation='none')
fig.colorbar(cax,label=r'$\langle\Delta J_R \rangle$ (kpc km/s)',ax=axs[0])
axs[0].set_ylabel('$\\Delta t$ (Myr)')

final_heatmap = Jphi
final_heatmap[final_heatmap <= 0] = np.nan
norm = matplotlib.colors.LogNorm(vmin=np.nanmin(final_heatmap), vmax=np.nanmax(final_heatmap))
# norm = matplotlib.colors.LogNorm(vmin=np.nanmin(final_heatmap), vmax=5)
cax=axs[2].imshow(final_heatmap.T, cmap='viridis',norm=norm, aspect='auto', origin='lower', interpolation='none')
axs[2].vlines(x=[5,10,50,100,200,300,400],ymin=0,ymax=465,ls='--',color='grey')
fig.colorbar(cax,label=r'$\langle\Delta J_{\phi} \rangle$ (kpc km/s)',ax=axs[2])
axs[2].set_ylabel('$\\Delta t$ (Myr)')

final_heatmap = Jz
final_heatmap[final_heatmap <= 0] = np.nan
norm = matplotlib.colors.LogNorm(vmin=np.nanmin(final_heatmap), vmax=np.nanmax(final_heatmap))
# norm = matplotlib.colors.LogNorm(vmin=np.nanmin(final_heatmap), vmax=5)
cax=axs[1].imshow(final_heatmap.T, cmap='viridis',norm=norm, aspect='auto', origin='lower', interpolation='none')
axs[1].vlines(x=[5,10,50,100,200,300,400],ymin=0,ymax=465,ls='--',color='grey')
fig.colorbar(cax,label=r'$\langle\Delta J_z \rangle$ (kpc km/s)',ax=axs[1])

plt.ylabel('$\\Delta t$ (Myr)')
plt.xlabel('Age (Myr)')
plt.xlim(0,465)
plt.ylim(0,465)
plt.tight_layout()
# plt.savefig('/g/data/jh2/ax8338/action/heatmap/paper_plots/heatmap_absall_lines.pdf')
plt.show()
