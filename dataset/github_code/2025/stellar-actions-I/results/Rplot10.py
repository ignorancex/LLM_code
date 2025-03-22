##getting pre-calculated data from the data directory to get the plot 10 of the paper- dependence of action change on local birth density

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from analysis.utils import fit_line

#relative change in action (only as function of delta_t) for stars born in different regions of density
dense_R= np.load('stellar-actions-I/data/dense_JR.npz')
sparse_R10 = np.load('stellar-actions-I/data/sparse_JR.npz')
sparse_R = np.load('stellar-actions-I/data/verysparse_JR.npz')

delta_t_bins = dense_R['dt']
JR_d = dense_R['JR']
JR_s= sparse_R['JR']
JR_s10= sparse_R10['JR']

dense_z= np.load('stellar-actions-I/data/dense_Jz.npz')
sparse_z10 = np.load('stellar-actions-I/data/sparse_Jz.npz')
sparse_z = np.load('stellar-actions-I/data/verysparse_Jz.npz')

Jz_d = dense_z['Jz']
Jz_s= sparse_z['Jz']
Jz_s10 = sparse_z10['Jz']


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

colormap = cm.get_cmap('viridis', len(range(1, 8, 1)))
colors = colormap(np.linspace(0, 1, len(range(1, 8, 1))))

fig, axs = plt.subplots(2, 1, figsize=(6,8),sharex='col')
final_heatmap=JR_d
axs[0].plot(delta_t_bins[:k], (final_heatmap**2)[:k], '-',c='m',alpha=1, label="dense (< smoothing length)")
axs[0].plot(delta_t_bins[:k], (JR_s10**2)[:k], '--',c='darkblue',alpha=1, label="sparse (90th percentile)")

final_heatmap=JR_s
axs[0].plot(delta_t_bins[:k], (final_heatmap**2)[:k], '--',c='c',alpha=1, label="very sparse (99th percentile)")

axs[0].plot(delta_t_bins[t1R:t2R][::4], y1[::4], '*',c='black',markersize=9,alpha=0.5,label=rf'$\langle \delta J_R^2 \rangle = 2 ({m1/2:.4f}) \Delta t + ({b1:.2f})$')

axs[0].set_ylabel(r"$\langle\delta J_R^2\rangle$")


final_heatmap=Jz_d
axs[1].plot(delta_t_bins[:k], (final_heatmap**2)[:k], '-',c='m',alpha=1, label=r"dense (< smoothing length)")
axs[1].plot(delta_t_bins[:k], (Jz_s10**2)[:k], '--',c='darkblue',alpha=1, label="sparse (90th percentile)")
final_heatmap=Jz_s
axs[1].plot(delta_t_bins[:k], (final_heatmap**2)[:k], '--',c='c',alpha=1, label="very sparse (99th percentile)")

axs[1].plot(delta_t_bins[t1z:t2z][::4], y[::4], '*',c='black',markersize=9,alpha=0.5, label=rf'$\langle \delta J_z^2 \rangle = 2 ({m/2:.4f}) \Delta t+ ({b:.2f})$')
axs[1].set_ylabel(r"$\langle\delta J_z^2\rangle$")
axs[1].set_xlabel(r"$\Delta t$ (Myr)")

axs[0].grid(True)
axs[0].legend()
axs[1].grid(True)
axs[1].legend()
plt.tight_layout()
# plt.savefig('/g/data/jh2/ax8338/action/heatmap/paper_plots/density_combined1.pdf')
plt.show()

