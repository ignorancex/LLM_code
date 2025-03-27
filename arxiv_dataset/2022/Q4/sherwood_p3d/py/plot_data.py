import numpy as np
import matplotlib.pyplot as plt
import sherwood_simulation as she_sim
import measured_power

# specify simulation from Sherwood suite
L_hMpc=160
n_part=2048
label='L{}_N{}'.format(L_hMpc,n_part)
sim = she_sim.SherwoodSimulation(L_hMpc=L_hMpc,n_part=n_part)

# specify grid of skewers 
snapshot_num=9
n_xy=1024
n_z=2048
z=she_sim.redshift_from_snapshot(snapshot_num)
# combined measurement
axis=0
skewers=she_sim.Grid(simulation=sim,snapshot_num=snapshot_num,
                                n_xy=n_xy,n_z=n_z,axis=axis)

# specify grid of halos
add_rsd=True

# mass bin
all_mass=True
if all_mass:
    logMh_min=None
    logMh_max=None
    mass_label='all'
else:
    logMh_min=11.5
    logMh_max=14.0
    mass_label='{}_{}'.format(logMh_min,logMh_max)
halos=she_sim.HaloGrid(simulation=sim,snapshot_num=snapshot_num,
                          logMh_min=logMh_min,logMh_max=logMh_max,
                          n_xy=n_xy,n_z=n_z,axis=axis,add_rsd=add_rsd)

# get P3D measurements (standard binning)
data_F=measured_power.get_power_from_grid(grid=skewers,power_type="flux_p3d")
data_X=measured_power.get_power_from_grid(grid=halos,power_type="cross_p3d")
data_H=measured_power.get_power_from_grid(grid=halos,power_type="halo_p3d")

# show only few mu bins
downsample=5
# short-cuts for convenience
n_mu=data_F['n_mu_bins']
cm=plt.get_cmap('jet')
mu_bin_edges = np.linspace(0., 1.,n_mu + 1)

def plot_p3d(data,ax,label):
    for i in range(0,n_mu,downsample):
        col=cm(i/n_mu)
        # identify bins we want to keep in plot
        nan_mask=np.isnan(data['mu'][:,i])
        kmax_mask=data['k_hMpc'][:,i][~nan_mask] > data['k_hMpc_max']
        keep=~nan_mask
        keep[~nan_mask]=~kmax_mask
        k=data['k_hMpc'][:,i][keep]
        mu=data['mu'][:,i][keep]
        if data['power_type'] == 'cross_p3d':
            P3D=-data['p3d_hMpc'][:,i][keep]
            mu_label=r"%.2f $\leq \mu \leq$ %.2f" % (mu_bin_edges[i],
                                                        mu_bin_edges[i+1])
        else:
            P3D=data['p3d_hMpc'][:,i][keep]
            mu_label=None
        # these are only approximated errorbars for cross-correlations!
        error=P3D/np.sqrt(0.5*data['counts'][:,i][keep])
        if data['power_type'] == 'halo_p3d':
            P3D-=data['shot_noise']
        kfac=k**3
        ax.fill_between(k,(P3D-error)*kfac,(P3D+error)*kfac,alpha=.3,
                label=mu_label,color=col)
        ax.plot(k,(P3D-error)*kfac,lw=3,alpha=.1,color=col)
        ax.plot(k,(P3D+error)*kfac,lw=3,alpha=.1,color=col)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel(label)
    if data['power_type'] == 'cross_p3d':
        ax.legend(loc="best",numpoints=1,fancybox=True)

fig, axs = plt.subplots(4, sharex=True, sharey=False,figsize=[8,10])
fig.suptitle('Measured power spectra, z={}'.format(z))
auto_label=r"$k^3 P_F(k,\mu)$"
cross_label=r"$ -k^3 P_X(k,\mu)$"
halo_label=r"$k^3 P_H(k,\mu)$"
plot_p3d(data_F,axs[0],label=auto_label)
plot_p3d(data_X,axs[1],label=cross_label)
plot_p3d(data_H,axs[2],label=halo_label)
plt.xlabel("k (h/Mpc)")

# plot cross-correlation coefficient
for i in range(0,n_mu,downsample):
    col=cm(i/n_mu)
    # identify bins we want to keep in plot
    nan_mask=np.isnan(data_F['mu'][:,i])
    kmax_mask=data_F['k_hMpc'][:,i][~nan_mask] > data_F['k_hMpc_max']
    keep=~nan_mask
    keep[~nan_mask]=~kmax_mask
    P_F=data_F['p3d_hMpc'][:,i][keep]
    P_X=data_X['p3d_hMpc'][:,i][keep]
    P_H=data_H['p3d_hMpc'][:,i][keep]
    shot_noise=data_H['shot_noise']
    k=data_F['k_hMpc'][:,i][keep]
    axs[3].plot(k,-P_X/np.sqrt(P_F*(P_H-shot_noise)),ls='-',color=col)
    axs[3].set_xscale('log')
    axs[3].set_ylim([0,1])
    axs[3].set_ylabel(r'$- P_X / \sqrt{P_F ~ P_H} (k,\mu)$')

plt.tight_layout()
plt.savefig('measured_p3d_{}.png'.format(mass_label))
plt.show()

