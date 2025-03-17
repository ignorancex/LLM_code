import numpy as np
import matplotlib.pyplot as plt
import sherwood_simulation as she_sim
import measured_power
import model_density


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
                          n_xy=n_xy,n_z=n_z,axis=axis,add_rsd=True)

# get P3D measurements (standard binning)
data_F=measured_power.get_power_from_grid(grid=skewers,power_type="flux_p3d")
data_X=measured_power.get_power_from_grid(grid=halos,power_type="cross_p3d")
data_H=measured_power.get_power_from_grid(grid=halos,power_type="halo_p3d")

# linear matter power
model = model_density.LinearDensityModel()

# two panels,top flux p3d and bottom cross
plt.rcParams.update({'font.size': 10})
fig, axs = plt.subplots(2,sharex=True,sharey=False,figsize=[8,6])
fig.suptitle('Flux and cross power spectra at z={}'.format(z))

# show only few mu bins
downsample=5
# short-cuts for convenience
n_mu=data_F['n_mu_bins']
cm=plt.get_cmap('jet')
mu_edges = np.linspace(0., 1.,n_mu + 1)

# plot flux_p3d
for i in range(0,n_mu,downsample):
    col=cm(i/n_mu)
    # identify bins we want to keep in plot
    nan_mask=np.isnan(data_F['mu'][:,i])
    kmax_mask=data_F['k_hMpc'][:,i][~nan_mask] > data_F['k_hMpc_max']
    keep=~nan_mask
    keep[~nan_mask]=~kmax_mask
    P_mu=data_F['p3d_hMpc'][:,i][keep]
    k_mu=data_F['k_hMpc'][:,i][keep]
    mu=data_F['mu'][:,i][keep]
    error=P_mu/np.sqrt(0.5*data_F['counts'][:,i][keep])
    P_L=model.linP_hMpc(z=z,k_hMpc=k_mu)
    label=r"%.2f $\leq \mu \leq$ %.2f" % (mu_edges[i],mu_edges[i+1])
    kwargs={'color':col}
    axs[0].fill_between(k_mu,(P_mu-error)/P_L,(P_mu+error)/P_L,alpha=.3,
            label=label,**kwargs)
    axs[0].plot(k_mu,(P_mu-error)/P_L,lw=3,alpha=.1,**kwargs)
    axs[0].plot(k_mu,(P_mu+error)/P_L,lw=3,alpha=.1,**kwargs)

axs[0].set_ylabel(ylabel=r"$P_F(k,\mu) / P_L(k)$")
axs[0].set_xscale("log")
axs[0].set_xlim([0.03,20])
axs[0].legend(loc=1,fancybox=True)

# plot cross_p3d
for i in range(0,n_mu,downsample):
    col=cm(i/n_mu)
    nan_mask=np.isnan(data_F['mu'][:,i])
    kmax_mask=data_F['k_hMpc'][:,i][~nan_mask] > data_F['k_hMpc_max']
    keep=~nan_mask
    keep[~nan_mask]=~kmax_mask
    k_mu=data_X['k_hMpc'][:,i][keep]
    mu=data_X['mu'][:,i][keep]
    P_L=model.linP_hMpc(z=z,k_hMpc=k_mu)
    # get cross-power and errorbars
    P_X=data_X['p3d_hMpc'][:,i][keep]
    P_F=data_F['p3d_hMpc'][:,i][keep]
    P_H=data_H['p3d_hMpc'][:,i][keep]
    error = np.sqrt(P_X**2+P_F*P_H)/np.sqrt(data_X['counts'][:,i][keep])
    label=r"%.2f $\leq \mu \leq$ %.2f" % (mu_edges[i],mu_edges[i+1])
    axs[1].fill_between(k_mu,-(P_X-error)/P_L,-(P_X+error)/P_L,alpha=.3,
            label=label,color=col)
    axs[1].plot(k_mu,-(P_X-error)/P_L,lw=3,alpha=.1,color=col)
    axs[1].plot(k_mu,-(P_X+error)/P_L,lw=3,alpha=.1,color=col)

axs[1].set_ylabel(ylabel=r"$-P_X(k,\mu) / P_L(k)$")
axs[1].set_xlabel(r"$k \, [h/Mpc]$")
axs[1].set_xscale("log")
axs[1].set_xlim([0.03,20])
axs[1].legend(loc=1,fancybox=True)

plt.tight_layout()
plt.savefig('flux_cross_z{}.png'.format(z))
plt.show()
plt.close()
