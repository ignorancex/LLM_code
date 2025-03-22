### third plot in the paper- fig 3- variation in potential
### data not available in public so code just for reference

from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sc
import os
os.chdir('/g/data/jh2/ax8338/action/action_function/')
import functions as f

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 15

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

filepath='/g/data/jh2/ax8338/fit_grid/all_565_fitgrid.h5'
i=200
print('Reading fitgrid and fitpot files')
fit_grid,fit_pot = f.read_fitgridhdf5(i,filepath)

fit_grid_new, fit_pot_new,R_res,z_res= f.reduce_res(fit_grid,fit_pot,0,50,20000,2000)
print(f'Doing RPhi calc for i={i} Myr')

x=np.unique(fit_grid_new[abs(fit_grid_new[:,1])<2][:,0])
#averaging over the potential for same R values- different z values.
y=fit_pot_new[abs(fit_grid_new[:,1])<2].reshape(int(fit_grid_new[abs(fit_grid_new[:,1])<2].shape[0]/len(x)), len(x)).mean(axis=0)
y1=fit_pot_new[(abs(fit_grid_new[:,1])<502)&(abs(fit_grid_new[:,1])>500)].reshape(int(fit_grid_new[(abs(fit_grid_new[:,1])<502)&(abs(fit_grid_new[:,1])>500)].shape[0]/len(x)), len(x)).mean(axis=0)

#Gaussian filtering
y_smooth = gaussian_filter1d(y,30,mode='nearest')
y1_smooth = gaussian_filter1d(y1,30,mode='nearest')

#bspline fitting
t,c,kb = interpolate.splrep(x,y_smooth,s=200,k=3)
RPhi = interpolate.BSpline(t,c,kb,extrapolate=True)
t1,c1,kb1 = interpolate.splrep(x,y1_smooth,s=200,k=3)
RPhi1 = interpolate.BSpline(t1,c1,kb1,extrapolate=True)
#defining finer grid
xx = np.linspace(0.1,20000,100000)
yy = RPhi(xx)
yy1 = RPhi1(xx)

## Phi z
def phi_z(R_g,window=101, g=2):
    fit_grid_new_z,fit_pot_new_z, _,_ = f.reduce_res(fit_grid,fit_pot,1,10,20000,2000)
    mask_derz = (abs(fit_grid_new_z[:, 0] - R_g) < 1)
    xz = np.unique(fit_grid_new_z[mask_derz][:, 1])
    yz = np.mean(fit_pot_new_z[mask_derz].reshape(len(xz), int(fit_grid_new_z[mask_derz].shape[0] / len(xz))), axis=1)
    
    # Smooth the potential in the z direction
    Phi_gaussz = sc.savgol_filter(yz, window, 2)
    Phi_gaussz = gaussian_filter1d(Phi_gaussz,g,mode='nearest')
    Phi_gaussz = gaussian_filter1d(Phi_gaussz,g,mode='nearest')
    Phi_gaussz = gaussian_filter1d(Phi_gaussz, g, mode='nearest')
    tz, cz, kfitz = interpolate.splrep(xz, Phi_gaussz, s=50, k=3)
    zPhi_b = interpolate.BSpline(tz, cz, kfitz, extrapolate=True)
    return zPhi_b

xx_z =np.linspace(-1000,1000,2000)
zPhi = phi_z(3000)
zPhi1 = phi_z(4000)
yy_z = zPhi(xx_z)
yy_z1 = zPhi1(xx_z)

#plot
fig, axs = plt.subplots(2, 1, figsize=(6,8))
axs[0].plot(xx/1000,yy,label='z = 0 pc')
axs[0].plot(xx/1000,yy1,label ='z $= \pm 500$ pc')
axs[0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0)) 
axs[0].set_xlabel('R(kpc)')
axs[0].legend()
axs[0].set_ylabel('$\Phi (km/s)^{2}$')

axs[1].plot(xx_z,yy_z,label='R = 3 kpc')
axs[1].plot(xx_z,yy_z1,label ='R = 4 kpc')
axs[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0)) 
axs[1].set_ylabel('$\Phi (km/s)^{2}$')
axs[1].set_xlabel('z (pc)')
axs[1].legend()

plt.tight_layout()
plt.savefig('/g/data/jh2/ax8338/action/heatmap/paper_plots/phi_vertical.pdf')
