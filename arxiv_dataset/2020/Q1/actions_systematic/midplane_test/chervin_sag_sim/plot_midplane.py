import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from matplotlib import rc
import matplotlib as mpl

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

textwidth = 7.10000594991
columnwidth = 3.35224200913

tb_c = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
'#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac']

nbootstrap = 1000
np.random.seed(162)

rcut = 0.5
zcut = 1.0
nspoke = 50

out = np.load('out_Disk_200.npy')
theta = np.array(out[:,0])
result = np.array(out[:,1:7])
fit = np.array(out[:,7])

midplane_est = np.array(result[:,0])
err_low = np.array(result[:,1])
err_high = np.array(result[:,2])

midplane_vel = np.array(result[:,3])
err_vel_low = np.array(result[:,4])
err_vel_high = np.array(result[:,5])
import matplotlib.pyplot as plt
plt.plot(theta/np.pi, midplane_est*1000)

glist = ['Disk_200', 'Disk_400', 'Disk_600', 'Disk_690']
column = [0,1,2,3]
time = [r'$t=2.0\,\text{Gyr}$', r'$t=4.0\,\text{Gyr}$', 
        r'$t=6.0\,\text{Gyr}$', r'$t=6.9\,\text{Gyr}$']

fig,axs = plt.subplots(2,4,figsize=((2/3)*textwidth,4))
for gal,col,tm in zip(glist,column,time):
    out = np.load('out_'+gal+'.npy')
    theta = out[:,0]
    result = out[:,1:7]
    fit = out[:,7]

    midplane_est = result[:,0]
    err_low = result[:,1]
    err_high = result[:,2]

    midplane_vel = result[:,3]
    err_vel_low = result[:,4]
    err_vel_high = result[:,5]
    
    axs[0,col].plot(theta/np.pi, midplane_est*1000, c=tb_c[0])
    axs[0,col].plot(theta/np.pi, err_low*1000, c=tb_c[0], ls='dashed', alpha=0.5)
    axs[0,col].plot(theta/np.pi, err_high*1000, c=tb_c[0], ls='dashed', alpha=0.5)
    axs[0,col].fill_between(theta/np.pi, err_high*1000, err_low*1000, color=tb_c[0], alpha=0.25)
    
    axs[1,col].set_xlabel(r'$\phi/\pi$')

    axs[0,col].text(0.05, 0.88, tm, horizontalalignment='left', verticalalignment='center', transform = axs[0,col].transAxes)

    axs[0,col].set_xlim(0, 2)
    axs[1,col].set_xlim(0, 2)

    axs[0,col].set_ylim(-200, 200)
    axs[1,col].set_ylim(-200, 200)

    axs[1,col].plot(theta/np.pi, (midplane_est-fit)*1000, c=tb_c[0])
    axs[1,col].plot(theta/np.pi, (err_low-fit)*1000, c=tb_c[0], ls='dashed', alpha=0.5)
    axs[1,col].plot(theta/np.pi, (err_high-fit)*1000, c=tb_c[0], ls='dashed', alpha=0.5)
    axs[1,col].fill_between(theta/np.pi, (err_high-fit)*1000, (err_low-fit)*1000, color=tb_c[0], alpha=0.25)

    midplane_up = np.percentile(midplane_est - fit, 95)
    midplane_low = np.percentile(midplane_est - fit, 5)
    print(gal, midplane_up - midplane_low)
axs[0,0].set_ylabel("midplane (pc)")
axs[1,0].set_ylabel("midplane (pc)")
axs[0,0].set_title("Chervin's Simulation")

fig.tight_layout()
plt.savefig('midplane.pdf')

fig, ax = plt.subplots(2, 2, sharex=True, figsize=((2/3)*textwidth,3))

for x in ax.flatten():
    x.locator_params('y', nbins=5)

# now make paper plot, with just fit
for gal,x,tm in zip(glist,ax.flatten(),time):
    out = np.load('out_'+gal+'.npy')
    theta = out[:,0]
    result = out[:,1:7]
    fit = out[:,7]

    midplane_est = result[:,0]
    err_low = result[:,1]
    err_high = result[:,2]

    midplane_vel = result[:,3]
    err_vel_low = result[:,4]
    err_vel_high = result[:,5]

    x.text(0.05, 0.88, tm, 
               horizontalalignment='left', 
               verticalalignment='center', 
               transform = x.transAxes)

    x.set_xlim(0, 2)
    x.set_xlim(0, 2)

    x.set_ylim(-200, 200)
    x.set_ylim(-200, 200)

    x.locator_params('x', nbins=5)

    x.plot(theta/np.pi, (midplane_est-fit)*1000, c=tb_c[0])
    x.plot(theta/np.pi, (err_low-fit)*1000, c=tb_c[0], ls='dashed', alpha=0.5)
    x.plot(theta/np.pi, (err_high-fit)*1000, c=tb_c[0], ls='dashed', alpha=0.5)
    x.fill_between(theta/np.pi, (err_high-fit)*1000, (err_low-fit)*1000, color=tb_c[0], alpha=0.25)

ax[1][0].set_xlabel(r'$\phi/\pi$')
ax[1][1].set_xlabel(r'$\phi/\pi$')
ax[0][0].set_ylabel(r'$\text{midplane}\,[\,\text{pc}\,]$')
ax[1][0].set_ylabel(r'$\text{midplane}\,[\,\text{pc}\,]$')
# ax[0].set_title("Chervin's Simulation")

fig.tight_layout()
plt.savefig('midplane_fit_chervinsim.pdf')

