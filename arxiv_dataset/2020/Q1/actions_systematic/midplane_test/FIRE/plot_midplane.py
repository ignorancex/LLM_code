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


glist = ['m12i', 'm12f', 'm12m']

fig, ax = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(textwidth,3.2))
# first make midplane comparison plot
for gal,ax_col in zip(glist, ax.transpose()):
    out = np.load('output/out_'+gal+'.npy')
    theta = out[:,0]
    result = out[:,1:7]
    fit = out[:,7]

    midplane_est = result[:,0]
    err_low = result[:,1]
    err_high = result[:,2]

    midplane_vel = result[:,3]
    err_vel_low = result[:,4]
    err_vel_high = result[:,5]

    ax_col[0].plot(theta/np.pi, midplane_est*1000, c=tb_c[0])
    ax_col[0].plot(theta/np.pi, err_low*1000, c=tb_c[0], ls='dashed', alpha=0.5)
    ax_col[0].plot(theta/np.pi, err_high*1000, c=tb_c[0], ls='dashed', alpha=0.5)
    ax_col[0].fill_between(theta/np.pi, err_high*1000, err_low*1000, color=tb_c[0], alpha=0.25)

    ax_col[1].set_xlabel(r'$\phi/\pi$')

    ax_col[0].text(0.05, 0.88, r'\texttt{'+gal+r'}', 
               horizontalalignment='left', 
               verticalalignment='center', 
               transform = ax_col[0].transAxes)

    ax_col[0].set_xlim(0, 2)
    ax_col[1].set_xlim(0, 2)

    ax_col[0].set_ylim(-200, 200)
    ax_col[1].set_ylim(-200, 200)

    ax_col[0].locator_params('x', nbins=5)
    ax_col[1].locator_params('y', nbins=5)

    print(gal, '90 perc: ', np.percentile((midplane_est-fit)*1000, 95) - np.percentile((midplane_est-fit)*1000, 5))

    ax_col[1].plot(theta/np.pi, (midplane_est-fit)*1000, c=tb_c[0])
    ax_col[1].plot(theta/np.pi, (err_low-fit)*1000, c=tb_c[0], ls='dashed', alpha=0.5)
    ax_col[1].plot(theta/np.pi, (err_high-fit)*1000, c=tb_c[0], ls='dashed', alpha=0.5)
    ax_col[1].fill_between(theta/np.pi, (err_high-fit)*1000, (err_low-fit)*1000, color=tb_c[0], alpha=0.25)

    midplane_up = np.percentile(midplane_est - fit, 95)
    midplane_low = np.percentile(midplane_est - fit, 5)
    print(gal, midplane_up - midplane_low)

ax[0][0].set_ylabel(r'$\text{midplane}\,[\,\text{pc}\,]$')
ax[1][0].set_ylabel(r'$\text{midplane}\,[\,\text{pc}\,]$')

fig.tight_layout()
plt.savefig('midplane.pdf')





fig, ax = plt.subplots(1, 3, sharex=True, figsize=(textwidth,1.8))
# now make paper plot, with just fit
for gal,x in zip(glist,ax):
    out = np.load('output/out_'+gal+'.npy')
    theta = out[:,0]
    result = out[:,1:7]
    fit = out[:,7]

    midplane_est = result[:,0]
    err_low = result[:,1]
    err_high = result[:,2]

    midplane_vel = result[:,3]
    err_vel_low = result[:,4]
    err_vel_high = result[:,5]

    out_force = np.load('output/out_force_'+gal+'.npy')
    theta_force = out_force[:,0]
    result_force = out_force[:,1:7]
    fit_force = out_force[:,7]

    midplane_est_force = result_force[:,0]
    err_low_force = result_force[:,1]
    err_high_force = result_force[:,2]

    x.set_xlabel(r'$\phi/\pi$')

    x.text(0.05, 0.88, r'\texttt{'+gal+r'}', 
               horizontalalignment='left', 
               verticalalignment='center', 
               transform = x.transAxes)

    x.locator_params('x', nbins=5)
    x.locator_params('y', nbins=5)

    x.set_xlim(0, 2)
    x.set_xlim(0, 2)

    x.set_ylim(-200, 200)
    x.set_ylim(-200, 200)

    x.plot(theta/np.pi, (midplane_est-fit)*1000, c=tb_c[0])
    x.plot(theta/np.pi, (err_low-fit)*1000, c=tb_c[0], ls='dashed', alpha=0.5)
    x.plot(theta/np.pi, (err_high-fit)*1000, c=tb_c[0], ls='dashed', alpha=0.5)
    x.fill_between(theta/np.pi, (err_high-fit)*1000, (err_low-fit)*1000, color=tb_c[0], alpha=0.25)
    
    #x.plot(theta_force/np.pi, (midplane_est_force-fit_force)*1000, c=tb_c[1])
    #x.plot(theta_force/np.pi, (err_low_force-fit_force)*1000, c=tb_c[1], ls='dashed', alpha=0.5)
    #x.plot(theta_force/np.pi, (err_high_force-fit_force)*1000, c=tb_c[1], ls='dashed', alpha=0.5)
    #x.fill_between(theta_force/np.pi, (err_high_force-fit_force)*1000, (err_low_force-fit_force)*1000, color=tb_c[1], alpha=0.25)

ax[0].set_ylabel(r'$\text{midplane}\,[\,\text{pc}\,]$')

fig.tight_layout()
plt.savefig('midplane_fit.pdf')
