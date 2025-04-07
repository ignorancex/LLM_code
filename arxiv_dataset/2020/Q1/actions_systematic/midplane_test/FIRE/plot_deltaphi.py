import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from scipy.interpolate import interp1d

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
Rsolar = 8.2

glist = ['m12i', 'm12f', 'm12m']

zoff_100_list = [61.47799653, 925.58008488]
zoff_10_list = [6.19967419, 47.17989471]
clist = [tb_c[7], tb_c[8]]

theta = np.load('output/theta.npy')

def print_dphi(dphi, rng):
    interp = interp1d(dphi, rng)
    def to_min(p):
        return np.abs(interp(p)-zoff_10_list[0]/1000)
    res = minimize(to_min, 0, bounds=[[0, 2.*np.pi]])
    print('dphi/pi:', res.x/np.pi)
    return res.x

def rotate(l, n):
    return np.append(l[n:], l[:n])

def get_range_vs_dphi(theta, midplane):
    dphi_list_list = []
    r_list_list = []
    for i in range(len(theta)):
        m = rotate(midplane, i)
        t = rotate(theta, i)
        t -= t[0]
        t = np.mod(t + np.pi, 2.*np.pi) - np.pi
        dphi_list = [0]
        r_list = [0]
        for j in rotate(range(int(len(theta)/2.0)), 1):
            p = np.append(t[-j:], t[0:j+1]) 
            rl = np.append(m[-j:], m[0:j+1])
            dphi = np.max(p) - np.min(p)
            r = np.max(rl) - np.min(rl)
            dphi_list.append(dphi)
            r_list.append(r)
        dphi_list_list.append(dphi_list)
        r_list_list.append(r_list)
    return np.array(dphi_list_list), np.array(r_list_list)


fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(textwidth,2))
# now make paper plot, with just fit
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

    ax_col.set_xlabel(r'$\Delta \phi/\pi$')

    ax_col.text(0.05, 0.88, r'\texttt{'+gal+r'}', 
               horizontalalignment='left', 
               verticalalignment='center', 
               transform = ax_col.transAxes)

    ax_col.set_xlim(0, 2)

    ax_col.set_ylim(0, 300)

    for z, c in zip(zoff_10_list, clist):
        ax_col.axhline(z, color=c)


    m = midplane_est - fit

    dphi_list_list, r_list_list = get_range_vs_dphi(theta, m)
    for dphi_list, r_list in zip(dphi_list_list, r_list_list):
        ax_col.plot(dphi_list/np.pi, r_list*1000,  c=tb_c[-1], alpha=0.15, lw=1)

    dphi_mean = np.mean(dphi_list_list, axis=0)
    r_mean = np.mean(r_list_list, axis=0) 
    r_sigma = np.std(r_list_list, axis=0, ddof=1)
    r_up = r_mean + r_sigma
    r_low = r_mean - r_sigma
    ax_col.plot(dphi_mean/np.pi, r_mean*1000, c=tb_c[0])
    ax_col.plot(dphi_mean/np.pi, r_up*1000, alpha=0.75, ls='dashed', c=tb_c[0])
    ax_col.plot(dphi_mean/np.pi, r_low*1000, alpha=0.75, ls='dashed', c=tb_c[0])

    out = np.transpose([dphi_mean, r_mean, r_sigma])
    np.save('output/r_vs_dphi_'+gal+'.npy', out)

    # chord length
    def tick_function(l, round=True):
        a = 2*Rsolar*np.sin(0.5*(l/2)*np.pi)
        if round:
            a = np.round(a, 1)
        return a

    t = print_dphi(dphi_mean, r_mean)
    print(tick_function(t/np.pi, round=False))

    ax_col.locator_params('x', nbins=5)

    ax2 = ax_col.twiny()
    #ax2.invert_xaxis()
    ax2.set_xticks(ax_col.get_xticks())
    ax2.set_xbound(ax_col.get_xbound())
    ax2.set_xticklabels(tick_function(ax_col.get_xticks()))
    #ax2.set_xlabel(r'$\nu\,[\,\text{MHz}\,]$')
    ax2.set_xlabel(r'$\text{chord length}\,[\,\text{kpc}\,]$')



ax[0].set_ylabel(r'$\text{range}\,[\,\text{pc}\,]$')

fig.tight_layout()
fig.savefig('range_dphi.pdf')
