import sys
sys.path.append('../')

from schmactions import schmactions

import numpy as np

import astropy.units as u
import gala.dynamics as gd
import gala.potential as gp
import pickle

from scipy.interpolate import interp1d
from scipy.optimize import minimize

import matplotlib as mpl
from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

textwidth = 7.10000594991
columnwidth = 3.35224200913

tb_c = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
        '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac']

Rsolar = 8.2

#cluster min, max mass
mc_min = 100
mc_max = 10000
mc_list = np.linspace(mc_min, mc_max, 1000)

# find mass enclosed within Rsolar
mw = gp.MilkyWayPotential()
q = gd.PhaseSpacePosition(pos=[Rsolar, 0, 0] * u.kpc, vel=[0, 0, 0] * u.km/u.s)
M_enclosed = float(mw.mass_enclosed(q).to_value(u.Msun))

dJzJz_target = np.power(mc_list / M_enclosed, 1/3)

# load in results
fname_list = ['thin', 'thick']
name_list = ['thin-disk', 'thick-disk']

def zoffset_gen(offlist, dJzJz):
    interp = interp1d(offlist, dJzJz)

    def to_minimize(off, target):
        return np.abs(interp(float(off)) - target)

    offlist_target = []
    for this_dJzJz in dJzJz_target:
        this_off = float(minimize(to_minimize, 0, args=(this_dJzJz,)).x)
        offlist_target.append(this_off)
    offlist_target = np.array(offlist_target)
    return offlist_target

def load_orbit(name, init_pos, init_vel):
    s = schmactions(init_pos, init_vel)
    zout = pickle.load(open('zout_'+name+'.p', 'rb'))
    res = pickle.load(open('true_res_'+name+'.p', 'rb'))
    J0, J1, J2 = res['actions'].to_value(u.kpc*u.km/u.s)

    zact = np.array([ s.extract_actions(r) for r in zout['act_result'] ]) 
    zup = np.percentile(zact, 95, axis=1)
    zlow = np.percentile(zact, 5, axis=1)
    dz = (zup - zlow)/2.0

    offlist = np.array(zout['offset_list'])[:,2]
    dJzJz = dz[:,2]/J2
    return offlist, dJzJz

init_pos = [8, 0, 0] * u.kpc
init_vel_list = [[0, -190, 10] * u.km/u.s,
                 [0, -190, 50] * u.km/u.s,
                 [0, -190, 190] * u.km/u.s]
clist = [tb_c[7], tb_c[8], tb_c[0]]

# set up figure
fig, ax = plt.subplots(1, 1, figsize=(columnwidth, 2.5))

for fname, name, init_vel, c in zip(fname_list, name_list, init_vel_list, clist):
    offlist, dJzJz = load_orbit(fname, init_pos, init_vel)

    offlist_target = zoffset_gen(offlist, dJzJz)

    ax.plot(mc_list, offlist_target*1000, c=c, label=name)

ax.set_xlabel(r'$m_c\,[\,M_{\odot}\,]$')
ax.set_ylabel(r'$z\,\text{offset}\,[\,\text{pc}\,]$')

ax.set_xscale('log')
ax.set_yscale('log')

ax.set_ylim((10, 2000))
ax.set_xlim((mc_min, mc_max))

ax.legend(frameon=False, title='orbit')

fig.tight_layout()
fig.savefig('cluster_offset.pdf')
plt.close()

# now begin making Rn vs mc plot

glist = ['m12i', 'm12f', 'm12m']
fig, ax_list = plt.subplots(1, 3, figsize=(textwidth, 2.7))

def chord(l):
        return np.round(2*Rsolar*np.sin(0.5*(l/2)*np.pi), 1)

for ax, gal in zip(ax_list, glist):
    dat = np.load('output/r_vs_dphi_'+gal+'.npy')
    dphi = dat[:,0]
    rng = dat[:,1]
    rng_sigma = dat[:,2]
    chord_length = chord(dphi/np.pi)
    for fname, name, init_vel, c in zip(fname_list, name_list, init_vel_list, clist):
        offlist, dJzJz = load_orbit(fname, init_pos, init_vel)

        offlist_target = zoffset_gen(offlist, dJzJz)

        interp = interp1d(rng, chord_length)
        interp_low = interp1d(rng-rng_sigma, chord_length)
        interp_high = interp1d(rng+rng_sigma, chord_length)

        def to_minimize(off, target, interp):
            # print(off)
            return np.abs(interp(float(off)) - target)

        chord_max = np.max(chord_length)

        Rn_target = []
        Rn_low_target = []
        Rn_high_target = []
        for this_off_target in offlist_target:
            if this_off_target >= np.max(rng):
                Rn_target.append(chord_max)
            else:
                this_Rn = interp(this_off_target)
                Rn_target.append(this_Rn)

            if this_off_target >= np.max(rng-rng_sigma):
                Rn_low_target.append(chord_max)
            else:
                this_Rn = interp_low(this_off_target)
                Rn_low_target.append(this_Rn)

            if this_off_target >= np.max(rng+rng_sigma):
                Rn_high_target.append(chord_max)
            else:
                this_Rn = interp_high(this_off_target)
                Rn_high_target.append(this_Rn)
        Rn_target = np.abs(Rn_target)
        Rn_low_target = np.abs(Rn_low_target)
        Rn_high_target = np.abs(Rn_high_target)

        ax.plot(mc_list, Rn_target, c=c, label=name)
        ax.plot(mc_list, Rn_low_target, c=c, ls='dashed')
        ax.plot(mc_list, Rn_high_target, c=c, ls='dashed')

        if gal=='m12f':
            l = 0.05
        elif gal=='m12m':
            l = 0.4
        elif gal=='m12i':
            l = 0.12
        ax.text(l, 0.88, r'\texttt{'+gal+r'}', 
               horizontalalignment='left', 
               verticalalignment='center', 
               transform = ax.transAxes)

for ax in ax_list:
    ax.set_xlabel(r'$m_c\,[\,M_{\odot}\,]$')
    ax.set_xscale('log')
    ax.set_ylim((0.2, 20))
    ax.set_xlim((mc_min, mc_max))
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())

ax_list[0].set_ylabel(r'$R_n[\,\text{kpc}\,]$')

ax.legend(frameon=False, title='orbit')

fig.tight_layout()
fig.savefig('Rn_vs_mc.pdf')
plt.close()
