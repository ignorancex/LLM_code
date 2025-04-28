# this script just automates the plotting of the sensitivities
import numpy as np
import random

import os
import scipy.stats as sts
import pandas as pd

import sys
import pickle
from matplotlib import pyplot as plt
sys.path.append('./../packages')
data_path = "../data/"
post_path = "../data/"
from hunt import perf_pval, count_from_df, nbkgmax, nsigmax
from style import le_size, la_size, ti_size, majt_size, mint_size, colormap, color_list, mfixed_labs, figwidth


from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--smear', type=int, help="small or large?", default=0)
parser.add_argument('--Nsam', type=int, help="how many points to sample (larger for smoother)", default=10000)
parser.add_argument('--mfixed', type=float, help="fixed mass for signal", default=1.)
parser.add_argument('--opt_thr', type=float, help="threshold for plots (must be in list used before)", default=0.6)


args = parser.parse_args()

smear = args.smear
mfixed = args.mfixed
Nsamples = args.Nsam
Nobs = 10000 # fixed by file size
opt_thro = args.opt_thr # can fix the threshold as input 
opt_thrp = args.opt_thr # can fix the threshold as input


if smear ==0:
    sigs, labsm = [0.001,0.01,0.005, 0.005 ], "small" # small smearing case
elif smear == 1:
    sigs, labsm = [0.001,0.05,0.01,  0.01  ], "large"

# from before, could be changed at will
if mfixed == 0.2:
    opt_bin = 4

elif mfixed == 1.:
    opt_bin = 11 # could be put differently for small and large

elif mfixed == 4.:
    if smear:
        opt_bin = 30 # could be put differently for small and large
    elif smear ==0:
        opt_bin = 48



# increase number of spaces for better visualization
mubspace = np.linspace(0.1, 6, 25)
musspace = np.linspace(0.1, 4, 25)


# we parallelize the process over the normalizations
def TS_par(dfc_opt, mub,mus):
    nbkgsig = np.array([sts.poisson(mub).rvs(Nsamples), sts.poisson(mus).rvs(Nsamples)])
    nnurbkg = np.array([sts.poisson(mub).rvs(Nsamples), np.zeros(Nsamples)])

    countB, countS = count_from_df(dfc_opt, nnurbkg, nbkgsig)

    return([mub, mus, perf_pval(countB, countS).meanp(), perf_pval(countB, countS).meanlogp(), perf_pval(countB, countS).meansigma()])


with open("../performances/counts/count_bump_"+labsm+"_m_"+str(mfixed)+".pickle", "rb") as fp:  
    countsS = pickle.load(fp)
dfc = pd.DataFrame(countsS, columns=["nbkg", "nsig", "nbins", "counts"])
dfc_opt = dfc.loc[dfc["nbins"]==opt_bin]
TS_bump = np.array((Parallel(n_jobs=num_cores)(delayed(TS_par)(dfc_opt, mub, mus) for mub in mubspace for mus in musspace )))

with open("../performances/counts/count_cl_"+labsm+"_m_"+str(mfixed)+".pickle", "rb") as fp:  
    countsS = pickle.load(fp)
dfc = pd.DataFrame(countsS, columns=["nbkg", "nsig", "thresh", "counts"])
dfc_opt = dfc.loc[round(dfc["thresh"],2)==opt_thro]
TS_ECO = np.array((Parallel(n_jobs=num_cores)(delayed(TS_par)(dfc_opt, mub, mus) for mub in mubspace for mus in musspace )))


with open("../performances/counts/count_cl_post_"+labsm+"_m_"+str(mfixed)+".pickle", "rb") as fp:  
    countsS = pickle.load(fp)
dfc = pd.DataFrame(countsS, columns=["nbkg", "nsig", "thresh", "counts"])
dfc_opt = dfc.loc[round(dfc["thresh"],2)==opt_thrp]
TS_EPO = np.array((Parallel(n_jobs=num_cores)(delayed(TS_par)(dfc_opt, mub, mus) for mub in mubspace for mus in musspace )))

mux, muy = np.meshgrid(musspace, mubspace)

# plot the sensitivities
def plot_heat(axs, arr0, arr1, arr2, ifeat):

    minv, maxv = np.min([np.min(arr0[:,ifeat]), np.min(arr1[:,ifeat]), np.min(arr2[:,ifeat])]), np.max([np.max(arr0[:,ifeat]), np.max(arr1[:,ifeat]), np.max(arr2[:,ifeat])])

    a0 = axs[0].pcolormesh(muy, mux, (arr0[:,ifeat]).reshape([len(mubspace), len(musspace)]), vmin = minv, vmax=maxv, cmap=colormap)
    a1 = axs[1].pcolormesh(muy, mux, (arr1[:,ifeat]).reshape([len(mubspace), len(musspace)]), vmin = minv, vmax=maxv, cmap=colormap)
    a2 = axs[2].pcolormesh(muy, mux, (arr2[:,ifeat]).reshape([len(mubspace), len(musspace)]), vmin = minv, vmax=maxv, cmap=colormap)



    axs[0].set_title("Bump hunt", fontsize=ti_size)
    axs[1].set_title("ECO hunt ", fontsize=ti_size)
    axs[2].set_title("EPO hunt ", fontsize=ti_size)



    axs[1].yaxis.set_tick_params(labelleft=True)
    axs[2].yaxis.set_tick_params(labelleft=True)

    for icol in range(3):
            axs[icol].tick_params(axis='both', which='major', labelsize=majt_size)
            axs[icol].tick_params(axis='both', which='minor', labelsize=mint_size)
            axs[icol].set_ylabel(r"$\mu_s$", fontsize = la_size)
            axs[icol].set_xlabel(r"$\mu_b$", fontsize = la_size)
            axs[icol].set_xlim(mubspace.min(),mubspace.max())
            axs[icol].set_ylim(musspace.min(),musspace.max())
    
    if ifeat ==3:
        pl_levels =  [2, 4, 6, 8, 10]
    elif ifeat ==4:
        pl_levels = [1, 2, 3]

    CS = axs[0].contour(muy, mux,    TS_bump[:,ifeat].reshape([len(mubspace), len(musspace)]), levels =pl_levels, colors='black', linestyles='--', linewidths = 1.2)
    axs[0].clabel(CS, inline=True, fontsize=10)

    CS = axs[1].contour(muy, mux,  TS_ECO[:,ifeat].reshape([len(mubspace), len(musspace)]), levels = pl_levels, colors='black', linestyles='--', linewidths = 1.2)
    axs[1].clabel(CS, inline=True, fontsize=10)

    CS = axs[2].contour(muy, mux, TS_EPO[:,ifeat].reshape([len(mubspace), len(musspace)]), levels = pl_levels, colors='black', linestyles='--', linewidths = 1.2)
    axs[2].clabel(CS, inline=True, fontsize=10)

    cbar_ax = fig.add_axes([0.88, 0.11, 0.02, 0.77])
    cbar = fig.colorbar(a0, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=12)
    if ifeat == 3:
        cbar.set_label(r"$-\langle \log p_b\rangle_{p_s}$", fontsize=16)
    elif ifeat == 4:
        cbar.set_label(r"$\langle Z \rangle_s$", fontsize=16)

fig, axs = plt.subplots(ncols = 3,  figsize=(figwidth,5.2), sharey = True, gridspec_kw={"wspace": 0.275})
plot_heat(axs, TS_bump, TS_ECO, TS_EPO, 4)
fig.subplots_adjust(right=0.85)
fig.suptitle(r"$\langle Z \rangle_s$"+" sensitivity for $m_a=${}GeV and ".format(mfixed)+labsm+" uncertainty", fontsize=ti_size)


plt.savefig("../figures/sigma_thr_"+str(opt_thro)+"_"+labsm+"_m_"+str(mfixed)+".pdf", bbox_inches='tight')

from scipy.ndimage import gaussian_filter

nsteps = 8
prec = 2

# plot the improvement in sensitivity
def plot_improv(ax, arr0, arr1, ifeat):

    minv, maxv = np.min([np.min(TS_ECO[:,ifeat]-TS_bump[:,ifeat]), np.min(TS_EPO[:,ifeat]-TS_bump[:,ifeat])]), np.max([np.max(TS_ECO[:,ifeat]-TS_bump[:,ifeat]), np.max(TS_EPO[:,ifeat]-TS_bump[:,ifeat])])
    stepsize = np.round((maxv-minv)/nsteps, prec)
    levels = round(minv, prec)+[stepsize*i for i in range (nsteps+1)]

   
    plot_data = gaussian_filter((arr0[:,ifeat]-arr1[:,ifeat]), 0) # can change smoothing kernel here, no smoothing by default
    CS = ax.contour(muy, mux, plot_data.reshape([len(mubspace), len(musspace)]), levels = levels)
    ax.clabel(CS, inline=True, fontsize=10)
    ax.set_ylabel(r"$\mu_s$", fontsize = la_size)
    ax.set_xlabel(r"$\mu_b$", fontsize = la_size)
    ax.tick_params(axis='both', which='major', labelsize=majt_size)
    ax.tick_params(axis='both', which='minor', labelsize=mint_size)

fig, axs = plt.subplots(ncols = 2, figsize=(12,5), sharey = True)

plot_improv(axs[0], TS_ECO, TS_bump, 4)
plot_improv(axs[1], TS_EPO, TS_bump, 4)

axs[1].yaxis.set_tick_params(labelleft=True)

axs[0].set_title("ECO hunt over bump hunt", fontsize=ti_size)
axs[1].set_title("EPO hunt over bump hunt", fontsize=ti_size)
fig.suptitle(r"$\langle Z \rangle_s$"+" sensitivity improvement for $m_a=${}GeV and ".format(mfixed)+labsm+" uncertainty", fontsize=ti_size)

plt.savefig("../figures/sigma_thr_"+str(opt_thro)+"_improv_"+labsm+"_m_"+str(mfixed)+".pdf", bbox_inches='tight')
