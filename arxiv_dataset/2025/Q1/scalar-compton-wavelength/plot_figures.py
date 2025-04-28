#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code to plot the figures in the the f(R) Compton wavelength paper.

Created: April 2024

Author: Bradley March
"""

# %%
# Python Preamble

# import relevant modules
import time
from os.path import join as joinpath
from os.path import exists as pathexists
from os import makedirs
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# import user created modules
from Solvers.fR2D import fR2DSolver
import Packages.fR_functions as fRf
import Packages.galaxy_relations as galf
import Packages.compt_functions as compt
from constants import pc, kpc, M_sun, rho_m

# plotting parameters
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 14
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['image.cmap'] = 'inferno'
cmap = plt.colormaps.get_cmap(plt.rcParams['image.cmap'])
size = 6
dpi = 300
savefigure = False

# check if folder to save figure exists and create if not
if not pathexists("figures"):
    makedirs("figures")

# Set up interactive figure windows
mpl.use("TkAgg")
plt.ion()

# get grid size parameters
r_min, r_max = galf.r_min, galf.r_max

# %%
# utility plotting functions


def get_full_polars(th: np.ndarray, r: np.ndarray, fg: np.ndarray,
                    extend_theta: bool = True, rmaxplotted: float = None
                    ) -> tuple:
    """
    Takes polar coordinates (th, r) and value to be plotted (fg), 
    extends over the theta coordinate (assuming periodicity at boundary) 
    to plot the full extent of the semi-circle, if extend_theta is True. 
    Also cuts off values at the max radius, rmaxplotted is not None.

    Parameters:
    th (numpy array): The theta coordinates.
    r (numpy array): The radial coordinates.
    fg (numpy array): The value to be plotted.
    extend_theta (bool): Whether to extend the theta coordinate.
    rmaxplotted (float): The maximum radius to plot.

    Returns:
    tuple: numpy arrays containing the extended theta, radial, 
        and value arrays.
    """

    # convert coordinates to vectors (if grid-like)
    if th.ndim == 2:
        th = th[0, :]
    if r.ndim == 2:
        r = r[:, 0]

    # add extra theta coordinate to plot over full semi-circle
    if extend_theta:
        th = np.hstack((th[-1] + np.pi, th, th[0] - np.pi))
        fg = np.hstack((fg[:, -1][:, None], fg, fg[:, 0][:, None]))

    # cut off the radial extent at rmaxplotted
    if rmaxplotted:
        mask = r < rmaxplotted
        r = r[mask]
        fg = fg[mask, :]

    return th, r, fg


# %% Figure 1
# Plots log values of:
# field solution            Compton wavelength
# screened field solution   density profile
tstart = time.time()

logMvir = np.log10(1.5e12)
logfR0 = -6
N_r, N_th = 512, 101

Mvir = M_sun * 10**logMvir
fR0 = -10**logfR0

# set up grid structure
grid = fR2DSolver(r_min, r_max, N_r, N_th)

# get densities
rhos = galf.get_densities(Mvir, grid.r, grid.theta, total=True)

# calculate fR, screened fR, and Lc
fR = fRf.load_solution(logfR0, logMvir, N_r, N_th)
fRscr = compt.calc_screened_fR(fR0, rhos['total'])
Lc = compt.calc_Compton_wavelength(fR, fR0)

datalist = [fRscr/fR0, rhos['total'], fR/fR0, Lc/pc]
datalabels = [r"$\log_{10}(f_R^\mathrm{scr}\ /\ f_{R0})$",
              r"$\log_{10}(\delta\rho\ /\ \mathrm{kg\,m^{-3}})$",
              r"$\log_{10}(f_R\ /\ f_{R0})$",
              r"$\log_{10}(\lambda_C\ /\ \mathrm{pc})$"]

# set max radial extent
rmaxplotted = 10*kpc
# get theta extended, raidally cutoff, coordinates & data
dataelist = []
for data in datalist:
    # extend data along theta direction and cutoff at radial extent
    the, re, datae = get_full_polars(
        grid.theta, grid.r, data, extend_theta=True, rmaxplotted=rmaxplotted)
    # log data & store for plotting
    dataelist.append(np.log10(datae))

# get vmax for fRscr
scrvmax = dataelist[2].max()

# set up figure
asp = 1
fig = plt.figure(figsize=(size, size / asp))

# panel sizes/positions
dX, dY = 0.45, 0.45
X1, Y1 = 0.16, 0.02
X2, Y2 = X1 + 0.228, Y1 + 0.5
# colorbar sizes/positions
cdX, cdY = 0.08, dY
cX1, cY1 = 0.15, 0.02
cX2, cY2 = 1 - cX1 - cdX, cY1 + 0.5

# create axes
ax1 = fig.add_axes([X1, Y1, dX, dY], projection='polar')
ax2 = fig.add_axes([X2, Y1, dX, dY], projection='polar')
ax3 = fig.add_axes([X1, Y2, dX, dY], projection='polar')
ax4 = fig.add_axes([X2, Y2, dX, dY], projection='polar')
axes = [ax1, ax2, ax3, ax4]
# create colorbar axes
cax1 = fig.add_axes([cX1, cY1, cdX, cdY])
cax2 = fig.add_axes([cX2, cY1, cdX, cdY])
cax3 = fig.add_axes([cX1, cY2, cdX, cdY])
cax4 = fig.add_axes([cX2, cY2, cdX, cdY])
caxes = [cax1, cax2, cax3, cax4]

# set general properties for panel axes
for ind, ax in enumerate(axes):
    ax.grid(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_theta_offset(np.pi / 2)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    # reverse direction for LHS panels
    ax.set_theta_direction((-1)**(ind))

# plot results
ticklocations = ['left', 'right']
for ind, ax in enumerate(axes):
    # get data
    datae = dataelist[ind]

    if ind == 0:
        vmax = scrvmax
    else:
        vmax = None

    # plot data
    im = ax.pcolormesh(the, re/kpc, datae, vmax=vmax, shading='auto')

    # create colorbar
    cbar = plt.colorbar(
        im, cax=caxes[ind], ticklocation=ticklocations[ind % 2], format='%1.1f')
    cbar.ax.set_ylabel(datalabels[ind], rotation=90+180*ind % 2)

plt.show()

if savefigure:
    fig.savefig(joinpath("figures", "Fig_1_fR_Lc_fRscr_rho.png"), dpi=dpi)
    print("Saved figure")

tfinish = time.time()
print(f"Figure 1 took {tfinish - tstart:.2f}s")

# %% Figure 2
# Plot showing Compton wavelength vs stellar separation
tstart = time.time()

logMvir = np.log10(1.5e12)
logfR0 = -6
N_r, N_th = 512, 101

Mvir = M_sun * 10**logMvir
fR0 = -10**logfR0

# set up grid structure
grid = fR2DSolver(N_r=N_r, N_th=N_th, r_min=r_min, r_max=r_max)
r = grid.r[:, 0]

# get densities
rhos = galf.get_densities(Mvir, grid.r, grid.theta,
                          splashback_cutoff=True, total=True)

# cutoff stellar density when it falls below the cosmic mean density
rhos['SD'][rhos['SD'][:, grid.disc_idx] < rho_m] = 0

# calculate stellar separation
sep = compt.calc_mean_separation(rhos['SD'][:, grid.disc_idx], Mp=M_sun)

# calculate Compton wavelength
fR = fRf.load_solution(logfR0, logMvir, N_r, N_th)[:, grid.disc_idx]
Lc = compt.calc_Compton_wavelength(fR, fR0)
Lcscr = compt.calc_screened_Compton_wavelength(fR0,
                                               rhos['total'][:, grid.disc_idx])

# plot results
asp = 3/2
fig, ax = plt.subplots(figsize=(size, size/asp))
ax.plot(r/kpc, Lc/pc, color=cmap(0.7),
        linestyle='-', label='$\lambda_\mathrm{C}$')
ax.plot(r/kpc, Lcscr/pc, color=cmap(0.7), linestyle='--',
        label='$\lambda_\mathrm{C}^\mathrm{scr}$')
ax.plot(r/kpc, sep/pc, color=cmap(0.2), linestyle='-.', label='S')
ax.set_xlabel("Galactocentric Radius [kpc]")
ax.set_ylabel("Length Scale [pc]")
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylim(0.5*Lcscr.min()/pc, 2*Lcscr.max()/pc)
ax.legend(loc='lower right')
fig.tight_layout()

plt.show()

if savefigure:
    fig.savefig(joinpath("figures", "Fig_2_Lc_vs_separation.png"), dpi=dpi)
    print("Saved figure")

tfinish = time.time()
print(f"Figure 2 took {tfinish - tstart:.2f}s")

# %% Figure 3
# Percent of stellar mass within valid regime (Lc > sep).
tstart = time.time()

dfR0 = 0.1
dMvir = 0.1
logfR0_Ar = np.arange(-8, -5+dfR0/2, dfR0)
logMvir_Ar = np.arange(10, 13.5+dMvir/2, dMvir)
N_r, N_th = 512, 101

# set up grid structure
grid = fR2DSolver(N_r=N_r, N_th=N_th, r_min=r_min, r_max=r_max)

# set up arrays to store results
percent_valid = np.zeros([logfR0_Ar.size, logMvir_Ar.size])
central_field = np.zeros_like(percent_valid)

# calculate the percent of stellar mass in regions where Lc > sep
for j, logMvir_j in enumerate(logMvir_Ar):
    Mvir = M_sun * 10**logMvir_j
    rhos = galf.get_densities(Mvir, grid.r, grid.theta, total=True)
    # cutoff stellar density when it falls below the cosmic mean density
    rhos['SD'][rhos['SD'] < rho_m] = 0
    sep = compt.calc_mean_separation(rhos['SD'], Mp=M_sun)
    M_sd = np.sum(rhos['SD'])
    for i, logfR0_i in enumerate(logfR0_Ar):
        fR0 = -10**logfR0_i

        # load solver solution
        fR = fRf.load_solution(logfR0_i, logMvir_j, N_r, N_th)

        # gather central field value to determine fully unscreend solutions
        central_field[i, j] = np.max(fR[0, :]) / fR0

        # calculate Lc
        Lc = compt.calc_Compton_wavelength(fR, fR0)

        # calculate indices where continuous density is valid
        valid_inds = Lc > sep

        # calculate percent of stellar mass within valid regime
        percent_valid[i, j] = np.sum(rhos['SD'][valid_inds]) / M_sd

# mask fully unscreened solutions
percent_valid = np.ma.masked_where(central_field > 1e-3, percent_valid)

# plot parameter space plot of results
asp = 4/3
fig, ax = plt.subplots(figsize=(size, size/asp))
im = ax.pcolormesh(logMvir_Ar, logfR0_Ar, 100*percent_valid,
                   shading='nearest', vmin=0, vmax=100)
cbar = plt.colorbar(im)
cbar.ax.set_ylabel(
    r"$\%$ of stellar mass in region where $\lambda_\mathrm{C}>S$")
ax.set_xlabel(r"$\log_{{10}}(M_\mathrm{{vir}}/M_\odot)$")
ax.set_ylabel(r"$\log_{{10}}(|f_{{R0}}|)$")

# add annotations for fully unscreened solutions
ax.annotate("Fully Unscreened", (10.5, -6.3), c='k', rotation=35, fontsize=16)
ax.annotate("Screened", (11.75, -7.3), c='w', rotation=35, fontsize=16)

fig.tight_layout()

plt.show()

if savefigure:
    fig.savefig(joinpath("figures", "Fig_3_percent_valid_stellar.png"), dpi=dpi)
    print("Saved figure")

tfinish = time.time()
print(f"Figure 3 took {tfinish - tstart:.2f}s")

# %% Figure 4
# Analysis of difference between rs^DM and rs^DM+SD
tstart = time.time()

dfR0 = 0.1
dMvir = dfR0
logfR0_Ar = np.arange(-8, -5+dfR0/2, dfR0)
logMvir_Ar = np.arange(10, 13.5+dMvir/2, dMvir)
N_r, N_th = 512, 101

# set up grid structure
grid = fR2DSolver(N_r=N_r, N_th=N_th, r_min=r_min, r_max=r_max)

# load arrays and prefill to -1 (fully unscreened solutions)
rs_DM_Ar = -np.ones([logfR0_Ar.size, logMvir_Ar.size])
rs_full_Ar = -np.ones_like(rs_DM_Ar)

for j, logMvir_j in enumerate(logMvir_Ar):
    for i, logfR0_i in enumerate(logfR0_Ar):
        # load rs values for DM only and DM + SD cases
        rs_DM = fRf.get_rs(logfR0_i, logMvir_j, N_r, N_th, SD=False)
        rs_full = fRf.get_rs(logfR0_i, logMvir_j, N_r, N_th, SD=True)
        # take the value at the disc index (if not fully unscreened)
        if isinstance(rs_DM, np.ndarray):
            rs_DM_Ar[i, j] = rs_DM[grid.disc_idx]
        if isinstance(rs_full, np.ndarray):
            rs_full_Ar[i, j] = rs_full[grid.disc_idx]

# mask arrays where fully unscreened in either DM only or DM +SD cases
rs_DM_Ar = np.ma.masked_where(rs_DM_Ar == -1, rs_DM_Ar)
rs_full_Ar = np.ma.masked_where(rs_full_Ar == -1, rs_full_Ar)
diff = rs_DM_Ar / rs_full_Ar

# plot parameter space plot of (rs^DM+SD - rs^DM) and (rs^DM / rs^DM+SD)
asp = 3/2
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(size, size/asp), sharey=True)
# plot rs^DM+SD - rs^DM
im1 = ax1.pcolormesh(logMvir_Ar, logfR0_Ar, np.ma.log10(
    (rs_full_Ar-rs_DM_Ar)/kpc), shading='nearest')
cbar1 = plt.colorbar(im1, ax=ax1, location='top', aspect=10, format='%1.1f')
cbar1.ax.set_xlabel(
    r"$\log_{10}((r_s^\mathrm{DM+SD}-r_s^\mathrm{DM})/\mathrm{kpc})$",
    labelpad=10)
# plot rs^DM / rs^DM+SD
im2 = ax2.pcolormesh(logMvir_Ar, logfR0_Ar, diff, shading='nearest')
cbar2 = plt.colorbar(im2, ax=ax2, location='top', aspect=10, format='%1.1f')
cbar2.ax.set_xlabel(r"$r_s^\mathrm{DM}\ /\ r_s^\mathrm{DM+SD}$", labelpad=10)
# format figure
ax1.set_xlabel(r"$\log_{{10}}(M_\mathrm{{vir}}/M_\odot)$")
ax2.set_xlabel(r"$\log_{{10}}(M_\mathrm{{vir}}/M_\odot)$")
ax1.set_ylabel(r"$\log_{{10}}(|f_{{R0}}|)$")
fig.tight_layout()

plt.show()

if savefigure:
    fig.savefig(joinpath("figures", "Fig_4_rs_DM_vs_rs_DM_SD.png"), dpi=dpi)
    print("Saved figure")

tfinish = time.time()
print(f"Figure 4 took {tfinish - tstart:.2f}s")
