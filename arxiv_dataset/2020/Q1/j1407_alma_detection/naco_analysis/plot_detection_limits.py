# -*- coding: utf-8 -*-

__author__ = "Alexander Bohn"

import numpy as np
import os
from astropy.io import fits
from astropy.modeling import models,fitting
import photutils as pu
from matplotlib import pyplot as plt
from astropy.table import Table
from astropy.io import ascii
from scipy.integrate import dblquad
from astropy.stats import sigma_clip
from scipy.interpolate import interp1d
import math as ma
from AtmospherModelReading import AtmosphereModel
import astropy.units as u
import astropy.constants as c

def set_plot_parameters():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['xtick.labelsize'] = 30
    plt.rcParams['ytick.labelsize'] = 30
    plt.rcParams['xtick.major.size'] = 3
    plt.rcParams['xtick.major.width'] = 2
    plt.rcParams['xtick.minor.size'] = 3
    plt.rcParams['xtick.minor.width'] = 2
    plt.rcParams['ytick.major.size'] = 3
    plt.rcParams['ytick.major.width'] = 1
    plt.rcParams['ytick.minor.size'] = 3
    plt.rcParams['ytick.minor.width'] = 1
    plt.rcParams['xtick.major.pad'] = 10
    plt.rcParams['ytick.major.pad'] = 10
    plt.rcParams['axes.linewidth'] = 1
    plt.rcParams['axes.labelsize'] = 40
    plt.rcParams["axes.titlesize"] = 40
    plt.rcParams['figure.figsize'] = (15, 12)
    plt.rcParams['figure.dpi'] = 400
    plt.rcParams['lines.linewidth'] = 5
    plt.rcParams['lines.markersize'] = 10
    plt.rcParams['legend.fontsize'] = 25
    plt.rcParams['legend.markerscale'] = 1

target = "J1407"
par = 0.007183474683431726
dist = 1./par
pixscale = 0.02719
filt = "L_prime"
instrument = "NaCo"
atmosphere_model = "dusty"

a = lambda M, m, T: (c.G*(M+m)*T**2/(4.*np.pi**2))**(1./3.)

# Stellar parameters
star_age = 0.016
star_mag = 9.2 # TODO find better estimate
star_metalicity = 0.

pca_numbers = np.array([1,5,10])

model = AtmosphereModel("AMES-%s"%atmosphere_model,
                        metallicity=0.,
                        instrument=instrument,
                        magnitude_ref="Vega")

model_4 = model.get_model(age=star_age)
masses = (model_4["M/Ms"].data * u.M_sun).to(u.M_jup).value[:-1]

# Define directory
path_in = "/Volumes/BOHN/work/data/targets/J1407/2019-03-01/NACO/L_prime/0_2/Results/detection_limits/"

# Prepare plots
# set_plot_parameters()

for pca in pca_numbers:

    f_res, ax1_res = plt.subplots()

    y_lim = (4.5,9.5)


    M_star = star_mag + 5. * (1. - np.log10(dist))

    # Load magnitude to mass conversion
    mag = model_4["L\'"].data[:-1]
    mag_to_mass = interp1d(mag, masses,bounds_error=False,fill_value=np.inf)

    y_ticks_mass = mag_to_mass(M_star+np.arange(y_lim[0],y_lim[1]+1,1))

    data_1 = np.loadtxt(os.path.join(path_in, "detection_limits_pca_%s.dat" % pca))
    # data_2 = np.loadtxt(os.path.join(path,"detection_limits_pca_5.dat"))

    separation_1 = data_1[:,0]
    contrast_1 = data_1[:,1]

    ax1_res.plot(separation_1,contrast_1,linewidth=2.,label="1.5h, VLT/NACO, L'",zorder=2)

    # plot settings
    ax2_res = ax1_res.twiny()
    ax1_res.set_xlabel('Projected separation [arcsec]')
    ax1_res.set_ylabel(r"$5\sigma$ magnitude contrast [mag]")
    ax1_res.invert_yaxis()
    ax1_res.set_xlim((0, separation_1[-1]))
    ax1_res.set_ylim((y_lim[1], y_lim[0]))
    # ax1_res.grid(True)
    ax1_res.legend(loc=0)

    # Create second x-axis with AU
    ax1Xs = ax1_res.get_xticks()

    ax2_res.set_xticks(np.arange(0,dist*separation_1[-1],25))
    ax2_res.set_xlim((0, dist*separation_1[-1]))
    ax2_res.set_xlabel('Projected separation [au]')

    # Create second y-axis with M_jup
    ax3_res = ax1_res.twinx()
    ax3_res.invert_yaxis()
    ax3_res.set_yticks(ax1_res.get_yticks())
    ax3_res.set_ylim(ax1_res.get_ylim())
    ax3_res.set_yticklabels(np.ceil(y_ticks_mass))
    ax3_res.set_ylabel(r'$5\sigma$ mass limit [$M_{jup}$]')

    f_res.savefig(os.path.join(path_in, "%s_%s_detection_limits_%s_pca_%s.pdf" % (target, filt, atmosphere_model, pca)))
    plt.close(f_res)
    # plt.show()