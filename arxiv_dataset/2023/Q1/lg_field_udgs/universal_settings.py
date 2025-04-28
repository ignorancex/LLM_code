#!/usr/bin/env python3

import os

import numpy as np

# Simulations
sim_n_part = 8192
sim_ids = ['09_18', '17_11', '37_11']

# Cosmology
h = 0.6777  # Hubble parameter
# rho_crit = 127.46  # Msun / kpc^3

# Analysis settings
n_sightings = 15000  # Number of mock SDSS observations per halo
MV_bins = np.arange(-20, -10, 0.5)  # Luminosity function bins
fiducial_lg_mass = 8.e12
target_lg_masses = np.asarray(
    [0.3e13, 0.4e13, 0.5e13, 0.6e13, 0.7e13, 0.8e13, 0.9e13])
zoa_extent_list = [5., 10., 15.]  # |b| <= x [deg]
obs_mw_r200 = 230.  # kpc
obs_m31_r200 = 275.  # kpc

# Survey settings
sdss_mu_lim = 29.  # Surface brightness limit (r-band) [mag / arcsec2]
des_mu_lim = 30.  # Surface brightness limit (r-band) [mag / arcsec2]
lsst_mu_lim = 31.  # Surface brightness limit (r-band) [mag / arcsec2]
sdss_app_mv_lim = 16.  # Apparent magnitude limit (V-band) [mag]
des_app_mv_lim = 17.5  # Apparent magnitude limit (V-band) [mag]
hsc_app_mv_lim = 20.  # Apparent magnitude limit (V-band) [mag]
lsst_app_mv_lim = 21.5  # Apparent magnitude limit (V-band) [mag]

# UDG selection criteria
b_mvir_ratio = 0.5  # M_baryon / M_virial
min_star_particles = 50  # Minimum number of star particles
reff = 1.  # Effective radius [kpc]
reff2 = 1.5  # Effective radius [kpc]
mu_e = 23.5  # Surface brightness within reff [mag arcsec^-2]
mu_e2 = 24.  # Surface brightness within reff [mag arcsec^-2]
max_mstar = 1.e9  # Max stellar mass [Msun]
min_mstar = 1.e6  # Min stellar mass [Msun]
d_lg = 2.5  # Radius of the Local Group [Mpc]

# File location settings
data_dir = "data"
generated_data_file_template = os.path.join(data_dir,
                                            '{}_{}_generated_paper_data.hdf5')
lg_galaxy_data_file = os.path.join(data_dir, 'mcconnachie_savino_subset.csv')
fattahi_data_file = os.path.join(data_dir, 'fattahi_2020_N_vs_M.csv')
survey_data_file = os.path.join(data_dir, 'survey_data.csv')
udg_file_template = os.path.join(data_dir, '{}_{}_z0_paper_data.hdf5')
k08_mv_vs_d_file = os.path.join(data_dir, 'k08_mag_d_data.csv')
k08_mu_vs_d_file = os.path.join(data_dir, 'k08_mu_d_data.csv')
mass_in_lg_file = os.path.join(data_dir,
                               '8192_lg_{}Mpc_volume_mass.csv'.format(d_lg))

# Plot settings
ax_limit_edge_adjustment = [0.95, 1.05]
sim_styles = {
    '09_18': {
        'marker': '.',
        'color': '#34B83B'
    },
    '37_11': {
        'marker': '^',
        'color': '#7E317B'
    },
    '17_11': {
        'marker': 'd',
        'color': 'cornflowerblue'
    }
}
