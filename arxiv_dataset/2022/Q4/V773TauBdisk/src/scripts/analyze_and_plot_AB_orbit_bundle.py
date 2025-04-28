# adapted from analyse_and_plot_AB_orbit.ipynb
import paths
import numpy as np
import emcee
import corner

import orbitize
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
from orbitize import driver
plt.rcParams.update({'font.size': 16})

import orbitize.results
myResults_AB = orbitize.results.Results()
myResults_AB.load_results(paths.data / 'v773_tau_AB.hdf5')

fig, ax = plt.subplots(1,1,figsize=(8,4))
orbit_figure = myResults_AB.plot_orbits(
    start_mjd=53440, # minimum MJD for colorbar (choose first data epoch)
    sep_pa_end_year = 2040,
    plot_astrometry=True
)

plt.savefig(paths.figures / 'v773_tau_b_orbits-long_burn.pdf')

# calcualte the means period for AB and next eclipse

# How to get orbital period in the orbitize! standard basis
sma = myResults_AB.post[:, myResults_AB.param_idx['sma1']]
mtot_fit = myResults_AB.post[:, myResults_AB.param_idx['mtot']]
mtot = (2.91+2.35) # Solar masses, total mass
period = np.sqrt(sma**3/mtot_fit) # years, period

fig, ax = plt.subplots(1,1,figsize=(6,6))
ax.hist(period,bins=20);
plt.savefig('_check_orbital_period_AB.pdf')

period_dist = np.percentile(period, [16, 50, 84])
q = np.diff(period_dist)
txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{+{2:.3f}}} yr"
txt = txt.format(period_dist[1], q[0], q[1], "P")
print(txt)

from astropy.time import Time

t_ecl = Time(55450.,format='mjd')
print('previous eclipse was at {:.2f}'.format(t_ecl.jyear))
print("Midpoint of next eclipse is at {:.2f}".format(t_ecl.jyear+period_dist[1]))

# remove degenerate orbits in \omega and \Omega

# How to get orbital period in the orbitize! standard basis
sma = myResults_AB.post[:, myResults_AB.param_idx['sma1']]
ecc = myResults_AB.post[:, myResults_AB.param_idx['ecc1']]
inc = myResults_AB.post[:, myResults_AB.param_idx['inc1']]
aop = myResults_AB.post[:, myResults_AB.param_idx['aop1']]
pan = myResults_AB.post[:, myResults_AB.param_idx['pan1']]
# ['sma1', 'ecc1', 'inc1', 'aop1', 'pan1', 'tau1', 'plx', 'mtot']
# aop == little omega
# pan1 == big Omega
mgood = (pan > (270*3.141/180)) * (pan<(350*3.141/180))


import copy
cleanorb = copy.deepcopy(myResults_AB)
fig, ax = plt.subplots(1,1,figsize=(6,6))
corner_figure = cleanorb.plot_corner()
plt.savefig('_check_triangle_plot_AB.pdf')

# print out orbital parameters in latex
from IPython.display import display, Math

for par_label in cleanorb.labels:
    mcmc = np.percentile(cleanorb.post[:, cleanorb.param_idx[par_label]], [16, 50, 84])

    # convert angles from radians to degrees
    if par_label == 'inc1' or par_label == 'aop1' or par_label == 'pan1':
        mcmc = mcmc * 180/np.pi

    q = np.diff(mcmc)
    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{+{2:.3f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], par_label)
    print(txt)



# How to get orbital period in the orbitize! standard basis
sma = cleanorb.post[:, cleanorb.param_idx['sma1']]
mtot_fit = cleanorb.post[:, cleanorb.param_idx['mtot']]
period = np.sqrt(sma**3/mtot_fit) # years, period
plx = cleanorb.post[:, cleanorb.param_idx['plx']]

# calculate t_periastron
tau_ref_epoch = cleanorb.tau_ref_epoch
tp = orbitize.basis.tau_to_tp(cleanorb.post[:, cleanorb.param_idx['tau1']], tau_ref_epoch, period)

tpp = Time(tp,format='mjd')

fig, ax = plt.subplots(1,3,figsize=(15,5))
ax[0].hist(period,bins=20);
ax[1].hist(tpp.jyear-period,bins=20);

period_dist = np.percentile(period, [16, 50, 84])
q = np.diff(period_dist)
txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{+{2:.3f}}} yr"
txt = txt.format(period_dist[1], q[0], q[1], "P")
print(txt)

tperi_dist = np.percentile(tpp.jyear-period, [16, 50, 84])
q = np.diff(tperi_dist)
txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{+{2:.3f}}} yr"
txt = txt.format(tperi_dist[1], q[0], q[1], "t_{peri}")
print(txt)

sma_mas = sma * plx

ax[2].hist(sma_mas,bins=20);

smmas = np.percentile(sma_mas, [16, 50, 84])
q = np.diff(smmas)
txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{+{2:.3f}}} mas"
txt = txt.format(smmas[1], q[0], q[1], "a")
print(txt)

ax[2].set_xlabel("Semimajor axis [mas]")
ax[1].set_xlabel("t_{periastron} [yr]")
ax[0].set_xlabel("Orbital period [yr]")

plt.savefig("_check_AB_P_Tperi_a_histograms.pdf")
