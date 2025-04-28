# adapted from analyse_and_plot_C_orbit.ipynb


import paths
import numpy as np
import emcee
import corner

import orbitize
import matplotlib.pyplot as plt

import warnings
#warnings.filterwarnings('ignore')
from orbitize import driver
plt.rcParams.update({'font.size': 16})

import orbitize.results
myResults_C = orbitize.results.Results()
myResults_C.load_results(paths.data / 'v773_tau_C_AB_bary_preselected_orbparams.hdf5')

fig, ax = plt.subplots(1,1,figsize=(8,4))
orbit_figure = myResults_C.plot_orbits(
    start_mjd=55440, # minimum MJD for colorbar (choose first data epoch)
    sep_pa_end_year = 2040
)

plt.savefig("_check_C_orbits.pdf")

# Eggleton criterion from https://ui.adsabs.harvard.edu/abs/1995ApJ...455..640E/abstract
# Eggleton and Kiseleva (1995)

m1 = 2.91
m2 = 2.35
m3 = 0.7

qin = m1/m2
qout = (m1+m2)/m3

Q1 = np.power(qout,1./3)

print('Qin is {:.2f}'.format(qin))
print('Qout is {:.2f}'.format(qout))


# Equation 1 from Eggleton and Kiseleva (1995)
Ymin0 = 1 + (3.7/Q1) + (2.2/(1+Q1)) + (1.4/np.power(qin,1./3))*(Q1-1)/(Q1+1)

print('Value of $Y_{{min}}^0$ is {:.2f}'.format(Ymin0))

# Ymin0 = periastron distance of outer orbit/apastron distance of inner orbit

# periastron distance of outer orbit = Ymin0 * apastron distance of inner orbit

apastron_inner = 15.3 * (1+0.10) # from A orbital fitting
print("Apastron for AB binary orbit is about {:.1f} au".format(apastron_inner))
minimum_peri_outer = Ymin0 * apastron_inner
print("Minimum periastron distance for C is {:.2f}au".format(minimum_peri_outer))

# What is the period of the C system?

sma = myResults_C.post[:, myResults_C.param_idx['sma1']]
mtot_fit = myResults_C.post[:, myResults_C.param_idx['mtot']]
mtot = (2.91+2.35) # Solar masses, total mass
period = np.sqrt(sma**3/mtot_fit) # years, period


fig, ax = plt.subplots(1,1,figsize=(6,4))
ax.hist(period,bins=20,range=(0,1000));
ax.set_xlabel("Period of C [yr]")
ax.set_ylabel("N")
plt.savefig("_check_period_of_C.pdf")

period_dist = np.percentile(period, [16, 50, 84])
q = np.diff(period_dist)
txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{+{2:.3f}}} yr"
txt = txt.format(period_dist[1], q[0], q[1], "P")
print("Orbital period of C with no constraints on periastron:")
print(txt)

# Calculate the periastron distance of all the orbits
# we rule out orbits with periastrons smaller than the Eggleton criterion

# How to get orbital period in the orbitize! standard basis
sma = myResults_C.post[:, myResults_C.param_idx['sma1']]
ecc = myResults_C.post[:, myResults_C.param_idx['ecc1']]
inc = myResults_C.post[:, myResults_C.param_idx['inc1']]

peri = sma*(1-ecc)

peri_min = minimum_peri_outer
mgood = (peri>peri_min)

mtot_fit = myResults_C.post[:, myResults_C.param_idx['mtot']]
period2 = np.sqrt(sma[mgood]**3/mtot_fit[mgood]) # years, period

fig, ax = plt.subplots(1,1,figsize=(6,4))
ax.hist(period,bins=20,range=(0,1000));
ax.set_xlabel("Period of C [yr]")
ax.set_ylabel("N")
ax.set_title("Periods of C with Eggletion criterion rejecting small periastrons")
plt.savefig("_check_period_of_C_peri_rejected.pdf")

period_dist = np.percentile(period2, [16, 50, 84])
q = np.diff(period_dist)
txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{+{2:.3f}}} yr"
txt = txt.format(period_dist[1], q[0], q[1], "P")
print("Orbital period of C with Eggleton constraint on periastron:")
print(txt)

import copy
cleanorb = copy.deepcopy(myResults_C)

fig, ax = plt.subplots(1,1,figsize=(6,4))

orbit_figure = cleanorb.plot_orbits(
    start_mjd=50140, # minimum MJD for colorbar (choose first data epoch)
    sep_pa_end_year = 2025
)
plt.savefig(paths.figures / 'v773_tau_c_orbits-long_burn.pdf')

fig, ax = plt.subplots(1,1,figsize=(6,4))
corner_figure = cleanorb.plot_corner(mod180=True,param_list=['sma1','ecc1','inc1'],labels=['a [au]','e','inclination (deg)'], range=(0.995,0.995,0.98))

plt.savefig(paths.figures / 'v773_c_a_e_i_corner_plot.pdf')

# print out orbital elements for C
print("# Orbital elements for C")

print(cleanorb.labels)
from IPython.display import display, Math

for par_label in cleanorb.labels:
    mcmc = np.percentile(cleanorb.post[:, cleanorb.param_idx[par_label]], [16, 50, 84])

    # convert angles from radians to degrees
    if par_label == 'inc1' or par_label == 'aop1' or par_label == 'pan1':
        mcmc = mcmc * 180/np.pi

    q = np.diff(mcmc)
    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{+{2:.3f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], par_label)
    #display(Math(txt))
    print(txt)

# Calculate t_peri, a and P distributions and marginalising over them 

# How to get orbital period in the orbitize! standard basis
sma = cleanorb.post[:, cleanorb.param_idx['sma1']]
mtot_fit = cleanorb.post[:, cleanorb.param_idx['mtot']]
period = np.sqrt(sma**3/mtot_fit) # years, period
plx = cleanorb.post[:, cleanorb.param_idx['plx']]



# calculate t_periastron
tau_ref_epoch = cleanorb.tau_ref_epoch
tp = orbitize.basis.tau_to_tp(cleanorb.post[:, cleanorb.param_idx['tau1']], tau_ref_epoch, period)

from astropy.time import Time

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

ax[2].set_xlabel("Semimajor axis [mas]")
ax[1].set_xlabel("t_{periastron} [yr]")
ax[0].set_xlabel("Orbital period [yr]")

plt.savefig("_check_C_P_Tperi_a_histograms.pdf")

smmas = np.percentile(sma_mas, [16, 50, 84])
q = np.diff(smmas)
txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{+{2:.3f}}} mas"
txt = txt.format(smmas[1], q[0], q[1], "a")
print(txt)
