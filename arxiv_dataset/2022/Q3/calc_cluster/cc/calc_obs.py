# CAUTION: only run this if you have ~100 GB of space on the hard drive for the output
# Conduct all the operations on models grids that produce 
# 	observables on (t, M, r, omega, i).
# This includes 
#	refinement of (M, omega, i) grids at r = 0 and different t,
# 	computation of the observables on (M, r, omega, i) grids at different t,
# 	checking that observable differences between neighboring models are small enough in the r dimension,
#	checking that the differences between neighboring isochrones are small enough.

import sys, os, time, pickle
# PARS imports
sys.path.append(os.path.abspath(os.path.join('..', 'paint_atmospheres')))
from pa.lib import surface as sf
from pa.lib import util as ut
from pa.opt import grid as gd
# cluster parameters imports
from lib import mist_util as mu
from lib import dens_util as du
import load_data as ld
import config as cf
# Python imports
import numpy as np
import gc 

# pre-compute Roche model volume versus PARS's omega
# and PARS's omega versus MESA's omega
sf.calcVA()
sf.calcom()

# Load and filter MIST models
print('Loading MIST...', end='')
start = time.time()
st = mu.Set('../data/mist_grid.npy')

st.select_MS() # select main sequence
st.select_Z(cf.Z) # select metallicity
st.select_valid_rotation() # select rotation with omega < 1
st.set_omega0() # set omega from omega_M; ignore the L_edd factor

# choose isochrone ages so that the space of age prior parameters with appreciable likelihoods
# is covered sufficiently finely
nt = 11
it = 102
tM = np.unique(st.t)[it : it + nt] # MIST ages around 9.159
st.select(np.isin(st.t, tM)) # select the ages in the model set
# split each interval between MIST ages into 4 equal parts
ts = [np.linspace(tM[i], tM[i+1], 5) for i in range(nt - 1)] # this is a list of ndarrays
# also split the first 5 intervals [t_M, t_M + delta_t], such that t_M is an original MIST age,
# put the new age a fourth of the way from t_M to t_M + delta_t 
for i in range(5):
	ts_new = (3./4) * ts[i][0] + (1./4) * ts[i][1]
	ts[i] = np.insert(ts[i], 1, ts_new) 
t = np.unique(np.concatenate(ts)) # refined ages
is_tM = np.isin(t, tM) # whether the refined age is an original MIST age

# use something along the following lines if the program stalls
ages = 2 # 1 or 2
ind = np.where(is_tM)[0] # indices of MIST ages
i = ind[int(len(ind)/2)] # middle MIST index
if ages == 1: t = t[:i+1]
elif ages == 2: t = t[i:]
is_tM = np.isin(t, tM) # whether the refined age is an original MIST age

# check that initial masses aren't multi-valued at constant (EEP, omega0, age)
EEP = np.unique(st.EEP)
oM0 = np.unique(st.oM0)
for tv in t:
	for eep in EEP:
		for o0 in oM0:
			x = st.Mini[(st.EEP == eep) & (st.oM0 == o0) & (st.t == tv)]
			if len(x) > 1: print('multivalued initial masses:', tv, eep, o0, x)

print('%.2f' % (time.time() - start) + ' seconds.')

# non-rotating models at full mass range
stc = st.copy(); stc.select(stc.omega0 == 0)  

print('Loading PARS...', end='', flush=True)
start = time.time()
with open('../data/pars_grid_ZM' + str(cf.Z).replace('-', 'm').replace('.', 'p') + '.pkl', 'rb') as f: 
	pars = pickle.load(f)
print('%.2f' % (time.time() - start) + ' seconds.' + '\n', flush=True)
mu.Grid.pars = pars # give a PARS grid reference to the grid class
# apply the lower mass cut-off for the primaries according the region of interest on the CMD 
st1 = st.copy(); st1.select_age( t[-1] ) # pick the highest age
Mmin = mu.Mlim(st1)
st.select_mass(Mmin=Mmin)
print('minimum mass = ' + '%.4f' % Mmin, flush=True)

nsig = cf.nsig - 1 # number of standard deviations to extend Gaussian kernels
npts = ld.obs.shape[0] # number of data points
ndim = ld.obs.shape[1] # number of observable dimensions

diff_ages = [] # initialize the differences between ages in observable space
it_0 = 0 # first age index
for it in range(it_0, len(t)):
	### refinement of (M, omega, i) grids at r = 0 and different t
	print('\nt = ' + '%.4f' % t[it], end=':')
	if is_tM[it]: print(' original model grid age.')
	else: print(' new intermediate age.')
	t_str = '_t' + ('%.4f' % t[it]).replace('.', 'p')
	# select age
	st1 = st.copy(); st1.select_age( t[it] ) # primary model set
	stc1 = stc.copy(); stc1.select_age( t[it] )	# non-rotating model set for the companions
	# refine model grids
	start = time.time(); print('refining the mass, omega and inclination grids...', flush=True)
	if is_tM[it]:
		grid = mu.refine_coarsen(st1)
		# save the omega and inclination grids for the use in the next age
		omega0_grid = grid.omega0
		inc_grid = grid.inc
	else:
		# refine with the omega and inclination grids fixed
		grid = mu.refine_coarsen(st1, o0=omega0_grid, inc=inc_grid)
	print('%.2f' % (time.time() - start) + ' seconds.', flush=True)

	EEP = grid.get_EEP()

	start = time.time(); print('combining observables...', flush=True)

	# get non-rotating companion magnitudes on a M * r grid; use the full 0 <= r <= 1 grid here
	mag = mu.companion_grid(cf.r, grid.Mini, stc1, pars, cf.A_V, cf.modulus)
	# combine the magnitudes of the non-rotating companion and its primary;
	# companion dimensions: initial primary mass, binary mass ratio, filter
	# primary dimensions: initial primary mass, initial omega, inclination, filter
	# final dimensions: initial primary mass, binary mass ratio, initial omega, inclination, filter
	mag_binary = \
		mu.combine_mags(mag[..., np.newaxis, np.newaxis, :], grid.mag[:, np.newaxis, ...])
	# insert the unary magnitudes, which evaluated to NAN above
	mag_binary[:, 0, ...] = grid.mag
	# compute the observables of binary models;
	# dimensions: initial primary mass, binary mass ratio, initial omega, inclination, mag / color / vsini;
	# the last dimension remains the same if three filters are now magnitude, color and vsini
	# print('Combining primary and companion magnitudes: ' + str(time.time() - start) + ' seconds.') 
	obs_binary = np.full( mag_binary.shape, np.nan )
	obs_binary[..., 0] = mag_binary[..., 1] # F555W magnitude
	obs_binary[..., 1] = mag_binary[..., 0] - mag_binary[..., 2] # F435W - F814W color
	obs_binary[..., 2] = grid.vsini[:, np.newaxis, ...] # vsini

	# differences in observables along the r dimension
	diff = np.abs(np.diff(obs_binary, axis=1))
	diff = np.moveaxis(diff, 1, -2) # move the r axis from original location to just before observables
	# mask along the r dimension where the differences are not all NaN
	m = ~np.all(np.isnan(diff), axis=(0, 1, 2)) # sum over mass, omega and i 
	# maximum observable differences along the r dimension
	dm = np.full(m.shape, np.nan)
	dm[m] = np.nanmax(diff[..., m], axis=(0, 1, 2))
	# largest companions may be so close to TAMS that magnitude differences in the r dimension are too large
	ind = np.argwhere( ~np.less_equal(dm, cf.dmax * cf.std[0], where=~np.isnan(dm)) ).flatten()
	if ind.size == 0:
		print('all observable differences are small in the r dimension')
	else:
		# left boundary of the first interval in r dimension with excessive magnitude difference
		print('observable differences are small up to r = ' + str(cf.r[:-1][ind[0]]))
		# cull the magnitude and r arrays
		r = cf.r[:ind[0] + 1]
		obs_binary = obs_binary[:, :ind[0] + 1, ...]

	# compute the distance from the previous-age isochrone if it exists
	if it > it_0:
		emin = int(np.floor(min(np.nanmin(EEP_prev), np.nanmin(EEP))))
		emax = int(np.ceil(max(np.nanmax(EEP_prev), np.nanmax(EEP))))
		EEPrange = np.array(range(emin, emax + 1))
		numbers = []
		diff = []
		for eep in EEPrange:
			# indices of mass + omega locations in the EEP grids where EEP rounds to some integer
			i0 = np.argwhere(np.around(EEP_prev) == eep)
			i1 = np.argwhere(np.around(EEP) == eep)
			# observables of corresponding models: look at all r and all i
			obs0 = obs_binary_prev[i0.T[0], :, i0.T[1]].reshape(-1, 3)
			obs1 = obs_binary[i1.T[0], :, i1.T[1]].reshape(-1, 3)
			obs0 = obs0[~np.isnan(obs0[:, 0])]
			obs1 = obs1[~np.isnan(obs1[:, 0])]
			if obs0.shape[0] > 0 and obs1.shape[0] > 0: 
				diff_means = np.mean(obs1, axis=0) - np.mean(obs0, axis=0)
				numbers.append(obs0.shape[0] * obs1.shape[0]) # number of model pairs
				diff.append(diff_means)
		numbers = np.array(numbers)
		diff = np.array(diff)
		d = np.nansum(diff * numbers[:, np.newaxis], axis=0) / np.nansum(numbers)
		diff_age = {'prev': t[it - 1], 'curr': t[it], 'mean_dist': d/cf.std, 'dist': diff/cf.std[np.newaxis, :]}
		diff_ages.append(diff_age)
		# print('magnitude, color, vsini minimum-error distances from the previous isochrone: ' +\
		# 	', '.join('%.4f' % x for x in d/cf.std))
		# print('EEPs: ' + np.array2string(EEPrange))
		# print('distances from the previous isochrone at all EEPs: ' + np.array2string(diff/cf.std[np.newaxis, :]))

	# make copies of observables and EEPs for the next iteration
	if it > it_0: del obs_binary_prev # mark the old version of previous observables for garbage collection
	obs_binary_prev = obs_binary.copy()
	EEP_prev = EEP.copy()

	print('%.2f' % (time.time() - start) + ' seconds.', flush=True)

	# save the binary observables
	with open(cf.obs_dir + 'obs' + t_str + '.pkl', 'wb') as f:
		pickle.dump([obs_binary, t[it], grid.Mini, r, grid.omega0, grid.inc], f)

	# mark large variables for cleanup
	del mag_binary
	del obs_binary
	gc.collect() # collect garbage / free up memory    

# save the distances between isochrones in observable space
with open(cf.obs_dir + 'dist' + str(ages) + '.pkl', 'wb') as f:
	pickle.dump(diff_ages, f)
