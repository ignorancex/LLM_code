import sys, os, time, pickle

from lib import dens_util as du
import config as cf
import load_data as ld
# Python imports
import numpy as np
import glob
import pdb
from scipy.optimize import root
from scipy.special import erf
from scipy.ndimage import map_coordinates
from scipy.interpolate import interp1d

eps = np.finfo(float).eps # 1e-300

nsig = cf.nsig - 1 # number of standard deviations to extend Gaussian kernels
npts = ld.obs.shape[0] # number of data points
ndim = ld.obs.shape[1] # number of observable dimensions

# load the data point densities
# dimensions: age, multiplicity population, rotational population, data point
filelist = list(np.sort(glob.glob(cf.points_dir + '*.pkl')))
t = None
for filepath in filelist:
	with open(filepath, 'rb') as f: 
		pts1, t1, om_sigma = pickle.load(f)
		if t is None: # if there are no ages from before
			points = pts1; t = t1
		else:
			m = ~np.isin(t1, t) # where the new ages are not in exisiting ages
			points = np.concatenate((points, pts1[m]), axis=0)
			t = np.concatenate((t, t1[m]))
# age steps for mid-point Riemann sum integration; dimension: age
delta_t = np.concatenate((np.array([t[1] - t[0]]), (t[2:] - t[:-2])/2, np.array([t[-1] - t[-2]])))

# background probability density
back = np.empty(npts)
mn = np.isnan(ld.obs[:, -1]) # mask of observations with no vsini
m0 = ld.obs[:, -1] == -1 # mask of observations with vsini = 0
mv = ~mn & ~m0 # observations with valid vsini
# on the CMD
back[mn] = 1 / cf.volume_cm 
# at the vsini = 0 boundary
back[m0] = cf.v0err * cf.std[-1] / (np.sqrt(2 * np.pi) * cf.volume) 
# everywhere else
back[mv] = ( 1 + erf(ld.obs[mv, -1] / (np.sqrt(2) * ld.std[mv, -1])) ) / (2 * cf.volume)

# if cf.mix:
# 	m = (t >= cf.t0min) & (t <= cf.t0max)
# 	ft = points[m] # the points are already on a grid of age parameters
# 	t0_ar = t[m] # the first age-related parameter is the age intercept 
# 	t1_ar = cf.a_ar # the second age-related parameter is related to the slope
# else:

## compute likelihoods on a grid of age, metallicity, 
## rotational population and multiplicity population proportions
## a range of age priors
# smaller of the two distances between range boundaries and available age grid boundaries,
# divided by the number of standard deviations in half the Gaussian age prior;
# this is the largest possible standard deviation of this prior
sigma_max = np.minimum( cf.tmin - t[0], t[-1] - cf.tmax ) / nsig
# means of the priors
if t[-1] <= cf.tmin: # if the largest age is below the target range
	t_mean = np.array([t[-1]]) # compute for the largest age only
elif t[0] >= cf.tmax: # if the smallest age is above the target range
	t_mean = np.array([t[0]]) # compute for the smallest age only
else: # else, compute on the intersection of the target range and allowed range
	t_mean = np.linspace(np.maximum(cf.tmin, t[0]), np.minimum(cf.tmax, t[-1]), cf.n, dtype=float)
# standard deviations of the priors
if sigma_max <= cf.smin: # if maximum sigma is below the target sigma range
	t_sigma = np.array([sigma_max]) # compute for the one value of maximum sigma
else: # else, compute on the intersection of the target range and allowed range
	t_sigma = np.linspace(cf.smin, np.minimum(sigma_max, cf.smax), cf.n)

print( 'age means: ' + ', '.join(['%.4f' % t for t in t_mean]) )
print( 'age standard deviations: ' + ', '.join(['%.4f' % t for t in t_sigma]) )
print( 'average age grid step: ' + '%.5f' % np.mean(delta_t) )
print( 'means grid step: ' + '%.5f' % (t_mean[1] - t_mean[0]) )
print( 'deviations grid step: ' + '%.5f' % (t_sigma[1] - t_sigma[0]) )
print( 'slow rotator proportion step: ' + '%.5f' % (cf.w0[1] - cf.w0[0]) )
print( 'fast rotator proportion step: ' + '%.5f' % (cf.w2[1] - cf.w2[0]) )

# construct the normalized age priors
# dimensions: age, prior mean, prior sigma
t_pr = np.exp( -0.5 * (t[:, np.newaxis, np.newaxis] - t_mean[np.newaxis, :, np.newaxis])**2 \
						 / t_sigma[np.newaxis, np.newaxis, :]**2 )
# assume integration is Riemann, with the given delta ages; multiply by the deltas
t_pr *= delta_t[:, np.newaxis, np.newaxis]
# normalize so that the sum is equal to 1
t_pr /= np.sum(t_pr, axis=0)
# marginalize probability densities at data point locations by the age prior,
# dimensions of output: age prior mean, age prior sigma, multiplicity, rotational population, data point 
ft = np.sum(t_pr[..., np.newaxis, np.newaxis, np.newaxis] * points[:, np.newaxis, np.newaxis, ...], axis=0)
t0_ar = t_mean # the first age-related parameter is the mean of the distribution
t1_ar = t_sigma # the second parameter is the standard deviation

# derivatives of log likelihood w.r.t. q and w.r.t. b at (q, b) = x
# likelihood is the product of (1 + Ai * q + Bi * q * b) over i
def dlogL(x, A, B):
	# keep q away from boundaries, where division by zero is possible
	if (x[0] >= 1.): q = 1 - eps
	elif (x[0] <= 0.): q = eps
	else: q = x[0]
	# keep b away from the zero boundary
	if (x[1] <= 0.): b = eps
	elif (x[1] >= 1. ): b = 1
	else: b = x[1]
	## derivatives with respect to q and b
	# to compute each derivative, work with a quantity that, when zero, makes the derivative zero;
	# explicitly compute the derivative wherever the quantity is non-zero 
	y = A + B * b; m = y != 0
	dlogL_dq = np.sum(1 / (q + 1 / (A[m] + B[m] * b)) )
	m = B != 0
	dlogL_db = np.sum(1 / (b + (1 / q + A[m]) / B[m]) )
	return [dlogL_dq, dlogL_db]

# derivative of log likelihood w.r.t q at fixed b
def dlogL_dq(q, b, A, B):
	return dlogL([q, b], A, B)[0]

# derivative of log likelihood w.r.t b at fixed q
# (b is the first argument)
def dlogL_db(b, q, A, B):
	return dlogL([q, b], A, B)[1]

# likelihood factors as a function of q, b, A and B
def L(q, b, A, B):
	return 1 + (A + B * b) * q

# grids on [0, 1] that have about nout total points outside [x0, x1]
# and nin total points inside [x0, x1]
def grid(x0, x1, nin=11, nout=11):
	fine = np.linspace(x0, x1, nin)
	# coarse steps
	step = (1 - x1 + x0) / nout
	if step > 0:
		# each coarse grid should have at least 2 entries if it has any measure
		# otherwise it should have one entry
		if x0 > 0: coarse0 = np.linspace(0, x0, int(x0 / step) + 2)
		else: coarse0 = np.array([0])
		if 1 - x1 > 0: coarse1 = np.linspace(x1, 1, int((1 - x1) / step) + 2)
		else: coarse1 = np.array([1])
	else:
		coarse0 = np.array([0])
		coarse1 = np.array([1])
	return np.concatenate( (coarse0[:-1], fine, coarse1[1:]) )

# given a likelihood and it1 integration weights on a grid of q and b and a requested proportion,
# return bounds on both parameters that ensure that the likelihood integrated 
# between these bounds is above the requested proportion
def bounds(l, w, q, b, lp):
	# get the range of q and b values where likelihood is appreciable
	fcp = du.CI_func(l, weights=w)
	# if the domain of the spline function includes the requested likelihood proportion
	if fcp.x.min() < lp and fcp.x.max() > lp:  
		# indices of points above a certain likelihood that contain the likelihood proportion
		ind = np.where(l >= float(fcp(lp)))
		iq = [ind[0].min(), ind[0].max()]
		ib = [ind[1].min(), ind[1].max()]
		# make a correction due to the discreteness of the grid
		if iq[0] > 0: iq[0] -= 1
		if iq[1] < len(q) - 1: iq[1] += 1
		if ib[0] > 0: ib[0] -= 1
		if ib[1] < len(b) - 1: ib[1] += 1
		return [q[iq[0]], q[iq[1]], b[ib[0]], b[ib[1]]]
	else: # return the least conservative bounds
		return [1, 0, 1, 0]

# indices of an array of length l, downsampled by factor n, with last element's index present
def ds(l, n): 
	x = list(range(0, l, n))
	if l % n == 0:
		x = x + [l - 1]
	return x

## marginalize in q under uniform prior for combinations of population proportions and age prior parameters
ll = np.full( (len(cf.w0), len(cf.w2), len(t0_ar), len(t1_ar)), np.nan )
qm = np.full_like( ll, np.nan ) # array of maximum-likelihood q
bm = np.full_like( ll, np.nan ) # array of maximum-likelihood b

start = time.time()
# the range of q and b where likelihood is appreciable;
# this will narrow
q0 = b0 = 0; q1 = b1 = 1
# the following variable is 0 while we are narrowing the fine integration range,
# then 1 for the final integrations, then 2 when we should exit the loop
run = 0 
while run < 2:
	if run == 0: # narrowing the integration range
		# put together a grid in q that is fine where likelihood is appreciable
		# and coarse elsewhere, based on the bounds of the fine area from previous run
		q = grid(q0, q1, nin=11, nout=3)
		b = grid(b0, b1, nin=11, nout=3)
		# initialize the new bounds we will determine in this run
		q0n, q1n, b0n, b1n = 1, 0, 1, 0
		# downsample factor of parameter arrays
		n = 2
	elif run == 1: # performing the final integrations with the current fine area bounds
		# make a grid that is extra-fine where likelihood is appreciable
		# and extra-coarse elsewhere
		q = grid(q0, q1, nin=21, nout=6)
		b = grid(b0, b1, nin=21, nout=6)
		# downsample factor of parameter arrays
		n = 1
		# global highest log likelihood on q, b, w_0, w_1, t0_ar and t1_ar
		LLmax = -np.inf

	# construct q and b weights according to the variable-step trapezoidal rule;
	# dimensions: q, b
	wq = du.trap(q)[:, np.newaxis]
	wb = du.trap(b)[np.newaxis, :]
	count = 0 # count of the age priors
	t0_ind = ds(len(t0_ar), n)
	t1_ind = ds(len(t1_ar), n)
	for it0 in t0_ind:
		for it1 in t1_ind:
			f = ft[it0, it1]
			for i in ds(len(cf.w0), n):
				w0 = cf.w0[i]
				j_range = np.searchsorted(cf.w2, 1 - w0, side='right')
				for j in ds(j_range, n):
					w2 = cf.w2[j]
					# unary probability densities for all data points
					f0 = f[0, 0, :] * w0 + f[0, 1, :] * (1 - w0 - w2) + f[0, 2, :] * w2
					# binary probability densities for all data points
					f1 = f[1, 0, :] * w0 + f[1, 1, :] * (1 - w0 - w2) + f[1, 2, :] * w2
					# coefficients of q and q * b in the likelihood factors (the remaining term is 1)
					A = f0 / back - 1
					B = (f1 - f0) / back

					# likelihood factors on a grid of q, b, and data points, 
					# this step and the next take the most time;
					# dimensions: q, b, data point
					lf = L( q[:, np.newaxis, np.newaxis], b[np.newaxis, :, np.newaxis], \
							A[np.newaxis, np.newaxis, :], B[np.newaxis, np.newaxis, :])

					### likelihood vs q and b, divided by the maximum likelihood; 
					### two options to prevent overflow
					if cf.overflow == 'log': 
						l = np.zeros( (len(q), len(b)) )
						# mask where the likelihoods are non-zero because none of the data point factors is zero
						m = ~np.any(lf == 0, axis=-1)
						## to prevent overflow when multiplying factors,
						## compute the exponent of the sum of logarithms of the factors, instead of a product; 
						## to prevent overflow when exponentiating, 
						## subtract the maximum of the sum before taking the exponent;
						# log-likelihood on the q-b grid, at locations where likelihood is non-zero
						ll_qb = np.sum(np.log(lf[m]), axis=-1)
						# correction to be added to log likelihood later is the maximum log likelihood on the q-b grid
						ll_corr = ll_qb.max()
						ll_qb -= ll_corr # subtract the maximum log likelihood
						l[m] = np.exp(ll_qb) # compute the likelihood where it is non-zero
					elif cf.overflow == 'root':
						# compute precise likelihood at a local maximum on (q, b),
						# assuming likelihood is unimodal in these variables
						sol = root(dlogL, [0.5, 0.5], args=(A, B))
						qmax, bmax = sol.x
						# bring the maximum likelihood parameters back into their range
						if bmax > 1: bmax = 1 
						elif bmax <= 0: bmax = eps
						if qmax >= 1: qmax = 1 - eps 
						elif qmax <= 0: qmax = eps
						# likelihood factors at maximum-likelihood q and b at individual data points
						lf_max = L(qmax, bmax, A, B)
						# the nth root of maximum likelihood on (q, b), 
						# where n is the number of data points
						nLmax = np.prod(np.power(lf_max, 1. / npts ))	
						# reduce the factors by the approximate nth root of maximum likelihood
						lf /= nLmax
						# multiply the reduced factors together to get the likelihood 
						# divided by a constant that is probably the maximum likelihood; 
						# to further reduce the effect of underflow, take products in groups
						N = 10
						products = [np.prod(x, axis=-1) for x in np.array_split(lf, N, axis=-1)]
						l = np.prod(np.stack(products), axis=0)
						# correction to be added to log of the likelihood integral later
						ll_corr = npts * np.log(nLmax)	

					lp = 0.999 # proportion of likelihood that should be within the fine integration area
					if run == 0: # if this is one of the narrowing runs
						if np.count_nonzero(l > 0) > 0:
							# get new, possibly narrower bounds on q and b between which the 
							# likelihood integrates to a high proportion of its total 
							q0new, q1new, b0new, b1new = bounds(l, wq * wb, q, b, lp)
							# update the bounds determined by this run
							q0n = min(q0n, q0new)
							q1n = max(q1n, q1new)
							b0n = min(b0n, b0new)
							b1n = max(b1n, b1new)
					elif run == 1: # if this is the final run
						# update the global highest likelihood 
						# if the highest likelihood at these tm, ts, w0 and w2 is higher;
						# also update the maximum-likelihood factors of individual data points here
						qi, bi = np.unravel_index(np.nanargmax(l), l.shape)
						llmax = np.log(l[qi, bi]) + ll_corr
						if llmax > LLmax: 
							LLmax = llmax; LF_max = lf[qi, bi] * back 
						# set the maximum-likelihood q and b to those where
						# likelihood is maximum on the grid
						iq, ib = np.unravel_index(np.argmax(l), l.shape)
						qm[i, j, it0, it1] = q[iq]
						bm[i, j, it0, it1] = b[ib]
						# weighted likelihood
						lw = l * wq * wb 
						# total integrated likelihood
						lint = np.sum( lw ) 
						## check the proportion of likelihood within the fine integration area
						# indices corresponding the the fine integration area
						q0i = np.argwhere(q == q0)[0][0]
						q1i = np.argwhere(q == q1)[0][0]
						b0i = np.argwhere(b == b0)[0][0]
						b1i = np.argwhere(b == b1)[0][0]
						lprop = np.sum( lw[q0i : q1i + 1, b0i : b1i + 1] ) / lint
						if lprop < lp:
							print('\t proportion of likelihood within integration area is too small.')
						# record the logarithm of the integral, with logarithm of maximum likelihood added back;
						# add back the likelihood that is maximum on the q-b grid
						ll[i, j, it0, it1] = np.log(lint) + ll_corr

			count += 1
			if count % 100 == 0:
				print(str(count) + ' / ' + str(len(t0_ind) * len(t1_ind)) + ' age-related priors; ' + \
					'%.2f' % (time.time() - start) + ' seconds since last time check.', flush=True )
				start = time.time()

	print('this run code: ' + str(run), flush=True)
	print('fine area bounds used in this run:', q0, q1, b0, b1, flush=True)

	if run == 0: # if this was a narrowing run
		print('fine area bounds found in this run:', q0n, q1n, b0n, b1n, flush=True)
		# distance between the fine area bounds determined in this run and the previous ones
		dist = np.sqrt(np.sum(np.subtract([q0n, q1n, b0n, b1n], [q0, q1, b0, b1])**2) / 4)
		print('distance between the two sets of bounds: ' + str(dist), flush=True)
		# if the fine area bounds determined in this run 
		# did not change significantly from the previous run, 
		# indicate that the next run should be the final one
		if dist < 0.1: run = 1
		# update the bounds
		q0, q1, b0, b1 = q0n, q1n, b0n, b1n
	elif run == 1: # if this was the final run
		# indicate that no more runs are necessary
		run = 2

	print('next run code: ' + str(run), flush=True)

# print('marginalization in q on a grid of w_0, w_1 and b: ' + '%.2f' % (time.time() - start) + ' seconds.')
w0i, w2i, t0i, t1i = np.unravel_index(np.nanargmax(ll),ll.shape)
print('max ln marginalized likelihood: ' + str(np.nanmax(ll)) + ' at w_0 = ' + '%.4f' % cf.w0[w0i] + \
	', w_1 = ' + '%.4f' % (1 - cf.w0[w0i] - cf.w2[w2i]) + ', w_2 = ' + '%.4f' % cf.w2[w2i] +\
	', t0_ar (e.g. mu_t) = ' + '%.4f' % t0_ar[t0i] + ', t1_ar (e.g. sigma_t) = ' + '%.4f' % t1_ar[t1i])
print('min ln marginalized likelihood: ' + str(np.nanmin(ll)))
print('max ln likelihood: ' + '%.4f' % LLmax + ', at q = ' + '%.4f' % qm[w0i, w2i, t0i, t1i] + \
	' and b = ' +  '%.4f' % bm[w0i, w2i, t0i, t1i])

suffix = str(cf.Z).replace('-', 'm').replace('.', 'p') + \
	'_os' + '_'.join([('%.2f' % n).replace('.','') for n in om_sigma]) + '.pkl'
	# '_os' + '_'.join([('%.2f' % n).replace('.','') for n in om_sigma]) + '.pkl', 'wb') as f:

# package the likelihoods, the ML q values and the rotational distribution standard deviations
with open(cf.like_dir + 'pkl/ll_' + suffix, 'wb') as f:
	pickle.dump([ll, qm, bm, t0_ar, t1_ar, cf.w0, cf.w2, om_sigma], f)

# package the likelihood factors of individual data points with the corresponding cluster model parameters
with open(cf.like_dir + 'pkl/lf_' + suffix, 'wb') as f:
	pickle.dump([LF_max, \
		qm[w0i, w2i, t0i, t1i], bm[w0i, w2i, t0i, t1i], \
		t0_ar[t0i], t1_ar[t1i], cf.w0[w0i], cf.w2[w2i], om_sigma], f)
