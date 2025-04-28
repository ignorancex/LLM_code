# Conduct all the operations on observable grids that produce 
# 	isochrone probability densities at data points for each rotational/multiplicity population.
# This includes 
#	placing the prior on the observable grid, convolving it with the minimum-error kernel and coarsening,
#	integrating the convolved prior with the residual error kernel for each data point.

# PARS imports
import sys, os, time, pickle
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
import glob

nsig = cf.nsig - 1 # number of standard deviations to extend Gaussian kernels
npts = ld.obs.shape[0] # number of data points
nrot = len(cf.om_mean) # number of rotational populations
nmul = len(cf.mult) # number of multiplicity populations
ndim = ld.obs.shape[1] # number of observable dimensions

### quantities needed for the computation of probability densities at point locations;
### assumes all observable grids have the same step
# residual standard deviation of data points, sigma^2 - sigma_0^2, in coarse pixels
res = ld.std**2 - cf.std[np.newaxis, :]**2 
res[ np.less(res, 0, where=~np.isnan(res)) ] = 0 # correct for round-off
sigma = np.sqrt(res) / (ld.step[np.newaxis, :] * cf.downsample)

filelist = list(np.sort(glob.glob(cf.obs_dir + 'obs*.pkl'))) # observables 
t = np.full(len(filelist), np.nan)
# probability densities at data point locations
# dimensions: age, multiplicity population, rotational population, data point
points = np.full( (len(filelist), nmul, nrot, npts), np.nan )
for it in range(len(filelist)):
	with open(filelist[it], 'rb') as f: 
		obs_binary, age, Mini, r, omega0, inc = pickle.load(f)
	print('\nt = ' + '%.4f' % age)
	t_str = '_t' + ('%.4f' % age).replace('.', 'p')
	t[it] = age

	# arrays of ordinate multipliers (weights) for the numerical integration in model space;
	# these include the varying discrete distances between adjacent abscissas;
	# dimensions: mass, r, omega, inclination
	w_Mini = du.trap(Mini)[:, np.newaxis, np.newaxis, np.newaxis]
	w_r = du.trap(r)[np.newaxis, :, np.newaxis, np.newaxis] # use the culled r grid here
	w_omega0 = du.trap(omega0)[np.newaxis, np.newaxis, :, np.newaxis]
	w_inc = du.trap(inc)[np.newaxis, np.newaxis, np.newaxis, :]
	# non-uniform priors in non-omega model dimensions
	pr_Mini = (Mini**-2.35)[:, np.newaxis, np.newaxis, np.newaxis]
	pr_inc = np.sin(inc)[np.newaxis, np.newaxis, np.newaxis, :]
	# overall prior without the omega distribution: prior on r is flat
	pr0 = pr_Mini * pr_inc * (w_Mini * w_r * w_omega0 * w_inc)
	# omega distribution prior; 
	# dimensions: rotational population, omega
	pr_om = np.exp(-0.5*((omega0[np.newaxis, :] - cf.om_mean[:, np.newaxis]) / cf.om_sigma[:, np.newaxis])**2)
	# dimensions: rotational population, mass, r, omega, inclination
	pr_om = pr_om[:, np.newaxis, np.newaxis, :, np.newaxis]

	# VCM, CM and vsini <= 0 densities for different rotational and multiplicity populations
	densities = [ [None for i in range(nmul)] for j in range(nrot)]
	densities_cmd = [ [None for i in range(nmul)] for j in range(nrot)]
	densities_v0 = [ [None for i in range(nmul)] for j in range(nrot)]
	# multiplicities
	for k, mult in enumerate(cf.mult):
		start = time.time() 
		m_str = '_m' + str(k)

		if (mult == 'unary'): # pick out the unary prior
			obs_models = obs_binary[:, 0, ...]
			pr_noom = pr0[:, 0, ...]
		else:
			obs_models = obs_binary
			pr_noom = pr0

		ind = [] # index of each model in the data space grid
		for i in range(len(ld.nobs)): # for each data space dimension
			ind.append( np.searchsorted(ld.obs_grids[i], obs_models[..., i], side='right').flatten() - 1 )
		ind = np.array(ind) # make a list into a numpy array
		# choose models that are not NAN and not -1 or len(observable grid dimension) - 1, i.e. outside the grid	
		m = np.all( (ind != ld.nobs[:, np.newaxis] - 1) & (ind != -1), axis=0 ) & \
			np.all( ~np.isnan(obs_models), axis=-1 ).flatten()
		ind = ind[:, m]

		# rotational populations
		for j in range(len(cf.om_mean)):
			rot_str = '_rot' + str(j)
			# prior on the model grid
			pr = pr_noom * pr_om[j]
			# transfer the prior from the model grid to the grid of observables
			pr_obs = np.zeros(ld.nobs, dtype=np.float32) # initialize the prior on observable grid
			np.add.at(pr_obs, tuple(ind), pr.flatten()[m]) 
			# print('\tPlacing the binary prior on a fine grid: ' + '%.2f' % (time.time() - start) + ' seconds.') 
			# package the prior density with the grids of observables 
			density = du.Grid(pr_obs, [x.copy() for x in ld.obs_grids], cf.ROI, cf.norm, Z=cf.Z, age=age)		
			# convolve and normalize the prior with the minimum-error Gaussians in each observable dimension
			min_kernels = [ du.Kernel(cf.std[i] / density.step[i], nsig, ds=cf.downsample) \
							for i in range(len(cf.std)) ]
			for i in range(len(density.obs)): # convolve in each observable dimension
				kernel = min_kernels[i]
				# check that the kernel, evaluated at the ROI boundaries, fits within the grid
				if density.check_roi(i, kernel): 
					density.convolve(i, kernel, ds=cf.downsample) # convolve
			# normalize so that the density adds up to 1 over the RON
			density.normalize() 
			# calculate the dependence of probability change on standard deviation of further convolving kernel
			density.dP_sigma(nsig)
			densities[j][k] = density
			# compute and save the CMD density
			density_cmd = density.copy()
			density_cmd.marginalize(2) # when marginalizing in vsini, don't need to normalize
			densities_cmd[j][k] = density_cmd
			# for data points where vsini is at the lower ROI boundary, convolve in vsini with the residual error kernel;
			# do not re-normalize after the convolution; integrate the probability beyond the lower boundary
			s = cf.std[-1] * np.sqrt(cf.v0err**2 - 1) # residual sigma = sqrt( sigma^2 - sigma_0^2 )
			kernel = du.Kernel(s / density.step[-1], nsig)
			density_v0 = density.copy()
			if density_v0.check_roi(-1, kernel): 
				density_v0.convolve(-1, kernel)
			density_v0.integrate_lower(2)
			densities_v0[j][k] = density_v0
		print(mult + ' convolutions: ' + str(time.time() - start) + ' seconds.', flush=True) 
	# at the first time point, 
	# calculate residual kernels and corresponding slices for individual data points
	if it == 0: kernels, slices = du.calc_kernels(densities[0][0], sigma, nsig)

	# save the convolved priors; this takes up lots of memory, only do it if you want to plot these densities
	with open(cf.dens_dir + 'density' + t_str + '.pkl', 'wb') as f:
		pickle.dump(densities, f)

	start = time.time() 
	# calculate the probability densities at data point locations
	f = np.zeros( (npts, nrot, nmul), dtype=float ) 
	# maximum absolute de-normalization
	max_dp = 0
	start = time.time()
	for j in range(nrot): # for each rotational population
		for k in range(nmul): # for each multiplicity population
			for i in range(npts): # for each star
				# status w.r.t. the vsini measurement
				if np.isnan(ld.obs[i, -1]): density1 = densities_cmd[j][k] # sigma_vsini = infinity				
				elif ld.obs[i, -1] == -1: 	density1 = densities_v0[j][k] # vsini = v_0 = 0
				else: 						density1 = densities[j][k] # vsini > v_0 
				# integration with the kernel, which is normalized up to the product of step sizes
				dens = np.sum(kernels[i] * density1.dens[slices[i]])
				# scale by density step sizes; 
				# this normalizes the density to integrate to 1 over the RON, instead of summing to 1
				dens /= np.prod(density1.step) 
				# normalization correction for this data point in this density grid
				norm = 1.
				for d in range(density1.dim):
					s = sigma[i, d] * density1.step[d] # standard deviation in units of the observable
					dP_spline = density1.correction[d] 
					if dP_spline is not None: # the spline function exists 
						if (s <= dP_spline.x[d]): # if not above the range of the spline
							dp = float( dP_spline(s) ) # evaluate the spline
						else: # extrapolate linearly from the last two points
							x0 = dP_spline.x[-1]; y0 = dP_spline.y[-1]
							x1 = dP_spline.x[-2]; y1 = dP_spline.y[-2]
							dp = y0 + (s - x0) * (y1 - y0) / (x1 - x0)
						norm *= 1 / (1 + dp) # update the re-normalization factor
						if max_dp < np.abs(dp): max_dp = np.abs(dp) # update maximum de-normalization
				# cluster model density at this data point for this rotational and multiplicity populations
				# dimensions: age, multiplicity population, rotational population, data point
				points[it, k, j, i] = float(dens * norm)
	print('data point densities: ' + str(time.time() - start) + ' seconds.', flush=True)
	# save the data point densities at these ages for these rotational population distributions; 
	# do this at every age, in case the program crashes; delete the previously saved file every time
	file = cf.points_dir + 'points_os' + ('_'.join(['%.2f' % n for n in cf.om_sigma])).replace('.','') + \
		( '_t' + '%.4f' % t[0] + '_' + '%.4f' % age ).replace('.','p') + '.pkl'
	with open(file, 'wb') as f: pickle.dump([points[:it+1], t[:it+1], cf.om_sigma], f)
	if it > 0: os.remove(prev_file)
	prev_file = file
	# mark large variables for cleanup
	del pr_obs
	del pr0
	del pr_noom
	del pr
	del m
	gc.collect() # collect garbage / free up memory    
	# # look at the sizes of the largest variables
	# for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
	# 						 key= lambda x: -x[1])[:10]:
	# 	print("{:>30}: {:>8}".format(name, mu.sizeof_fmt(size)))

