import sys, os, time, warnings
sys.path.append(os.path.abspath(os.path.join('..', 'paint_atmospheres')))
from pa.lib import surface as sf
from pa.lib import util as ut
from pa.opt import grid as gd

import load_data as ld
import config as cf

import numpy as np
from scipy.interpolate import griddata
from matplotlib import pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.size'] = 12

def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

# combine magnitude arrays that can be broadcast to the same shape;
# this takes a while because we have to evaluate logarithms
def combine_mags(mag1, mag2):
	return -2.5 * np.log10( 10**(-mag1/2.5) + 10**(-mag2/2.5) )

# a set of MIST models
class Set:

	# load the MIST models
	def __init__(self, filename=None, age=None, Z=None):
		self.age = age
		self.Z = Z
		if filename is not None:
			# logZ oM0 EEP log10_isochrone_age_yr initial_mass star_mass log_L log_L_div_Ledd log_Teff\
			# log_R surf_avg_omega surf_r_equatorial_div_r surf_r_polar_div_r surf_avg_omega_div_omega_crit
			md = np.load(filename, allow_pickle=False)
			# select only the variables of interest
			self.models = md[:, [0, 1, 2, 3, 4, 5, 6, 7, 9, 13]]
			self.set_vars()
		else:
			self.clear_vars()
		self.omega0 = None

	def copy(self):
		st = Set()
		st.models = np.copy(self.models)
		st.age = self.age
		st.Z = self.Z
		st.set_vars()
		if self.omega0 is not None:
			st.omega0 = np.copy(self.omega0)
		return st

	# set model parameter arrays
	def set_vars(self):	
		self.logZm = self.models[:, 0] # logarithmic metallicity in MIST
		self.oM0 = self.models[:, 1] # initial omega_MESA
		self.EEP = self.models[:, 2] # eeps
		self.t = self.models[:, 3] # log age in years
		self.Mini = self.models[:, 4] # initial mass in solar masses
		self.M = self.models[:, 5] # mass in solar masses
		self.logL = self.models[:, 6] # log(luminosity) in solar luminosities
		self.logL_div_Ledd = self.models[:, 7] # log of the Eddington ratio
		self.R = 10**self.models[:, 8] # volume-averaged radius
		self.oM = self.models[:, 9] # observed Omega / Omega_c
		self.oMc = self.oM * np.sqrt(1 - 10**self.logL_div_Ledd) # Omega / Omega_c, corrected for luminosity

	# set omega: ignore the correction due to the Eddington limit here; 
	# it should be less than ~1% for an intermediate-age cluster
	def set_omega0(self):
		oM0 = self.oM0
		nn = ~np.isnan(oM0)
		omega0 = np.full_like(oM0, np.nan)
		omega0[nn] = sf.omega(oM0[nn])
		self.omega0 = omega0

	# clear parameter arrays
	def clear_vars(self):
		self.logZm = np.empty((0,))
		self.oM0 = np.empty((0,))
		self.EEP = np.empty((0,))
		self.t = np.empty((0,))
		self.Mini = np.empty((0,))
		self.M = np.empty((0,))
		self.logL = np.empty((0,))
		self.logL_div_Ledd = np.empty((0,))
		self.R = np.empty((0,))
		self.oM = np.empty((0,))
		self.oMc = np.empty((0,))
		self.omega0 = np.empty((0,))

	# select a set of models
	def select(self, mask):
		self.models = self.models[mask, :]
		if self.omega0 is not None:
			self.omega0 = self.omega0[mask]
		self.set_vars()

	# select models with valid rotation
	def select_valid_rotation(self):
		m = (self.oMc >= sf.omin) & (self.oMc <= sf.omax) # mask out models with negative or super-critical rotation
		m = m & (self.oM0 < 0.8) # mask out models likely to be super-critical on the pre-main sequence
		self.select(m)

	def select_MS(self):
		m = (self.EEP >= 202) & (self.EEP < 454)
		self.select(m)

	def select_mass(self, Mmin=0, Mmax=np.inf):
		m = (self.Mini >= Mmin) & (self.Mini <= Mmax)
		self.select(m)

	def select_Z(self, Z):
		self.Z = Z
		if Z in self.logZm:		
			m = (self.logZm == Z)
			self.select(m)
		# else: # interpolate in metallicity, possibly keeping age, EEP and omega0 constant			

	def select_age(self, age):
		self.age = age
		if age in self.t:
			self.select(self.t == age)
		else: # interpolate in age, keeping EEP and omega0 constant
			## remove all ages other than the immediate neighbors of the target
			t = np.unique(self.t)
			i = np.searchsorted(t, age)
			self.select(np.isin(self.t, np.array([t[i - 1], t[i]])))
			## interpolate
			EEP = np.unique(self.EEP)
			oM0 = np.unique(self.oM0)
			points = [ self.EEP, self.t ] 	# points from which to interpolate
			xi = [ EEP, [age] ]				# points at which to interpolate
			if oM0.shape[0] > 1: # if omega is an interpolation variable
				points.insert(1, self.oM0)
				xi.insert(1, oM0)
			xi = np.meshgrid( *xi, sparse=True, indexing='ij' )
			models = griddata( tuple(points), self.models, tuple(xi), method='linear')
			models = models.reshape(-1, models.shape[-1])
			m = ~np.any(np.isnan(models), axis=-1)
			self.models = models[m]
			# correct for possible round-off error in initial omega_M
			self.models[:, 1][ self.models[:, 1] < 0 ] = 0
			# correct for possible round-off error in age
			self.models[:, 3] = age
			# set variables
			self.set_vars()
			# calculate initial omega
			if self.omega0 is not None: self.set_omega0()

# given a set of models and the boundaries of the observable space region where we need model priors,
# calculate the lowest mass / highest magnitude cut-off
def Mlim(st):
	# for any given mass, a star with the smallest magnitude is the one with the smallest omega
	# (so that it has been moving towards TAMS fastest), largest age (so that it's closest to TAMS),
	# smallest inclination (so that we see the star pole-on), and largest (i.e. equal-mass) companion.
	# construct the corresponding grid
	Mi = np.array(np.linspace(st.Mini.min(), st.Mini.max(), 100)) 
	omega0 = np.array([st.omega0.min()])  
	inc = np.array([0]) 
	grid = Grid(st, Mi, omega0, inc, cf.A_V)
	# combine primary and companion magnitudes 
	mag = combine_mags(grid.obs[..., 0], np.expand_dims(grid.obs[:, 0, :, 0], 1))
	# mask that is true where magnitude is below maximum
	m = np.full_like(mag, False, dtype=bool) 
	np.less_equal(mag, ld.obs1[np.newaxis, np.newaxis, 0], where=~np.isnan(mag), out=m)
	# indices of masses above the lower mass cut-off
	j = np.nonzero( np.any(m, axis=-1) )[0] 
	if len(j) > 0: # if there are masses above the cut-off
		jmin = j.min();
		# subtract one from the minimum mass index, if possible 
		if jmin > 0:
			jmin-=1
		Mmin = grid.Mini[jmin]
	else:
		Mmin = np.nan
	return Mmin

# given a set of models, produce a (M, omega, t, i) model grid with proper observable spacings at a given age;
# if omega and inclination grids are given, refine only in mass dimension;
# this function sometimes gets stuck, in this case, restart
def refine_coarsen(st, o0=None, inc=None):
	# get maximum differences for all dimensions
	def diffs(grid):
		return np.array([np.nanmax(grid.get_maxdiff(i)) for i in range(grid.ndim)])
	# get grid lengths for all dimensions
	def lengths(grid):
		return np.array([len(getattr(grid, grid.ivars[i])) for i in range(grid.ndim)])

	Mi = np.unique(st.Mini) # original mass grid
	dims = list(range(Grid.ndim)) # dimensions to refine
	if o0 is None: o0 = np.array([st.omega0.min(), st.omega0.max()]) # small omega grid
	else: dims.remove(1) # do not refine in omega dimension	
	if inc is None:	inc = np.linspace(0, np.pi/2, 2) # small inclination grid
	else: dims.remove(2) # do not refine in inclination dimension

	grid = Grid(st, Mi, o0, inc, cf.A_V)
	grid.coarsen(0, dmax=cf.dmax) # coarsen the mass grid, in case it's too fine to start with
	md = diffs(grid) 	# get maximum differences for all dimensions
	
	while np.any(md[dims] > cf.dmax): # while any of the maximum differences are above the observable difference cutoff 
		for i in dims: # for each of the dimensions (M, omega, i)
			while (md[i] > cf.dmax): # while this dimension is not fine enough
				ivar = grid.ivars[i]
				var = getattr(grid, ivar) # get the grid in this dimension
				if var.shape[0] > 1: # if the grid has more than one value
					# print('Refining the ' + ivar + ' dimension.', flush=True)
					while md[i] > cf.dmax: # while the maximum difference in this dimension is above the cutoff
						grid.refine(i, dmin=cf.dmax) # refine this dimension
						md = diffs(grid) # update maximum differences
					# print('Coarsening the ' + ivar + ' dimension.', flush=True)
					grid.coarsen(i, dmax=cf.dmax)
					md = diffs(grid)
	gl = lengths(grid)
	print(md, flush=True)
	print(gl, flush=True)

	return grid

# a grid of MIST models at some age and metallicity
class Grid:
	# independent variables that define the grid, axes of the corresponding array of observables
	ivars = ['Mini', 'omega0', 'inc']
	# labels of the independent variables
	lvars = [r'M_0', r'\omega_0', r'i']
	# number of dimensions
	ndim = len(ivars)
	# PARS grid
	pars = None

	def __init__(self, \
			st=None, Mi=None, o0=None, inc=None, A_V=None, verbose=False):
		# independent model parameters
		self.Mini = Mi
		self.omega0 = o0
		self.inc = inc
		self.A_V = A_V
		self.st = st # set of MIST models
		self.age = st.age
		self.Z = st.Z
		# interpolate to set the dependent parameters
		if st is not None:
			self.calc_obs(verbose=verbose)
		else:
			self.mag = None
			self.vsini = None
			self.obs = None

	# clear the dependent model variables that are used for calculations
	def clear_vars(self):
		del self.M
		del self.L
		del self.omega
		del self.Req

	# return a version of the object for pickling;
	# this copy only has the independent star model variables, the observables, and the cluster variables;
	# it does not have the original MIST models or the PARS grid.
	def pickle(self):
		grid = Grid(Mi=self.Mini, o0=self.omega0, inc=self.inc, A_V=self.A_V)
		grid.mag = np.copy(self.mag).astype(np.float32)
		grid.vsini = np.copy(self.vsini).astype(np.float32)
		grid.age = self.age
		grid.Z = self.Z
		del grid.st
		return grid

	# Interpolates between MIST models in initial mass, omega and age
	#	to find other model parameters; computes omega and equatorial radii;
	# 	records a set of inclinations as well
	# Makes use of the following:
	#	MIST model set at some age and metallicity
	#	initial masses
	# 	initial omegas
	# 	ages
	# Note: when you are certain you don't need to interpolate in age here, remove
	# 	all comments that used to be program code and have the 'age' boolean variable in them
	def interp(self):
		st = self.st
		points = [ st.Mini, st.omega0 ] # points from which to interpolate
		xi = [ self.Mini, self.omega0 ] # points at which to interpolate
		dims = [ 0, 1 ] # corresponding dimensions
		xi = np.meshgrid( *xi, sparse=True, indexing='ij' )
		
		## calculate the dependent variables via interpolation in the independent variables; 
		## use the linear method because there are small discontinuities at a limited range of masses
		M = griddata( tuple(points), st.M, tuple(xi), method='linear' ) # mass
		# remove intermediate grid points where all masses are NaN
		for i in dims:
			# mask that shows at which grid points not all entries are NaN 
			m = ~np.all(np.isnan(M), axis=tuple(np.delete(dims, i)))
			m[0] = m[-1] = True # keep the original boundary grid points
			var = getattr(self, self.ivars[i]) # get this parameter grid (e.g. Mini, omega0)
			setattr(self, self.ivars[i], var[m]) # delete the NaN grid points
			M = np.compress(m, M, axis=i) # delete the NaN grid points from the mass array
		self.M = M
		# re-calculate the points at which we interpolate
		xi = [ self.Mini, self.omega0 ]
		xi = np.meshgrid( *xi, sparse=True, indexing='ij' )
		# calculate the rest of the dependent variables
		self.L = 10**griddata( tuple(points), st.logL, tuple(xi), method='linear' )
		oM = griddata( tuple(points), st.oM, tuple(xi), method='linear' )
		logL_div_Ledd = griddata( tuple(points), st.logL_div_Ledd, tuple(xi), method='linear' )
		R = griddata( tuple(points), st.R, tuple(xi), method='linear' )
		oMc = oM * np.sqrt(1 - 10**logL_div_Ledd)
		# mitigate round-off error from interpolation
		notnan = ~np.isnan(oMc)
		oMc[np.less(oMc, sf.omin, where=notnan) & notnan] = sf.omin
		oMc[np.greater(oMc, sf.omax, where=notnan) & notnan] = sf.omax
		# PARS' omega and equatorial radius
		sh = oMc.shape
		shf = np.prod(sh)
		omega = np.full(shf, np.nan)
		Req = np.full(shf, np.nan)
		nn = ~np.isnan(oMc)
		nnf = nn.flatten()
		omega[nnf] = sf.omega(oMc[nn])
		Req[nnf] = R.flatten()[nnf] * np.cbrt((4 * np.pi / 3) / sf.V(omega[nnf]))
		omega = omega.reshape(sh)
		Req = Req.reshape(sh)
		self.omega = omega
		self.Req = Req

	def calc_obs(self, verbose=False):
		if verbose:
			print('Calculating the observables for ' + str(len(self.Mini)) + ' x ' +\
				str(len(self.omega0)) + ' x ' + str(len(self.inc)) + ' = ' +\
				'{:,}'.format(len(self.Mini) * len(self.omega0) * len(self.inc)) + ' models...')
			start = time.time()
		self.interp() # calculate the dependent model variables
		# construct points for interpolating from the PARS grid;
		# 	each point is a set of values used by the PARS grid in which we interpolate
		#	(e.g. [tau, omega, inclination, gamma] 
		#	- all the dimensions of a PARS grid except metallicity, reddening and the bands)
		# these points are on a grid of MESA independent variables 
		# 	(e.g. initial mass, initial omega),
		#	plus additional parameters for PARS that MESA models don't have 
		#	(e.g. inclination)
		tau = ut.tau(self.L, self.Req) # has dimensions of the grid of MESA independent variables
		gamma = ut.gamma(self.M, self.Req) # has dimensions of the grid of MESA independent variables
		pars_dims = len(self.pars.dims) - 3 # number of interpolated PARS dimensions
		# start with arrays of PARS interpolated variables on a grid of 
		# MESA independent variables and inclination
		points = np.full(self.M.shape + ( len(self.inc), pars_dims ), np.nan)
		points[..., 0] = tau[..., np.newaxis] # add axes for non-MESA independent variables
		points[..., 1] = self.omega[..., np.newaxis] # add axes for non-MESA independent variables
		points[..., 2] = self.inc # if this is the only MESA-independent variable, no new axes necessary here
		points[..., 3] = gamma[..., np.newaxis] # add axes for non-MESA independent variables
		# points[..., 4] = ut.logZp_from_logZm(self.st.Z) # same metallicity for each point
		# points[..., 5] = self.A_V # same AV for each point
		# record the shape of the grid and flatten it, keeping the arrays of PARS variables intact
		sh = points.shape[:-1] 
		points = points.reshape( (-1, pars_dims) )
		# interpolate from the PARS grid;
		# the result is an an array of magnitudes, e.g. [F435W, F555W, F814W]
		mag = gd.interp4d(self.pars, points, ut.logZp_from_logZm(self.st.Z), self.A_V)
		# reshape magnitudes back to the grid of evolutionary model parameters
		mag = mag.reshape(sh + (len(self.pars.bands), ))
		# correct for radius and distance;
		# the radius array needs extra dimensions due to bands and PARS parameters
		# that evolutionary models don't have (e.g. inclination)
		self.mag = gd.correct(mag, self.Req[..., np.newaxis, np.newaxis], cf.modulus)
		self.vsini = ut.vsini1(self.M[..., np.newaxis], self.Req[..., np.newaxis], \
			self.omega[..., np.newaxis], self.inc[np.newaxis, np.newaxis, :]) / 1e5
		self.obs = np.stack( (self.mag[..., 1], self.mag[..., 0] - self.mag[..., 2], self.vsini), axis=-1 )
		self.clear_vars() # clear the dependent model variables
		if verbose:
			print('\t' + str(time.time() - start) + ' seconds.')

	# Get the maximum (observable difference / std) in a focal model dimension 
	def get_maxdiff(self, axis):
		# absolute difference in sigmas along the axis
		diff = np.abs(np.diff(self.obs, axis=axis)) / cf.std
		# move the focal model axis to the front		
		diff = np.moveaxis(diff, axis, 0)
		if diff.shape[0] > 0: # if the focal axis has more than one element
			# flatten all but the focal axis
			diff = diff.reshape(diff.shape[0], -1)
			# suppress the error for all-NAN slices, which can happen at the edges of the grid
			warnings.filterwarnings('ignore') 
			# maximum difference across observables and non-focal model dimensions
			maxdiff = np.nanmax(diff, axis=1)
			# go back to default error reports
			warnings.filterwarnings('default')
		else:
			maxdiff = np.array([0])
		return maxdiff

	# Subdivide each interval into n subintervals, where n is the ceiling of the largest 
	# 	(observable difference / (std * dmin)), where dmin is a class variable
	# Inputs:
	#	model dimension to refine
	#	standard deviations of the observables
	#	refinement factor; make it smaller for a finer grid; default is 1.0
	def refine(self, axis, dmin=1.0):
		# get the number of intervals to split each interval into
		def get_ns(axis, dmin):
			maxdiff = self.get_maxdiff(axis)
			# compute the number of intervals to split each interval into
			notnan = ~np.isnan(maxdiff)
			ns = np.ones_like(maxdiff, dtype=int) 
			ns[notnan] = np.ceil(maxdiff[notnan] / dmin).astype(int)
			return ns

		vp = 1e-4 # precision below which model differences are treated as non-existent
		var = getattr(self, self.ivars[axis]) # get the model parameter list
		vardiff = np.diff(var) # differences between neighboring models
		ns = get_ns(axis, dmin)
		# split intervals: reverse so that indices remain valid
		for i in np.flip(np.where((ns > 1.) & (vardiff >= vp))[0]): # indices where to split the array
			# split the interval at each location and insert the result 
			var = np.insert(var, i+1, np.linspace(var[i], var[i+1], ns[i]+1)[1:-1])
		setattr(self, self.ivars[axis], var) # update the model parameter list
		self.calc_obs() # calculate the observables
		
		# at the locations where the observable differences are large, but model differences are not,
		# add models with intermediate observables
		vardiff = np.diff(var) # differences between neighboring models
		ns = get_ns(axis, dmin)
		for i in np.flip(np.where((ns > 1.) & (vardiff < vp))[0]): # indices where to split the array
			# split the interval at each location and insert the result 
			var = np.insert(var, i+1, np.linspace(var[i], var[i+1], ns[i]+1)[1:-1])
			self.mag = np.insert(self.mag, i+1, np.linspace(self.mag[i], self.mag[i+1], ns[i]+1)[1:-1], axis=axis)
			self.vsini = np.insert(self.vsini, i+1, np.linspace(self.vsini[i], self.vsini[i+1], ns[i]+1)[1:-1], axis=axis)
			self.obs = np.insert(self.obs, i+1, np.linspace(self.obs[i], self.obs[i+1], ns[i]+1)[1:-1], axis=axis)
		setattr(self, self.ivars[axis], var) # update the model parameter list

	# Coarsen the model grid in a given focal dimension 
	def coarsen(self, axis, dmax=1.0):
		obs = self.obs
		mag = self.mag
		vsini = self.vsini		
		# move the focal model axis to the front
		obs = np.moveaxis(obs, axis, 0) 
		mag = np.moveaxis(mag, axis, 0)
		vsini = np.moveaxis(vsini, axis, 0) 
		# model parameter grid 
		var = getattr(self, self.ivars[axis])
		maxdiff = self.get_maxdiff(axis)
		ind = np.argsort(maxdiff) # indices of sorted differences, NAN are at the end
		i = 0 # index in the above array of indices
		j = ind[i] # current index in the array of maximum differences
		stop = False
		# check for the stopping condition
		if np.isnan(maxdiff[j]): stop = True
		elif maxdiff[j] >= dmax: stop = True
		while not stop:
			# differences due to the merging of the focal interval with neighbors to the left or right
			maxright = np.nan
			maxleft = np.nan
			# if there is an interval to the right of the focal interval
			if j < len(maxdiff) - 1: 
				# subtract the observables at the lower bound of the corresponding interval
				# from those at the upper bound of the interval to the right; compute the maximum
				warnings.filterwarnings('ignore') # suppress the error for all-NAN slices
				maxright = np.nanmax( np.abs(obs[j+2,...] - obs[j,...]) / cf.std )
				warnings.filterwarnings('default') # go back to default error reports
				# only consider the merge if the combined differences are below the maximum allowable difference
				if maxright > dmax: maxright = np.nan
			# if there is an interval to the left of the focal interval
			if j > 0:
				# get maximum differences due to the merging of the focal interval with the interval to the left
				warnings.filterwarnings('ignore') # suppress the error for all-NAN slices
				maxleft = np.nanmax( np.abs(obs[j+1,...] - obs[j-1,...]) / cf.std )
				warnings.filterwarnings('default') # go back to default error reports
				if maxleft > dmax: maxleft = np.nan
			# determine whether to merge with the left neighbor, the right neighbor or not at all
			if ~np.isnan(maxleft):
				if ~np.isnan(maxright):
					if maxleft <= maxright:
						merge = 'left'
					else:
						merge = 'right'
				else:
					merge = 'left'
			elif ~np.isnan(maxright):
				merge = 'right'
			else:
				merge = None
			# if not merging, advance the index of indices
			if merge is None:
				i += 1
				if i < len(ind):
					j = ind[i]
				else:
					stop = True
			elif merge == 'left':
				## merge with the interval to the left: 
				# delete the left bound from the observables arrays, 
				obs = np.delete(obs, j, axis=0)
				mag = np.delete(mag, j, axis=0)
				vsini = np.delete(vsini, j, axis=0)
				# replace the maximum differences for the focal interval and the left neighbor 
				# with the combined maximum differences 
				maxdiff[j-1] = maxleft
				maxdiff = np.delete(maxdiff, j)
				# delete the model variable value at the left boundary of the focal interval
				var = np.delete(var, j)
			elif merge == 'right':
				## merge with the interval to the right: 
				# delete the right bound from the observables arrays, 
				obs = np.delete(obs, j+1, axis=0)
				mag = np.delete(mag, j+1, axis=0)
				vsini = np.delete(vsini, j+1, axis=0)
				# replace the maximum differences for the focal interval and the right neighbor 
				# with the combined maximum differences 
				maxdiff[j] = maxright
				maxdiff = np.delete(maxdiff, j+1)
				# delete the model variable value at the right boundary of the focal interval
				var = np.delete(var, j+1)
			if merge is not None:
				# re-set the sorted maximum differences
				ind = np.argsort(maxdiff)
				i = 0 
				j = ind[i]
			# check for the stopping condition
			if np.isnan(maxdiff[j]): stop = True
			elif maxdiff[j] >= dmax: stop = True
		# move the focal model axis back into its place
		self.obs = np.moveaxis(obs, 0, axis) 
		self.mag = np.moveaxis(mag, 0, axis)
		self.vsini = np.moveaxis(vsini, 0, axis)
		setattr(self, self.ivars[axis], var) # set the model parameter list

	# get EEP on the model grid		
	def get_EEP(self):
		st = self.st
		points = [ st.Mini, st.omega0 ] # points from which to interpolate
		xi = [ self.Mini, self.omega0 ] # points at which to interpolate
		xi = np.meshgrid( *xi, sparse=True, indexing='ij' )
		EEP = griddata( tuple(points), st.EEP, tuple(xi), method='linear')
		return EEP

# companion magnitudes on a grid of initial mass of the primary and binary mass ratio
# Inputs:
#	mass ratio grid
#	primary mass grid
#	set of models
# 	single age
# 	PARS grid of observables
#	reddening
#	distance modulus
# Note: when you are certain you don't need to interpolate in age here, remove the 
#	comments that have the age boolean in them, which used to be program code
def companion_grid(r, Mini, st, pars, A_V, modulus):
	Mc = Mini[:, np.newaxis] * r[np.newaxis, :] # companion mass on a grid of primary mass and binary ratio
	points = ( st.Mini, ) # points from which to interpolate
	xi = Mc # points at which to interpolate

	## compute all three magnitudes of non-rotating companions 
	## on a grid of primary initial mass and binary mass ratio;
	## about half of the values won't exist because companion mass doesn't go low enough in the model grids
	# MIST model dependent variables
	M = griddata( points, st.M, xi, method='linear' )
	L = 10**griddata( points, st.logL, xi, method='linear' )
	logL_div_Ledd = griddata( points, st.logL_div_Ledd, xi, method='linear' )
	R = griddata( points, st.R, xi, method='linear' )
	# PARS grid variables
	tau = ut.tau(L, R) 
	gamma = ut.gamma(M, R) 
	pars_dims = len(pars.dims) - 3 # number of interpolated PARS dimensions
	# start with arrays of PARS interpolated variables on a grid of 
	# MESA independent variables and inclination
	points = np.full(M.shape + (pars_dims, ), np.nan)
	points[..., 0] = tau
	points[..., 1] = 0 # omega is zero
	points[..., 2] = 0 # inclination can be zero
	points[..., 3] = gamma 
	# record the shape of the grid and flatten the grid almost completely,
	# keeping the arrays of PARS variables intact
	sh = points.shape[:-1] 
	points = points.reshape( (-1, pars_dims) )
	# interpolate from the PARS grid;
	# the result is an an array of magnitudes, e.g. [F435W, F555W, F814W]
	mag = gd.interp4d(pars, points, ut.logZp_from_logZm(st.Z), A_V)
	# reshape magnitudes back to the grid of evolutionary model parameters
	mag = mag.reshape(sh + (len(pars.bands), ))
	# correct for radius and distance;
	# the radius array needs extra dimensions due to bands
	mag = gd.correct(mag, R[..., np.newaxis], modulus)
	return mag