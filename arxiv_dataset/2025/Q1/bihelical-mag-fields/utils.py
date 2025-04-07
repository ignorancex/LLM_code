import numpy as np
import os
import warnings
import sys

def test_small_im(arr):
	assert max(np.abs(np.imag(arr)/np.real(arr))) < 1e-10

def real(arr):
	test_small_im(arr)
	return np.real(arr)

def jackknife(arr, axis):
	"""
	Jackknife estimate of the error in the mean of arr (along axis)
	
	Arguments:
		arr: numpy array
		axis: int
	"""
	tr = [axis]
	tr.extend([i for i in range(arr.ndim) if i!=axis])
	arr = np.transpose(arr, axes=tr)
	
	n = arr.shape[0]
	if n < 2:
		raise ValueError(f"Cannot estimate error with only one sample. {arr.shape = }, {axis = }")
	
	means = []
	for i in range(n):
		rest = [j for j in range(n) if j!=i]
		means.append(np.average(arr[rest], axis=0))
	means = np.array(means)
	
	m1 = np.average(means, axis=0)
	m2 = np.average(means**2, axis=0)
	var = m2 - m1**2 #biased estimator of variance
	std = np.sqrt(var*n/(n-1.5)) #less biased estimator of standard deviation, assuming errors are Gaussian and uncorrelated between samples. Correction factor copied from Wikipedia (https://en.wikipedia.org/wiki/Standard_deviation#Unbiased_sample_standard_deviation ; accessed 1:55 PM IST, 04-Aug-2023)
	
	return m1, std

class fig_saver():
	"""
	savefig: bool
	savedir: string, path to save the figure
	"""
	def __init__(self, savefig, savedir):
		self.savefig = savefig
		self.savedir = savedir
	
	def __call__(self, fig, name, **kwargs):
		if not self.savefig:
			return
		
		if not os.path.exists(self.savedir):
			#Create directory if it does not exist
			os.makedirs(self.savedir)
		elif not os.path.isdir(self.savedir):
			raise FileExistsError(f"Save location {self.savedir} exists but is not a directory.")
		
		try:
			import git
			repo = git.Repo(path = os.path.dirname(__file__), search_parent_directories = True)
			
			if "metadata" in kwargs.keys():
				raise ValueError("Git was found. Do not specify metadata manually.")
			
			scriptname = sys.argv[0]
			id_str = f"{os.path.basename(scriptname)} at git commit {repo.head.object.hexsha}"
			
			if name[-4:] == ".pdf":
				kwargs['metadata'] = {'Creator': id_str}
			elif name[-4:] == ".png":
				kwargs['metadata'] = {'Software': id_str}
			else:
				raise ValueError(f"Could not infer file type for name {name}")
		except Exception as e:
			warnings.warn(f"Git info will not be saved in the image. {repr(e)}")
		
		loc = os.path.join(self.savedir, name)
		loc_dir = os.path.dirname(loc)
		if not os.path.exists(loc_dir):
			os.makedirs(loc_dir)
		fig.savefig(loc, **kwargs)

def rebin(k_vec, spec, bin_boundaries, axis=0, norm=True):
	"""
	Given an array spec such that the values along axis correspond to values at corresponding entries of k_vec, rebin those values into bins specified by the list bin_boundaries. If norm=True, rebinned values are normalized assuming a uniform measure.
	
	Arguments:
		k_vec: array
		spec: array
		bin_boundaries: array
		axis: int
		norm: bool
	"""
	
	spec = np.swapaxes(spec, 0, axis)
	
	old_bin_widths = (np.roll(k_vec, -1) - np.roll(k_vec, 1))/2
	old_bin_widths[0] = k_vec[1] - k_vec[0]
	old_bin_widths[-1] = k_vec[-1] - k_vec[-2]
	
	new_bin_widths = bin_boundaries[1:] - bin_boundaries[:-1]
	
	n_bins = len(bin_boundaries)-1
	rebinned = np.zeros([n_bins, *np.shape(spec)[1:]], dtype=spec.dtype)
	ib = 0
	for ik, k in enumerate(k_vec):
		if bin_boundaries[ib+1] <= k:
			ib += 1
		if ib >= n_bins:
			break
		if not bin_boundaries[ib] <= k < bin_boundaries[ib+1]:
			raise RuntimeError(f"Something wrong with specified bins.\n{ik = }\n{ib = }\n{k = }\n{bin_boundaries = }\n{k_vec = }")
		
		if norm:
			rebinned[ib] += spec[ik]*old_bin_widths[ik]/new_bin_widths[ib]
		else:
			rebinned[ib] += spec[ik]
	
	rebinned = np.swapaxes(rebinned, 0, axis)
	return rebinned

def downsample_half(k, E, H, calc_spec, axis=0):
	"""
	Given values of E and H at wavenumbers k, rebin to double the k-spacing.
	
	Arguments:
		k: numpy array
		E: numpy array
		H: numpy array
		calc_spec: instance of spectrum.calc_spec
		axis: int
	"""
	k_old = k
	dk = k_old[1] - k_old[0] #assumes k are equispaced.
	k = k_old[::2]
	bin_bounds = np.linspace(-dk,k_old[-1]+dk,len(k)+1)
	E = rebin(k_old, E, bin_bounds, axis=axis, norm=False)
	H = rebin(k_old, H, bin_bounds, axis=axis, norm=False)
	
	return k, *calc_spec.scale_EH(E,H,2)

class ErrorFill():
	"""
	Attributes:
		line: the plotted line corresponding to the data
	"""
	def __init__(self, line):
		self.line = line

def errorfill(ax, x, y, yerr, marker = 'o', **kwargs):
	"""
	Uses fill_between to display errorbars in a nicer way than plt.errorbar.
	
	Arguments:
		ax: matplotlib.axes.Axes instance
		x: 1D array of float
		y: 1D array of float, same size as x
		yerr: 1D or 2D array of float, with the last dimension having the same size as x. If a 2D array, the first row is the - error, and the second row is the + error.
	
	Returns an ErrorFill instance.
	"""
	y = np.array(y)
	yerr = np.array(yerr)
	
	if yerr.ndim == 0:
		yerr_minus = np.full_like(x, yerr)
		yerr_plus = np.full_like(x, yerr)
	elif yerr.ndim == 1:
		yerr_minus = yerr
		yerr_plus = yerr
	elif yerr.ndim == 2:
		yerr_minus = yerr[0]
		yerr_plus = yerr[1]
		
		if len(yerr_minus) != len(x):
			raise ValueError(f"Dimensions of yerr {yerr.shape} and x {np.shape(x)} are not compatible.")
	else:
		raise ValueError(f"Cannot handle yerr with {yerr.ndim} dimensions")
	
	if len(x) == 1:
		#fill_between will not display the error bounds if the data series is only one element long. Fall back to errorbar for that case.
		warnings.warn("Falling back to plt.errorbar for single point.")
		
		l = ax.errorbar(
			x,
			y,
			yerr = [yerr_minus, yerr_plus],
			marker = marker,
			**kwargs,
			)
		
		return ErrorFill(l.lines[0])
	else:
		[l] = ax.plot(
			x,
			y,
			marker = marker,
			**kwargs,
			)
		
		ax.fill_between(
			x,
			y - yerr_minus,
			y + yerr_plus,
			alpha = 0.25,
			color = l.get_color(),
			linewidth = 0, #prevent an outline being drawn around the boundary of the region being filled.
			)
		
		return ErrorFill(l)

def smooth_boxcar(arr, l, axis=0):
	"""
	Smooth arr along axis with a boxcar profile of width 2l+1
	
	Arguments:
		arr: numpy array
		l: int
		axis: int
	"""
	if np.shape(arr)[axis] < l:
		raise ValueError(f"Given array is too small for boxcar of requested width (axis length {np.shape(arr)[axis]} < boxcar half-width {l})")
	
	arr = np.swapaxes(arr, axis, 0)
	
	smoothed = arr.copy()
	norm = np.ones_like(arr) #Keep track of what number to divide by at the end
	for i in range(1,l+1):
		smoothed[i:] += arr[:-i]
		norm[i:] += 1
		
		smoothed[:-i] += arr[i:]
		norm[:-i] += 1
	
	smoothed = smoothed/norm
	smoothed = np.swapaxes(smoothed, axis, 0)
	return smoothed

if __name__ == "__main__":
	from termcolor import cprint
	
	k = np.linspace(0,9,10)
	new_bin_boundaries = np.linspace(-0.5, 9.5, 11)
	a = np.ones_like(k)
	b = rebin(k, a, new_bin_boundaries)
	assert np.all(np.isclose(a,b))
	
	new_bin_boundaries = np.array([0,2,4,6,8,10])
	b = rebin(k, a, new_bin_boundaries)
	assert np.all(b == 1)
	
	#####
	cprint("All tests passed", attrs=['bold'])
