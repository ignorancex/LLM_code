import sys
sys.path.append('..')
import config as cf

import glob, pickle
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

plt.rcParams.update({
	"text.usetex": True,
	"font.family": "serif", 
	"font.serif": "Computer Modern",
    "font.size": 20
})

# labels of the independent variables
xlabels = [r'M_{\rm i}, M_\odot', 'r', r'\omega_{\rm i}', r'i, {\rm rad}']

# Get the maximum (observable difference / std) in a focal model dimension 
def get_maxdiff(axis, obs):
	# absolute difference in sigmas along the axis
	diff = np.abs(np.diff(obs, axis=axis)) / cf.std
	# move the focal model axis to the front		
	diff = np.moveaxis(diff, axis, 0)
	if diff.shape[0] > 0: # if the focal axis has more than one element
		# flatten all but the focal axis
		diff = diff.reshape(diff.shape[0], -1)
		# suppress the error for all-NAN slices, which can happen at the edges of the grid
		# warnings.filterwarnings('ignore') 
		# maximum difference across observables and non-focal model dimensions
		maxdiff = np.nanmax(diff, axis=1)
		# go back to default error reports
		# warnings.filterwarnings('default')
	else:
		maxdiff = np.array([0])
	return maxdiff

def plot_diff(axis, x, obs, filename):
	xlabel = xlabels[axis]
	# set it to midpoints between models
	x = (x[1:] + x[:-1]) / 2
	# difference with maximum modulus in sigmas along the axis
	maxdiff = get_maxdiff(axis, obs)
	plt.scatter(x, maxdiff, s=6)
	plt.axhline(3.0, color='grey', linestyle='--')
	plt.xlabel(r'$' + xlabel + r'$')
	plt.ylabel(r'$\max{\left|\,\Delta x / \sigma_x\,\right|}$')
	plt.tight_layout()
	plt.savefig(filename, dpi=200)
	plt.close()

filelist = list(np.sort(glob.glob('../' + cf.obs_dir + 'obs*.pkl'))) # observables 
for it in range(len(filelist)):
	with open(filelist[it], 'rb') as f: 
		obs, age, Mini, r, omega0, inc = pickle.load(f)
	t_str = '_t' + ('%.4f' % age).replace('.', 'p')
	# plot maximum differences versus model parameter
	plot_diff(0, Mini, obs, '../../data/model_grids/png/diff_vs_Mini' + t_str + cf.z_str + '.pdf')
	plot_diff(1, r, obs, '../../data/model_grids/png/diff_vs_r' + t_str + cf.z_str + '.pdf')
	plot_diff(2, omega0, obs, '../../data/model_grids/png/diff_vs_omega0' + t_str + cf.z_str + '.pdf')
	plot_diff(3, inc, obs, '../../data/model_grids/png/diff_vs_inc' + t_str + cf.z_str + '.pdf')
