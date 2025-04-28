# plot the dependence of probability change on sigma
import sys
sys.path.append('..')
import config as cf
import os, glob, pickle
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.size'] = 16
dims = ['mag', 'col']
dimensions = ['m', 'c']

filelist = list(np.sort(glob.glob('../../data/densities/pkl/*.pkl')))
prefix = '../../data/normalization/'
for filepath in filelist: # for each combination of age and metallicity
	# load the pre-computed density on a grid of observables
	with open(filepath, 'rb') as f:
		densities = pickle.load(f)
		for j in range(len(densities)):
			for k in range(len(densities[0])):
				density = densities[j][k]
				for axis in range(len(density.correction)):
					dP_spline = density.correction[axis]
					if dP_spline is not None:
						x = dP_spline.x; y = dP_spline.y
						xplot = np.linspace(x.min(), x.max(), 100)
						fig, ax = plt.subplots()
						ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
						ax.plot(xplot, dP_spline(xplot), c='g')
						ax.scatter(x, y, facecolors='w', edgecolors='k')
						ax.set_ylabel(r'$\delta$') #, labelpad=20)
						ax.set_xlabel(r"$\sigma'_{\!\!" + dimensions[axis] + "}$")
						ax.spines["top"].set_visible(False)
						ax.spines["right"].set_visible(False)
						textstr = '\n'.join((
							r'$A_{\rm V}=' + '%.2f' % cf.A_V + '$',
						    r'${\rm [M/H]}_{\rm M}=' + str(cf.Z) + '$',
						    r'$\log{\,t}=' + '%.4f' % density.age + '$',
							str(cf.rot_pop[j]) + r' rotation',
							r'$\sigma_{\rm \omega} = ' + '%.2f' % cf.om_sigma[j] + '$',
							str(cf.mul_pop[k]) ))
							# r'Dimension: ' + dimensions[axis] ))
						if y[-1] > y[0]: txt_x = 0.15
						else: txt_x = 0.70
						ax.text(txt_x, 1.05, textstr, transform=ax.transAxes, horizontalalignment='left',
						        verticalalignment='top', bbox=dict(facecolor='w', alpha=0.0, edgecolor='w'))
						plt.tight_layout()
						# write plot file
						suffix = ('%.4f' % density.age).replace('.', '') + '_om' + str(j) + '_mul' + str(k)
						plt.savefig(prefix + dims[axis] + '/dP' + suffix + '.pdf', dpi=300)
						plt.close()