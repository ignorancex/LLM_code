# plot the individual likelihood factors for the data points at maximum likelihood values
# of the cluster parameters
import sys
sys.path.append('..')
import load_data as ld
import config as cf
from lib import dens_util as du

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import ticker
import pickle, glob, os

mpl.rcParams['font.size'] = 14
ROI_kwargs = {'facecolor':'none', 'edgecolor':'grey', 'alpha':0.5, 'lw':1}
# color maps
cmapBig = mpl.cm.get_cmap('viridis', 512)
cmap = mpl.colors.ListedColormap(cmapBig(np.linspace(0, 1.0, 256)))

# patches to plot the boundaries of a 2D region
# x index = 1
# y index = 0
def plot_region(ax, region, region_kwargs):
	xmina, xmaxa = ax.get_xlim()
	ymina, ymaxa = ax.get_ylim()
	y = list(region[0]); ymin = y[0]; ymax = y[1] 
	x = list(region[1]); xmin = x[0]; xmax = x[1]
	if xmin == -np.inf: xmin = xmina; x.remove(-np.inf)
	if xmax == np.inf: xmax = xmaxa; x.remove(np.inf)
	if ymin == -np.inf: ymin = ymina; y.remove(-np.inf)
	if ymax == np.inf: ymax = ymaxa; y.remove(np.inf)
	kwargs = region_kwargs
	ax.hlines(y, xmin, xmax, **kwargs)
	ax.vlines(x, ymin, ymax, **kwargs)

# generic plotting function
def plot(LF_max, cmap, plot_type, base):
		marker_size = 2
		if 'cmd' in plot_type:
			xlab = 'c = F435W - F814W'; xp = ld.color; xi = 1; 
			ROI = np.delete(cf.ROI, 2, axis=0)
			if plot_type == 'cmd_nan':
				mask = np.isnan(ld.vsini)
				txt = r'$\sigma_v = \infty$' + '\n' + r'$n_{\rm m} = 1042$'
			elif plot_type == 'cmd_0':
				mask = (ld.vsini == -1)
				marker_size = 4
				txt = r'$v_{\rm e}\,\sin{i} = 0$' + '\n' + r'$n_{\rm z} = 74$'
			elif plot_type == 'cmd':
				mask = (ld.vsini > 0)
				txt = r'$v_{\rm e}\,\sin{i} > 0$' + '\n' + r'$n_{\rm p} = 1237$'
		elif 'vmd' in plot_type:
			xlab = r'$v = v_{\rm e}\,\sin{i}, \,\mathrm{km/s}$'; xp = ld.vsini;
			xp[xp == -1] == 0; xi = 2; 
			ROI = np.delete(cf.ROI, 1, axis=0)
			mask = (ld.vsini > 0)
			txt = r'$v_{\rm e}\,\sin{i} > 0$' + '\n' + r'$n_{\rm p} = 1237$'

		lf = np.log(LF_max)[mask]
		lf -= lf.max()

		cmap_min = min(-5.3, lf.min()) # attempt to use a common minumum for relative factors
		cmap_max = lf.max(); cb_format = '%.1f'
		norm = mpl.colors.Normalize(vmin=cmap_min, vmax=cmap_max, clip=True)

		fig, [ax, ax1] = plt.subplots(ncols=2, gridspec_kw={'width_ratios': [5, 1]})
		plt.subplots_adjust(left=0.13, right=0.8, bottom=0.13, top=0.95)
		ax.set_xlim(left=ld.obmin_plot[xi], right=ld.obmax_plot[xi])
		ax.set_ylim(bottom=ld.obmin_plot[0], top=ld.obmax_plot[0])
		ax.invert_yaxis()
		ax.set_ylabel(r'$m = {\rm F555W}$')
		ax.set_xlabel(xlab)
		scatter_plot = ax.scatter(xp[mask], ld.f555w[mask], s=marker_size, c=lf, cmap=cmap, norm=norm, alpha=1.0)
		plot_region(ax, ROI, ROI_kwargs)
		ax.spines["top"].set_visible(False)
		ax.spines["right"].set_visible(False)
		
		# color bar
		ax1.axis('off')
		cax = fig.add_axes([0.7, 0.1, 0.03, 0.4])
		
		ticks = ticker.LinearLocator(5)
		cb = fig.colorbar(mappable=scatter_plot, ax=ax1, cax=cax, norm=norm, orientation='vertical', ticks=ticks, \
			format=cb_format, alpha=1.0, shrink=0.6)
		cb.set_label(label=r'$\Delta\ln{\varrho_p}$', fontsize=18, rotation=0, labelpad=30, y=0.65)

		# text box
		textstr = '\n'.join((
			# r'$A_{\rm V}=' + cf.fstr(cf.A_V, 2) + '$',
			# r'${\rm [M/H]}_{\rm M}=' + str(cf.Z) + '$',
			txt,
			r'$\mu_{t} = \widehat{\mu}_{t}=' + cf.fstr(tmax, 3) + '$',
			r'$\sigma_{t} = \widehat{\sigma}_{t}=' + cf.fstr(smax, 3) + '$',
			# r'$\sigma_{\omega} = \{' + ', '.join([cf.fstr(n, 2) for n in om_sigma]) + '\}$',
			r'$w = \widehat{w} =$' + '\n' + r'$\quad\{' + cf.fstr(w0max, 2) + ', ' + cf.fstr(1 - w0max - w1max, 2) +\
				', ' + cf.fstr(w1max, 2) + '\}$',
			r'$q = \widehat{q} = $' + cf.fstr(qmax, 2),
			r'$b = \widehat{b} = $' + cf.fstr(bmax, 2)))
		ax.text(1.0, 1.0, textstr, transform=ax.transAxes, fontsize=14,
		        verticalalignment='top', bbox=dict(facecolor='w', alpha=1.0, edgecolor='w'))

		plt.savefig('../../data/likelihoods/png/' + plot_type + '_' + base + '.pdf', dpi=300)
		plt.close()

# ages
filelist = list(np.sort(glob.glob('../../data/likelihoods/pkl/lf*.pkl')))
for filepath in filelist:
	with open(filepath, 'rb') as f:
		LF_max, qmax, bmax, tmax, smax, w0max, w1max, om_sigma = pickle.load(f)

	# base file name
	base = os.path.basename(filepath).split('.')[0]
		
	LF_max[LF_max == 0] = 1e-300

	print('Plotting...')
	# plot four panels: 
	#	cmd for points without vsini
	#	cmd for points with vsini = 0
	#	cmd for points with vsini > 0
	#	vmd for points with vsini > 0
	for plot_type in ['cmd_nan', 'cmd_0', 'cmd', 'vmd']:
		plot(LF_max, cmap, plot_type, base)