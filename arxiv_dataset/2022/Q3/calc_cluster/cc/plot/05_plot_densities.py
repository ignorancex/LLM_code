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
cmapBig = mpl.cm.get_cmap('afmhot_r', 512)
cmap_hot = mpl.colors.ListedColormap(cmapBig(np.linspace(0, 0.8, 256)))

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
def plot(density, cmap, textstr, j, plot_type, base):
		if plot_type=='cmd':
			xlab = 'c = F435W - F814W'; xp = ld.color; xi = 1 
			cmap_lab = r'$\ln{\frac{{\rm d}\rho}{{\rm d}m\,{\rm d}c}}$'
			cmap_min = np.log(0.01); cmap_max = np.log(35); cb_format = '%.1f'
		elif plot_type=='vmd':
			xlab = r'$v = v_{\rm e}\,\sin{i}, \,\mathrm{km/s}$'; xp = ld.vsini; 
			xp[xp == -1] = 0; xi = 2 
			cmap_lab = r'$\ln{\frac{{\rm d}\rho}{{\rm d}m\,{\rm d}v}}$'
			cmap_min = np.log(5.5e-05); cmap_max = np.log(0.2); cb_format = '%.1f'

		# # mask for scatter that depends on rotational population
		# mask = (ld.vsini >= cf.vsini_bins[j]) & (ld.vsini < cf.vsini_bins[j+1])

		dens = density.density() # this version integrates to 1 over RON, instead of summing to 1
		dens[dens == 0] = 1e-300
		dens = np.log(dens)
		norm = mpl.colors.Normalize(vmin=cmap_min, vmax=cmap_max, clip=True)

		fig, [ax, ax1] = plt.subplots(ncols=2, gridspec_kw={'width_ratios': [5, 1]})
		plt.subplots_adjust(left=0.13, right=0.8, bottom=0.13, top=0.95)
		# interpolated shading for smoother plot; for true density values, use shading='nearest' 
		pcm = ax.pcolormesh(density.obs[1], density.obs[0], dens, cmap=cmap, norm=norm,\
			shading='gouraud') 
		ax.set_xlim(left=ld.obmin_plot[xi], right=ld.obmax_plot[xi])
		ax.set_ylim(bottom=ld.obmin_plot[0], top=ld.obmax_plot[0])
		ax.invert_yaxis()
		ax.set_ylabel(r'$m = {\rm F555W}$')
		ax.set_xlabel(xlab)
		# ax.scatter(xp[mask], ld.f555w[mask], s=3, c='k', alpha=1.0, lw=0, marker='o')
		ax.scatter(xp, ld.f555w, s=1, c='k', alpha=1.0, lw=0, marker=',')
		plot_region(ax, density.ROI, ROI_kwargs)
		ax.spines["top"].set_visible(False)
		ax.spines["right"].set_visible(False)
		
		# color bar
		ax1.axis('off')
		cax = fig.add_axes([0.7, 0.13, 0.03, 0.4])
		
		ticks = ticker.LinearLocator(5)
		cb = fig.colorbar(mappable=pcm, ax=ax1, cax=cax, norm=norm, orientation='vertical', ticks=ticks, \
			format=cb_format, alpha=1.0, shrink=0.6)
		cb.set_label(label=cmap_lab, fontsize=18, rotation=0, labelpad=35, y=0.65)

		# text box
		ax.text(1.0, 1.0, textstr, transform=ax.transAxes, # fontsize=14,
		        verticalalignment='top', bbox=dict(facecolor='w', alpha=1.0, edgecolor='w'))

		plt.savefig('../../data/densities/' + plot_type + '/' + base + '_om' + \
			str(j) + 'mul' + str(k) + '.pdf', dpi=300)
		plt.close()

# ages
filelist = list(np.sort(glob.glob('../../data/densities/pkl/*.pkl')))
t = [] 
nrot = len(cf.om_mean) # number of rotational populations
nmul = len(cf.mult) # number of multiplicity populations
densities_cmd = [ [ [None for k in range(len(filelist)) ] for i in range(nmul)] for j in range(nrot) ] 
densities_vsini = [ [ [None for k in range(len(filelist)) ] for i in range(nmul)] for j in range(nrot) ]
it = 0
for filepath in filelist:
	# load the density file
	with open(filepath, 'rb') as f:
		densities = pickle.load(f)
	t.append(densities[0][0].age)
	# base file name
	base = os.path.basename(filepath).split('.')[0]
	if len(base.split('p')[-1]) == 1:
		base = base + '0'
		
	for j in range(len(densities)): # rotational population
		for k in range(len(densities[j])): # multiplicity population
			density = densities[j][k]
			# CM density
			density_cmd = density.copy()
			density_cmd.marginalize(2) # when marginalizing in vsini, don't need to normalize
			densities_cmd[j][k][it] = density_cmd
			# VM density
			density_vsini = density.copy()
			density_vsini.marginalize(1)
			density_vsini.normalize() # so that the sum over the RON is 1
			densities_vsini[j][k][it] = density_vsini

			## text 

			textstr = '\n'.join((
				r'$A_{\rm V}=' + '%.2f' % cf.A_V + '$',
				r'${\rm [M/H]}_{\rm M}=' + str(cf.Z) + '$',
			    r'$\log{\,t}=' + base.split('_')[1].replace('p','.')[1:] + '$',
				str(cf.rot_pop[j]) + r' rotation',
				r'$\sigma_{\rm \omega} = ' + '%.2f' % cf.om_sigma[j] + '$',
				str(cf.mul_pop[k]) 
				))

			print('Plotting...')
			for plot_type in ['cmd', 'vmd']:
				if plot_type=='cmd': density_plot = density_cmd
				elif plot_type=='vmd': density_plot = density_vsini
				plot(density_plot, cmap_hot, textstr, j, plot_type, base)
	it += 1

## plot the minimum-error density at maximum-likelihood cluster age parameters
# define a prior in age
t = np.array(t)
# age steps for mid-point Riemann sum integration; dimension: age
delta_t = np.concatenate((np.array([t[1] - t[0]]), (t[2:] - t[:-2])/2, np.array([t[-1] - t[-2]])))
# parameters of the prior and the prior itself
t_mean = 9.160
t_std = 0.0225
t_pr = np.exp( -0.5 * (t - t_mean)**2 / t_std**2 )
# assume integration is Riemann, with the given delta ages; multiply by the deltas
t_pr *= delta_t
# normalize so that the prior adds up to 1
t_pr /= np.sum(t_pr) 
for j in range(len(densities_cmd)):
	for k in range(len(densities_cmd[j])):
		for it in range(len(densities_cmd[j][k])):
			densities_cmd[j][k][it].dens *= t_pr[it]
			densities_vsini[j][k][it].dens *= t_pr[it]
		density_cmd = du.add(densities_cmd[j][k])
		density_vsini = du.add(densities_vsini[j][k])
		
		## text
		# # text for scatter that depends on rotational population
		# if np.isinf(cf.vsini_bins[j+1]): 
		# 	upper_bound_text = '\infty'
		# else:
		# 	upper_bound_text = str(cf.vsini_bins[j+1])
		# vsini_text = r'$v_{\rm e}\,\sin{i}\, \in \,[' + str(cf.vsini_bins[j]) + ',' + \
		# 	upper_bound_text + ')\,$' + 'km/s'

		textstr = '\n'.join((
		    r'$A_{\rm V}=' + '%.2f' % cf.A_V + '$',
		    r'${\rm [M/H]}_{\rm M}=' + str(cf.Z) + '$',
		    r'$\mu_{t}=' + cf.fstr(t_mean, 3) + '$',
		    r'$\sigma_{t}=' + cf.fstr(t_std, 3) + '$',
			str(cf.rot_pop[j]) + r' rotation',
			r'$\sigma_{\rm \omega} = ' + cf.fstr(cf.om_sigma[j], 2) + '$',
			str(cf.mul_pop[k]) #,
			# vsini_text
			))

		print('Plotting...')
		for plot_type in ['cmd', 'vmd']:
			if plot_type=='cmd': density_plot = density_cmd
			elif plot_type=='vmd': density_plot = density_vsini
			plot(density_plot, cmap_hot, textstr, j, plot_type, 'density_dist')