# plot the original mist models at an age closest to the maximum likelihood age

# Python imports
import sys, os, time, pickle
import numpy as np
import gc 
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
# PARS imports
from pa.lib import surface as sf
from pa.lib import util as ut
from pa.opt import grid as gd
# cluster parameters imports
from lib import mist_util as mu
from lib import dens_util as du
import load_data as ld
import config as cf
# matplotlib imports
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import ticker

mpl.rcParams['font.size'] = 12
ROI_kwargs = {'facecolor':'none', 'edgecolor':'grey', 'alpha':0.5, 'lw':1}

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

def plot(texstr, x, y, xlabel, ylabel, filename):
	# index of the x axis in the observables
	if ('c' in xlabel): xi = 1; ROI = np.delete(cf.ROI, 2, axis=0)
	elif ('v' in xlabel): xi = 2; ROI = np.delete(cf.ROI, 1, axis=0)

	# color map
	cmap = mpl.cm.get_cmap('tab10')
	cmap_colors = cmap(oM0)

	fig, [ax, ax1] = plt.subplots(ncols=2, gridspec_kw={'width_ratios': [5, 1]})
	ax1.axis('off')
	plt.subplots_adjust(left=0.13, right=0.8, bottom=0.13, top=0.95)
	ax.set_xlim(left=ld.obmin_plot[xi], right=ld.obmax_plot[xi])
	ax.set_ylim(bottom=ld.obmin_plot[0], top=ld.obmax_plot[0])
	ax.invert_yaxis()
	ax.set_ylabel(ylabel)
	ax.set_xlabel(xlabel)

	# plots and legends
	poleon = []; sideon = []
	for i in range(len(oM0)):
		sc = ax.scatter(x[:, i, 0], y[:, i, 0], s=1, color=cmap_colors[i], alpha=0.1)
		poleon.append(sc)
	for i in range(len(oM0)):
		sc = ax.scatter(x[:, i, 1], y[:, i, 1], s=1, color=cmap_colors[i], alpha=1.0, label=str(oM0[i]))
		sideon.append(sc)
	first_legend = ax.legend(title=r'$\omega_{\rm Mi}$', loc='upper left', bbox_to_anchor=(1.05, 0.55), \
		markerscale=5, frameon=False, handles=sideon)
	ax.add_artist(first_legend)
	first_legend._legend_box.align = "left"
	second_legend = ax.legend(title='Inclination', loc='upper left', bbox_to_anchor=(1.05, 0.75), markerscale=5, \
		frameon=False, handles=[poleon[0], sideon[0]], labels=[str(0), r'$\pi / 2$'])
	second_legend._legend_box.align = "left"

	plot_region(ax, ROI, ROI_kwargs)
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)

	# text box
	ax.text(1.0, 1.0, textstr, transform=ax.transAxes, fontsize=12,
	        verticalalignment='top', bbox=dict(facecolor='w', alpha=1.0, edgecolor='w'))

	plt.savefig('../../data/model_grids/cvmd/' + filename, dpi=300)
	plt.close()

# pre-compute Roche model volume versus PARS's omega
# and PARS's omega versus MESA's omega
sf.calcVA()
sf.calcom()

# Load and filter MIST models
print('Loading MIST...', end='')
start = time.time()
st = mu.Set('../../data/mist_grid.npy')
print('%.2f' % (time.time() - start) + ' seconds.')

st.select_MS() # select main sequence
st.select_Z(cf.Z) # select metallicity
st.select_valid_rotation() # select rotation with omega < 1
st.set_omega0() # set omega from omega_M; ignore the L_edd factor

# select age = 9.15436242
t = np.unique(st.t)[107]
st.select_age(t) 

print('Loading PARS...', end='', flush=True)
start = time.time()
with open('../../data/pars_grid_ZM' + str(cf.Z).replace('-', 'm').replace('.', 'p') + '.pkl', 'rb') as f: 
	pars = pickle.load(f)
print('%.2f' % (time.time() - start) + ' seconds.', flush=True)
mu.Grid.pars = pars # give a PARS grid reference to the grid class
# apply the lower mass cut-off for the primaries according the region of interest on the CMD 
Mmin = mu.Mlim(st)
st.select_mass(Mmin=Mmin)
print('minimum mass = ' + '%.4f' % Mmin, flush=True)

Mini = np.unique(st.Mini)
omega0 = np.unique(st.omega0)
inc = np.linspace(0, np.pi/2, 2)
grid = mu.Grid(st, Mini, omega0, inc, cf.A_V)
mag = grid.obs[..., 0]
col = grid.obs[..., 1]
vsini = grid.obs[..., 2]

oM0 = np.unique(st.oM0)
oM0_3d = oM0[np.newaxis, :, np.newaxis]
sh = list(grid.obs.shape)[:-1]
oM0_3d = np.broadcast_to(oM0_3d, sh)

## plot

textstr = '\n'.join((
	r'$A_{\rm V}=' + '%.2f' % cf.A_V + '$',
	r'${\rm [M/H]}_{\rm M}=' + str(cf.Z) + '$',	
	r'$\log{\,t}=' + '%.4f' % t + '$',
	r'$\sigma_{\rm m} = ' + '%.3f' % cf.std[0] + '$',
	r'$\sigma_{\rm c} = ' + '%.3f' % cf.std[1] + '$'))
filename = 'mist_cmd_t' + ('%.4f' % t).replace('.', 'p') + '.pdf'
plot(textstr, col,  mag, 'c = F435W - F814W', r'$m = {\rm F555W}$', filename)

textstr = '\n'.join((
	r'$A_{\rm V}=' + '%.2f' % cf.A_V + '$',
	r'${\rm [M/H]}_{\rm M}=' + str(cf.Z) + '$',	
	r'$\log{\,t}=' + '%.4f' % t + '$',
	r'$\sigma_{\rm m} = ' + '%.3f' % cf.std[0] + '$',
	r'$\sigma_{\rm v} = ' + '%.0f' % cf.std[2] + '$ km/s'))
filename = 'mist_vmd_t' + ('%.4f' % t).replace('.', 'p') + '.pdf'
plot(textstr, vsini, mag, r'v = $v_{\rm e}\,\sin{i}$, km/s', r'$m = {\rm F555W}$', filename)
