import sys
sys.path.append('..')
import config as cf

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import ticker
from matplotlib.ticker import FormatStrFormatter
import pickle, glob, os

mpl.rcParams['font.size'] = 12
min_ll = -10
max_ll = 0
# color map
cmapBig = mpl.cm.get_cmap('Greys', 512)
cmap = mpl.colors.ListedColormap(cmapBig(np.linspace(0, 1, 256)))
norm = mpl.colors.Normalize(vmin=min_ll, vmax=max_ll, clip=True)

like_dir = '../../data/likelihoods/'
t0_label = r'$\mu_{\log_{10}{t}}$'
t1_label = r'$\sigma_{\log_{10}{t}}$'
t0_hat = r'$\widehat{\mu}_{\log_{10}{t}}$'
t1_hat = r'$\widehat{\sigma}_{\log_{10}{t}}$'

def plot(x, y, qm, bm, ll, xlabel, ylabel, textstr, filename):
	fig, [ax, ax1] = plt.subplots(ncols=2, gridspec_kw={'width_ratios': [8, 3]})
	fig.set_figwidth(8)
	plt.subplots_adjust(left=0.13, right=0.9, bottom=0.13, top=0.95)
	pcm = ax.pcolormesh(x, y, ll, cmap=cmap, shading='nearest', norm=norm) 
	ax.set_xlabel(xlabel, fontsize=14)
	ax.set_ylabel(ylabel, fontsize=14)
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)

	# maximum q values
	csq = ax.contour(x, y, qm, 3, colors='r', linestyles='dashed')
	ax.clabel(csq, fmt='%.3f')
	csq.collections[0].set_label(r'$\widehat{q}_{\rm local}$')
	# maximum b values
	csb = ax.contour(x, y, bm, 3, colors='b', linestyles='dashed')
	ax.clabel(csb, fmt='%.2f')
	csb.collections[0].set_label(r'$\widehat{b}_{\rm local}$')

	ax.legend(loc='upper left', bbox_to_anchor=(1.35, 0.63))

	# color bar
	ax1.axis('off')
	cax = fig.add_axes([0.7, 0.1, 0.03, 0.4])	
	ticks = ticker.LinearLocator(5)
	cb = fig.colorbar(mappable=pcm, ax=ax1, cax=cax, orientation='vertical', ticks=ticks, \
		alpha=1.0, shrink=0.6, norm=norm) # format='%.1f',
	cb.set_label(label=r'$\ln{\cal L}$', fontsize=18, rotation=0, labelpad=25, y=0.65)

	# text box
	ax.text(1.05, 1.0, textstr, transform=ax.transAxes, fontsize=14,
			verticalalignment='top', bbox=dict(facecolor='w', alpha=1.0, edgecolor='w'))

	plt.savefig(filename, dpi=300)
	plt.close()

# filelist = list(np.sort(glob.glob('../data/likelihoods/pkl/ll*.pkl')))
filelist = list(np.sort(glob.glob(like_dir + 'pkl/ll*.pkl')))
for filepath in filelist: # for each combination of age and metallicity
	# load the log likelihood and the maximum q
	with open(filepath, 'rb') as f:
		ll_4d, qm_4d, bm_4d, t0_ar, t1_ar, w0, w2, om_sigma = pickle.load(f)
	# get base file name
	base = os.path.basename(filepath).split('.')[0]
	if len(base.split('p')[-1]) == 1:
		base = base + '0'
	# set maximum likelihood to zero
	ll_4d -= np.nanmax(ll_4d) 	
	# indices of ML parameters
	w0m, w2m, t0m, t1m = np.unravel_index(np.nanargmax(ll_4d), ll_4d.shape)  

	# plot likelihood vs. rotational proportions, 
	# at maximum-likelihood binaries proportion and age priors
	ll = ll_4d[..., t0m, t1m] # log-likelihoods
	qm = qm_4d[..., t0m, t1m] # maximum q
	bm = bm_4d[..., t0m, t1m] # maximum b
	# text box text
	textstr = '\n'.join((
		r'$A_{\rm V}=' + '%.2f' % cf.A_V + '$',		
		r'${\rm [M/H]}_{\rm MIST}=' + str(cf.Z) + '$',
	    r'$\sigma_{\rm \omega} = \{' + ', '.join(['%.2f' % n for n in cf.om_sigma]) + '\}$',
		r'$\widehat{w} = \{' + '%.2f' % w0[w0m] + ', ' + '%.2f' % (1 - w0[w0m] - w2[w2m]) +\
			', ' + '%.2f' % w2[w2m] + '\}$',
	    r'' + t0_label + ' = ' + t0_hat + ' = ' + '%.4f' % t0_ar[t0m],
	    r'' + t1_label + ' = ' + t1_hat + ' = ' + '%.4f' % t1_ar[t1m],
		r'$\widehat{q} = $' + '%.3f' % qm[w0m, w2m],
		r'$\widehat{b} = $' + '%.2f' % bm[w0m, w2m]))
	filename = like_dir + 'png/' + base + '_rotation' + '.pdf'
	plot(cf.w2, cf.w0, qm, bm, ll, r'$w_2$', r'$w_0$', textstr, filename)

	# plot likelihood vs. age priors, 
	# at maximum-likelihood binaries proportion and rotational population proportions
	ll = ll_4d[w0m, w2m, ...] # log-likelihoods
	qm = qm_4d[w0m, w2m, ...] # maximum q
	bm = bm_4d[w0m, w2m, ...] # maximum b
	# text box text
	textstr = '\n'.join((	
		r'$A_{\rm V}=' + '%.2f' % cf.A_V + '$',	
		r'${\rm [M/H]}_{\rm MIST}=' + str(cf.Z) + '$',
		r'$\sigma_{\rm \omega} = \{' + ', '.join(['%.2f' % n for n in om_sigma]) + '\}$',
		r'$w = \widehat{w} = \{' + '%.2f' % w0[w0m] + ', ' + '%.2f' % (1 - w0[w0m] - w2[w2m]) +\
			', ' + '%.2f' % w2[w2m] + '\}$',
	    t0_hat + ' = ' + '%.4f' % t0_ar[t0m],
	    t1_hat + ' = ' + '%.4f' % t1_ar[t1m],
		r'$\widehat{q} = $' + '%.3f' % qm[t0m, t1m],
		r'$\widehat{b} = $' + '%.2f' % bm[t0m, t1m]))
	filename = like_dir + 'png/' + base + '_age' + '.pdf'
	plot(t1_ar, t0_ar, qm, bm, ll, t1_label, t0_label, textstr, filename)