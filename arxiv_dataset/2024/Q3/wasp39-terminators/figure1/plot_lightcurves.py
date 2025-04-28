import batman, catwoman
import os, sys
import pickle
from datetime import datetime
import numpy as np
import matplotlib
from matplotlib import cm
from matplotlib.patches import Arrow
from matplotlib.gridspec import GridSpec
from astropy.table import Table, Column, vstack
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

import juliet
from utils import *#(load_plt_params, convolve_model, convolve_model_xy,
                   #load_parula, ibin)

from utils import correct_lightcurve, get_lc, confidence_ellipse
# set the matplotlib parameters
pltparams = load_plt_params()
parula = load_parula()
COLOR = pltparams[pltparams['name']=='text.color']['value'][0]

fig = plt.figure(figsize=(13,11))
fig.set_facecolor('w')

# Load lightcurves and posteriors:
t, f, ferr, reg = np.loadtxt('co2_asymmetric_narrow/lc.dat', unpack = True, usecols = (0,1,2,4))
asymmetric_posteriors = pickle.load(open('co2_asymmetric_narrow/_dynesty_DNS_posteriors.pkl', 'rb'))
symmetric_posteriors = pickle.load(open('co2_symmetric_narrow/_dynesty_DNS_posteriors.pkl', 'rb'))

asymmetric_posteriors = asymmetric_posteriors['posterior_samples']
symmetric_posteriors = symmetric_posteriors['posterior_samples']

# Get systematics-corrected lightcurves:
amodel, fa, ferra = correct_lightcurve(t, f, ferr, reg, asymmetric_posteriors)
smodel, fs, ferrs = correct_lightcurve(t, f, ferr, reg, symmetric_posteriors)

# Now get mean catwoman and batman lightcurves:
params = batman.TransitParams()
cwparams = catwoman.TransitParams()
batman_lc, bm_full = get_lc(params, t, reg, symmetric_posteriors, method = 'batman', nsamples = 140)
catwoman_lc, cw_full = get_lc(cwparams, t.astype('float64'), reg, asymmetric_posteriors, method = 'catwoman', nsamples = 140)

gs = GridSpec(220+50+150+50, 2, figure=fig, wspace = 0.3)#, height_ratios=[2,2,1])#, wspace=0.05, hspace=0)

ax1_cw = fig.add_subplot(gs[:150, 0]) 
ax1_cw.set_ylabel('Relative flux')
ax2_cw = fig.add_subplot(gs[150+25:220+25, 0])
ax2_cw.set_ylabel('O-C (ppm)')
ax2_cw.set_xlabel('Time from mid-transit (hr)')
ax3_cw = fig.add_subplot(gs[220+50+50:, 0])
ax3_cw.set_xlabel('Evening depth (ppm)')
ax3_cw.xaxis.label.set_color('tomato')
ax3_cw.set_ylabel('Morning depth (ppm)')
ax3_cw.yaxis.label.set_color('cornflowerblue')

ax1_bm = fig.add_subplot(gs[:150, 1]) 
ax2_bm = fig.add_subplot(gs[150+25:220+25, 1]) 
ax2_bm.set_xlabel('Time from mid-transit (hr)')
ax3_bm = fig.add_subplot(gs[220+50+50:, 1]) 
ax3_bm.set_xlabel('Transit depth (ppm)')
ax3_bm.xaxis.label.set_color('grey')

# Deactivate some axes:
for ax in [ax1_cw, ax1_bm]:

    ax.tick_params(labelbottom=False)

for ax in [ax1_bm, ax2_bm, ax3_bm]:

    ax.tick_params(labelleft=False)

# Plot data:
t0 = 2459771.335647 
ax1_cw.errorbar((t-t0)*24, fa, ferra, fmt = '.', ms = 1, elinewidth = 1, color = 'black', alpha = 0.1)
ax1_cw.plot((t-t0)*24, catwoman_lc, color = 'purple', lw = 3)

ax1_bm.errorbar((t-t0)*24, fs, ferrs, fmt = '.', ms = 1, elinewidth = 1, color = 'black', alpha = 0.1)
ax1_bm.plot((t-t0)*24, batman_lc, color = 'grey', lw = 3)

# Plot residuals:
min_t, max_t = np.min((t-t0)*24), np.max((t-t0)*24)
ax2_cw.plot([min_t, max_t], [0, 0], '--', color = 'purple', lw = 3)
#ax2_cw.errorbar((t-t0)*24, (fa - catwoman_lc)*1e6, ferra*1e6, fmt = '.', ms = 1, elinewidth = 1, color = 'black', alpha = 0.1)

ax2_bm.plot([min_t, max_t], [0, 0], '--', color = 'grey', lw = 3)
#ax2_bm.errorbar((t-t0)*24, (fs - batman_lc)*1e6, ferrs*1e6, fmt = '.', ms = 1, elinewidth = 1, color = 'black', alpha = 0.1)

# Plot binned residuals:
cwt, cwbin, cwbin_err = juliet.utils.bin_data((t-t0)*24, (f - cw_full)*1e6, n_bin = 120)#120)
ax2_cw.errorbar(cwt, cwbin, cwbin_err, fmt = 'o', ms = 8, mec = 'purple', ecolor = 'purple', elinewidth = 2, mfc = 'white')

bmt, bmbin, bmbin_err = juliet.utils.bin_data((t-t0)*24, (f - bm_full)*1e6, n_bin = 120)#120)
ax2_bm.errorbar(bmt, bmbin, bmbin_err, fmt = 'o', ms = 8, mec = 'grey', ecolor = 'grey', elinewidth = 2, mfc = 'white')
ax2_bm.plot((t-t0)*24, (catwoman_lc - batman_lc)*1e6, color = 'purple', lw = 3)

# Plot catwoman - batman on the batman plot (likely too confusing?)
#ax2_bm.plot((t-t0)*24, (cw_full - bm_full)*1e6, color = 'purple', lw = 3, alpha = 0.3)

# Plot posteriors:
idx = np.random.choice(np.arange(len(asymmetric_posteriors['p1_p1'])), 3000, replace = False)

evening_depth, morning_depth = (asymmetric_posteriors['p1_p1'][idx]**2)*1e6*0.5, (asymmetric_posteriors['p2_p1'][idx]**2)*1e6*0.5

#ax3_cw.plot(evening_depth, morning_depth,
#            '.', color = 'purple', ms = 2, alpha = 0.01)

confidence_ellipse(evening_depth, morning_depth,
                   ax3_cw, facecolor='purple', n_std = 1.0, alpha = 0.3)

confidence_ellipse(evening_depth, morning_depth,
                   ax3_cw, facecolor='purple', n_std = 2.0, alpha = 0.3)

confidence_ellipse(evening_depth, morning_depth,
                   ax3_cw, facecolor='purple', n_std = 3.0, alpha = 0.3)

ax3_cw.plot([0,100000], [0,100000], '--', color = 'grey', lw = 3)

ax3_cw.set_xlim(10600, 10600+2500-200)
ax3_cw.set_ylim(10150-500, 10150+1500+500-200)
ax3_cw.set_yticks([10750-1000,10750,10750+1000])
ax3_cw.set_xticks([11750-1000,11750,11750+1000])

ax3_bm.hist((symmetric_posteriors['p_p1']**2)*1e6, color = 'grey')
ax3_bm.yaxis.set_tick_params(labelleft=False)
ax3_bm.set_yticks([])

# Set limits:
for ax in [ax1_cw, ax1_bm, ax2_cw, ax2_bm]:

    ax.set_xlim(min_t, max_t)

for ax in [ax2_cw, ax2_bm]:

    ax.set_ylim(-450,450)

plt.savefig('lc_model.pdf',
            dpi=250,#, rasterize=True,
            #transparent=True,
            bbox_inches='tight')
