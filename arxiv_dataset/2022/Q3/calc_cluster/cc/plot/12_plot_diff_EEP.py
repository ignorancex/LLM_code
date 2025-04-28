# plot a histogram of observable differences
# between t = 9.1544 and t = 9.1594 at constant EEP
import sys, os, time, pickle
sys.path.append(os.path.abspath(os.path.join('..')))
import numpy as np
import config as cf

from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif", 
    "font.serif": "Computer Modern",
    "font.size": 20
})

iodir = '../../data/'
with open(iodir + 'observables/dist1.pkl', 'rb') as f: dist1 = pickle.load(f)
with open(iodir + 'observables/dist2.pkl', 'rb') as f: dist2 = pickle.load(f)

print ('Plotting observable differences between age ' + '%.4f' % dist2[0]['prev'] + ' and age ' + '%.4f' % dist2[0]['curr'] + '.')

# values for the following two variables obtained from printout of observables calculations
diff = dist2[0]['dist']
diff_avg = np.array( [d['mean_dist'][0] for d in dist1 + dist2] )

# plot histograms for one age neighbor pair, all EEPs and observables
bins = np.linspace(-3,4,15)
majorLocator = MultipleLocator(1.0)
majorFormatter = FormatStrFormatter('%d')
minorLocator = MultipleLocator(0.5)

plt.hist(diff, bins=bins, \
    label=['magnitude', 'color', r'$v_{\rm e}\sin{i}$'], alpha=0.8)
plt.legend(loc=(0.07, 0.6), frameon=False)
plt.axvline(-3.0, color='grey', linestyle='--')
plt.axvline(3.0, color='grey', linestyle='--')
ax = plt.gca()
ax.xaxis.set_major_locator(majorLocator)
ax.xaxis.set_major_formatter(majorFormatter)
ax.xaxis.set_minor_locator(minorLocator)
# plt.xticks(np.linspace(-3,4,8))
plt.xlabel(r'$\Delta \boldmath{x} / \boldmath{\sigma}_{\boldmath{x}}$')
plt.tight_layout()
plt.savefig(iodir + 'diff_EEP.pdf', dpi=300)
plt.close()

# plot the histogram just for magnitudes, for all age neighbor pairs
bins = np.linspace(-1,4,11)
plt.hist(diff_avg, bins=bins, histtype='step', lw=2)
ax = plt.gca()
ax.xaxis.set_major_locator(majorLocator)
ax.xaxis.set_major_formatter(majorFormatter)
ax.xaxis.set_minor_locator(minorLocator)
# plt.xticks(np.linspace(-1,5,7))
plt.xlabel(r'$\Delta m / \sigma_m$')
plt.tight_layout()
plt.savefig(iodir + 'delta_m_delta_t.pdf', dpi=300)
plt.close()