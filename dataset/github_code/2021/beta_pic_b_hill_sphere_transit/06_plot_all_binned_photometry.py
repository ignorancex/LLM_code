import numpy as np
import matplotlib.pyplot as plt
import betapic as bp
from astropy.table import Table

import os, sys
import datetime
runtime = os.path.abspath((sys.argv[0])) + " run at " + datetime.datetime.now().strftime("%c")
tyb = dict(color='black', fontsize=16)
pltfmt = dict(color='blue', fmt='.', alpha=0.4, mec=None, mew=0)

f_brite = Table.read('binned_flux_brite.dat', format='ascii.ecsv')
f_astep = Table.read('binned_flux_astep.dat', format='ascii.ecsv')
f_bring = Table.read('binned_flux_bring.dat', format='ascii.ecsv')

f_hst = Table.read('./data/hst/betapic_hst_phot.csv', format='ascii.csv')

figc, (a1, a2, a3, a4) = plt.subplots(4,1,figsize=(10,6),
                                      sharex=True,
                                      sharey=True)

a1.errorbar(f_brite['time'],  f_brite['flux'], yerr=f_brite['ferr'], **pltfmt)
a2.errorbar(f_astep['time'],  f_astep['flux'], yerr=f_astep['ferr'], **pltfmt)
a3.errorbar(f_bring['time'],  f_bring['flux'], yerr=f_bring['ferr'], **pltfmt)
a4.scatter(f_hst['BJD']-2400000., f_hst['Norm_flux']-1.00)

for (a, name) in zip((a1,a2,a3,a4),['BRITE','ASTEP','BRING','HST']):

    # red dotted zero line
    a.axhline(ls='dotted',lw=1, color='red',zorder=10)

    # Hill sphere
    bp.addhill(a)

    # instrument label
    a.text(0.05, 0.70, name, ha='left', va='bottom', transform=a.transAxes, **tyb)
    
a1.set_xlim(57600, 58350)
a1.set_ylim(-0.04,0.04)

aa = figc.add_subplot(111, frameon=False)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.grid(False)
plt.xlabel("Epoch [MJD]", fontsize=18)

aa.xaxis.set_label_coords(0.5, -0.08)

plt.ylabel("Change in normalised flux", fontsize=18)
aa.yaxis.set_label_coords(-0.08, 0.5)

###a1.text(0.98, 0.90, runtime, ha='right', va='bottom', transform=a1.transAxes, fontsize=8)
#plt.draw()
plt.savefig('figs/all_binned_photometry.pdf', bbox_inches='tight')

