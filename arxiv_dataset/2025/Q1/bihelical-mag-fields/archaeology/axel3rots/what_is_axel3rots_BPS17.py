"""
The data in the file axel3rots.sav were used in BPS2017. Nishant says it was on a latitude-longitude grid. Here, I try to figure out what this data is by comparing it with the data I downloaded from HMI (and my rebinned version of that data.

From the results, it is clear that axel3rots contains data on a sin_latitude✕longitude grid (as mentioned in the paper).

Interestingly, the amplitude of the magnetic field also differs!
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import scipy.io

#Following files are present only on the work PC
bp2161_petrie = scipy.io.readsav("../nishant_idl_scripts/examples/gordon_petrie/sav/axel3rots.sav")['bp2161']
bp2161 = fits.getdata("../images/hmi.b_synoptic_small.2161.Bp.fits")
bp2161_rebin = fits.getdata("../images/hmi.b_synoptic_small.rebinned.2161.Bp.fits")

fig,axs = plt.subplots(3, 1)
axs[0].set_title("axel3rots")
axs[1].set_title("hmi.b_synoptic_small")
axs[2].set_title("rebinned")
fig.suptitle(r"$B_\phi$, CR 2161")

for data, ax in zip([bp2161_petrie, bp2161, bp2161_rebin], axs):
	vmin = -100
	vmax = 100
	levels = np.linspace(vmin,vmax,101)
	im = ax.contourf(data, cmap='bwr', levels=levels, vmin=vmin, vmax=vmax, extend='both')
	c = plt.colorbar(im, ax=ax)
	nlat, nlon = np.shape(data)
	
	ax.axhline((nlat-1)/2, c='k', ls="--", lw=1)
	ax.axhline((nlat-1)*0.6, c='k', ls="--", lw=1)
	ax.axhline((nlat-1)*0.4, c='k', ls="--", lw=1)

fig.set_size_inches(4,7)
fig.tight_layout()

fig.savefig("what_is_axel3rots.png", dpi=500)
