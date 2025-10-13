import numpy as np
from astropy import units
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.timeseries import LombScargle

rc = Table.read('../rcParams.txt', format='csv')
for name, val in zip(rc['name'], rc['value']):
    plt.rcParams[name] = val

# Load in the relevant data
data1 = np.load('../data/stacked_3I_2-3_v4.npy', allow_pickle=True).item()
tpf1 = data1['subtracted'] * 1.0
err1 = data1['err_sub'] * 1.0

data2 = np.load('../data/stacked_3I_1-2_v4.npy', allow_pickle=True).item()
tpf2 = data2['subtracted'] * 1.0
err2 = data2['err_sub'] * 1.0

#################
# Plot the data #
#################

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))

q = data1['good_frames'] == 0
im1 = ax1.imshow(np.nanmedian(tpf1[q], axis=0), aspect='auto',
                 origin='lower', vmin=0)


q = data2['good_frames'] == 0
im2 = ax2.imshow(np.nanmedian(tpf2[q], axis=0), aspect='auto',
                 origin='lower', vmin=0, vmax=0.5)


ims = [im1, im2]
fc = 'w'

for i, ax in enumerate([ax1, ax2]):
    ax.plot(9,9,'o', ms=70, color="none", markeredgecolor='w', markeredgewidth=3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(x=5.75, y=4, s='3I/ATLAS', color=fc, fontweight='bold')
    ax.axhline(0.5, 0.06, 0.15, color=fc, zorder=10, lw=3)
    ax.text(0.5, 0.8, "41''", color=fc)

ax1.set_title('May 7 - 17, 2025', fontsize=16)
ax2.set_title('May 20 - June 2, 2025', fontsize=16)

#plt.show()

plt.savefig('../figures/press_release_image.pdf', bbox_inches='tight', dpi=300)
