import numpy as np
import pylab as plt
from pdf_trans import spin2risco, risco2spin
from matplotlib.ticker import MultipleLocator

'''
python plot_rISCO_spin.py
'''

# Output file name
outdir = '../ms/figures/'
fout   = outdir + 'rISCO_spin.eps'

# Calculate r_ISCO(a)
a_min, a_max = -1, 1
N = 5001
a = np.linspace(a_min, a_max, N)
r_ISCO = np.zeros(N)
for i in np.arange(N):
    r_ISCO[i] = spin2risco(a=a[i])

# Plotting preferences
dpi = 300
xsize, ysize = 8.4, 8.4
left, right, bottom, top = 0.15, 0.97, 0.15, 0.97
fs, fs_sm, lw, pad, tlmaj, tlmin = 28, 24, 2, 12, 10, 5
fs_in, fs_sm_in, lw_in, pad_in, tlmaj_in, tlmin_in = 20, 16, 2, 6, 10, 5
xmin, xmax = a_min, a_max
ymin, ymax = 1.0, 10.0
xinc_maj, xinc_min = 0.5, 0.1
yinc_maj, yinc_min = 1.0, 0.5
xmajorLocator = MultipleLocator(xinc_maj)
xminorLocator = MultipleLocator(xinc_min)
ymajorLocator = MultipleLocator(yinc_maj)
yminorLocator = MultipleLocator(yinc_min)
xinc_maj_in, xinc_min_in = 0.02, 0.01
yinc_maj_in, yinc_min_in = 0.5,  0.25
xmajorLocator_in = MultipleLocator(xinc_maj_in)
xminorLocator_in = MultipleLocator(xinc_min_in)
ymajorLocator_in = MultipleLocator(yinc_maj_in)
yminorLocator_in = MultipleLocator(yinc_min_in)

fig = plt.figure(figsize=(xsize, ysize), dpi=dpi)
fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
ax = fig.add_subplot(111)
ax.set_xlabel(r"${\rm Black\ Hole\ Spin\ Parameter,}\ a$", fontsize=fs, labelpad=pad)
ax.set_ylabel(r"${\rm Innermost\ Stable\ Circular\ Orbit,}\ r_{\rm ISCO}$", fontsize=fs, labelpad=pad*2)
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.tick_params('both', direction='in', length=tlmaj, width=lw, which='major', labelsize=fs_sm, pad=pad)
ax.tick_params('both', direction='in', length=tlmin, width=lw, which='minor')
ax.xaxis.set_major_locator(xmajorLocator)
ax.xaxis.set_minor_locator(xminorLocator)
ax.yaxis.set_major_locator(ymajorLocator)
ax.yaxis.set_minor_locator(yminorLocator)

ax.plot(a, r_ISCO, linewidth=lw, linestyle='solid', color='black')

avals = [-0.5, 0.0, 0.5, 0.90]
for i in np.arange(4):
    a0 = avals[i]
    r0 = spin2risco(a=a0)
    ax.plot([a0, a0], [ymin, r0], linewidth=lw, linestyle='dotted', color='black')
    ax.plot([xmin, a0], [r0, r0], linewidth=lw, linestyle='dotted', color='black')

# Inset Plot
ax_inset = fig.add_axes([0.525, 0.725, 0.4, 0.2])  # left, bottom, width, height
ax_inset.plot(a, r_ISCO, 'k-', linewidth=lw)
ax_inset.set_xlabel(r"${\rm Black\ Hole\ Spin,}\ a$", fontsize=fs_in, labelpad=pad_in)
ax_inset.set_ylabel(r"$r_{\rm ISCO}$",   fontsize=fs_in, labelpad=pad_in)
ax_inset.set_xlim(0.95, 1.0)
ax_inset.set_ylim(ymin, 2.0)
ax_inset.tick_params('both', direction='in', labelsize=fs_sm_in, length=tlmaj_in, width=lw, which='major', pad=pad_in)
ax_inset.tick_params('both', direction='in', labelsize=fs_sm_in, length=tlmin_in, width=lw, which='minor')
ax_inset.xaxis.set_major_locator(xmajorLocator_in)
ax_inset.xaxis.set_minor_locator(xminorLocator_in)
ax_inset.yaxis.set_major_locator(ymajorLocator_in)
ax_inset.yaxis.set_minor_locator(yminorLocator_in)

a0 = 0.998
r0 = spin2risco(a=a0)
ax_inset.plot([a0, a0], [ymin, r0], linewidth=lw, linestyle='dotted', color='black')
ax_inset.plot([xmin, a0], [r0, r0], linewidth=lw, linestyle='dotted', color='black')

fig.savefig(fout, bbox_inches=0, dpi=dpi)
plt.close()
