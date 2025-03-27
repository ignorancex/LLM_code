import numpy as np
from astropy import time
import matplotlib.pyplot as plt
import matplotlib as mpl
print(mpl.matplotlib_fname())
from astropy.table import Table, vstack
from matplotlib.colors import LogNorm,ListedColormap


# adding a date stamp in the corner of the figure

import os, sys
import datetime
runtime = os.path.abspath((sys.argv[0])) + " run at " + datetime.datetime.now().strftime("%c")
tyb = dict(color='black', fontsize=8)

# plotting all photometry

# add timestamp to plot
tya = dict(color='black', fontsize=8)

fname_brite = 'brite_all.fits'
data_brite = Table.read(fname_brite)

x_brite = data_brite['time']
y_brite = data_brite['flux']

fname_astep = 'astep_all.fits'

data_astep = Table.read(fname_astep)

x_astep_all = data_astep['time']
y_astep_all = data_astep['flux']

fname_bring = 'bring_all.fits'

data_bring = Table.read(fname_bring)

x_bring = data_bring['time']
y_bring = data_bring['flux']

# panels in the plots made
t_start = 57750.

fig = plt.figure(figsize=(15,5))

ax = fig.add_subplot(111)

local_midday = 0.68 # local offset for the middle of the day.


t_plot_start_epoch = t_start - 100.
t_plot_end_epoch = t_start + 600.

def river2(t, f, t_offset=0.68, scaler=100):
    'convert time series to a river plot assume f is around zero'
    day_count = np.floor(t - t_offset)
    f_moved = (f*scaler) + day_count
    return (t - day_count - t_offset, f_moved, day_count)

def circsel(x, y, yc=57780., xc=0.5, xw=0.4, yw=10.):
    'completely bonkers filter to select an ellipse in the river diagram'
    mask = np.sqrt(((x-xc)/xw)**2 + ((y-yc)/yw)**2)

    m = (mask<1)
    xsel = x[m]
    ysel = y[m]
    return (xsel, ysel, m)

(x_aste, y_aste, day_count_aste) = river2(x_astep_all, y_astep_all, scaler=100)

# select 2017 ASTEP photoemtry mask
(xm2, ym2, mask2) = circsel(x_aste, day_count_aste, yc=57920,yw=80, xc=0.48, xw=0.35)
#ax.scatter(xm2, ym2, color='orange')

# select 2018 ASTEP photoemtry mask
(xm3, ym3, mask3) = circsel(x_aste, day_count_aste, yc=57920+365, yw=80, xc=0.48, xw=0.35)
#ax.scatter(xm3, ym3, color='pink')

x_astep_masked = x_aste[mask2+mask3]
y_astep_masked = y_aste[mask2+mask3]
x_astep_photom_masked = x_astep_all[mask2+mask3]
y_astep_photom_masked = y_astep_all[mask2+mask3]

ax.set_xlim(t_plot_start_epoch, t_plot_end_epoch)

def rebintimeseries(t, f, tedges):
    from astropy.stats import sigma_clip
    tc_bin = np.array([])
    fc_bin = np.array([])
    fsigc_bin = np.array([])
    fc_npoi = np.array([])

    t_lower = tedges[:-1]
    t_upper = tedges[1:]
    t_midpoint = (t_lower+t_upper)/2.

    for t_mid, t_low, t_high in zip(t_midpoint, t_lower, t_upper):
        # select points from flux that have
        selpoints = ((t > t_low) * (t < t_high))
        fluxsel = f[selpoints]
        #fluxesel = flux_c[np.where(np.abs(time_c-t)<(dt/2))]
        if (fluxsel.size > 3):
        #    print('time {} has {} points'.format(t_mid, fluxsel.size))

            meanflux = sigma_clip(fluxsel, sigma=3, maxiters=2, masked=True)
            meane = np.ma.mean(meanflux)
            if np.isfinite(meane):
                tc_bin = np.append(tc_bin, t_mid)
                fc_bin = np.append(fc_bin, np.ma.mean(meanflux))
                fsigc_bin = np.append(fsigc_bin, np.ma.std(meanflux))
                fc_npoi = np.append(fc_npoi, np.sum(selpoints))

    return (tc_bin, fc_bin, fsigc_bin, fc_npoi)

# epochs for rebinned flux
t_rebin = np.arange(t_plot_start_epoch, t_plot_end_epoch, 0.05)

# BRING
(t_bring_binned, f_bring_binned, f_bring_sig_binned, n_poi) = rebintimeseries(x_bring, y_bring, t_rebin)

mc = (f_bring_sig_binned < 0.01)

t_bring_sel, f_bring_sel, f_bring_sig_sel = t_bring_binned[mc], f_bring_binned[mc], f_bring_sig_binned[mc]

ax.errorbar(t_bring_sel, f_bring_sel, yerr=f_bring_sig_sel, fmt='.', alpha=0.5)

dat = Table( [t_bring_sel, f_bring_sel, f_bring_sig_sel], names=('time','flux','ferr')  )
dat.write('binned_flux_bring.dat', format='ascii.ecsv',  overwrite=True)

# ASTEP
(t_astep_binned, f_astep_binned, f_astep_sig_binned, n_poi) = rebintimeseries(x_astep_photom_masked, y_astep_photom_masked, t_rebin)

ma = (f_astep_sig_binned < 0.007)
t_astep_sel, f_astep_sel, f_astep_sig_sel = t_astep_binned[ma], f_astep_binned[ma], f_astep_sig_binned[ma]

ax.errorbar(t_astep_sel, f_astep_sel, yerr=f_astep_sig_sel, fmt='.', alpha=0.5)

dat = Table( [t_astep_sel, f_astep_sel, f_astep_sig_sel], names=('time','flux','ferr')  )

dat.write('binned_flux_astep.dat', format='ascii.ecsv', overwrite=True)

# BRITE
(t_brite_binned, f_brite_binned, f_brite_sig_binned, n_poi) = rebintimeseries(x_brite, y_brite, t_rebin)

mb = (f_brite_sig_binned < 0.01)
t_brite_sel, f_brite_sel, f_brite_sig_sel = t_brite_binned[mb], f_brite_binned[mb], f_brite_sig_binned[mb]

ax.errorbar(t_brite_sel, f_brite_sel, yerr=f_brite_sig_sel, fmt='.', alpha=0.5)

dat = Table( [t_brite_sel, f_brite_sel, f_brite_sig_sel], names=('time','flux','ferr')  )


#
#
# show histograms of photometric errors
#

fig2, (a1, a2, a3) = plt.subplots(3,1,figsize=(4,11))

a1.hist(f_astep_sig_binned,bins=20,range=[0,0.02])
a1.set_title('ASTEP')
a2.hist(f_brite_sig_binned,bins=20,range=[0,0.02])
a2.set_title('BRITE')
a3.hist(f_bring_sig_binned,bins=20,range=[0,0.02])
a3.set_title('BRING')

dat.write('binned_flux_brite.dat', format='ascii.ecsv', overwrite=True)

ax.set_xlabel('Time [HJD]')
ax.set_ylabel('Flux')

plt.draw()
fig.savefig('daily_average_per_instrument.jpg', dpi=200, transparent=True, bbox_inches='tight')
plt.show()
