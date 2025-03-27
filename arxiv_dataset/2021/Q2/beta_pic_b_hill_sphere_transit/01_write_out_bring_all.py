from astropy.io import ascii
from astropy.table import Table
import numpy as np

fname_bring = 'data/bring/Reduced_2110269_v20190625.fits'

fname_out = 'bring_all.fits'

data_bring = Table.read(fname_bring)

# JD == Julian Date
# Raw == with initial reduction (spatio-temporal calibration, interpixel calibration), but no further detrending (moon/daily/lst/…). Units are V-magnitude
# Reduced == with initial reduction (spatio-temporal calibration, interpixel calibration), detrended (moon/daily/lst/…). Units are delta-V-magnitude (median should be approximately zero).

# convert to MJD
time_all = data_bring['jd']-2400000.5

# bad values have nan in the 'reduced' column

m = ~np.isnan(data_bring['reduced'])

time = time_all[m]

#
y_bring_all = np.power(10.,(-data_bring['reduced']/2.5)) - 1.

y_bring = y_bring_all[m]

t = Table([time, y_bring], names=('time','flux'))

t.write(fname_out)
print('read in {} and wrote out {}'.format(fname_bring,fname_out))
