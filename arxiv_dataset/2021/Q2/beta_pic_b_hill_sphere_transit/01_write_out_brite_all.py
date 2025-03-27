from astropy.io import ascii
from astropy.table import Table, vstack
import numpy as np

# BRITE residual light curves are in millimagnitudes with the mean magnitude subtracted.
# The first column is HJD (mid exposure) - 2456000.0 in days.

fname_out = 'brite_all.fits'

t1 = ascii.read('data/brite/betaPic-BRITE-R-2015.dat')
t2 = ascii.read('data/brite/betaPic-BRITE-R-2016-17.dat')
t3 = ascii.read('data/brite/betaPic-BRITE-R-2017-18.dat')
print(t1['col1'])
t_b = vstack([t1,t2,t3])

# convert to MJD
time = t_b['col1'] + 2456000.0 - 2400000.5

# the second column is in millimagnitudes
# convert to normalised flux - 1
f1 = np.power(10.,( ((t_b['col2']/1000.)/-2.5))) - 1.

t = Table([time, f1], names=('time','flux'))

t.write(fname_out)
print('read in and wrote out {}'.format(fname_out))

