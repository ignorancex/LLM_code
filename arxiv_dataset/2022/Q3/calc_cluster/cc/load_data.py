# Load NGC1846 data, filter it
import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'paint_atmospheres')))
import config as cf

import numpy as np
from matplotlib import pyplot as plt
from scipy import special
from scipy import optimize

dirname = os.path.dirname(__file__)
datadir = os.path.join(dirname, '../data/')

# load data on stars with vsini
file = datadir + 'ngc1846_vsini.txt'
data0 = np.loadtxt(file).T
data0 = data0[0:12] # select fields
# record the range of these stars in RA and dec; this should be close to the MUSE field of view (FOV)
ramin = data0[-2].min()
ramax = data0[-2].max()
demin = data0[-1].min()
demax = data0[-1].max()
# delete the entries without valid IDs, observables or sky coordinates, 
# as well as those outside the color-magnitude window
mag = data0[2]
col = data0[1] - data0[3] 
m = np.all(data0 != 99, axis=0) & \
	(mag <= cf.ROI[0][1]) & (mag >= cf.ROI[0][0]) & (col <= cf.ROI[1][1]) & (col >= cf.ROI[1][0])
data0 = data0[:, m]
# record the observable fields
ID = data0[0].astype(int)
f435w = data0[1]
f555w = data0[2]
f814w = data0[3]
f435w_err = data0[4]
f555w_err = data0[5]
f814w_err = data0[6]
vsini = data0[7]
vsini_uperr = data0[8]
vsini_loerr = data0[9]
vsini_err = (vsini_loerr + vsini_uperr) / 2

# load data on low-vsini stars
file = datadir + 'ngc1846_lowvsini.txt'
data1 = np.loadtxt(file).T
data1 = data1[0:12] # select fields
# delete the entries without valid IDs, observables or sky coordinates, 
# those outside the color-magnitude window or outside the MUSE FOV,
# as well as those that had data with normal vsini 
mag = data1[2]
col = data1[1] - data1[3] 
ra = data1[-2]
dec = data1[-1]
ID1 = data1[0].astype(int)
m = np.all(data1 != 99, axis=0) & \
	(mag <= cf.ROI[0][1]) & (mag >= cf.ROI[0][0]) & (col <= cf.ROI[1][1]) & (col >= cf.ROI[1][0]) & \
	(ra <= ramax) & (ra >= ramin) & (dec <= demax) & (dec >= demin) & \
	~np.isin(ID1, ID)
data1 = data1[:, m]
# merge with previous data
ID_vsini = np.concatenate( (ID, ID1) )
f435w = np.concatenate( (f435w, data1[1]) )
f555w = np.concatenate( (f555w, data1[2]) )
f814w = np.concatenate( (f814w, data1[3]) )
f435w_err = np.concatenate( (f435w_err, data1[4]) )
f555w_err = np.concatenate( (f555w_err, data1[5]) )
f814w_err = np.concatenate( (f814w_err, data1[6]) )
# -1 in vsini and its error means vsini measurement was either below zero or couldn't be distinguished from zero
vsini = np.concatenate( (vsini, np.full_like(data1[7], -1, dtype=float)) ) 
vsini_err = np.concatenate( (vsini_err, np.full_like(data1[8], -1, dtype=float)) )

# load data on stars without vsini in the MUSE FOV
file = datadir + 'ngc1846_full.txt'
data2 = np.loadtxt(file).T
data2 = data2[0:9] # select fields
# delete the entries without valid IDs, observables or sky coordinates, 
# those outside the color-magnitude window or outside the MUSE FOV,
# as well as those that had data with vsini or low vsini
mag = data2[2]
col = data2[1] - data2[3]
ra = data2[-2]
dec = data2[-1]
ID2 = data2[0].astype(int)
m = np.all(data2 != 99, axis=0) & \
	(mag <= cf.ROI[0][1]) & (mag >= cf.ROI[0][0]) & (col <= cf.ROI[1][1]) & (col >= cf.ROI[1][0]) & \
	(ra <= ramax) & (ra >= ramin) & (dec <= demax) & (dec >= demin) & \
	~np.isin(ID2, ID_vsini)
data2 = data2[:, m]

# merge all the data; set vsini-related fields appropriately
vsini_nan = np.full(data2.shape[1], np.nan)
ID = np.concatenate( (ID, ID2) )
f435w = np.concatenate( (f435w, data2[1]) )
f555w = np.concatenate( (f555w, data2[2]) )
f814w = np.concatenate( (f814w, data2[3]) )
f435w_err = np.concatenate( (f435w_err, data2[4]) )
f555w_err = np.concatenate( (f555w_err, data2[5]) )
f814w_err = np.concatenate( (f814w_err, data2[6]) )
vsini = np.concatenate( (vsini, np.copy(vsini_nan)) )
vsini_err = np.concatenate( (vsini_err, np.copy(vsini_nan)) )
# get color
color = f435w - f814w
color_err = np.sqrt(f435w_err**2 + f814w_err**2)

## compute the boundaries of the color, magnitude and vsini space where we will need model priors
obs = np.stack( (f555w, color, vsini), axis=1 )
std = np.stack( (f555w_err, color_err, vsini_err), axis=1 )
std = np.maximum(std, cf.std[np.newaxis,:]) # apply the minimum bound on error
stdr = cf.std[np.newaxis,:] / std # ratio of minimum error to actual error
ker = cf.nsig * std * ( stdr + np.sqrt(1 - stdr**2) ) # half an error kernel
## data space grids with ranges that will allow 
## probability leakage checks and error kernel integration for individual stars;
## with coarse steps at least as small as the minimum standard deviations
obs0 = np.minimum.reduce([ 	cf.ROI[:, 0], np.nanmin(obs - ker, axis=0), \
							cf.ROI[:, 0] - cf.denorm_err * cf.nsig * cf.std ])
obs1 = np.maximum.reduce([ 	cf.ROI[:, 1], np.nanmax(obs + ker, axis=0), \
							cf.ROI[:, 1] + cf.denorm_err * cf.nsig * cf.std ]) 
nobs = cf.downsample * ( np.ceil((obs1 - obs0) / cf.std + 1).astype(int) ) # number of entries
obs_grids = [] # observable grids
step = np.empty( len(nobs) ) # their steps
for i in range(len(nobs)):
	ogrid = np.linspace(obs0[i], obs1[i], nobs[i]) # grid for this observable
	obs_grids.append(ogrid)
	step[i] = ogrid[1] - ogrid[0]
# plotting boundaries - these are inside the area where minimum-error density is defined
obmax_plot = cf.ROI[:, 1] + cf.plot_err * cf.nsig * cf.std
obmin_plot = cf.ROI[:, 0] - cf.plot_err * cf.nsig * cf.std
