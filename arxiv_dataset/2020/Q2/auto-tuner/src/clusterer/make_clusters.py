"""
Author: Renato Durrer
Created: 06.05.2019

"""
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import bisect
import Labber
from src.utils.funcs import data_creator
from scipy.optimize import curve_fit
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.preprocessing import StandardScaler
import itertools
from src.utils.visualization import latexify
from scipy import linalg
import matplotlib as mpl

from sklearn import mixture

############
# get data #
############

folderpath = '../../data/coarse/cluster_test/'
datapaths = [os.path.normpath(os.path.join(folderpath, x))
             for x in os.listdir(folderpath) if (not x.endswith('_cal.hdf5') and x.endswith('.hdf5'))]

calpaths = [os.path.normpath(os.path.join(folderpath, x))
            for x in os.listdir(folderpath) if x.endswith('_cal.hdf5')]

labber_data = Labber.LogFile(datapaths[1])
raw_data = data_creator(labber_data)

labber_cal = Labber.LogFile(calpaths[0])
raw_cal = data_creator(labber_cal)
# in practise one would need to calibrate the qpc and then measure

#################
# Calibrate QPC #
#################

# get gate configuration for operating point, i.e. x_0
LPG0 = None
MPG0 = None
QPCM0 = None

for channel in labber_cal.getStepChannels():
    if channel['name'] == 'LPG0':
        LPG0 = channel['values'][0]
    if channel['name'] == 'MPG0':
        MPG0 = channel['values'][0]

for channel in labber_data.getStepChannels():
    if channel['name'] == 'QPC_M':
        QPCM0 = channel['values'][0]

# interpolate QPC signal as calibrated
IQPC_cal = CubicSpline(raw_cal['QPC_M'][::-1], raw_cal['I QPC'][::-1])

# check the spline

x_data = np.linspace(-1.2, 0, 500)


##########################
# obatin cross couplings #
##########################

# go to x_0, and wiggle with both plunger gates a bit
# problem: x_0 is not present in measurement data -> take closest point. But correction needed...


def find_nearest(a, a0):
    idx = np.abs(a - a0).argmin()
    return a.flat[idx]

# find nearest gate config to x_0, called meas_lpg0 & meas_mpg0
meas_lpg0 = find_nearest(raw_data['LPG0'][:, 0], LPG0)
meas_mpg0 = find_nearest(raw_data['MPG0'][0, :], MPG0)

# get the respective index in the measurement data
idx_l = np.where(raw_data['LPG0'][:, 0] == meas_lpg0)[0][0]
idx_m = np.where(raw_data['MPG0'][0, :] == meas_mpg0)[0][0]

# get the measured QPC current at that point
IQPC0_meas = raw_data['I QPC'][idx_l, idx_m]

# get the differences in measurements
if np.argmin([np.abs(raw_data['I QPC'][idx_l+1, idx_m] - IQPC0_meas), np.abs(raw_data['I QPC'][idx_l-1, idx_m] - IQPC0_meas)]):
    del_I_PG_L = IQPC0_meas - raw_data['I QPC'][idx_l-1, idx_m]
    I_PG_L = raw_data['I QPC'][idx_l - 1, idx_m]
else:
    del_I_PG_L = raw_data['I QPC'][idx_l+1, idx_m] - IQPC0_meas
    I_PG_L = raw_data['I QPC'][idx_l + 1, idx_m]
dif_L = np.abs((raw_data['LPG0'][idx_l, idx_m] - raw_data['LPG0'][idx_l+2, idx_m]) / 2)

if np.argmin([np.abs(raw_data['I QPC'][idx_l, idx_m+1] - IQPC0_meas), np.abs(raw_data['I QPC'][idx_l, idx_m-1] - IQPC0_meas)]):
    del_I_PG_M = IQPC0_meas - raw_data['I QPC'][idx_l, idx_m-1]
    I_PG_M = raw_data['I QPC'][idx_l, idx_m-1]
else:
    del_I_PG_M = raw_data['I QPC'][idx_l, idx_m+1] - IQPC0_meas
    I_PG_M = raw_data['I QPC'][idx_l, idx_m+1]
dif_M = np.abs((raw_data['MPG0'][idx_l, idx_m] - raw_data['MPG0'][idx_l, idx_m+2]) / 2)

# find the corresponding values in the spline
# later the offset won't be needed anymore
offset = IQPC0_meas - IQPC_cal(QPCM0)
opt_M = lambda x: IQPC_cal(x+QPCM0) - IQPC_cal(QPCM0) + del_I_PG_M
opt_L = lambda x: IQPC_cal(x+QPCM0) - IQPC_cal(QPCM0) + del_I_PG_L

delta_M = bisect(opt_M, -50*dif_M, 50*dif_M, maxiter=100)
delta_L = bisect(opt_L, -50*dif_L, 50*dif_L, maxiter=100)

coup_M = delta_M / dif_M
coup_L = delta_L / dif_L


# define IQPC as a function of PG's
def QPC(PG_1, PG_2, QPC_G=QPCM0, coup_1_1=0.11898617, coup_2_1=0.18459571, off=0.00313063):
    x = QPC_G + coup_1_1*PG_1 + coup_2_1*PG_2 + off
    return IQPC_cal(x)
    # return IQPC_cal(QPC_G) + 6.4e-9*PG_1 + 3.7e-9*PG_2 + off


dim_x = len(raw_data['I QPC'][:, 0])
dim_y = len(raw_data['I QPC'][0, :])

def fitter(x, y0, y2, y4):
    return 1e12 * QPC(x[0], x[1], coup_1_1=y0,  coup_2_1=y2, off=y4)

# remove outliers
# idea: points that have higher absolute value than 50*abs(median) are outliers

med = np.median(raw_data['I QPC'])
threshold = 500*np.abs(med)

x0 = [raw_data['LPG0'][257:, 192:].flatten(), raw_data['MPG0'][257:, 192:].flatten()]
y0 = raw_data['I QPC'][257:, 192:].flatten()
x0[0] = x0[0][np.abs(y0) < threshold]
x0[1] = x0[1][np.abs(y0) < threshold]
y0 = y0[np.abs(y0) < threshold]

params = curve_fit(fitter, x0, 1e12*y0, p0=[0.12, 0.132, 0.075], method='lm')

# predict the QPC current and plot it agains measured values
current_L = raw_data['I QPC'][:, idx_m]
PGL = raw_data['LPG0'][:, idx_m]

current_M = raw_data['I QPC'][480, :]
PGM = raw_data['MPG0'][480, :]

current_pred_L = QPC(PGL, MPG0, coup_1_1=params[0][0], coup_2_1=params[0][1], off=params[0][2])
current_pred_M = QPC(raw_data['LPG0'][480, 480], PGM, coup_1_1=params[0][0], coup_2_1=params[0][1], off=params[0][2])

plt.figure()
plt.plot(PGL, current_L, color='red', label='measured')
plt.plot(PGL, current_pred_L, color='blue', label='predict')
plt.legend(loc='best')
plt.title('LPG0')
plt.show()

plt.figure()
plt.plot(PGM, current_M, color='red', label='measured')
plt.plot(PGM, current_pred_M, color='blue', label='predict')
plt.legend(loc='best')
plt.title('MPG0')
plt.show()

np_current = np.zeros((dim_x, dim_y))

indexx = np.zeros((dim_x, dim_y))
indexy = np.zeros((dim_x, dim_y))

for x in range(dim_x):
    for y in range(dim_y):
        m_current = raw_data['I QPC'][x, y]
        p_current = QPC(raw_data['LPG0'][x, y], raw_data['MPG0'][x, y], coup_1_1=params[0][0], coup_2_1=params[0][1], off=params[0][2])
        np_current[x, y] = (p_current - m_current) + 1e-10
        indexx[x, y] = x
        indexy[x, y] = y

med = np.median(np_current)
vmin = med - 1e-10
vmax = med + 5e-10

plt.figure()
plt.pcolormesh(raw_data['LPG0'][:, 0], raw_data['MPG0'][0, :], np_current, vmin=vmin, vmax=vmax)
plt.savefig('processed_meas_signal.jpg')
plt.show()

latexify(fig_width=7.4, fig_height=7.4)

# remove outliers
upper_boundary = np.quantile(np_current, 0.999)
lower_boundary = np.quantile(np_current, 0.001)
selection = np_current[(np_current > lower_boundary) & (np_current < upper_boundary)]

# defome min and max values for the histogram
min_value = 1e9 * (lower_boundary - 1e-10)
max_value = 1e9 * (upper_boundary - 1e-10)

# define bins for the histogram
bins = np.linspace(min_value, max_value, 250)
x_data = 1e9*(selection.flat[:] - 1e-10)

# create histogram
plt.figure()
plt.hist(x_data, bins=bins)
plt.xlabel(r'$\Delta I_{\mathrm{QPC}} \ \mathrm{[nA]}$')
plt.ylabel(r'$\mathrm{Counts}$')
plt.tight_layout()
plt.savefig('delta_I_QPC_hist.pdf')

latexify()
np_current[abs(np_current) > threshold] = 1
###################
# Create Clusters #
###################
n = 5

features = 1e10 * np_current.flatten()[::n]
spatial_features = np.zeros((len(features), 2))
spatial_features[:, 0] = indexx.flatten()[::n]
spatial_features[:, 1] = indexy.flatten()[::n]

all_features = np.array([features, spatial_features[:, 0], spatial_features[:, 1]])
all_features = np.transpose(all_features)
features = features.reshape(-1, 1)

# make clusters based on current only
clusterer = GaussianMixture(13, reg_covar=1e-9)
I_labels = clusterer.fit_predict(features)

# rescale data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(all_features)
# make clusters including spatial information, use clusters from current for initialization
means = np.ones((len(clusterer.means_), 3)) * 50
means[:, 0] = clusterer.means_[:, 0]
means = scaler.transform(means)
clusterer_3d = GaussianMixture(13, reg_covar=1e-9, covariance_type='full', means_init=means)
labels = clusterer_3d.fit_predict(scaled_features)
print(clusterer_3d.means_)

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange', 'brown', 'red'])


def plot_results(X, Y_, means, covariances, index, title):
    plt.figure()
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):

        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.plot(X[Y_ == i], np.zeros((len(X[Y_ == i]), )), 'o', markersize=1, color=color)
    plt.title(title)
    plt.show()

#plot_results(features, clusterer.predict(features), clusterer.means_, clusterer.covariances_, 0,
#             'Gaussian Mixture')

flattened = 1e10 * np_current.flatten()
to_predict = scaler.transform(np.transpose(np.array([flattened, indexx.flatten(), indexy.flatten()])))
flattened = flattened.reshape(-1,1)

classes_1d = clusterer.predict(flattened)
classes_1d = classes_1d.reshape(np.shape(np_current))

classes_3d = clusterer_3d.predict(to_predict)
probabs = np.max(clusterer_3d.predict_proba(to_predict), axis=1)

classes_3d = classes_3d.reshape(np.shape(np_current))
probabs = probabs.reshape(np.shape(np_current))

plotter = -5 * np.ones(np.shape(classes_3d))

confidence = 0
plotter[probabs > confidence] = classes_3d[probabs > confidence]

fraction = np.count_nonzero(probabs > confidence) / len(classes_3d)**2

# plot 1d clusters
plt.figure()
plt.pcolormesh(raw_data['LPG0'][:, 0], raw_data['MPG0'][0, :], classes_1d, linewidth=0, rasterized=True)
plt.xlabel(r'$\mathrm{LPG0 \ [V]}$')
plt.ylabel(r'$\mathrm{MPG0 \ [V]}$')
plt.tight_layout()
plt.show()
plt.show()

# plot 3d clusters
plt.figure()
plt.pcolormesh(raw_data['LPG0'][:, 0], raw_data['MPG0'][0, :], plotter, linewidth=0, rasterized=True)
plt.savefig('clusters.jpg')
plt.show()

print('Fraction: {}'.format(fraction))
