#!/usr/bin/env python3

import numpy as np

# Values used in the paper
use_minX = np.array([-7.9118004, 0., -9.394201, 0., -3.9944992, 0., -4.2058992, 0., -2.851099, 0., -6.1702003, 0., -4.963501, 0., -6.359, 0., -5.72029], dtype=np.float32)
use_maxX = np.array([5.9019985, 0.5281896, 5.8084, 0.46895373, 2.9131012, 0.52544963, 3.900301, 0.45075417, 3.905901, 0.5185917, 4.9472, 0.4172655, 6.077201, 0.5891852, 7.9728994, 0.46186885, 3.2700593], dtype=np.float32)

# Default output npy filename
outfn = "infer.npy"

# the 17 columns as input features
# column  4: g-r in mean PSF AB magnitude
# column  5: uncertainty of the column 4
# column  6: g-r in mean Kron AB magnitude
# column  7: uncertainty of the column 6
# column  8: r-i in mean PSF AB magnitude
# column  9: uncertainty of the column 8
# column 10: r-i in mean Kron AB magnitude
# column 11: uncertainty of the column 10
# column 12: i-z in mean PSF AB magnitude
# column 13: uncertainty of the column 12
# column 14: i-z in mean Kron AB magnitude
# column 15: uncertainty of the column 14
# column 16: z-y in mean PSF AB magnitude
# column 17: uncertainty of the column 16
# column 18: z-y in mean Kron AB magnitude
# column 19: uncertainty of the column 18
# column 20: E(B-V)
phot_data = np.genfromtxt("example_inference_data.csv", delimiter=",", 
dtype=np.float32, usecols=range(3,20))
# the second column = spectroscopic redshift which is not required for inference.
# For inference, simply put some number.
zspec = np.genfromtxt("example_inference_data.csv", delimiter=",", 
dtype=np.float32, usecols=(1))
# the third column = uncertainty of spectroscopic redshift which is not required for inference.
# For inference, simply put some number.
zerr = np.genfromtxt("example_inference_data.csv", delimiter=",", 
dtype=np.float32, usecols=(2))

X = phot_data
X[:,-1] = np.log(X[:,-1])

labels = np.zeros(len(zspec))
Y = np.vstack((labels, zspec, zerr)).astype(np.float32).T

normedX = np.zeros(X.shape)
n_features = X.shape[1]
for feature_ind in range(0, n_features):
    normedX[:,feature_ind] = (X[:,feature_ind]-use_minX[feature_ind])/(use_maxX[feature_ind]-use_minX[feature_ind])*2.-1.
normed = np.hstack((Y, normedX.astype(np.float32)))
print(normed.shape)

np.save(outfn, normed)
