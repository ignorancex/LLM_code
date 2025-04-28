#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Load data functions

"""
import pickle
import numpy as np
import torch

# 1k dataset -> 'data/data_norm_sdss16_SNR10_random_1.fits'
# 20k dataset -> 'data/data_HST_1220_5000_2A.pickle'

#hdu = fits.open('data/data_norm_sdss16_SNR10_all.fits')  
#hdu = fits.open('data/data_norm_sdss16_SNR10_random_1.fits')

# -------------------------------------------------------------------------------
# initialize parameters
# -------------------------------------------------------------------------------

def save_spectra_labels_small():
    
    f = open('data/data_HST_1220_5000_2A.pickle', 'rb') 
    data, data_ivar = pickle.load(f)
    f.close()

    for qq in range(data.shape[0]): 
     
        # remove one object for test set, rest as training set
        ind_test = np.zeros(data.shape[0], dtype = bool)
        ind_test[qq] = True
        ind_train = ~ind_test
    
        # -------------------------------------------------------------------------------
        # re-scale all input data to be Gaussian with zero mean and unit variance
        # -------------------------------------------------------------------------------
        
        qs = np.nanpercentile(data[ind_train, :], (2.5, 50, 97.5), axis=0)
        
        pivots = qs[1]
        scales = (qs[2] - qs[0]) / 4.
        scales[scales == 0] = 1.
        data_scaled = (data - pivots) / scales
        #data_ivar_scaled = data_ivar * scales**2 
        
        # -------------------------------------------------------------------------------
        # prepare rectangular training data
        # -------------------------------------------------------------------------------
        
        # spectra
        X = data_scaled[:, 7:] 
        #X_var = 1 / data_ivar_scaled[:, 7:]
        
        # labels
        inds_label = np.zeros(data_scaled.shape[1], dtype = bool)git@github.com:vr308/QuasarGPLVM.git
        inds_label[0] = True # black hole mass
        inds_label[1] = True # redshift
        inds_label[6] = True # Lbol
        Y = data_scaled[:, inds_label] 
        #Y_var = (1 / data_ivar_scaled[:, inds_label])
        
        data_scaled = np.hstack((X,Y))
        return data_scaled
        
def load_spectra_labels_small():
    
       data = np.loadtxt(fname='data/small_quasar.csv', dtype=float, delimiter=',')
       return data

def load_spectra_labels(hdu):
    
    issues = hdu[4].data
    wave = hdu[0].data 
    X = hdu[1].data[issues == 0.]
    X_ivar = hdu[2].data[issues == 0.]
    masks = hdu[3].data[issues == 0.]
    Y = hdu[5].data[issues == 0.]
    Y_ivar = hdu[6].data[issues == 0.]
    snr = hdu[7].data[issues == 0.]
    
    # set missing values to NaN
    X[masks == 0.] = np.nan
    X_ivar[masks == 0.] = np.nan
    
    if X.shape[0] > 1000:
        # slice at wave = 1216
        X = X[:,wave > 1216]
        wave = wave[wave > 1216]
    
    # remove redshift feature - column 0 
    #Y = Y[:,1:]
    
    ## The Y_ivar column for bhm (3rd column, index 2) is not inverted - fixing that below
    Y_ivar[:,2] = 1/Y_ivar[:,2]

    means_X = np.nanmean(X, axis = 0)
    means_Y = np.nanmean(Y, axis = 0)
    std_X = np.nanstd(X, axis = 0)
    std_Y = np.nanstd(Y, axis = 0)
    
    X = (X - means_X) / std_X
    Y = (Y - means_Y) / std_Y
    
    return X, Y, means_X, std_X, means_Y, std_Y, X_ivar, Y_ivar, snr, wave

def load_synthetic_labels(Y_test, Y_test_orig, means_Y, std_Y, Y_ivar, device):
    
    norm_means = torch.nanmean(Y_test, dim=0)
    Y_synthetic = norm_means.repeat(300,1)

    ## simulate data in real unit ranges 
    
    synthetic_bhm = torch.linspace(8.5,10,100).to(device)
    synthetic_lumin = torch.linspace(46.25,48,100).to(device)
    synthetic_edd = torch.linspace(-1.5,0.5,100).to(device)

    ## plug in the synthetic range data in the respective columns 
    
    Y_synthetic[0:100, 1] = (synthetic_lumin - means_Y[1])/std_Y[1]
    Y_synthetic_lumin = Y_synthetic[0:100]
    
    Y_synthetic[100:200, 2] = (synthetic_bhm - means_Y[2])/std_Y[2]
    Y_synthetic_bhm = Y_synthetic[100:200]
    
    Y_synthetic[200:300, 3] = (synthetic_edd - means_Y[3])/std_Y[3]
    Y_synthetic_edd = Y_synthetic[200:300]
    
    # extract X & Y measurement uncertainty 
    
    Y_sigma = np.sqrt(1/Y_ivar)
    
    #Y_sigma[:,2] = np.sqrt(Y_ivar[:,2])
    
    return Y_synthetic_lumin, Y_synthetic_bhm, Y_synthetic_edd, Y_sigma

