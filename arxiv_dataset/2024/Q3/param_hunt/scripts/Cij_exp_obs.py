# script to export the Cij for ECO hunt
import numpy as np
import random

import tensorflow as tf
from tensorflow import keras

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import os
import scipy.stats as sts
import pandas as pd

for gpu in tf.config.experimental.list_physical_devices('GPU'):
      tf.config.experimental.set_memory_growth(gpu, True)

import sys
import pickle
from matplotlib import pyplot as plt
sys.path.append('./../packages')
data_path = "../data/"
post_path = "../data/"
from feat_extractor import feature_extract, post_extract 
from architecture import class_model, class_model_cnn

def reg_log(x, pref):
    y=np.copy(x)
    y[y<pref]=pref
    y = (y-pref)/(np.max(y, axis=(1,2)).reshape(-1,1,1)-pref)
    return y

# just define the geometry
nsigmax = 8
nbkgmax = 13

malp_min, malp_max = 0.1, 4.5
tmalp_min, tmalp_max = 0.05, 500 # SHiP
z_min, z_max, z_cal, l_x, l_y= 32, 82, 93, 2, 3 # SHiP

x_min, x_max = -l_x, l_x
y_min, y_max = -l_y, l_y

par_lab = "m_"+str(malp_min)+"_"+str(malp_max)+"_tm_"+str(tmalp_min)+"_"+str(tmalp_max)+"_"
geo_lab = "c_"+str(z_min)+"_"+str(z_max)+"_"+str(l_x)+"_"+str(l_y)

bkgfile = data_path + "event_bkg_13_"+par_lab+geo_lab+".csv"


for smear in [0, 1]:
    if smear ==0:
        sigs, labsm = [0.001,0.01,0.005, 0.005 ], "small" # small smearing case
    elif smear == 1:
        sigs, labsm = [0.001,0.05,0.01,  0.01  ], "large"

    dirfolder = "../models/model_01_"+labsm+"_0"
    model = keras.models.load_model(dirfolder+"/model.tf")
    xzscaler = pickle.load(open(dirfolder+'/xzscaler.pkl', 'rb'))

    np.random.seed(42)
    feats=feature_extract(bkgfile, sigs[0], sigs[1], sigs[2], sigs[3], Eres=1,  x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
    features = np.vstack(([feats.extract_llo(iOBS) for iOBS in range(nbkgmax)]))  
    XB = features.T

    # apply ECO hunt on pairs of bkg events
    Cij = []
    for i in range(nbkgmax):
        for j in range(i+1, nbkgmax):
            Xpair = np.hstack((XB[:, int(10*i):int(10*(i+1))], XB[:, int(10*j):int(10*(j+1))]))
            Xpair = xzscaler.transform(Xpair)
            Cij_temp = np.array(model(Xpair)).flatten()
            Cij_temp = np.hstack((np.array([i,j]),Cij_temp))
            Cij.append(Cij_temp)
    CijBB = np.array(Cij)

    np.save("../performances/Cijs/CijBB_"+labsm+".npy", CijBB)

    # apply ECO hunt on pairs of signal events and a bkg+signal pair
    for mfixed in [0.2, 1., 4.]:
        sigfile = data_path + "event_sig_8_m_"+str(mfixed)+"_"+str(mfixed)+"_tm_1.0_1.0_"+geo_lab+".csv"
        np.random.seed(42)
        feats=feature_extract(sigfile, sigs[0], sigs[1], sigs[2], sigs[3], Eres=1,  x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
        features = np.vstack(([feats.extract_llo(iOBS) for iOBS in range(nsigmax)]))  
        XS = features.T

        Cij = []
        for i in range(nsigmax):
            for j in range(i+1, nsigmax):
                Xpair = np.hstack((XS[:, int(10*i):int(10*(i+1))], XS[:, int(10*j):int(10*(j+1))]))
                Xpair = xzscaler.transform(Xpair)
                Cij_temp = np.array(model(Xpair)).flatten()
                Cij_temp = np.hstack((np.array([i,j]),Cij_temp))
                Cij.append(Cij_temp)
        CijSS = np.array(Cij)
        np.save("../performances/Cijs/CijSS_"+str(mfixed)+"_"+labsm+".npy", CijSS)


        Cij = []
        for i in range(nbkgmax):
            for j in range(nsigmax):
                Xpair = np.hstack((XB[:, int(10*i):int(10*(i+1))], XS[:, int(10*j):int(10*(j+1))]))
                Xpair = xzscaler.transform(Xpair)
                Cij_temp = np.array(model(Xpair)).flatten()
                Cij_temp = np.hstack((np.array([i,j]),Cij_temp))
                Cij.append(Cij_temp)
        CijBS = np.array(Cij)
        np.save("../performances/Cijs/CijBS_"+str(mfixed)+"_"+labsm+".npy", CijBS)


