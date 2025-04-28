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

for smear in [0, 1]:
    if smear ==0:
        sigs, labsm = [0.001,0.01,0.005, 0.005 ], "small" # small smearing case
    elif smear == 1:
        sigs, labsm = [0.001,0.05,0.01,  0.01  ], "large"

    bkgfile = post_path +  "post_bkg_13_"+labsm+".csv"


    dirfolder = "../models/model_post_01_"+labsm+"_0" 
    model = keras.models.load_model(dirfolder+"/model.tf")
    with open(dirfolder+'/meta_par.pkl', 'rb') as file_t:
        meta_par = pickle.load(file_t)

    np.random.seed(42)
    feats=post_extract(bkgfile)
    XB= feats.extract_post(nbkgmax)
    XB = XB.reshape(10000,nbkgmax,200)

    # apply EPO hunt on pairs of bkg events
    Cij = []
    for i in range(nbkgmax):
        for j in range(i+1, nbkgmax):
            Xpair = np.hstack((XB[:,i,:].reshape(-2,1,200),XB[:,j,:].reshape(-2,1,200)))
            Xpair= reg_log(Xpair, meta_par["pref"])
            Cij_temp = np.array(model(Xpair)).flatten()
            Cij_temp = np.hstack((np.array([i,j]),Cij_temp))
            Cij.append(Cij_temp)
    CijBB = np.array(Cij)

    np.save("../performances/Cijs/CijBB_post_"+labsm+".npy", CijBB)

    # apply EPO hunt on pairs of bkg events
    for mfixed in [0.2, 1., 4.]:
        sigfile = post_path +  "post_sig_8_"+labsm+"_m_"+str(mfixed)+".csv"
        np.random.seed(42)
        feats=post_extract(sigfile)
        XS= feats.extract_post(nsigmax)
        XS = XS.reshape(10000,nsigmax,200)

        Cij = []
        for i in range(nsigmax):
            for j in range(i+1, nsigmax):
                Xpair = np.hstack((XS[:,i,:].reshape(-2,1,200),XS[:,j,:].reshape(-2,1,200)))
                Xpair= reg_log(Xpair, meta_par["pref"])
                Cij_temp = np.array(model(Xpair)).flatten()
                Cij_temp = np.hstack((np.array([i,j]),Cij_temp))
                Cij.append(Cij_temp)
        CijSS = np.array(Cij)
        np.save("../performances/Cijs/CijSS_post_"+str(mfixed)+"_"+labsm+".npy", CijSS)


        Cij = []
        for i in range(nbkgmax):
            for j in range(nsigmax):
                Xpair = np.hstack((XB[:,i,:].reshape(-2,1,200),XS[:,j,:].reshape(-2,1,200)))
                Xpair= reg_log(Xpair, meta_par["pref"])
                Cij_temp = np.array(model(Xpair)).flatten()
                Cij_temp = np.hstack((np.array([i,j]),Cij_temp))
                Cij.append(Cij_temp)
        CijBS = np.array(Cij)
        np.save("../performances/Cijs/CijBS_post_"+str(mfixed)+"_"+labsm+".npy", CijBS)


