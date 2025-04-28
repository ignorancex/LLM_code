# this script uses a trained classifier to extract psoteriors
import numpy as np

import tensorflow as tf
from tensorflow import keras

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import sys
import os
import pickle
import pandas as pd

import scipy.stats as sts
from scipy.integrate import trapezoid

sys.path.append('./../packages')
data_path = "../data/"
post_path = "../data/"

for gpu in tf.config.experimental.list_physical_devices('GPU'):
      tf.config.experimental.set_memory_growth(gpu, True)
from feat_extractor import feature_extract
from architecture import class_model

np.seterr(invalid='ignore') # remove warning divide by 0


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--smear', type=int, help="small or large?", default=0)
parser.add_argument('--ifile', type=int, help="train, bkg or sig file", default=0)
parser.add_argument('--icol', type=int, help="column of event", default=0)
parser.add_argument('--Nsam', type=int, help="how many samples", default=100)
parser.add_argument('--Nlin', type=int, help="how many lin points", default=200)
parser.add_argument('--mfixed', type=float, help="fixed mass for signal", default=0)

args = parser.parse_args()

smear = args.smear
ifile = args.ifile
icol = args.icol
nlin = args.Nlin # number of grid points
Nsam = args.Nsam
mfixed = args.mfixed


if smear ==0:
    sigs, labsm = [0.001,0.01,0.005, 0.005 ], "small" # small smearing case
elif smear == 1:
    sigs, labsm = [0.001,0.05,0.01,  0.01  ], "large"



malp_min, malp_max = 0.1, 4.5
tmalp_min, tmalp_max = 0.05, 500 # SHiP
z_min, z_max, z_cal, l_x, l_y= 32, 82, 93, 2, 3 # SHiP

x_min, x_max = -l_x, l_x
y_min, y_max = -l_y, l_y

par_lab = "m_"+str(malp_min)+"_"+str(malp_max)+"_tm_"+str(tmalp_min)+"_"+str(tmalp_max)+"_"
geo_lab = "c_"+str(z_min)+"_"+str(z_max)+"_"+str(l_x)+"_"+str(l_y)

def prior_m():
    return sts.uniform(loc=np.log10(malp_min), scale= np.log10(malp_max/malp_min))

def prior_tm():
    return sts.uniform(loc=np.log10(tmalp_min), scale= np.log10(tmalp_max/tmalp_min))

# load model
dirfolder = "../models/model_post_"+labsm+"_0"
model = keras.models.load_model(dirfolder+"/model.tf", compile=False)
# we pick the two training files which will be converted in posteriors

#from which file to extract the posterior
if ifile == 1:
    trainfile = os.path.expanduser('~')+"/Desktop/Gitlab/classbump/data/event_train_11_"+par_lab+geo_lab+".csv"
elif ifile == 0:
    trainfile = os.path.expanduser('~')+"/Desktop/Gitlab/classbump/data/event_train_00_"+par_lab+geo_lab+".csv"
elif ifile == 8:
    trainfile = os.path.expanduser('~')+"/Desktop/Gitlab/classbump/data/event_sig_8_m_"+str(mfixed)+"_"+str(mfixed)+"_tm_1.0_1.0_"+geo_lab+".csv"
elif ifile == 13:
    trainfile = os.path.expanduser('~')+"/Desktop/Gitlab/classbump/data/event_bkg_13_"+par_lab+geo_lab+".csv"


feats=feature_extract(trainfile, sigs[0], sigs[1], sigs[2], sigs[3], Eres=1,  x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
x0 = feats.extract_llo(icol).T
if ifile == 8:
    malp, tmalp = feats.extract_model(0) # signal model parameters are the same in the whole file
else:
    malp, tmalp = feats.extract_model(icol)

xzscaler = pickle.load(open(dirfolder+'/xzscaler.pkl', 'rb'))
mrange = np.linspace(np.log10(malp_min), np.log10(malp_max), nlin) # no point in extending beyond the range due to the prior


# assuming scaler and x0 have been already imported
# extract posterior from X (including normalization)
def postvals(mass_range, model, iS):
    X = np.hstack((mass_range.reshape(-1,1), np.array([x0[iS],]*len(mass_range))))
    X = xzscaler.transform(X)
    score = np.array(model(X))
    LER = score/(1-score)
    postvals = LER*(prior_m().pdf(mass_range).reshape(-1,1))
    norm = trapezoid(postvals.T, mass_range.reshape(1,-1))
    return postvals/norm

# scan over elements (not vectorialized or parallelized)
posts = []
for iS in range(Nsam):
    posts.append(np.append(np.array([malp[iS], tmalp[iS]]), np.log(postvals(mrange, model, iS))))

# will contain -inf, can be regularized later
df = pd.DataFrame(np.array(posts))
if mfixed:
    df.to_csv(post_path+"post_sig_"+str(ifile)+"_"+labsm+"_m_"+str(mfixed)+"_"+str(icol)+".csv", index=False)
else:
    df.to_csv(post_path+"post_prior_"+str(ifile)+"_"+labsm+"_"+str(icol)+".csv", index=False)
