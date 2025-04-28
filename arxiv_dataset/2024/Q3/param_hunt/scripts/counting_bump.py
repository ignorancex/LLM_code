# returns the TS distribution for bumphunt
import numpy as np
import os


import sys
import pickle
import argparse

sys.path.append('./../packages')
data_path = "../data/"
post_path = "../data/"
from feat_extractor import feature_extract
from hunt import  extract_counts, nbkgmax, nsigmax, mgg, bphunt

from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()

np.seterr(invalid='ignore') # remove warning divide by 0

def reg_log(x, pref):
    y=np.copy(x)
    y[y<pref]=pref
    y = (y-pref)/(np.max(y, axis=(1,2)).reshape(-1,1,1)-pref)
    return y



parser = argparse.ArgumentParser()
parser.add_argument('--nbins', type=int, help="number of nthresh", default=5)
parser.add_argument('--mfixed', type=float, help="mass of the signal", default=1.)
parser.add_argument('--smear', type=int, help="small or large smearing", default=0)


args = parser.parse_args()

nbin_max= args.nbins
mfixed = args.mfixed
smear = args.smear


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


perfs = []
sigfile = data_path + "event_sig_8_m_"+str(mfixed)+"_"+str(mfixed)+"_tm_1.0_1.0_"+geo_lab+".csv"
bkgfile = data_path + "event_bkg_13_"+par_lab+geo_lab+".csv"

# we define the single step process which can be parallelized
def processInput(nbkg, nsig, nbin):
    nobs = nbkg + nsig
    if nobs < 2:
        countsS = np.array([nobs,10000]).reshape(-1,1)
        return([nbkg, nsig, nbin, countsS])
    else:
        if nbkg:
            np.random.seed(42)
            feats=feature_extract(bkgfile, sigs[0], sigs[1], sigs[2], sigs[3], Eres=1,  x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
            features = np.vstack(([feats.extract_llo(iOBS) for iOBS in range(nbkg)]))  
            x0 = features.T

        if nsig:
            np.random.seed(42)
            feats=feature_extract(sigfile, sigs[0], sigs[1], sigs[2], sigs[3], Eres=1,  x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
            features = np.vstack(([feats.extract_llo(iOBS) for iOBS in range(nsig)]))  
            x1 = features.T

        if nbkg*nsig:
            XS = np.hstack((x0,x1))
            del x0, x1, feats, features
        if ((nsig==0) and (nbkg>1)):
            XS = x0
            del x0, feats, features
        if ((nbkg==0) and (nsig>1)):
            XS = x1
            del x1, feats, features

        # compute the invariant mass from the features
        mggS = mgg(XS)

        # bin and count
        countsS = bphunt(np.log10(mggS), nbin, [-1.5, 0.8]).over(3)
        return([nbkg, nsig, nbin, countsS])
perfs = (Parallel(n_jobs=num_cores)(delayed(processInput)(nbkg, nsig, nbin) for nbkg in range(nbkgmax+1) for nsig in range(nsigmax+1) for nbin in range(4,nbin_max)))
        
with open("../performances/counts/count_bump_"+labsm+"_m_"+str(mfixed)+".pickle", "wb") as fp:  
    pickle.dump(perfs, fp)

