# extract the performances
# this is used to choose the threshold
import numpy as np
import random

import os
import scipy.stats as sts
import pandas as pd

import sys
import pickle
sys.path.append('./../packages')
data_path = "../data/"
post_path = "../data/"
from hunt import perf_pval, count_from_df

from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mfixed', type=float, help="mass of the signal", default=1.)
parser.add_argument('--smear', type=int, help="small or large smearing", default=0)


args = parser.parse_args()

Nsamples = 250000
mfixed = args.mfixed
smear = args.smear

if smear ==0:
    sigs, labsm = [0.001,0.01,0.005, 0.005 ], "small" # small smearing case
elif smear == 1:
    sigs, labsm = [0.001,0.05,0.01,  0.01  ], "large"

# we select some levels of background and signal normalization
mubspace = np.linspace(0.5, 6, 12)
musspace = np.linspace(0.5, 4, 8)


# we parallelize the process over the normalizations
def TS_par(dfc_opt, mub,mus):
    nbkgsig = np.array([sts.poisson(mub).rvs(Nsamples), sts.poisson(mus).rvs(Nsamples)])
    nnurbkg = np.array([sts.poisson(mub).rvs(Nsamples), np.zeros(Nsamples)])

    countB, countS = count_from_df(dfc_opt, nnurbkg, nbkgsig)

    return([mub, mus, perf_pval(countB, countS).meanp(), perf_pval(countB, countS).meanlogp(), perf_pval(countB, countS).meansigma()])

# we import the counts and extract the performances (for ECO)
perf_ECO = []
with open("../performances/counts/count_cl_"+labsm+"_m_"+str(mfixed)+".pickle", "rb") as fp:  
    countsS = pickle.load(fp)
dfc = pd.DataFrame(countsS, columns=["nbkg", "nsig", "thresh", "counts"])
for opt_thro in np.linspace(0.04,0.96, 24):
    dfc_opt = dfc.loc[round(dfc["thresh"],2)==round(opt_thro,2)]
    TS_ECO = np.array((Parallel(n_jobs=num_cores)(delayed(TS_par)(dfc_opt, mub, mus) for mub in mubspace for mus in musspace )))
    perf_ECO.append(np.hstack((opt_thro*np.ones(len(TS_ECO)).reshape(-1,1),TS_ECO)))
perf_ECO = np.array(perf_ECO)
perf_ECO = perf_ECO.reshape(perf_ECO.shape[0]*perf_ECO.shape[1],perf_ECO.shape[2])

# we import the counts and extract the performances (for EPO)
perf_EPO = []
with open("../performances/counts/count_cl_post_"+labsm+"_m_"+str(mfixed)+".pickle", "rb") as fp:  
    countsS = pickle.load(fp)
dfc = pd.DataFrame(countsS, columns=["nbkg", "nsig", "thresh", "counts"])
for opt_thrp in np.linspace(0.04,0.96, 24):
    dfc_opt = dfc.loc[round(dfc["thresh"],2)==round(opt_thrp,2)]
    TS_EPO = np.array((Parallel(n_jobs=num_cores)(delayed(TS_par)(dfc_opt, mub, mus) for mub in mubspace for mus in musspace )))
    perf_EPO.append(np.hstack((opt_thrp*np.ones(len(TS_EPO)).reshape(-1,1),TS_EPO)))
perf_EPO = np.array(perf_EPO)
perf_EPO = perf_EPO.reshape(perf_EPO.shape[0]*perf_EPO.shape[1],perf_EPO.shape[2])


np.savetxt("../performances/TS/perf_full_cl_"+labsm+"_m_"+str(mfixed)+".csv", perf_ECO)
np.savetxt("../performances/TS/perf_full_cl_post_"+labsm+"_m_"+str(mfixed)+".csv", perf_EPO)
