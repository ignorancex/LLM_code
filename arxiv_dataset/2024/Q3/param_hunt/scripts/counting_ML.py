# this exports the TS distribution for the ML approaches
# but works on Cij, so no need to call the models
# can be done purely on (parallelized) CPU
import numpy as np
import os

import sys
import pickle
import argparse

sys.path.append('./../packages')
data_path = "../data/"
post_path = "../data/"
from hunt import  extract_counts, nbkgmax, nsigmax

from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()

np.seterr(invalid='ignore') # remove warning divide by 0

parser = argparse.ArgumentParser()
parser.add_argument('--post', type=int, help="Use posteriors", default=0)
parser.add_argument('--nsteps', type=int, help="number of nthresh", default=5)
parser.add_argument('--mfixed', type=float, help="mass of the signal", default=1.)
parser.add_argument('--smear', type=int, help="small or large smearing", default=0)


args = parser.parse_args()

post = args.post
N_steps= args.nsteps
mfixed = args.mfixed
smear = args.smear

if smear ==0:
    sigs, labsm = [0.001,0.01,0.005, 0.005 ], "small" # small smearing case
elif smear == 1:
    sigs, labsm = [0.001,0.05,0.01,  0.01  ], "large"

# import the already computed Cij
labpost="_"
if post:
    labpost = "_post_"
CijSS = np.load("../performances/Cijs/CijSS"+labpost+str(mfixed)+"_"+labsm+".npy")
CijBS = np.load("../performances/Cijs/CijBS"+labpost+str(mfixed)+"_"+labsm+".npy")
CijBB = np.load("../performances/Cijs/CijBB"+labpost+labsm+".npy")

# use routine to extract TS from Cij given a certain threshold
def processInput(nbkg, nsig, thresh):
    nobs = nbkg + nsig
    if nobs < 2:
        countsS = np.array([nobs,10000]).reshape(-1,1)
        return([nbkg, nsig, thresh, countsS])
    else:
        countsS =  extract_counts(CijSS, CijBS, CijBB, nbkg, nsig, thresh)
        return([nbkg, nsig, thresh, countsS])
perfs = (Parallel(n_jobs=num_cores)(delayed(processInput)(nbkg, nsig, thresh) for nbkg in range(nbkgmax+1) for nsig in range(nsigmax+1) for thresh in np.linspace(0.04, 0.96, N_steps)))

with open("../performances/counts/count_cl"+labpost+labsm+"_m_"+str(mfixed)+".pickle", "wb") as fp:  
    pickle.dump(perfs, fp)
