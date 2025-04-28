import numpy as np
import pandas as pd

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ship', type=int, help="short or ship?", default=0)
parser.add_argument('--smear', type=int, help="small or large?", default=0)
parser.add_argument('--ifile', type=int, help="train, bkg or sig file", default=0)
parser.add_argument('--mfixed', type=float, help="fixed mass for signal", default=0)

args = parser.parse_args()

iSHIP = args.ship
smear = args.smear
ifile = args.ifile
mfixed = args.mfixed

if iSHIP==0:
    subfolder_lab="short/"
elif iSHIP:
    subfolder_lab="ship/"
    
if smear ==0:
    sigs, labsm = [0.001,0.01,0.005, 0.005 ], "small" # small smearing case
elif smear == 1:
    sigs, labsm = [0.001,0.05,0.01,  0.01  ], "large"


ncols = ifile
if ifile ==0:
    ncols =2
elif ifile == 1:
    ncols =2

dfs = []
if mfixed:
    for icol in range(ncols):
        dfs.append(pd.read_csv(subfolder_lab+"post_sig_"+str(ifile)+"_"+labsm+"_m_"+str(mfixed)+"_"+str(icol)+".csv"))
else:
    for icol in range(ncols):
        dfs.append(pd.read_csv(subfolder_lab+"post_prior_"+str(ifile)+"_"+labsm+"_"+str(icol)+".csv"))

dftot = pd.concat(dfs, axis=1)

if mfixed:
    dftot.to_csv(subfolder_lab+"post_sig_"+str(ifile)+"_"+labsm+"_m_"+str(mfixed)+".csv", index=False)
else:
    if ((ifile ==0) or (ifile ==1)):
        dftot.to_csv(subfolder_lab+"post_train_"+str(ifile)+"_"+labsm+".csv", index=False)
    else:
        dftot.to_csv(subfolder_lab+"post_bkg_"+str(ifile)+"_"+labsm+".csv", index=False)

