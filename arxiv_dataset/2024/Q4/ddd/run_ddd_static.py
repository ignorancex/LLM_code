#############################################################################################
# Driver script for Delaunay Density Diagnositic with static data set
#   as described in the paper
#   Algorithm XXXX: The Delaunay Density Diagnostic
#       under review at ACM Transactions on Mathematical Software
#       (original title: Data-driven geometric scale detection via Delaunay interpolation)
#   by Andrew Gillette and Eugene Kur
#   Version 2.0, Jan 2024
#
# This script does the following:
#   1) Run multiple trials of delaunay_density_diagnostic.py for a static data set 
#       (xyz coordinates of topography data as described in the paper).
#           Output consists of one file per trial of the form zz*seed*.csv
#   2) Save list of all output files into a txt file called allfiles.multi.
#   3) Call generate_ddd_figures.py on allfiles.multi, which does the following: 
#           Generate figure showing MSD and grad-MSD rate as a function 
#               of average sample spacing.  
#           Output figure is displayed and then saved as ddd-figure-static.png
#############################################################################################

import subprocess
import os
import numpy as np

if any(fname.endswith('.csv') for fname in os.listdir()):
    print("==> Current working directory is", os.getcwd())
    print("==> Remove all .csv files from current working directory, then run again.")
    exit()

if any(fname.endswith('allfiles.multi') for fname in os.listdir()):
    print("==> Current working directory is", os.getcwd())
    print("==> Remove allfiles.multi from current working directory, then run again.")
    exit()

jobid = 123456
numtrials = 100

staticdatapath = 'staticdata/topog_ex_10k.npy'

for seed in range(numtrials):
    print("\n ==> Starting ddd for static data =",staticdatapath, " seed=", seed+1, "\n")
    subprocess.run(["python", "delaunay_density_diagnostic.py", "--jobid", str(jobid),
                    "--staticdatapath", staticdatapath, 
                    "--numrates", str(3),
                    "--minpctile", str(10),
                    # "--numtestperdim", str(10),
                    "--seed", str(seed+1)
                    ])
                    # "--logbase", str(1.2)])
#

allfiles = []
for x in os.listdir():
    if x.endswith(".csv"):
        allfiles.append(str(x))

n_fnames = ["{}\n".format(i) for i in allfiles]
with open('allfiles.multi', 'w') as fp:
    fp.writelines(n_fnames)

subprocess.run(["python", "generate_ddd_figures.py", "--infile", "allfiles.multi", "--outfile", "ddd-figure-static.png", "--mindens", "0"])


## Alternate modality example:
##  Generate 10k samples of paraboloid in 2D on [-125,125]^2
##  python delaunay_density_diagnostic.py --save2static --numtrainpts 10000 --zoomexp 2 --dim 2 --fn paraboloid
