#############################################################################################
# Driver script for Delaunay Density Diagnositic (python version)
#   as used in the paper
#   Algorithm XXXX: The Delaunay Density Diagnostic
#       under review at ACM Transactions on Mathematical Software
#       (original title: Data-driven geometric scale detection via Delaunay interpolation)
#   by Andrew Gillette and Eugene Kur
#   Version 2.0, Jan 2024
#
# This script does the following:
#   1) Run multiple trials of delaunay_density_diagnostic.py for Griewank in dim 2.
#           The outer loop adjusts the scale of inquiry through the "zoom exponent" option
#               but leaves the center of the domain as the origin.
#           The inner loop adjusts the global seed used for randomization.
#           Output consists of one file per trial of the form zz*zoom*seed*.csv
#   2) Save list of all output files into a txt file called allfiles.multi.
#   3) Call generate_ddd_figures.py on allfiles.multi, which does the following: 
#           Generate figure showing MSD and grad-MSD rate as a function 
#               of average sample spacing.  
#           Output figure is displayed and then saved as ddd-figure-griewank.png
#############################################################################################

import subprocess
import os
import numpy as np

if any(fname.endswith('.csv') for fname in os.listdir()):
    print("==> Current working directory is", os.getcwd())
    print("==> Remove all .csv files from current working directory, then run again.")
    exit()

jobid = 123456
zoomstep = 0.4
minzoom = 0.0
maxzoom = 4.0
numtrials = 10
numtestperdim = 20
zoomexps = np.linspace(minzoom,maxzoom,num=int(maxzoom/zoomstep))

for zoomexp in zoomexps:
    for seed in range(numtrials):
        print("\n ==> Starting ddd trial with zoom exponent =",zoomexp, " seed=", seed, "\n")
        subprocess.run(["python", "delaunay_density_diagnostic.py", "--jobid", str(jobid), "--seed", str(seed), "--zoomexp", str(zoomexp), "--numtestperdim", str(numtestperdim)])
    #
#

allfiles = []
for x in os.listdir():
    if x.endswith(".csv"):
        allfiles.append(str(x))

n_fnames = ["{}\n".format(i) for i in allfiles]
with open('allfiles.multi', 'w') as fp:
    fp.writelines(n_fnames)

subprocess.run(["python", "generate_ddd_figures.py", "--infile", "allfiles.multi", "--outfile", "ddd-figure-griewank.png", "--mindens", "50", "--logscalex"])
