"""
Author: Theresa Wagner <theresa.wagner@mathematik.tu-chemnitz.de>

Corresponding publication:
"A Preconditioned Interior Point Method for Support Vector Machines Using an
ANOVA-Decomposition and NFFT-Based Matrix–Vector Products"
by T. Wagner, John W. Pearson, M. Stoll (2023)

Execute this file to reproduce the results presented in Figure 2.
"""
from nfftsvmipm.func_prec_timings import main_precond_timings

##################################################################################
## READ PARSED ARGUMENTS

import argparse

# Create the argument parser
parser = argparse.ArgumentParser(description="Run preconditioner timings test with configurable parameters.")

# Add arguments
parser.add_argument('--kernel', type=int, default=1, choices=[1, 3], 
                    help="Kernel type: 1 for Gaussian, 3 for Matérn(1/2), default=1.")
parser.add_argument('--Ndata', type=int, default=[100000], 
                    help="List of subset size candidates, where 0 corresponds to the entire data set, default=[100000].")
parser.add_argument('--prec', nargs='+', type=str, default=["chol_greedy", "chol_rp", "rff", "nystrom"], choices=["chol_greedy", "chol_rp", "rff", "nystrom"],
                    help="List of preconditioner type candidates, default=['chol_greedy', 'chol_rp', 'rff', 'nystrom'].")
parser.add_argument('--rank', nargs='+', type=int, default=[50, 200, 1000], 
                    help="List of target preconditioner rank candidates, default=[50, 200, 1000].")

# Parse arguments
args = parser.parse_args()

# Assign the parsed arguments to the parameters

# kernel definition
kernel = args.kernel
# subset sizes
if isinstance(args.Ndata, int):
    Ndata = [args.Ndata]
else:
    Ndata = args.Ndata
# list of preconditioner candidates
if isinstance(args.prec, str):
    prec_list = [args.prec]
else:
    prec_list = args.prec
# list of target preconditioner rank candidates
if isinstance(args.rank, int):
    rank_list = [args.rank]
else:
    rank_list = args.rank
    
####################
## CHOOSE PARAMETERS

# choose data set
data = "cod_rna"

####################
# include nfftsvmipm folder into the path
import sys
import os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the nfftsvmipm folder (one level up from the current directory)
nfftsvmipm_dir = os.path.join(current_dir, '..', 'nfftsvmipm')

# Add the home directory to sys.path
sys.path.insert(0, nfftsvmipm_dir)

####################
# initialize dict for results
dict_cholgr = {r: [] for r in rank_list}
dict_cholrp = {r: [] for r in rank_list}
dict_rff = {r: [] for r in rank_list}
dict_nystr = {r: [] for r in rank_list}

for n in Ndata:
    print("Solving for dimension", n)
    ###########################
    # work with cod-rna dataset
    from data_SVMipm import cod_rna
            
    X_train, X_test, y_train, y_test = cod_rna(n)
    
    wind_param_list = [[[0, 2, 3], [7, 6, 5], [4, 1]], [[7.199384124614643, 5.753872018666903, 6.129828390835456], 0.3407055254938164]]
    ###########################
    
    print("\nDataset:", data)
    print("--------\nShape train data:", X_train.shape)
    print("Shape test data:", X_test.shape)
    
    ###########################
    
    for prec in prec_list:
        print("Solving for preconditioner", prec)
        
        for D_prec in rank_list:
            print("Solving for rank", D_prec)
    
            # measure precond timings
            precond_time = main_precond_timings(X_train, X_test, y_train, y_test, wind_param_list, prec, D_prec, kernel)
            
            # save values to dict
            if prec == "chol_greedy":
                dict_cholgr[D_prec].append(precond_time)
            elif prec == "chol_rp":
                dict_cholrp[D_prec].append(precond_time)
            elif prec == "rff":
                dict_rff[D_prec].append(precond_time)
            elif prec == "nystrom":
                dict_nystr[D_prec].append(precond_time)
               
            print("###############################################################")
        
            print("\nDataset:", data)
            print("--------\nShape train data:", X_train.shape)
            print("Shape test data:", X_test.shape)
            print("prec:", prec)
            print("D_prec:", D_prec)
            print("results prec time:", precond_time)
            
            print("###############################################################")
        
            print("\nResults Precond Timings:")
            print("------------------------\n")
            print("dict_cholgr:", dict_cholgr)
            print("dict_cholrp:", dict_cholrp)
            print("dict_rff:", dict_rff)
            print("dict_nystr:", dict_nystr)
