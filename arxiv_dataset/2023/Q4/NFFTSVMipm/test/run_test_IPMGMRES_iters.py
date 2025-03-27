"""
Author: Theresa Wagner <theresa.wagner@mathematik.tu-chemnitz.de>

Corresponding publication:
"A Preconditioned Interior Point Method for Support Vector Machines Using an
ANOVA-Decomposition and NFFT-Based Matrix–Vector Products"
by T. Wagner, John W. Pearson, M. Stoll (2023)

Execute this file to reproduce the results presented in Figure 3.
"""

import numpy as np

from nfftsvmipm.class_NFFTSVMipm import RandomSearch

##################################################################################
## READ PARSED ARGUMENTS

import argparse

# Create the argument parser
parser = argparse.ArgumentParser(description="Run test on IPM/GMRES iterations with configurable parameters.")

# Add arguments
parser.add_argument('--kernel', type=int, default=1, choices=[1, 3], 
                    help="Kernel type: 1 for Gaussian, 3 for Matérn(1/2), default=1.")
parser.add_argument('--Ndata', nargs='+', type=int, default=[0], 
                    help="List of subset sizes to consider, where 0 corresponds to the entire data set, default=[0].")
parser.add_argument('--prec', nargs='+', type=str, default=["chol_greedy", "chol_rp", "rff", "nystrom"], choices=["chol_greedy", "chol_rp", "rff", "nystrom"],
                    help="List of preconditioner types candidates, default=['chol_greedy', 'chol_rp', 'rff', 'nystrom'].")
parser.add_argument('--rank', nargs='+', type=int, default=[50, 200, 1000], 
                    help="List of target preconditioner ranks candidates, default=[50, 200, 1000].")
parser.add_argument('--S', nargs='+', type=float, default=[1e-2, 1e-1, 1, 1e+1, 1e+2], 
                    help="List of sigma/length-scale candidates, default=[1e-2, 1e-1, 1, 1e+1, 1e+2].")
parser.add_argument('--C', type=float, default=0.4, 
                    help="Regularization parameter, default=0.4.")
parser.add_argument('--mis_thres', type=float, default=0.0, 
                    help="Mutual information score threshold, default=0.0.")
parser.add_argument('--window_scheme', type=str, default="mis", choices=["mis", "consec", "random"],
                    help="Window scheme: 'mis', 'consec', or 'random', default='mis'.")
parser.add_argument('--weight_scheme', type=str, default="equally weighted", choices=["equally weighted", "no weights"], 
                    help="Weight scheme: 'equally weighted' or 'no weights'.")
parser.add_argument('--IPMiter', type=int, default=100, 
                    help="Maximum number of IPM iterations, default=100.")
parser.add_argument('--GMRESiter', type=int, default=100, 
                    help="Maximum number of GMRES iterations, default=100.")
parser.add_argument('--ipmpar', nargs=3, type=float, default=[0.2, 1e-3, 1e-6], 
                    help="IPM parameters as a list: [sigma_br, tol, Gtol], default=[0.2, 1e-3, 1e-6].")
parser.add_argument('--dratio', type=float, default=1.0, 
                    help="Propoertion of features to include, default=1.0.")
parser.add_argument('--data', type=str, default="cod_rna", choices=["susy", "cod_rna", "higgs"], 
                    help="Data sets to use, default='cod_rna'.")

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
# list of length-scale candidates
if isinstance(args.S, int):
    Slist = [args.S]
else:
    Slist = args.S
# regularization parameter
C = args.C
# mutual information score threshold
mis_thres = args.mis_thres
# window scheme
window_scheme = args.window_scheme
# weight scheme
weight_scheme = args.weight_scheme
# maximum number of IPM iterations
iter_ip = args.IPMiter
# maximum number of GMRES iterations
Gmaxiter = args.GMRESiter
# ipm parameters: ipmpar=[sigma_br,tol,Gtol]
ipmpar = args.ipmpar
# dratio
dr = args.dratio
# data set
data = args.data

####################
## CHOOSE PARAMETERS

# set number of iterations in RandomSearch
iRS = 1 # only one run since parameters are fixed

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

####################
print("\n####################################################\n")
print("############################")
print("Solving for data = ", data)
print("############################")
  
for n in Ndata:
    print("\n####################################################\n")
    print("############################")
    print("Solving for n = ", n)
    print("############################")

    if data == "higgs":
        from data_SVMipm import higgs
            
        X_train, X_test, y_train, y_test = higgs(n)
		
    elif data == "susy":
        from data_SVMipm import susy
		
        X_train, X_test, y_train, y_test = susy(n)
		
    elif data == "cod_rna":
        from data_SVMipm import cod_rna
		
        X_train, X_test, y_train, y_test = cod_rna(n)
		
    print("\nDataset:", data)
    print("--------\nShape train data:", X_train.shape)
    print("Shape test data:", X_test.shape)
    
    for prec in prec_list:
        print("Solving for preconditioner", prec)
        
        for D_prec in rank_list:
            print("Solving for rank", D_prec)
    
            acc = []
            ipmiters = []
            gmresiters = []
    
            for S in Slist:
                print("\n####################################################\n")
                print("############################")
                print("Solving for sigma = ", S)
                print("############################")
	    
                ##############################
    	        ## RandomSearch for NFFTSVMipm

    	        # define bounds for individual sigmas for every window
                lb_sigma = S
                ub_sigma = S

    	        # define bounds for C
                lb_C = C
                ub_C = C

    	        # setup RandomSearch model for SVMipm classifier
                model = RandomSearch(classifier="NFFTSVMipm", kernel=kernel, lb=[lb_sigma,lb_C], ub=[ub_sigma, ub_C], max_iter_rs=iRS, mis_threshold=mis_thres, window_scheme=window_scheme, d_ratio=dr, weight_scheme=weight_scheme, sigma_br=ipmpar[0], D_prec=D_prec, prec=prec, iter_ip=iter_ip, tol=ipmpar[1], Gmaxiter=Gmaxiter, Gtol=ipmpar[2])

    	        ## run classification task
                results_ipm = model.tune(X_train, y_train, X_test, y_test)
                print("\nRandomSearch for NFFTSVMipm")
                print("IPM Parameters:", ipmpar)
                print("d_ratio:", dr)
                print("Best Parameters:", results_ipm[0])
                print("Best Result:", results_ipm[1])
                print("Best Runtime Fit:", results_ipm[2])
                print("Best Runtime Predict:", results_ipm[3])
                print("Best Total Runtime:", results_ipm[2] + results_ipm[3])
                print("Mean Runtime Fit:", results_ipm[4])
                print("Mean Runtime Predict:", results_ipm[5])
                print("Mean Total Runtime:", results_ipm[4] + results_ipm[5])
                print("Best IPMiter:", results_ipm[9])
                print("Best GMRESiter:", results_ipm[7])
                print("Mean GMRESiter:", results_ipm[8])
                
                acc.append(results_ipm[1][0])
                ipmiters.append(results_ipm[9])
                gmresiters.append(np.mean(results_ipm[7]))
	    
    	    # save values to dict
            if prec == "chol_greedy":
                dict_cholgr[D_prec].append([acc,ipmiters,gmresiters])
            elif prec == "chol_rp":
                dict_cholrp[D_prec].append([acc,ipmiters,gmresiters])
            elif prec == "rff":
                dict_rff[D_prec].append([acc,ipmiters,gmresiters])
            elif prec == "nystrom":
                dict_nystr[D_prec].append([acc,ipmiters,gmresiters])
        	    
            #################################################################################
            #################################################################################
            ## print results in comparison
            print("\n########################################################################")
            print("\nResults NFFTSVMipm:")
            print("------------------------\n")
            print("dict_cholgr:", dict_cholgr)
            print("dict_cholrp:", dict_cholrp)
            print("dict_rff:", dict_rff)
            print("dict_nystr:", dict_nystr)