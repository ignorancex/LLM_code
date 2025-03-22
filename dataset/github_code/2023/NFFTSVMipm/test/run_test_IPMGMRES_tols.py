"""
Author: Theresa Wagner <theresa.wagner@mathematik.tu-chemnitz.de>

Corresponding publication:
"A Preconditioned Interior Point Method for Support Vector Machines Using an
ANOVA-Decomposition and NFFT-Based Matrix–Vector Products"
by T. Wagner, John W. Pearson, M. Stoll (2023)

Execute this file to reproduce the results presented in Table 2.
"""

import numpy as np

from nfftsvmipm.class_NFFTSVMipm import RandomSearch

##################################################################################
## READ PARSED ARGUMENTS

import argparse

# Create the argument parser
parser = argparse.ArgumentParser(description="Run test on IPM/GMRES convergence tolerance with configurable parameters.")

# Add arguments
parser.add_argument('--kernel', type=int, default=1, choices=[1, 3], 
                    help="Kernel type: 1 for Gaussian, 3 for Matérn(1/2), default=1.")
parser.add_argument('--Ndata', nargs='+', type=int, default=[0], 
                    help="List of subset sizes to consider, where 0 corresponds to the entire data set, default=[0].")
parser.add_argument('--prec', type=str, default="chol_greedy", choices=["chol_greedy", "chol_rp", "rff", "nystrom"],
                    help="Preconditioner types, default='chol_greedy'.")
parser.add_argument('--rank', type=int, default=200, 
                    help="Target preconditioner rank, default=200.")
parser.add_argument('--iRS', type=int, default=25, 
                    help="Number of iterations in RandomSearch, default=25.")
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
parser.add_argument('--sbr', type=float, default=0.2, 
                    help="Sigma barrier reduction IPM parameter, default=0.2.")
parser.add_argument('--dratio', type=float, default=1.0, 
                    help="Proportion of features to include, default=1.0.")
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
# preconditioner
prec = args.prec
# target preconditioner rank
Dprec = args.rank
# number of iterations in RandomSearch
iRS = args.iRS
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
# sigma_br parameter
sbr = args.sbr
# dratio
dr = args.dratio
# data set
data = args.data

####################
## CHOOSE PARAMETERS

# define list of candidates for IPM/GMRES tolerance combination
tol_list = [[1e-1,1e-4], [1e-2,1e-5], [1e-3,1e-6], [1e-4,1e-7]]

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
dict_acc = {t[0]: [] for t in tol_list}
dict_ipmiters = {t[0]: [] for t in tol_list}
dict_gmresiters = {t[0]: [] for t in tol_list}
dict_bestparam = {t[0]: [] for t in tol_list}
dict_bestfit = {t[0]: [] for t in tol_list}
dict_bestpred = {t[0]: [] for t in tol_list}

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
	    
	#################################################################################
    ## RandomSearch for NFFTSVMipm

    # define bounds for individual sigmas for every window
    lb_sigma = np.sqrt(1/1e+2)
    ub_sigma = np.sqrt(1/1e-2)

    # define bounds for C
    lb_C = 0.1
    ub_C = 0.7

    for t in tol_list:
    
        print("\n###################################################\n")
        print("############################")
        print("Solving for tol = ", t)
        print("############################")

        # define IPM parameters: ipmpar=[sigma_br,tol,Gtol]
        ipmpar = [sbr,t[0],t[1]]

        # setup RandomSearch model for SVMipm classifier
        model = RandomSearch(classifier="NFFTSVMipm", kernel=kernel, lb=[lb_sigma,lb_C], ub=[ub_sigma, ub_C], max_iter_rs=iRS, mis_threshold=mis_thres, window_scheme=window_scheme, d_ratio=dr, weight_scheme=weight_scheme, sigma_br=ipmpar[0], D_prec=Dprec, prec=prec, iter_ip=iter_ip, tol=ipmpar[1], Gmaxiter=Gmaxiter, Gtol=ipmpar[2])

    	## run classification task
        results_ipm = model.tune(X_train, y_train, X_test, y_test)
        print("\nRandomSearch for NFFTSVMipm")
        print("IPM Parameters:", ipmpar)
        print("d_ratio:", dr)
        print("tols:", t)
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
        
	    # save values to dict
        dict_acc[t[0]].append((results_ipm[1])[0])
        dict_ipmiters[t[0]].append(results_ipm[9])
        dict_gmresiters[t[0]].append(np.mean(results_ipm[7]))
        dict_bestparam[t[0]].append(results_ipm[0])
        dict_bestfit[t[0]].append(results_ipm[2])
        dict_bestpred[t[0]].append(results_ipm[3])
        
        #################################################################################
        #################################################################################
        ## print results in comparison
        print("\n########################################################################")
        print("\nResults NFFTSVMipm:")
        print("------------------------\n")
        print("dict_acc:", dict_acc)
        print("dict_ipmiters:", dict_ipmiters)
        print("dict_gmresiters:", dict_gmresiters)
        print("dict_bestparam", dict_bestparam)
        print("dict_bestfit", dict_bestfit)
        print("dict_bestpred", dict_bestpred)
