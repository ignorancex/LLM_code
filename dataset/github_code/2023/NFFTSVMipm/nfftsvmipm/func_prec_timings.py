"""
Author: Theresa Wagner <theresa.wagner@mathematik.tu-chemnitz.de>

Corresponding publication:
"A Preconditioned Interior Point Method for Support Vector Machines Using an
ANOVA-Decomposition and NFFT-Based Matrix–Vector Products"
by T. Wagner, John W. Pearson, M. Stoll (2023)
"""

import numpy as np
import scipy
import time
import fastadj2

from .precond import pivoted_chol_rp
from .data_preprocessing import data_preprocess

##################################################################################

def preprocess(X_train, y_train, X_test, windows):
    """
    Balance train and z-score normalize train and test data and determine weights for the sum of kernels.
        
    Parameters
    ----------
    X_train : ndarray
        The training data.
    y_train : ndarray
        The target vector incorporating the true labels for the training data.
    X_test : ndarray
        The test data.
    windows : list
        The list of windows determining the feature grouping.
        
    Returns
    -------
    X_train : ndarray
        The balanced and z-score normalized train data.
    y_train : ndarray
        The corresponding target vector to the balanced train data.
    X_test : ndarray
        The z-score normalized test data.
    weights : float
        The weight for the weighted sum of kernels.
    """
    ############################################################
    # preprocess data: balance train data and z-score normalize
    X_train, y_train, X_test = data_preprocess(X_train, y_train, X_test, balance=True)
    
    ##################################
    
    ## compute kernel weights: equally weighted such that weights sum up to 1
    kweights = 1.0/len(windows)
    
    weights = kweights
    
    return X_train, y_train, X_test, weights

##################################################################################

def init_fast_matvec(X_train, windows, sigma, kernel):
    """
    Set up computations with the adjacency matrix and create adjacency matrix object.
        
    Parameters
    ----------
    X_train : ndarray
        The training data.
    windows: list
        The list of windows determining the feature grouping.
    sigma : float
        Sigma parameter for the RBF kernel.
    kernel : int
        The indicator of the chosen kernel definition.
        kernel=1 denotes the Gaussiam kernel, kernel=3 the Matérn(1/2) kernel.
    
    Returns
    -------
    adj_mats : object
        The adjacency matrix object.
    """
    ## setup computations with the adjacency matrices
    if kernel == 1:
        adj_mats = [fastadj2.AdjacencyMatrix(X_train[:,windows[l]], np.sqrt(2)*sigma[l], setup="default", kernel=kernel, diagonal=1.0) for l in range(len(windows))]
    elif kernel == 3:
        adj_mats = [fastadj2.AdjacencyMatrix(X_train[:,windows[l]], sigma[l], setup="default", kernel=kernel, diagonal=1.0) for l in range(len(windows))]
        
    return adj_mats
        

##################################################################################

def fast_matvec(adj_mats, p, windows, weights):
    """
    Approximate matrix-vector product A*p, where A = K1 + ... + KP, with equal weights 
    
    Parameters
    ----------
    adj_mats : object
        The adjacency matrix object for multiplying the matrix A by a vector from the right.
    p : ndarray
        The vector, whose product A*p with the matrix A shall be approximated.
    windows : list
        The list of windows determining the feature grouping.
    weights : float
        The weight for the weighted sum of kernels.

    Returns
    -------
    Ap : ndarray
        The approximated matrix-vector product A*p.
    """    
    # perform kernel-vector multiplication
    Ap_i = np.asarray([adj_mats[l].apply(p) for l in range(len(windows))])
    
    # add weights and sum weighted sub-kernels up
    Ap = weights * np.sum(Ap_i, axis=0)
        
    return Ap

##################################################################################

def setup_precond(X_train, y_train, prec, D_prec, windows, sigma, weights, kernel):
    """
    Set up preconditioners for the kernel matrix whose action on a vector is
    defined by the adjacency matrix object adj_mats and measure the time needed
    for construction.
    
    Parameters
    ----------
    X_train : ndarray
        The training data.
    y_train : ndarray
        The corresponding target vector.
    prec : str
        The preconditioner that shall be used to precondition the kernel matrix.
    D_prec : int
        The desired rank of the preconditioner.
    windows : list
        The list of windows determining the feature grouping.
    sigma : 
        Sigma parameter for the RBF kernel.
    weights : float
        The weight for the weighted sum of kernels.
    kernel : int
        The indicator of the chosen kernel definition.
        kernel=1 denotes the Gaussiam kernel, kernel=3 the Matérn(1/2) kernel.

    Returns
    -------
    precond_time : float
        The time needed for constructing the preconditioner.
    """    
    # start timing preconditioner setup
    start_precond = time.time()
    
    if prec != "rff":
        # set up computations with adjacency matrix
        adj_mats = init_fast_matvec(X_train, windows, sigma, kernel)
        KER_fast = lambda p: fast_matvec(adj_mats, p, windows, weights)

    #####################
    ## PRECONDITIONING
    #####################
    # pivoted Cholesky (greedy)
    if prec == "chol_greedy":
        MM = D_prec
        n = len(y_train)

        Ldec = pivoted_chol_rp(MM,KER_fast,n,"greedy")
        
    ########################
    # pivoted Cholesky (rp)
    elif prec == "chol_rp":
        MM = D_prec
        n = len(y_train)

        Ldec = pivoted_chol_rp(MM,KER_fast,n,"rp")
        
    ######################
    # random Fourier features
    elif prec == "rff":
        
        # initialize array of decompositions
        Ldec = []
        
        # 1 rff decomposition per window
        for l in range(len(windows)):
            
            # generate D_prec iid samples from p(w)
            W = np.sqrt(2/(sigma[l]**2))*np.random.normal(size=(D_prec,(X_train[:,windows[l]]).shape[1]))
            # generate D_prec iid samples from Uniform(0,2*pi)
            b = 2*np.pi*np.random.rand(D_prec)
            
            Zl = np.sqrt(2/D_prec) * np.cos(((X_train[:,windows[l]]).dot(W.conj().T) + b[np.newaxis,:]))
        
            Ldec.append(Zl)
        
        Ldec = np.concatenate(Ldec, axis=1)
    
    ###########################
    # Nyström decomposition
    elif prec == "nystrom":
        k = D_prec
        G = np.random.randn(X_train.shape[0],k)
        
        Y_ny = np.zeros((G.shape))
        for i in range(k):
            Y_ny[:,i] = KER_fast(G[:,i])
        Q = np.linalg.qr(Y_ny)[0]
        
        AQ = np.zeros((Q.shape))
        for j in range(k):
            AQ[:,j] = KER_fast(Q[:,j])    
            
        QaAQ = Q.T @ AQ

        LL, D, per = scipy.linalg.ldl(QaAQ)
        D = D.clip(min = 1e-2)
        L = LL@scipy.linalg.sqrtm(D)
        
        Ldec = np.zeros((X_train.shape[0],k))
        Ldec=np.linalg.solve(L.T,AQ.T)
        Ldec=Ldec.T

    ########################
    
    # stop timing preconditioner setup
    precond_time = time.time() - start_precond
    
    return precond_time

##################################################################################
    
def main_precond_timings(X_train, X_test, y_train, y_test, wind_param_list, prec, D_prec, kernel):
    """
    The main function for measuring the time for constructing a preconditioner
    for the kernel matrix.
    
    Parameters
    ----------
    X_train : ndarray
        The training data.
    X_test : ndarray
        The test data.
    y_train : ndarray
        The target vector for the training data.
    y_test : ndarray
        The target vector for the test data.
    wind_param_list : list
        A list defining the feature windows, and the length-scale parameters and the regularization parameter.
        The parameter list is of the form: [[W1,...,WP], [[l1,...,lP],C]]
    prec : str
        The preconditioner that shall be used to precondition the kernel matrix.
    D_prec : int
        The desired rank of the preconditioner.
    kernel : int
        The indicator of the chosen kernel definition.
        kernel=1 denotes the Gaussiam kernel, kernel=3 the Matérn(1/2) kernel.

    Returns
    -------
    precond_time : float
        The time needed for constructing the preconditioner.
    """
    windows, params = wind_param_list
    sigma = params[0]
    
    # compute feature weights and normalize data points
    X_train, y_train, X_test, weights = preprocess(X_train, y_train, X_test, windows)
    
    # compute precond time
    precond_time = setup_precond(X_train, y_train, prec, D_prec, windows, sigma, weights, kernel)
    
    return precond_time
