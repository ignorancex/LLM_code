"""
Author: Theresa Wagner <theresa.wagner@mathematik.tu-chemnitz.de>

Corresponding publication:
"A Preconditioned Interior Point Method for Support Vector Machines Using an
ANOVA-Decomposition and NFFT-Based Matrix–Vector Products"
by T. Wagner, John W. Pearson, M. Stoll (2023)
"""

import numpy as np
import fastadj2

##################################################################################

def svm_predict_fastsum(X_test, alpha, y_train, X_train, sigma, windows, weights, kernel=1, fastadj_setup="default"):
    """
    Perform predictions based on the learned classifier parameter applying the NFFT-based fast summation approach.
    
    Parameters
    ----------
    X_test : ndarrray
        The test data.
    alpha : ndarray
        The learned classifier parameter.
    y_train : ndarray
        The training target vector.
    X_train : ndarray
         The training data.
    sigma : float
        Sigma parameter for the RBF kernel.
    windows : list
        The list of feature windows determining the feature grouping.
    weights : float
        The weight for the weighted sum of kernels.
    kernel : int, default = 1
        The indicator of the chosen kernel definition.
        kernel=1 denotes the Gaussiam kernel, kernel=3 the Matérn(1/2) kernel.
    fastadj_setup : str, default = "default"
        Defines the desired approximation accuracy of the NFFT fastsum method. It is one of the strings 'fine', 'default' or 'rough'.

    Returns
    -------
    YPred : ndarray
        Predictions for the test data.
    """
    ####################
    # setup NFFT-based fast summation
    N_sum = X_test.shape[0] + X_train.shape[0]
    arr = np.append(X_train, X_test, axis=None).reshape(N_sum,X_train.shape[1])
    
    p = np.append(alpha*y_train, np.zeros(X_test.shape[0]), axis=None).reshape(N_sum,)
    
    if isinstance(sigma, float):
        if kernel == 1:
            adj_vals = [fastadj2.AdjacencyMatrix(arr[:,windows[l]], np.sqrt(2)*sigma, setup=fastadj_setup, kernel=kernel, diagonal=1.0) for l in range(len(windows))]
        elif kernel == 3:
            adj_vals = [fastadj2.AdjacencyMatrix(arr[:,windows[l]], sigma, setup=fastadj_setup, kernel=kernel, diagonal=1.0) for l in range(len(windows))]
    else:
        if kernel == 1:
            adj_vals = [fastadj2.AdjacencyMatrix(arr[:,windows[l]], np.sqrt(2)*sigma[l], setup=fastadj_setup, kernel=kernel, diagonal=1.0) for l in range(len(windows))]
        elif kernel == 3:
            adj_vals = [fastadj2.AdjacencyMatrix(arr[:,windows[l]], sigma[l], setup=fastadj_setup, kernel=kernel, diagonal=1.0) for l in range(len(windows))]
    
    #####################
    ## predict responses
    
    # perform kernel-vector multiplication
    vals_i = np.asarray([adj_vals[l].apply(p) for l in range(len(windows))])
    # add weights and sum weighted sub-kernels up
    vals = weights * np.sum(vals_i, axis=0)
    
    # select predicted responses for test data
    YPred = np.sign(vals[-X_test.shape[0]:])
    
    return YPred