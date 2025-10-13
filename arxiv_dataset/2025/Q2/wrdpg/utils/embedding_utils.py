'''
Created on May 6, 2025

@author: bernardo
'''
import numpy as np

def align_Xs(X1,X2):
    """
    An auxiliary function that Procrustes-aligns two embeddings. 
    Parameters
    ----------
    X1 : an array-like with the embeddings to be aligned
    X2 : an array-like with the embeddings to align X1 to
    Returns
    -------
    X1_aligned : the aligned version of X1 to X2.
    """
    V,_,Wt = np.linalg.svd(X1.T@X2)
    U = V@Wt
    X1_aligned = X1@U
    return X1_aligned