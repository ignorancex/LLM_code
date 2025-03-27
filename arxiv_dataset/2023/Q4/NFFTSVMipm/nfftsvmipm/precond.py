"""
Author: Theresa Wagner <theresa.wagner@mathematik.tu-chemnitz.de>

Corresponding publication:
"A Preconditioned Interior Point Method for Support Vector Machines Using an
ANOVA-Decomposition and NFFT-Based Matrix–Vector Products"
by T. Wagner, John W. Pearson, M. Stoll (2023)
"""

import numpy as np

from scipy.linalg import lu_solve

#################################################################################

def pivoted_chol_rp(k, KER, n, alg):
    """
    Compute the pivoted Cholesky preconditioner decomposition matrix.
            
    Parameters
    ----------
    k : int
        Target preconditioner rank.
    KER : object
        LinearOperator for approximating a kernel vector product.
    n : int
        Number of training data points.
    alg : str
        The Cholesky preconditioner that shall be used.
        If alg=="rp", the randomized pivoted Cholesky preconditioner is chosen.
        If alg=="greedy" the greedy-based Cholesky preconditioner is chosen.   
        
    Returns
    -------
    Ldec : ndarray
        Cholesky preconditioner decomposition matrix.
    """
    diags =  np.copy(get_diag(KER,n))

    # row ordering, is much faster for large scale problems
    F = np.zeros((k,n))
    rows = np.zeros((k,n))
    rng = np.random.default_rng()
    arr_idx = []
    for i in range(k):
        if alg == "rp":
            idx = rng.choice(range(n), p = diags / sum(diags))
        elif alg == "greedy":
            idx = np.argmax(diags)
        arr_idx.append(idx)
        rows[i,:] = get_row(idx,KER,n)
        F[i,:] = (rows[i,:] - F[:i,idx].T @ F[:i,:]) / np.sqrt(diags[idx])
        diags -= F[i,:]**2
        diags = diags.clip(min = 0)
        
        if np.max(diags) < 1e-3:
            break
        
    Ldec = np.transpose(F)
        
    return Ldec

##################################################################################

def SMW_prec(inx, n, Lp, y, dhm, test, lu, piv):
    """
    Multiply the SMW preconditioner to a vector.
            
    Parameters
    ----------
    inx : ndarray
        The vector that is multiplied to the SMW preconditioner from the right.
    n : int
        Number of training data points.
    Lp : ndarray
        The low-rank decomposition matrix for the preconditioner.
    y : ndarray
        The training target vector.
    dhm : ndarray
        The diagonal barrier matrix Theta from equation (5) in the paper.
    test : ndarray
        Matrix I_k + ZD^-1Z^T from the paper.    
    lu : ndarray
        Matrix containing U in its upper triangle, and L in its lower triangle, for the pivoted LU decomposition PLU.
    piv : ndarray
        Pivot indices representing the permutation matrix P.
    
    Returns
    -------
    out : ndarray
        The result of multiplying the SMW preconditioner with a vector.
    """
    # (1,1) block
    temp = inx[:n]/dhm # theta^{-1}
    z = y*temp
    z = Lp.conj().T@ z
    z = lu_solve((lu,piv),z)
    z = Lp@z
    z = y*z
    z = z/dhm
    z = temp - z
    out = z

    # (2,2) block 
    out = np.append(out,-y.T@z-inx[n])    
    
    return out

##################################################################################
    
def get_diag(KER, n):
    """
    Get the diagonal of the matrix approximated by the linear operator KER.
    
    Note
    ----
    Diagonal entries of kernel matrix are all equal.
            
    Parameters
    ----------
    KER : object
        LinearOperator for approximating a kernel vector product.
    n : int
        Shape parameter of the square matrix approximated by the linear operator KER.
    
    Returns
    -------
    out : ndarray
        Diagonal of the matrix approximated by the linear operator KER.
    """
    ei = np.zeros(n,)
    ei[0] = 1
    a = np.transpose(KER(ei))
    
    out = a[0]*np.ones((n,))
    return out

##################################################################################

def get_row(i, KER, n):
    """
    Get the i-th row of the matrix approximated by the linear operator KER.
            
    Parameters
    ----------
    i : int
        Index of the row that shall be returned.
    KER : object
        LinearOperator for approximating a kernel vector product.
    n : int
        Shape parameter of the square matrix approximated by the linear operator KER.
        
    Returns
    -------
    row : ndarray
       I-th row of the matrix approximated by the operator KER. 
    """
    ei = np.zeros(n,)
    ei[i] = 1
    
    row = np.transpose(KER(ei))
    return row
    
