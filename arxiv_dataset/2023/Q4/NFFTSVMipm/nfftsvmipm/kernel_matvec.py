"""
Author: Theresa Wagner <theresa.wagner@mathematik.tu-chemnitz.de>

Corresponding publication:
"A Preconditioned Interior Point Method for Support Vector Machines Using an
ANOVA-Decomposition and NFFT-Based Matrix–Vector Products"
by T. Wagner, John W. Pearson, M. Stoll (2023)
"""

##################################################################################

def kernel_matvec(inx, KER, y):
    """
    Multiply the matrix YKY with a vector inx.
            
    Parameters
    ----------
    inx : ndarray
        The vector that is multiplied to YKY from the right.
    KER : object
        LinearOperator for approximating a kernel vector product with the matrix K.
    y : ndarray
        The diagonal entries of the diagonal matrix Y.

    Returns
    -------
    out : ndarray
        The resulting vector of the multiplication.
    """
    out = y * inx # elementwise multiplication
    out = KER(out)
    out = y * out # elementwise multiplication
    
    return out
