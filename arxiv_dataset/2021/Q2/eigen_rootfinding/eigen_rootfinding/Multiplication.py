import numpy as np
import itertools
from scipy.linalg import eig, schur
from eigen_rootfinding.polynomial import MultiCheb, MultiPower
from eigen_rootfinding.utils import memoize
from scipy.stats import ortho_group

def indexarray(matrix_terms, which, var):
    """Compute the array mapping monomials under multiplication by x_var

    Parameters
    ----------
    matrix_terms : 2d integer ndarray
        Array containing the monomials in order. matrix_terms[i] is the array
        containing the exponent for each variable in the ith multivariate
        monomial
    which : slice object
        object to index into the matrix_terms for the monomials we want to multiply by var
    var : int
        Variable to multiply by: x_0, ..., x_(dim-1)

    Returns
    -------
    arr : 1d integer ndarray
        Array containing the indices of the lower-degree monomials after multiplication
        by x_var
    """
    mults = matrix_terms[which].copy()
    mults[:, var] += 1
    return np.argmin(np.abs(mults[:, np.newaxis] - matrix_terms[np.newaxis]).sum(axis=-1), axis=1)

def indexarray_cheb(matrix_terms, which, var):
    """Compute the array mapping Chebyshev monomials under multiplication by x_var:

        T_1*T_0 = T_1
        T_1*T_n = .5(T_(n+1)+ T_(n-1))

    Parameters
    ----------
    matrix_terms : 2d integer ndarray
        Array containing the monomials in order. matrix_terms[i] is the array
        containing the degree for each univariate Chebyshev monomial in the ith
        multivariate monomial
    m : int
        Number of monomials of highest degree, i.e. those that do not need to be
        multiplied
    var : int
        Variable to multiply by: x_0, ..., x_(dim-1)

    Returns
    -------
    arr1 : 1d integer ndarray
        Array containing the indices of T_(n+1)
    arr2 : 1d
        Array containing the indices of T_(n-1)
    """
    up = matrix_terms[which].copy()
    up[:, var] += 1
    down = matrix_terms[which].copy()
    down[:, var] -= 1
    down[down[:, var]==-1, var] += 2
    arr1 = np.argmin(np.abs(up[:, np.newaxis] - matrix_terms[np.newaxis]).sum(axis=-1), axis=1)
    arr2 = np.argmin(np.abs(down[:, np.newaxis] - matrix_terms[np.newaxis]).sum(axis=-1), axis=1)
    return arr1, arr2

def ms_matrices(E, Q, matrix_terms, dim):
    """Compute the Möller-Stetter matrices in the monomial basis from a
    reduced Macaulay matrix

    Parameters
    ----------
    E : (m, k) ndarray
        Columns of the reduced Macaulay matrix corresponding to the quotient basis
    Q : (l, n) 2d ndarray
        Matrix whose columns give the quotient basis in terms of the monomial basis
    matrix_terms : 2d ndarray
        Array with ordered monomial basis
    dim : int
        Number of variables

    Returns
    -------
    M : (n, n, dim) ndarray
        Array containing the nxn Möller-Stetter matrices, where the matrix
        corresponding to multiplication by x_i is M[..., i]
    """
    n = Q.shape[1]
    m = E.shape[0]
    M = np.empty((n, n, dim),dtype=E.dtype)
    A = np.vstack((-E, Q))
    for i in range(dim):
        arr = indexarray(matrix_terms, slice(m,None), i)
        M[..., i] = Q.conj().T@A[arr]
    return M

def ms_matrices_cheb(E, Q, matrix_terms, dim):
    """Compute the Möller-Stetter matrices in the Chebyshev basis from a
    reduced Macaulay matrix

    Parameters
    ----------
    E : (m, k) ndarray
        Columns of the reduced Macaulay matrix corresponding to the quotient basis
    Q : (l, n) 2d ndarray
        Matrix whose columns give the quotient basis in terms of the Chebyshev basis
    matrix_terms : 2d ndarray
        Array with ordered Chebyshev basis
    dim : int
        Number of variables

    Returns
    -------
    M : (n, n, dim) ndarray
        Array containing the nxn Möller-Stetter matrices, where the matrix
        corresponding to multiplication by x_i is M[..., i]
    """
    n = Q.shape[1]
    m = E.shape[0]
    M = np.empty((n, n, dim),dtype=E.dtype)
    A = np.vstack((-E, Q))
    for i in range(dim):
        arr1, arr2 = indexarray_cheb(matrix_terms, slice(m,None), i)
        M[..., i] = .5*Q.T.conj()@(A[arr1]+A[arr2])
    return M

def ms_matrices_p(E, P, matrix_terms, dim, cut):
    """Compute the Möller-Stetter matrices in the power basis from a
    reduced Macaulay matrix (QRP method)

    Parameters
    ----------
    E : (m, k) ndarray
        Columns of the reduced Macaulay matrix corresponding to the quotient basis
    P : (, l) ndarray
        Array of pivots returned in QR with pivoting, used to permute the columns.
    matrix_terms : 2d ndarray
        Array with ordered Chebyshev basis
    dim : int
        Number of variables

    Returns
    -------
    M : (n, n, dim) ndarray
        Array containing the nxn Möller-Stetter matrices, where the matrix
        corresponding to multiplication by x_i is M[..., i]
    """
    r, n = E.shape
    matrix_terms[cut:] = matrix_terms[cut:][P]
    M = np.empty((n, n, dim),dtype=E.dtype)
    A = np.vstack((-E, np.eye(n)))
    for i in range(dim):
        arr = indexarray(matrix_terms, slice(r,None), i)
        M[..., i] = A[arr]
    return M

def ms_matrices_p_cheb(E, P, matrix_terms, dim, cut):
    """ Compute the Möller-Stetter matrices in the Chebyshev basis from a
    reduced Macaulay matrix (QRP method)

    Parameters
    ----------
    E : (m, k) ndarray
        Columns of the reduced Macaulay matrix corresponding to the quotient basis
    P : (, l) ndarray
        Array of pivots returned in QR with pivoting, used to permute the columns.
    matrix_terms : 2d ndarray
        Array with ordered Chebyshev basis
    dim : int
        Number of variables

    Returns
    -------
    M : (n, n, dim) ndarray
        Array containing the nxn Möller-Stetter matrices, where the matrix
        corresponding to multiplication by x_i is M[..., i]
    """
    r, n = E.shape
    matrix_terms[cut:] = matrix_terms[cut:][P]
    M = np.empty((n, n, dim),dtype=E.dtype)
    A = np.vstack((-E, np.eye(n)))
    for i in range(dim):
        arr1, arr2 = indexarray_cheb(matrix_terms, slice(r,None), i)
        M[..., i] = .5*(A[arr1] + A[arr2])
    return M

def sort_eigs(eigs, diag):
    """Sorts the eigs array to match the order on the diagonal
    of the Schur factorization

    Parameters
    ----------
    eigs : 1d ndarray
        Array of unsorted eigenvalues
    diag : 1d complex ndarray
        Array containing the diagonal of the approximate Schur factorization

    Returns
    -------
    w : 1d ndarray
        Eigenvalues from eigs sorted to match the order in diag
    """
    n = diag.shape[0]
    lst = list(range(n))
    arr = []
    for eig in eigs:
        i = lst[np.argmin(np.abs(diag[lst]-eig))]
        arr.append(i)
        lst.remove(i)
    return np.argsort(arr)

@memoize
def get_rand_combos_matrix(rows, cols, normal=False):
    """ Generates a rows by cols random matrix with orthogonal rows or columns,
    depending on if rows > cols or cols > rows.

    Parameters
    ----------
    rows : int
        Number of rows
    cols : int
        Number of columns
    normal : bool
        Optional. Whether or not to create a matrix using entries drawn
        from the standard normal distribution (N(0, 1)) or not. If it's
        False, it will return an orthogonal matrix.

    Returns
    -------
    C : (rows, cols) ndarray
        Matrix with orthgonal rows or columns, depending on if rows > cols or
        cols > rows if normal is False, otherwise a matrix with
        coefficients drawn from the standard normal (N(0, 1)).
    """
    np.random.seed(57)
    # TODO perhaps explore different types of random matrices?
    # randn was giving me conditioning problems
    if normal:
        C = np.random.normal(loc=0, scale=1, size=(rows, cols))
        return C
    size = max(rows, cols)
    C = ortho_group.rvs(size)
    return C[:rows, :cols]

@memoize
def get_Q_c(dim):
    """ Generates a once-chosen random orthogonal matrix and a random linear combination
    for use in the simultaneous eigenvalue compution.

    Parameters
    ----------
    dim : int
        Dimension of the system

    Returns
    -------
    Q : (dim, dim) ndarray
        Random orthogonal rotation
    c : (dim, ) ndarray
        Random linear combination
    """
    np.random.seed(103)
    Q = ortho_group.rvs(dim)
    c = np.random.randn(dim)
    return Q, c

def msroots(M):
    """Computes the roots to a system via the eigenvalues of the Möller-Stetter
    matrices. Implicitly performs a random rotation of the coordinate system
    to avoid repeated eigenvalues arising from special structure in the underlying
    polynomial system. Approximates the joint eigenvalue problem using a Schur
    factorization of a linear combination of the matrices.

    Parameters
    ----------
    M : (n, n, dim) ndarray
        Array containing the nxn Möller-Stetter matrices, where the matrix
        corresponding to multiplication by x_i is M[..., i]

    Returns
    -------
    roots : (n, dim) ndarray
        Array containing the approximate roots of the system, where each row
        is a root.
    """
    dim = M.shape[-1]

    # perform a random rotation with a random orthogonal Q
    Q, c = get_Q_c(dim)
    M = (Q@M[..., np.newaxis])[..., 0]

    eigs = np.empty((dim, M.shape[0]), dtype='complex')
    # Compute the matrix U that triangularizes a random linear combination
    U = schur((M*c).sum(axis=-1), output='complex')[1]

    for i in range(0, dim):
        T = (U.T.conj())@(M[..., i])@U
        w = eig(M[..., i], right=False)
        arr = sort_eigs(w, np.diag(T))
        eigs[i] = w[arr]

    # Rotate back before returning, transposing to match expected shape
    return (Q.T@eigs).T
