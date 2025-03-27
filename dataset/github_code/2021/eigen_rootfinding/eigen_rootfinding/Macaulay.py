import numpy as np
import itertools
from scipy.linalg import qr, solve_triangular, qr_multiply, svd
from eigen_rootfinding.polynomial import Polynomial, MultiCheb, MultiPower,\
                                         is_power
from eigen_rootfinding.utils import row_swap_matrix, slice_top, mon_combos,\
                                    mon_combosHighest, solve_linear
from matplotlib import pyplot as plt
from warnings import warn
from eigen_rootfinding.Multiplication import ms_matrices,ms_matrices_p,\
                                             ms_matrices_cheb,ms_matrices_p_cheb,\
                                             msroots,get_rand_combos_matrix

macheps = 2.220446049250313e-16

def plot_scree(s,tol):
    plt.semilogy(s,marker='.')
    plt.plot(np.ones(len(s))*tol)
    plt.show()

def macaulay_solve(polys, max_cond_num, verbose=False, return_all_roots=True,
                   method='svd', randcombos=False, normal=False):
    '''
    Finds the roots of the given list of multidimensional polynomials using
    a reduced Macaulay matrix to create Moller-Stetter mtarices

    Parameters
    ----------
    polys : list of polynomial objects
        Polynomials to find the common roots of.
    max_cond_num : float
        The maximum condition number of the Macaulay Matrix Reduction
    verbose : bool
        Prints information about how the roots are computed.
    return_all_roots : bool
        If True returns all the roots, otherwise just the ones in the unit box.
    method : str
        Which method to use to reduce the Macaulay matrix the system.
        Options are 'qrp','lq','svd'.
    randcombos : bool
        Whether or not to first take random linear combinations of the Macaulay matrix.
    normal : bool
        If randcombos is True, whether or not to use a matrix with entries
        drawn from the standard normal dsitribution when taking random
        linear combinations of the Macaulay matrix.

    returns
    -------
    roots : numpy array
        The common roots of the polynomials. Each row is a root.
    '''
    #We don't want to use Linear Projection right now
#    polys, transform, is_projected = polys, lambda x:x, False

    if len(polys) == 1:
        from eigen_rootfinding.OneDimension import solve
        return transform(solve(polys[0], MSmatrix=0))
    poly_type = is_power(polys, return_string = True)
    dim = polys[0].dim

    #By Bezout's Theorem. Useful for making sure that the reduced Macaulay Matrix is as we expect
    bezout_bound = np.prod([poly.degree for poly in polys])
    matrix, matrix_terms, cut = build_macaulay(polys, verbose)
    if randcombos:
        C = get_rand_combos_matrix(matrix.shape[1]-bezout_bound,matrix.shape[0], normal=normal)
        matrix = C@matrix

    roots = np.array([])

    # If cut is zero, then all the polynomials are linear and we solve
    # using solve_linear.
    if cut == 0:
        roots, cond = solve_linear([p.coeff for p in polys])
        # Make sure roots is a 2D array.
        roots = np.array([roots])
    else:
        # Attempt to reduce the Macaulay matrix
        if method == 'svd':
            res = reduce_macaulay_svd(matrix, cut, bezout_bound, max_cond_num)
            if res[0] is None:
                return res
            E, Q = res
        elif method == 'lq':
            res = reduce_macaulay_lq(matrix, cut, bezout_bound, max_cond_num)
            if res[0] is None:
                return res
            E, Q = res
        elif method == 'qrp':
            res = reduce_macaulay_qrp(matrix, cut, bezout_bound, max_cond_num)
            if res[0] is None:
                return res
            E, Q = res
        else:
            raise ValueError("Method must be one of 'svd', 'lq' or 'qrp'")

        # Construct the Möller-Stetter matrices
        # M is a 3d array containing the multiplication-by-x_i matrix in M[..., i]
        if poly_type == "MultiCheb":
            if method == 'lq' or method == 'svd':
                M = ms_matrices_cheb(E, Q, matrix_terms, dim)
            elif method == 'qrp':
                M = ms_matrices_p_cheb(E, Q, matrix_terms, dim, cut)
        else:
            if method == 'lq' or method == 'svd':
                M = ms_matrices(E, Q, matrix_terms, dim)
            elif method == 'qrp':
                M = ms_matrices_p(E, Q, matrix_terms, dim, cut)

        # Compute the roots using eigenvalues of the Möller-Stetter matrices
        roots = msroots(M)

    if return_all_roots:
        return roots
    else:
        # only return roots in the unit complex hyperbox
        return roots[[np.all(np.abs(root) <= 1) for root in roots]]

def add_polys(degree, poly, poly_coeff_list):
    """Adds polynomials to a Macaulay Matrix.

    This function is called on one polynomial and adds all monomial multiples of
     it to the matrix.

    Parameters
    ----------
    degree : int
        The degree of the Macaulay Matrix
    poly : Polynomial
        One of the polynomials used to make the matrix.
    poly_coeff_list : list
        A list of all the current polynomials in the matrix.
    Returns
    -------
    poly_coeff_list : list
        The original list of polynomials in the matrix with the new monomial
        multiplications of poly added.
    """

    poly_coeff_list.append(poly.coeff)
    deg = degree - poly.degree
    dim = poly.dim

    mons = mon_combos([0]*dim,deg)

    for mon in mons[1:]: #skips the first all 0 mon
        poly_coeff_list.append(poly.mon_mult(mon, returnType = 'Matrix'))
    return poly_coeff_list

def find_degree(poly_list, verbose=False):
    '''Finds the appropriate degree for the Macaulay Matrix.

    Parameters
    --------
    poly_list: list
        The polynomials used to construct the matrix.
    verbose : bool
        If True prints the degree
    Returns
    -----------
    find_degree : int
        The degree of the Macaulay Matrix.

    '''
    if verbose:
        print('Degree of Macaulay Matrix:', sum(poly.degree for poly in poly_list) - len(poly_list) + 1)
    return sum(poly.degree for poly in poly_list) - len(poly_list) + 1

def build_macaulay(initial_poly_list, verbose=False):
    """Constructs the unreduced Macaulay matrix.

    Parameters
    --------
    initial_poly_list: list
        The polynomials in the system we are solving.
    verbose : bool
        Prints information about how the roots are computed.
    Returns
    -----------
    matrix : 2d ndarray
        The Macaulay matrix
    matrix_terms : 2d integer ndarray
        Array containing the ordered basis, where the ith row contains the
        exponent/degree of the ith basis monomial
    cut : int
        Where to cut the Macaulay matrix for the highest-degree monomials
    """
    power = is_power(initial_poly_list)
    dim = initial_poly_list[0].dim
    poly_coeff_list = []
    degree = find_degree(initial_poly_list)

    # linear_polys = [poly for poly in initial_poly_list if poly.degree == 1]
    # nonlinear_polys = [poly for poly in initial_poly_list if poly.degree != 1]
    # #Choose which variables to remove if things are linear, and add linear polys to matrix
    # if len(linear_polys) >= 1: #Linear polys involved
    #     #get the row rededuced linear coefficients
    #     A, Pc = nullspace(linear_polys)
    #     varsToRemove = Pc[:len(A)].copy()
    #     #add to macaulay matrix
    #     for row in A:
    #         #reconstruct a polynomial for each row
    #         coeff = np.zeros([2]*dim)
    #         coeff[tuple(get_var_list(dim))] = row[:-1]
    #         coeff[tuple([0]*dim)] = row[-1]
    #         if not power:
    #             poly = MultiCheb(coeff)
    #         else:
    #             poly = MultiPower(coeff)
    #         poly_coeff_list = add_polys(degree, poly, poly_coeff_list)
    # else: #no linear
    #     A, Pc = None, None
    #     varsToRemove = []

    #add nonlinear polys to poly_coeff_list
    for poly in initial_poly_list:#nonlinear_polys:
        poly_coeff_list = add_polys(degree, poly, poly_coeff_list)

    #Creates the matrix
    # return (*create_matrix(poly_coeff_list, degree, dim, varsToRemove), A, Pc)
    return create_matrix(poly_coeff_list, degree, dim)#, varsToRemove)

def create_matrix(poly_coeffs, degree, dim):#, varsToRemove):
    ''' Builds a Macaulay matrix.

    Parameters
    ----------
    poly_coeffs : list.
        Contains numpy arrays that hold the coefficients of the polynomials to be put in the matrix.
    degree : int
        The degree of the Macaulay Matrix
    dim : int
        The dimension of the polynomials going into the matrix.
    Returns
    -------
    matrix : 2D numpy array
        The Macaulay matrix.
    matrix_terms : numpy array
        The ith row is the term represented by the ith column of the matrix.
    cut : int
        Number of monomials of highest degree
    '''
    bigShape = [degree+1]*dim

    matrix_terms, cut = sorted_matrix_terms(degree, dim)#, varsToRemove)

    #Get the slices needed to pull the matrix_terms from the coeff matrix.
    matrix_term_indexes = list()
    for row in matrix_terms.T:
        matrix_term_indexes.append(row)

    #Adds the poly_coeffs to flat_polys, using added_zeros to make sure every term is in there.
    added_zeros = np.zeros(bigShape,dtype=poly_coeffs[0].dtype)
    flat_polys = list()
    for coeff in poly_coeffs:
        slices = slice_top(coeff.shape)
        added_zeros[slices] = coeff
        flat_polys.append(added_zeros[tuple(matrix_term_indexes)])
        added_zeros[slices] = np.zeros_like(coeff,dtype=coeff.dtype)
    del poly_coeffs

    #Make the matrix. Reshape is faster than stacking.
    matrix = np.reshape(flat_polys, (len(flat_polys), len(matrix_terms)))

    #Sorts the rows of the matrix so it is close to upper triangular.
    matrix = row_swap_matrix(matrix)
    return matrix, matrix_terms, cut

def sorted_matrix_terms(degree, dim):#, varsToRemove):
    '''Finds the matrix_terms sorted in the term order needed for Macaulay reduction.
    So the highest terms come first, the x, y, z etc monomials last.
    Parameters
    ----------
    degree : int
        The degree of the Macaulay Matrix
    dim : int
        The dimension of the polynomials going into the matrix.
    Returns
    -------
    sorted_matrix_terms : numpy array
        The sorted matrix_terms. The ith row is the term represented by the ith column of the matrix.
    cuts : int
        Number of monomials of highest degree
    '''
    highest_mons = mon_combosHighest([0]*dim, degree)[::-1]

    other_mons = list()
    d = degree - 1
    while d > 1:
        other_mons += mon_combosHighest([0]*dim, d)[::-1]
        d -= 1

    #extra-small monomials: 1, x, y, etc.
    xs_mons = mon_combos([0]*dim, 1)[::-1]

    #trivial case
    if degree == 1:
        matrix_terms = np.reshape(xs_mons, (len(xs_mons), dim))
        cuts = 0
    #normal case
    else:
        matrix_terms = np.reshape(highest_mons+other_mons+xs_mons, (len(highest_mons+other_mons+xs_mons), dim))
        cuts = len(highest_mons)

    # for var in varsToRemove:
    #     B = matrix_terms[cuts[0]:]
    #     mask = B[:, var] != 0
    #     matrix_terms[cuts[0]:] = np.vstack([B[mask], B[~mask]])
    #     cuts = tuple([cuts[0] + np.sum(mask), cuts[1]+1])

    return matrix_terms, cuts

def reduce_macaulay_lq(M, cut, bezout_bound, max_cond=1e6):
    """Reduces the Macaulay matrix using the LQ method.

    Parameters:
    -----------
    matrix : 2d ndarray
        The Macaulay matrix
    cut : int
        Number of columns of max degree
    bezout_bound : int
        Number of roots of the system, determined by Bezout's theoerm
    max_cond : int or float
        Max condition number for the two condition number checks

    Returns:
    --------
    E : 2d ndarray
        The columns of the reduced Macaulay matrix corresponding to the quotient basis
    Q2 : 2d ndarray
        Matrix giving the quotient basis in terms of the monomial basis. Q2[:,i]
        being the coefficients for the ith basis element
    """
    # Compute numerical rank
    s = svd(M, compute_uv=False)
    tol = max(M.shape)*s[0]*macheps
    rank = len(s[s>tol])
    # Check if numerical rank doesn't match bezout bound
    bezout_rank = M.shape[1]-bezout_bound
    if rank < bezout_rank:
        warn("Rank of Macaulay Matrix does not match the Bezout bound. Expected rank {}, found rank {}. System potentially has infinitely many solutions.".format(bezout_rank,rank))
    elif rank > bezout_rank:
        warn('Rank of Macaulay Matrix does not match the Bezout bound. Expected rank {}, found rank {}.'.format(bezout_rank,rank))

    # Check condition number before first QR
    cond_num = np.linalg.cond(M[:,:cut])
    if cond_num > max_cond:
        return None, "Condition number of the Macaulay high-degree columns is {}".format(cond_num)

    # QR reduce the highest-degree columns
    Q,M[:,:cut] = qr(M[:,:cut])
    M[:,cut:] = Q.T.conj() @ M[:,cut:]
    Q = None
    del Q

    # If the matrix is "tall", compute an orthogonal transformation of the remaining
    # columns, generating a new polynomial basis
    if cut < M.shape[0]:
        Q = qr(M[cut:,cut:].T.conj(),pivoting=True)[0]
        Q2 = Q[:,-bezout_bound:]
        del Q
        M[:cut,-bezout_bound:] = M[:cut,cut:] @ Q2 # Apply column transform

    # Return the backsolved columns and coefficient matrix for the quotient basis
    return solve_triangular(M[:cut,:cut],M[:cut,bezout_rank:]),Q2

def reduce_macaulay_svd(M, cut, bezout_bound, max_cond=1e6):
    """Reduces the Macaulay matrix using the SVD method.

    Parameters:
    -----------
    matrix : 2d ndarray
        The Macaulay matrix
    cut : int
        Number of columns of max degree
    bezout_bound : int
        Number of roots of the system, determined by Bezout's theoerm
    max_cond : int or float
        Max condition number for the two condition number checks

    Returns:
    --------
    E : 2d ndarray
        The columns of the reduced Macaulay matrix corresponding to the quotient basis
    Q2 : 2d ndarray
        Matrix giving the quotient basis in terms of the monomial basis. Q2[:,i]
        being the coefficients for the ith basis element
    """
    # Compute numerical rank
    s = svd(M, compute_uv=False)
    tol = max(M.shape)*s[0]*macheps
    rank = len(s[s>tol])
    # Check if numerical rank doesn't match bezout bound
    bezout_rank = M.shape[1]-bezout_bound
    if rank < bezout_rank:
        warn("Rank of Macaulay Matrix does not match the Bezout bound. Expected rank {}, found rank {}. System potentially has infinitely many solutions.".format(bezout_rank,rank))
    elif rank > bezout_rank:
        warn('Rank of Macaulay Matrix does not match the Bezout bound. Expected rank {}, found rank {}.'.format(bezout_rank,rank))

    # Check condition number before first QR
    cond_num = np.linalg.cond(M[:,:cut])
    if cond_num > max_cond:
        return None, "Condition number of the Macaulay high-degree columns is {}".format(cond_num)

    # QR reduce the highest-degree columns
    Q,M[:,:cut] = qr(M[:,:cut])
    M[:,cut:] = Q.T.conj() @ M[:,cut:]
    Q = None
    del Q

    # If the matrix is "tall", compute an orthogonal transformation of the remaining
    # columns, generating a new polynomial basis
    if cut < M.shape[0]:
        V = svd(M[cut:,cut:])[2].T.conj()[:,-bezout_bound:]
        V2 = V[:,-bezout_bound:]
        del V
        M[:cut,bezout_rank:] = M[:cut,cut:] @ V2# Apply column transform

    # Return the backsolved columns and coefficient matrix for the quotient basis
    return solve_triangular(M[:cut,:cut],M[:cut,bezout_rank:]),V2

def reduce_macaulay_qrp(M, cut, bezout_bound, max_cond=1e6):
    """Reduces the Macaulay matrix using the QRP method.

    Parameters:
    -----------
    matrix : 2d ndarray
        The Macaulay matrix
    cut : int
        Number of columns of max degree
    bezout_bound : int
            Number of roots of the system, determined by Bezout's theoerm
    max_cond : int or float
        Max condition number for the two condition number checks

    Returns:
    --------
    E : 2d ndarray
        The columns of the reduced Macaulay matrix corresponding to the quotient basis
    P : 1d ndarray
        Array of pivots returned in QR with pivoting, used to permute the columns.
    """
    # Compute numerical rank
    s = svd(M, compute_uv=False)
    tol = max(M.shape)*s[0]*macheps
    rank = len(s[s>tol])
    # Check if numerical rank doesn't match bezout bound
    bezout_rank = M.shape[1]-bezout_bound
    if rank < bezout_rank:
        warn("Rank of Macaulay Matrix does not match the Bezout bound. Expected rank {}, found rank {}. System potentially has infinitely many solutions.".format(bezout_rank,rank))
    elif rank > bezout_rank:
        warn('Rank of Macaulay Matrix does not match the Bezout bound. Expected rank {}, found rank {}.'.format(bezout_rank,rank))

    # Check condition number before first QR
    cond_num = np.linalg.cond(M[:,:cut])
    if cond_num > max_cond:
        return None, "Condition number of the Macaulay high-degree columns is {}".format(cond_num)

    # QR reduce the highest-degree columns
    Q,M[:,:cut] = qr(M[:,:cut])
    M[:,cut:] = Q.T.conj() @ M[:,cut:]
    Q = None
    del Q

    # If the matrix is "tall", compute an orthogonal transformation of the remaining
    # columns, generating a new polynomial basis
    if cut < M.shape[0]:
        M[cut:,cut:],P = qr(M[cut:,cut:], mode='r', pivoting=True)
        M[:cut,cut:] = M[:cut,cut:][:,P] # Permute columns

    # Check condition number before backsolve
    cond_num_back = np.linalg.cond(M[:bezout_rank,:bezout_rank])
    if cond_num_back > max_cond:
        return None, "Condition number of the Macaulay primary submatrix is {}".format(cond_num)

    return solve_triangular(M[:bezout_rank,:bezout_rank],M[:bezout_rank,bezout_rank:]),P

def reduce_macaulay_p(M, cut, P, bezout_bound, max_cond=1e6):
    """Reduces the Macaulay matrix using a predetermined permutation of columns.

    Parameters:
    -----------
    matrix : 2d ndarray
        The Macaulay matrix
    cut : int
        Number of columns of max degree
    P : 1d ndarray
        Predetermined Array of pivots
    bezout_bound : int
            Number of roots of the system, determined by Bezout's theoerm
    max_cond : int or float
        Max condition number for the two condition number checks

    Returns:
    --------
    E : 2d ndarray
        The columns of the reduced Macaulay matrix corresponding to the quotient basis
    P : 1d ndarray
        Array of pivots
    """
    # Compute numerical rank
    s = svd(M, compute_uv=False)
    tol = max(M.shape)*s[0]*macheps
    rank = len(s[s>tol])
    # Check if numerical rank doesn't match bezout bound
    bezout_rank = M.shape[1]-bezout_bound
    if rank < bezout_rank:
        warn("Rank of Macaulay Matrix does not match the Bezout bound. Expected rank {}, found rank {}. System potentially has infinitely many solutions.".format(bezout_rank,rank))
    elif rank > bezout_rank:
        warn('Rank of Macaulay Matrix does not match the Bezout bound. Expected rank {}, found rank {}.'.format(bezout_rank,rank))

    # Check condition number before first QR
    cond_num = np.linalg.cond(M[:,:cut])
    if cond_num > max_cond:
        return None, "Condition number of the Macaulay high-degree columns is {}".format(cond_num)

    # QR reduce the highest-degree columns
    Q,M[:,:cut] = qr(M[:,:cut])
    M[:,cut:] = (Q.T.conj() @ M[:,cut:])[:,P]
    Q = None
    del Q

    # If the matrix is "tall", compute an orthogonal transformation of the remaining
    # columns, generating a new polynomial basis
    if cut < M.shape[0]:
        M[cut:,cut:] = qr(M[cut:,cut:])[1:]

    # Check condition number before backsolve
    cond_num_back = np.linalg.cond(M[:,:cut])
    if cond_num_back > max_cond:
        return None, "Condition number of the Macaulay primary submatrix is {}".format(cond_num)

    return solve_triangular(M[:bezout_rank,:bezout_rank],M[:bezout_rank,bezout_rank:]),P
