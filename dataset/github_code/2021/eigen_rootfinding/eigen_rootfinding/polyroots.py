import numpy as np
from eigen_rootfinding import OneDimension as oneD
from eigen_rootfinding.polynomial import is_power
from eigen_rootfinding.Macaulay import macaulay_solve
from eigen_rootfinding.Nullspace import nullspace_solve
from eigen_rootfinding.utils import match_poly_dimensions, ConditioningError

#todo test and decide what to use as default
def solve(polys, verbose=False, return_all_roots=True,
          max_cond_num=1.e6, method='svdmac', randcombos=False,
          normal=False):
    '''
    Finds the roots of the given list of polynomials.

    Parameters
    ----------
    polys : list of polynomial objects
        Polynomials to find the common roots of.
    verbose : bool
        Prints information about how the roots are computed.
    return_all_roots : bool
        If True returns all the roots, otherwise just the ones in the unit box.
    max_cond_num : float
        The maximum condition number of the Macaulay matrix reduction
    method : str
        Which method to use to solve the system. Options are:
        --1 dimension--
            'multeigvals':
                roots are computed as eigenvalues of multiplication matrix
            'diveigvals':
                roots are computed as eigenvalues of division matrix
            'multeigvecs':
                roots are from as eigenvectors of multiplication matrix
            'diveigvecs':
                roots are from as eigenvectors of division matrix
        --n dimensions--
            All methods compute roots as eigenvalues of Moller-Stetter (MS) matrices.
            The methods vary in how the MS matrix is created.
            'qrpmac','lqmac','svdmac':
                Via QRP, LQ or SVD of Macaulay matrix
            'qrpnull','lqnull','svdnull':
                Via QRP, LQ or SVD of nullspace of Macaulay matrix, which is computed via SVD
            'qrpfastnull','lqfastnull','svdfastnull': MS
                Via QRP, LQ or SVD of nullspace of Macaulay matrix, which is computed degree by degree
    randcombos : bool
        Whether or not to first take random linear combinations of the Macaulay matrix.
        Not allowed for fastnullspace reductions
    normal : bool
        If randcombos is True, whether or not to use a matrix with entries
        drawn from the standard normal dsitribution when taking random
        linear combinations of the Macaulay matrix.
    returns
    -------
    roots : numpy array
        The common roots of the polynomials. Each row is a root.
    '''
    polys = match_poly_dimensions(polys)
    # Determine polynomial type and dimension of the system
    poly_type = is_power(polys, return_string = True)
    dim = polys[0].dim

    if dim == 1:
        #default to using rotated companion matrix with eigenvalues
        if method=='svdmac' or method=='multeigvals': #todo update to final default method
            MSmatrix = 0
            eigvals = True
        elif method=='diveigvals':
            MSmatrix = -1
            eigvals = True
        elif method=='multeigvecs':
            MSmatrix = 0
            eigvals = False
        elif method=='diveigvecs':
            MSmatrix = -1
            eigvals = False
        else:
            raise ValueError('invalid method type for univariate solver')
        if len(polys) == 1:
            return oneD.solve(polys[0], MSmatrix=MSmatrix, eigvals=eigvals, verbose=verbose)
        else:
            zeros = np.unique(oneD.solve(polys[0], MSmatrix=MSmatrix, eigvals=eigvals, verbose=verbose))
            #Finds the roots of each succesive polynomial and checks which roots are common.
            for poly in polys[1:]:
                if len(zeros) == 0:
                    break
                zeros2 = np.unique(oneD.solve(poly, MSmatrix=MSmatrix, eigvals=eigvals, verbose=verbose))
                common = list()
                tol = 1.e-10
                for zero in zeros2:
                    spot = np.where(np.abs(zeros-zero)<tol)
                    if len(spot[0]) > 0:
                        common.append(zero)
                zeros = common
            return zeros
    else:
        if len(polys) != dim:
            raise ValueError('dimension must match len(polys)')
        if method in {'qrpnull','lqnull','svdnull',
                      'qrpfastnull','lqfastnull','svdfastnull'}:
            if method[-8:]=='fastnull':
                if randcombos:
                    #todo verify this is true
                    raise ValueError('Cannot do random linear combinations and fast nullspace together')
                return nullspace_solve(polys, return_all_roots=return_all_roots,
                                   method=method[:-8], nullmethod='fast',
                                   randcombos=randcombos, normal=normal)
            else:
                return nullspace_solve(polys, return_all_roots=return_all_roots,
                               method=method[:-4], nullmethod='svd',randcombos=randcombos,
                               normal=normal)
        elif method in {'qrpmac','lqmac','svdmac'}:
            res = macaulay_solve(polys, max_cond_num=max_cond_num, verbose=verbose,
                                 return_all_roots=return_all_roots, method=method[:-3],
                                 randcombos=randcombos, normal=normal)
            if res[0] is None:
                raise ConditioningError(res[1])
            else:
                return res
        else:
            raise ValueError('invalid method type for multivariate solver')
