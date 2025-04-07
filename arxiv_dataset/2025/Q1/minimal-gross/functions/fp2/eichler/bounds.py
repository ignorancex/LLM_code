from sage.all import *

def d3_bound(*args):
    r"""
    Given a Gram matrix of the minmimal Gross lattice tests if it the corresponding supersingular elliptic curve lies over Fp or not.

    The output is of the form [Boolean, upperbound].
    """
    if len(args) == 1:
        M = args[0]
        assert M.is_symmetric() and M.nrows() == 3 and M.ncols() == 3, "input must be a symmetric 3x3 matrix"
        R = PolynomialRing(QQ, 'p')
        p = R.gen()
        D1, D2, D3, x, y ,z = R(M[0][0]), R(M[1][1]), R(M[2][2]), R(M[0][1]), R(M[0][2]), R(M[1][2])
    elif len(args) == 2:
        M, p = args
        assert M.is_symmetric() and M.nrows() == 3 and M.ncols() == 3, "input must be a symmetric 3x3 matrix"
        assert is_pseudoprime(p), "p must be a (pseudo) prime"
        D1, D2, D3, x, y ,z = M[0][0], M[1][1], M[2][2], M[0][1], M[0][2], M[1][2]
    else:
        raise ValueError("Invalid number of arguments. Use d3_bound(M) or d3_bound(M, p).")
    
    fpbound = 8*p/7 + 7/4
    fp2bound = 3*p/5 + 5
    if p <= D3 <= fpbound:
        return [True, floor(fpbound)]
    elif 0 <= D3 <= fp2bound:
        return [False, floor(fp2bound)]
    else:
        return [False, False]

def minimal_basis_matrix(M):
    r"""
    Obtaining Gram matrix of the minimal Gross lattice from LLL-reduced matrix.

    Rearrange LLL-reduced Gram matrix such that D1 <= D2 <= D3 and x,y>0 (flip angles accordingly).
    """
    # Create a deep copy of the input matrix M
    Gred = deepcopy(M)
    # if D2 > D3 then swap D2 and D3, and x and y 
    if Gred[1, 1] > Gred[2, 2]:
        # Swap D2 and D3
        Gred[1, 1], Gred[2, 2] = Gred[2, 2], Gred[1, 1]
        # Swap x and y, 
        Gred[0, 1], Gred[0, 2] = Gred[0, 2], Gred[0, 1]
        Gred[1, 0], Gred[2, 0] = Gred[2, 0], Gred[1, 0]
    
    # make sure x and y are positive.
    if Gred[0, 1] < 0 and Gred[0, 2] >= 0:
        Gred[0, 1] = -Gred[0, 1]
        Gred[1, 0] = -Gred[1, 0]
        Gred[1, 2] = -Gred[1, 2]
        Gred[2, 1] = -Gred[2, 1]
    elif Gred[0, 1] >= 0 and Gred[0, 2] < 0:
        Gred[0, 2] = -Gred[0, 2]
        Gred[2, 0] = -Gred[2, 0]
        Gred[1, 2] = -Gred[1, 2]
        Gred[2, 1] = -Gred[2, 1]
    elif Gred[0, 1] < 0 and Gred[0, 2] < 0:
        Gred[0, 1] = -Gred[0, 1]
        Gred[1, 0] = -Gred[1, 0]
        Gred[0, 2] = -Gred[0, 2]
        Gred[2, 0] = -Gred[2, 0]
    
    return Gred