from sage.all import *

def get_matrix_type(*args):
    r"""
    Given a reduced Gram matrix of the Gross lattice tests if it belongs to 
    one of the four conjectured types over Fp.

    The output is of the form [Type,  [parameters]].

    Type 1: [t]
    [d    2*t          0]
    [2*t  4*(p+t^2)/d  0]
    [0    0            p]

    Type 2: [m, n]
    [d    2*m          2*n]
    [2*m  4*(p+m^2)/d  m  ]
    [2*n  m            p+n]

    Type 3: [u]
    [d   2*u           u             ]
    [2u  4*(p+u^2)/d   2*(p+u^2)/d   ]
    [u   2*(p+u^2)/d   p +(p + u^2)/d]

    Type 4: [a, b]
    [d    a             b              ]
    [a   (4*p+a^2)/d  -(2p-ab)/d       ]
    [b  -(2p-ab)/d      p + (p + b^2)/d]

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
        raise ValueError("Invalid number of arguments. Use get_matrix_type(M) or get_matrix_type(M, p).")

    if D3 == p and y == 0 and z == 0:
        if x%2 == 0:
            t = x//2
            if 4*(p+t**2)%D1 == 0 and D2 == 4*(p+t**2)//D1:
                return [1,[t]]
    elif x == 2*z:
        m = z
        if 4*(p+m**2)%D1 == 0 and  D2 == 4*(p+m**2)//D1:
            if y%2 == 0:
                n = y//2
                if D3 == p+n:
                    return [2,[m,n]]
    elif x == 2*y and D2 == 2*z:
        u = y
        if 4*(p+u**2)%D1 == 0 and  D2 == 4*(p+u**2)//D1:
            if ((D1+1)*p+u**2)%D1 == 0 and  D3 == ((D1+1)*p+u**2)//D1:
                return [3,[u]]
    elif (-2*p+x*y)%D1 == 0 and z == (-2*p+x*y)//D1:
            if (4*p+x**2)%D1 == 0 and  D2 == (4*p+x**2)//D1:
                if ((D1+1)*p+y**2)%D1 == 0 and  D3 == ((D1+1)*p+y**2)//D1:
                    return [4,[x,y]]
    else:
        return [0, [0]]