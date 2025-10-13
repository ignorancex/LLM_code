from . import chebpy
import numpy as np
from copy import deepcopy

def scaleNodes(x, dom):
    # SCALENODES   Scale the Chebyshev nodes X from [-1,1] to DOM.
    if dom[0] == -1 and dom[1] == 1:
        # Nodes are already on [-1, 1]
        return x

    # Scale the nodes:
    return dom[1]*(x + 1)/2 + dom[0]*(1 - x)/2

def chebpts(n, dom = chebpy.core.settings.DefaultPreferences.domain, type = 2):
    #chebpts    Chebyshev points.
    #   chebpts(N) returns N Chebyshev points of the 2nd-kind in [-1,1].
    #
    #   chebpts(N, D), where D is vector of length 2 and N is a scalar integer,
    #   scales the nodes and weights for the interval [D(1),D(2)].

    ##################################################################################
    #   [Mathematical reference]:
    #   Jarg Waldvogel, "Fast construction of the Fejer and Clenshaw-Curtis
    #   quadrature rules", BIT Numerical Mathematics, 46, (2006), pp 195-202.
    ##################################################################################

    # Create a dummy CHEBTECH of appropriate type to access static methods.
    if type == 1:
        f = chebpy.core.chebtech.Chebtech
    elif type == 2:
        f = chebpy.core.chebtech.Chebtech2
    else:
        raise Exception('CHEBFUN:chebpts:type, Unknown point type.') 

    if isinstance(n,int) or isinstance(n,np.int64) or len(n) == 1:         # Single grid
        # Call the static CHEBTECH.CHEBPTS() method:
        x = f._chebpts(n)
        # Scale the domain:
        x = scaleNodes(x, dom)
    else:                  # Piecewise grid.
        raise NotImplementedError
    return x

def legpoly(n, dom = chebpy.core.settings.DefaultPreferences.domain, normalize = False, prefs = chebpy.core.settings._preferences):
    """
    P = legpoly(N) computes a CHEBFUN of the Legendre polynomial of degree N on
    the interval [-1,1]. N can be a vector of integers, in which case the output
    is an array-valued CHEBFUN.

    P = legpoly(N, D) computes the Legendre polynomials as above, but on the
    interval given by the domain D, which must be bounded.

    P = LEGPOLY(N, D, True) normalises so that
    integral(P(:,j).*P(:,k)) = delta_{j,k}.

    For N <= 1000 LEGPOLY uses the standard recurrence relation.

    There is no current implementation of leg2cheb, or computation using
    weighted QR factorisation of a 2*(N+1) x 2*(N+1) Chebyshev Vandermonde
    matrix, or corresponding to legoply chebfun:
    https://github.com/chebfun/chebfun/blob/master/legpoly.m
    """

    nMax = max(n)
    u, indices = np.unique(n, return_index = True)
    P = np.zeros((nMax+1, max(n.shape)))
    x = chebpts(nMax+1)
    Lk_2, Lk_1 = np.ones((nMax+1,)), x           # P0 and P1, Initial condn for recurrence
    ind = 0
    for k in range(2,nMax+2):
        if u[ind] == k-2:
            if normalize:
                invnrm = np.sqrt((2*k - 1)/np.diff(dom))
                P[:,ind] = Lk_2*invnrm
            else:
                P[:,ind] = Lk_2
            
            # Update recurrence
            temp = Lk_1
            Lk_1 = (2 - 1/(k+1))*x*Lk_1 - (1 - 1/(k+1))*Lk_2
            Lk_2 = temp
            ind += 1
    return chebpy.chebfun(P[:,indices], dom, prefs = prefs)

def abstractQR(qMat, E, InnerProduct, Norm, tol = chebpy.core.settings.DefaultPreferences.eps):
    A = deepcopy(qMat)
    numCols = A.shape[1]
    R = np.zeros((numCols,numCols))
    V = A

    for k in range(numCols):
        # Scale
        scl = max(Norm(E[:,k]), Norm(A[:,k]))
        # Multiply the kth column of A with the basis in E:
        ex = InnerProduct(E[:,k], A[:,k])
        aex = np.abs(ex)

        # Adjust the sign of the kth column in E:

        if aex < tol*scl:
            s = 1
        else:
            s = -np.sign(ex/aex)
        E[:,k] = E[:,k] * s
        
        # Compute the norm of the kth column of A:
        r = np.sqrt(InnerProduct(A[:,k], A[:,k]))
        R[k,k] = r

        # Compute the reflection v:
        v = r*E[:,k] - A[:,k]

        # Make it more orthogonal:
        for i in range(k):
            ev = InnerProduct(E[:,i],v)
            v = v - E[:,i]*ev
        
        # Normalize:
        nv = np.sqrt(InnerProduct(v,v))
        if nv < tol*scl:
            v = E[:,k]
        else:
            v = (1/nv) * v

        # Store
        V[:,k] = v

        # Subtract v from the remaining columns of A:
        for j in range(k+1,numCols):
            # Apply the Householder reflection:
            av = InnerProduct(v, A[:,j])
            A[:,j] = A[:,j] - 2*v*av

            # Compute other nonzero entries in the current row and store them:
            rr = InnerProduct(E[:,k],A[:,j])
            R[k,j] = rr

            # Subtract off projections onto the current vector E[:,k]:
            A[:,j] = A[:,j] - E[:,k]*rr


        # # "Kind-of" vectorized version. Fix bugs
        # J = slice(k+1,numCols)
        # av = v.T * A[:,J]
        # A[:,J] = A[:,J] - (v * (av*2))
        
        # rr = E[:,k].T * A[:,J]
        # R[k,J] = rr.squeeze()
        # for j in range(k+1,numCols):
        #     # Subtract off projections onto the current vector E[:,k]:
        #     A[:,j] = A[:,j] - E[:,k]*R[k,j]

    # Form Q from the columns of V:
    Q = E
    for k in reversed(range(numCols)):
        for j in range(k,numCols):
            vq = InnerProduct(V[:,k],Q[:,j])
            Q[:,j] = Q[:,j] - 2*(V[:,k]*vq)

    return Q,R

    

