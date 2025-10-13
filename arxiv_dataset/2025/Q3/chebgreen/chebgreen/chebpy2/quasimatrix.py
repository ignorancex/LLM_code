from . import chebpy
import numpy as np
from abc import ABC, abstractmethod, abstractclassmethod
from .utils import chebpts, abstractQR
from .chebpy.core.plotting import import_plt

class Quasimatrix(ABC):
    """Create a Quasimatrix in order to implement most functionality for the chebfun2 constructor"""
    def __init__(self, data, transposed = False):
        # Currently only initialized with a numpy array of chebfuns
        self.data = np.array(data).reshape(-1)
        if self.data is not None:
            self.domain = self.data[0].domain
            self.prefs = self.data[0].prefs
        self.transposed = transposed
    
    # Disable matrix multiplication from numpy with broadcasting while right multiplying.
    __array_ufunc__ = None

    def __repr__(self):
        if not self.transposed:
            header = f"Quasimatrix of shape ({self.shape[0]} x {self.shape[1]}) with columns:\n"
        else:
            header = f"Quasimatrix of shape ({self.shape[0]} x {self.shape[1]}) with rows:\n"
        return header + self.data.__repr__()
            
    def __getitem__(self, key):
        x, y = key
        
        if not self.transposed:
            assert isinstance(y,slice) or isinstance(y,int) or isinstance(y,list) or isinstance(y,np.ndarray), \
                'Second index needs to be a slice or an integer'

            # !! Write checks for the shape and type of list/numpy array "y"

            if isinstance(x,int) or isinstance(x,float):
                return np.array([col(x) for col in self.data[y]])
            elif isinstance(x,np.ndarray) and (x.dtype == np.int64 or x.dtype == np.float64):
                return np.array([col(x) for col in self.data[y]]).T
            elif (x == slice(None)):
                return Quasimatrix(data = np.array(self.data[y]).reshape(-1), transposed = self.transposed)
            else:
                raise RuntimeError('The first index needs to be a float or a numpy array of floats')
        else:
            assert isinstance(x,slice) or isinstance(x, int) or isinstance(x,list) or isinstance(x,np.ndarray), \
                'First index needs to be a slice or an integer'

            # !! Write checks for the shape and type of list/numpy array "x"

            if isinstance(y,int) or isinstance(y,float):
                return np.array([row(y) for row in self.data[x]])
            elif isinstance(y,np.ndarray) and (y.dtype == np.int64 or y.dtype == np.float64):
                return np.array([row(y) for row in self.data[x]])
            elif y == slice(None):
                return Quasimatrix(data = np.array(self.data[x]).reshape(-1), transposed = self.transposed)
            else:
                raise RuntimeError('The second index needs to be a float or a numpy array of floats')
            
    def __setitem__(self, key, newvalue):
        x, y = key
        if not self.transposed:
            assert isinstance(y,slice) or isinstance(y,int), 'Second index needs to be a slice or an integer'

            if (x == slice(None)):
                if isinstance(y, int):
                    self.data[y] = newvalue.data.item()
                elif isinstance(y,slice):
                    start, stop, step = y.indices(len(self.data))
                    for ind, i in enumerate(range(start, stop, step)):
                        self.data[i] = newvalue.data[ind]
            else:
                raise RuntimeError('Can only set the value of a slice of the Column Quasimatrix.')
        else:
            assert isinstance(x,slice) or isinstance(x, int), 'First index needs to be a slice or an integer'

            if y == slice(None):
                if isinstance(x, int):
                    self.data[x] = newvalue.data.item()
                elif isinstance(x,slice):
                    start, stop, step = x.indices(len(self.data))
                    for ind, i in enumerate(range(start, stop, step)):
                        self.data[i] = newvalue.data[ind]
            else:
                raise RuntimeError('Can only set the value of a slice of the Row Quasimatrix.')    
    
    # Scalar addition and subtraction not implemented
    def __add__(self, qmat):
        # Addition of quasimatrices
        assert self.shape == qmat.shape, f"Cannot add a ({self.shape[0]} x {self.shape[1]}) matrix to a ({qmat.shape[0]} x {qmat.shape[1]}) matrix."
        return Quasimatrix(data = self.data + qmat.data, transposed = self.transposed)
    
    def __sub__(self, qmat):
        # Addition of quasimatrices
        assert self.shape == qmat.shape, f"Cannot subtract a ({qmat.shape[0]} x {qmat.shape[1]}) matrix from a ({self.shape[0]} x {self.shape[1]}) matrix."
        return Quasimatrix(data = self.data - qmat.data, transposed = self.transposed)

    def __mul__(self, G):
        """
        Multiplication of quasimatrices

        Here, I am assuming that the quasimatrices only comprises of chebfuns which
        are of the same rank within a quasimatrix and that each chebfun only has one
        single interval.
        Faster but uses more space.
        """
        F = self 

        if isinstance(G,int) or isinstance(G,float):
            return Quasimatrix(F.data * G, transposed = F.transposed)

        # Check if the inputs can be multiplied
        assert F.shape[1] == G.shape[0], f"Cannot mulitply a ({F.shape[0]} x {F.shape[1]}) matrix with a ({G.shape[0]} x {G.shape[1]}) matrix."
        
        if isinstance(G,np.ndarray) and (G.dtype == np.int64 or G.dtype == np.float64):
            if not F.transposed:
                return Quasimatrix(data = chebpy.chebfun(F.coeffrepr() @ G, domain = F.domain, prefs = F.prefs, initcoeffs = True), transposed = False) # Returns (inf x G.shape[1]) matrix
            else:
                RuntimeError('Invalid multiplication')  # This should never be reached because of shape check
        
        if F.shape[1] == np.inf: 
            assert F.domain == G.domain, "Cannot multiply quasimatrices on different domains"
            out = 0
            m, n = F.shape[0], G.shape[1]
            
            N = len(F.data[0].coeffs) + len(G.data[0].coeffs)
            Fvalues = np.zeros((m,N))
            Gvalues = np.zeros((N,n))
            
            prolongtemp = np.zeros(N)
            for i in range(m):
                f = F.data[i].funs[0]
                prolongtemp[:len(f.coeffs)] = f.coeffs
                f = f.onefun._coeffs2vals(prolongtemp)
                Fvalues[i,:len(f)] = f
            
            for j in range(n):
                g = G.data[j].funs[0]
                prolongtemp[:len(g.coeffs)] = g.coeffs
                g = g.onefun._coeffs2vals(prolongtemp)
                Gvalues[:len(g),j] = g
                
            w = chebpy.core.algorithms.quadwts2(N).reshape((1,-1))
            rescalingFactor = 0.5 * float(np.diff(F.domain))
            
            return (w * Fvalues) @ Gvalues * rescalingFactor
        else:
            raise NotImplementedError

    def __rmul__(self, F):
        """
        Multiplication of quasimatrices

        Here, I am assuming that the quasimatrices only comprises of chebfuns which
        are of the same rank within a quasimatrix and that each chebfun only has one
        single interval.
        Faster but uses more space.
        """
        G = self 

        if isinstance(F,int) or isinstance(F,float):
            return Quasimatrix(F * G.data, transposed = G.transposed)

        # Check if the inputs can be multiplied
        assert F.shape[1] == G.shape[0], f"Cannot mulitply a ({F.shape[0]} x {F.shape[1]}) matrix with a ({G.shape[0]} x {G.shape[1]}) matrix."

        if isinstance(F,np.ndarray) and (F.dtype == np.int64 or F.dtype == np.float64):
            if G.transposed:
                return Quasimatrix(data = chebpy.chebfun((F @ G.coeffrepr()).T, domain = G.domain, prefs = G.prefs, initcoeffs = True), transposed = True) # Returns (G.shape[0] x inf) matrix
            else:
                RuntimeError('Invalid multiplication')  # This should never be reached because of shape check
        
        if F.shape[1] == np.inf: 
            assert F.domain == G.domain, "Cannot multiply quasimatrices on different domains"
            out = 0
            m, n = F.shape[0], G.shape[1]
            
            N = len(F.data[0].coeffs) + len(G.data[0].coeffs)
            Fvalues = np.zeros((m,N))
            Gvalues = np.zeros((N,n))
            
            prolongtemp = np.zeros(N)
            for i in range(m):
                f = F.data[i].funs[0]
                prolongtemp[:len(f.coeffs)] = f.coeffs
                f = f.onefun._coeffs2vals(prolongtemp)
                Fvalues[i,:len(f)] = f
            
            for j in range(n):
                g = G.data[j].funs[0]
                prolongtemp[:len(g.coeffs)] = g.coeffs
                g = g.onefun._coeffs2vals(prolongtemp)
                Gvalues[:len(g),j] = g
                
            w = chebpy.core.algorithms.quadwts2(N).reshape((1,-1))
            rescalingFactor = 0.5 * float(np.diff(F.domain))
            
            return (w * Fvalues) @ Gvalues * rescalingFactor
        else:
            raise NotImplementedError        

    # Householder QR in value representation for L2-inner product
    def qr(self):
        assert self.shape[0] == np.inf, "QR decomposition is only computed for column Quasimatrices"
        
        # Compute parameters for the QR:
        tol = np.finfo(float).eps*np.max(self.vscale)
        n, numCols = max([len(cb.coeffs) for cb in self.data]), self.shape[1]

        # Make the discrete analogue of the Quasimatrix:
        N = 2*max(n, numCols)
        A = np.zeros((N,numCols))        

        prolongtemp = np.zeros(N)
        for j in range(numCols):
            a = self.data[j].funs[0]
            prolongtemp[:len(a.coeffs)] = a.coeffs
            a = a.onefun._coeffs2vals(prolongtemp)
            A[:len(a),j] = a

        # Create the Chebyshev nodes and quadrature weights:
        x = chebpts(N).reshape(-1)
        w = chebpy.core.algorithms.quadwts2(N).reshape((1,-1))

        # Define norm and inner products:
        InnerProduct = lambda f,g: w @ (f.reshape((-1,1)) * g.reshape((-1,1)))
        Norm = lambda f: np.max(np.abs(f))

        # Generate a discrete E (Legendre-Chebyshev-Vandermonde matrix) directly:
        E = np.ones(A.shape)
        E[:,1] = x[:]
        for k in range(2,numCols):
            E[:,k] = ((2*k-1)*x*E[:,k-1] - (k-1)*E[:,k-2]) / k
        # Scaling:
        for k in range(numCols):
            E[:,k] = E[:,k] * np.sqrt((2*k+1)/2)

        # E = np.ones(A.shape)
        # for k in range(numCols):
        #     E[:,k] = np.sin( (x - self.domain[0])/(np.diff(self.domain))*(k+1)*np.pi)

        # Note that the formulas may look different because they are corrected for zero-indexed arrays

        Q, R = abstractQR(A, E, InnerProduct, Norm, tol)

        # Adjust scaling
        scale = np.sqrt(2/(self.domain[1]-self.domain[0]))
        
        D = np.eye(Q.shape[1])*scale
        Q = Q @ D
        R = np.linalg.inv(D) @ R

        # Convert to Coefficient form to truncate the unnecessary
        Qcoeffs = np.hstack([chebpy.core.algorithms.vals2coeffs2(Q[:,i]).reshape(-1,1) for i in range(Q.shape[1])])
        Qcoeffs = Qcoeffs[:,:(int(N/2) + 1)]

        Q = Quasimatrix(data = chebpy.chebfun(Qcoeffs, domain = self.domain, prefs = self.prefs, initcoeffs = True), transposed = False)

        return Q,R
    
    # # Full abstract QR    
    # def qr(self):
    #     assert self.shape[0] == np.inf, "QR decomposition is only computed for column Quasimatrices"
    #     L = Quasimatrix(legpoly(n = np.linspace(0,self.shape[1], self.shape[1]).astype(int),
    #                 dom = self.domain,
    #                 normalize = True,
    #                 prefs = self.prefs))
        
    #     InnerProduct = lambda A,B: (A.T * B).item()
    #     Norm = lambda A: A.normest
    #     return abstractQR(self, L, InnerProduct, Norm, self.prefs.eps)
        
    ### Properties
    @property
    def shape(self):
        if self.transposed:
            return len(self.data), np.inf
        else:
            return np.inf, len(self.data)
    
    # Assuming all chebfuns have the same size
    def __len__(self):
        if len(self.data) == 0:
            return None
        return self.data[0].funs[0].size
    
    @property
    def vscale(self):
        if len(self.data) == 0:
            return None
        return [f.vscale for f in self.data]
    
    @property
    def normest(self):
        if len(self.data) == 0:
            return None
        return np.max(self.vscale)
        
    @property
    def T(self):
        # Tranpose of a quasimatrix
        return Quasimatrix(data = self.data, transposed = not self.transposed)

    def coeffrepr(self, N = None):
        if len(self.data) == 0:
            raise RuntimeError('Cannot construct a coefficient representation for an empty quasimatrix')
        F = self
        if self.transposed:
            m = F.shape[0]
            if N is None:
                N = np.max([row.funs[0].size for row in self.data])
            Fvalues = np.zeros((m,N))
            
            prolongtemp = np.zeros(N)
            for i in range(m):
                f = F.data[i].funs[0]
                prolongtemp[:len(f.coeffs)] = f.coeffs
                Fvalues[i,:len(prolongtemp)] = prolongtemp
            return Fvalues
        else:
            n = F.shape[1]
            if N is None:
                N = np.max([col.funs[0].size for col in self.data])
            Fvalues = np.zeros((N,n))
            
            prolongtemp = np.zeros(N)
            for j in range(n):
                f = F.data[j].funs[0]
                prolongtemp[:len(f.coeffs)] = f.coeffs
                Fvalues[:len(prolongtemp),j] = prolongtemp
            return Fvalues
        # if len(self.data) == 0:
        #     raise RuntimeError('Cannot construct a coefficient representation for an empty quasimatrix')
        # if self.transposed:
        #     return np.array([row.coeffs for row in self.data])
        # else:
        #     return np.array([col.coeffs for col in self.data]).T


    ###  Utilities
    def plot(self, fig = None, ax = None, **kwds):
        if isinstance(self.data,chebpy.core.chebfun.Chebfun):
            self.data.plot(**kwds)
        else:
            plt = import_plt()
            if plt:
                plt.ion()
                if fig is None:
                    fig = plt.figure(figsize = kwds.pop("figsize",None))
                if ax is None:
                    ax = plt.gca()
            cmap = plt.get_cmap("viridis", len(self.data))
            for i in range(len(self.data)):
                self.data[i].plot(fig = fig, ax = ax, color = cmap(i), **kwds)
    
    def plotcoeffs(self, fig = None, ax = None, **kwds):
        if isinstance(self.data,chebpy.core.chebfun.Chebfun):
            self.data.plotcoeffs(**kwds)
        else:
            plt = import_plt()
            if plt:
                plt.ion()
                if fig is None:
                    fig = plt.figure(figsize = kwds.pop("figsize",None))
                if ax is None:
                    ax = plt.gca()
            for i in range(len(self.data)):
                self.data[i].plotcoeffs(fig = fig, ax = ax, **kwds)