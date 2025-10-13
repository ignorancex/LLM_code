from . import chebpy
import numpy as np
from abc import ABC
from .preferences import Chebpy2Preferences
from .quasimatrix import Quasimatrix
from .chebpy.core.plotting import import_plt
from copy import deepcopy
import warnings

class Chebfun2(ABC):
    def __init__(self, g, domain = None, prefs = Chebpy2Preferences(), simplify = False, vectorize = False):
        """
        Constructor for the Chebfun2 class.
        ----------------------------------------------------------------------------------------------------------------
        Arguments:
            g: A python function which specifies the function to be approximated. This is evaluated on a grid of points to
                construct the Chebfun2 object. The function should be vectorized if vectorize is set to False.

            domain: A numpy array which specifies the domain of the function.

            prefs: A Chebpy2Preferences object which specifies the preferences for the approximation.

            simplify: A boolean which specifies whether to simplify the approximation using a singular value expansion.

            vectorize: A boolean which specifies whether to vectorize the function. 

        ----------------------------------------------------------------------------------------------------------------
        """
        # Default domain is [-1,1] x [-1, 1]
        if domain is None:
            self.domain = prefs.domain
        elif type(domain) is not np.ndarray:
            self.domain = np.array(domain)
        else:
            self.domain = domain

        self.prefs = prefs

        if isinstance(g, list):
            assert len(g) == 3, "Not enough parameters to initialize Chebfun2 with"
            assert isinstance(g[0],Quasimatrix) and isinstance(g[1], np.ndarray) and isinstance(g[2],Quasimatrix), \
                "Input format should be [Quasimatrix, Numpy array, Quasimatrix]"
            check = (g[0].shape[1] == g[1].shape[0] and g[2].shape[0] == g[1].shape[0])
            if check:
                assert check and not np.count_nonzero(g[1] == 0), \
                    " ".join("""Chebfun2 takes the input [C D R] such the bivariate function it
                             represents is f = C * diag(D) * R, where u is a column quasimatrix,
                             D has no zeros, and vt is a row quasimatrix.""".split())
            self.cols, self.pivotValues, self.rows = g[0], 1/g[1], g[2]
            self.domain = np.hstack([self.rows.domain,self.cols.domain])
        else:
            self.cols, self.rows, self.pivotValues, self.pivotLocations = self.constructor(lambda x,y: g(x,y), vectorize)

        # Simplify the Chebfun2 object using a singular value decomposition. We truncate to the rank floor(2*n/pi) where n is the number of pivot values.
        if simplify:
            U, S, Vt = self.svd()
            self.cols, self.pivotValues, self.rows = U, 1/S, Vt
            self.truncate(np.floor(2*len(self.pivotValues)/np.pi).astype(int))

        # Write a sampletest here

    # Prioritize the operator functions in Chebfun2 over other data types
    __array_ufunc__ = None

    # Representation of the Chebfun2 object
    def __repr__(self):
        header = f"chebfun2 object\n"
        toprow = "     domain       rank               corner values\n"
        rowdta = (f"[{self.domain[0]:.3f},{self.domain[1]:.3f}] x [{self.domain[2]:.3f},{self.domain[3]:.3f}]     {self.rank}       "
            f"[{self.cornervalues[0]:.3f} {self.cornervalues[1]:.3f} {self.cornervalues[2]:.3f} {self.cornervalues[3]:.3f}]\n")
        btmrow = f"vertical scale = {self.vscale:2f}"
        return header + toprow + rowdta + btmrow
    

    def constructor(self, g, vectorize = False):
        """
        Constructor for the Chebfun2 class.
        ----------------------------------------------------------------------------------------------------------------
        Arguments:
            g: A python function which specifies the function to be approximated. This is evaluated on a grid of points to
                construct the Chebfun2 object. The function should be vectorized if vectorize is set to False.

            vectorize: A boolean which specifies whether to vectorize the function. 
        ----------------------------------------------------------------------------------------------------------------
        Returns:
            cols: A column Quasimatrix which specifies the columns of the Chebfun2 object.

            rows: A row Quasimatrix object which specifies the rows of the Chebfun2 object.

            pivotValues: A numpy array which specifies the pivot values of the Chebfun2 object.

            pivotLocations: A numpy array which specifies the pivot locations of the Chebfun2 object.
        """

        # Define this somewhere in a config file
        prefx = self.prefs.prefx
        prefy = self.prefs.prefy
        minSample = self.prefs.minSample
        maxSample = np.array(np.power(2,[prefx.maxpow2,prefy.maxpow2]))
        maxRank = self.prefs.maxRank

        if prefy == None:
            prefy = prefx
        
        assert (prefx.tech == 'Chebtech2' and prefy.tech == 'Chebtech2'), "CHEBFUN:CHEBFUN2:constructor, Unrecognized technology"
        
        pseudoLevel = min(prefx.eps, prefy.eps)
        
        factor = 4.0        # Ratio between the size of matrix and #pivots.
        isHappy = 0         # If we are currently unresolved.
        failure = 0         # Reached max discretization size without being happy.

        minSample = np.power(2,np.floor(np.log2(minSample-1)))+1


        while not isHappy and not failure:
            grid = minSample

            # Sample function on a Chebyshev tensor grid:
            xx, yy = points2D(grid[0],grid[1],self.domain,prefx,prefy)
            vals = evaluate(g, xx, yy, vectorize)
            
            # Does the function blow up or evaluate to nan?:
            vscale = np.max(np.abs(vals[:]))

            if vscale == np.inf:
                raise RuntimeError('Function returned INF when evaluated')
            elif (vals[:] == np.nan).any():
                raise RuntimeError('Function returned NaN when evaluated')
            

            relTol, absTol = getTol(xx, yy, vals, self.domain, pseudoLevel)
            prefx.eps = relTol
            prefy.eps = relTol

            #### Phase 1:
            # Do GE with complete pivoting:
            pivotVal, pivotPos, rowVals, colVals, iFail = completeACA(vals, absTol, factor)

            strike = 1
            while iFail and (grid <= factor*(maxRank-1)+1).any() and strike < 3:
                # Refine sampling on tensor grid:
                grid[0], _ = gridRefine(grid[0], prefx)
                grid[1], _ = gridRefine(grid[1], prefy)
                xx, yy = points2D(grid[0],grid[1],self.domain,prefx,prefy)
                vals = evaluate(g, xx, yy) # Resample
                vscale = np.max(np.abs(vals[:]))
                
                #New Tolerance
                relTol, absTol = getTol(xx, yy, vals, self.domain, pseudoLevel)
                prefx.eps = relTol
                prefy.eps = relTol
                
                pivotVal, pivotPos, rowVals, colVals, iFail = completeACA(vals, absTol, factor)
                
                # If the function is 0+noise then stop after three strikes.
                if np.abs(pivotVal[0]) < 1e4*vscale*relTol:
                    strike += 1
            
            # If the rank of the function is above maxRank then stop.
            if (grid > factor*(maxRank-1)+1).any():
                failure = 1

                # raise RuntimeWarning('Chebpy2:Chebfun2:constructor:rank, Not a low-rank function.')
                warnings.warn('Chebpy2:Chebfun2:constructor:rank, Not a low-rank function.')
                
                if np.linalg.norm(colVals) == 0 or np.linalg.norm(rowVals) == 0:
                    colVals = 0
                    rowVals = 0
                    pivotVal = np.inf
                    pivotPos = np.array([0,0])
                    isHappy = 1

                cols = Quasimatrix(data = chebpy.chebfun(colVals, domain = np.array(self.domain[2:]), prefs = prefy), transposed = False) 
                rows = Quasimatrix(data = chebpy.chebfun(rowVals.T, domain = np.array(self.domain[:2]), prefs = prefx), transposed = True)
                pivotValues = pivotVal
                pivotLocations = pivotPos

                # Write a Sample Test
                
                return cols, rows, pivotValues, pivotLocations
            
            # Check if the column and row slices are resolved. Hardcoded for Chebtech2
            colTech = chebpy.core.chebtech.Chebtech2.initvalues(values = np.sum(colVals,axis = 1), interval = self.domain[2:])
            resolvedCols = chebpy.core.algorithms.happinessCheck(tech = colTech, vals = colVals, pref = prefy)
            rowTech = chebpy.core.chebtech.Chebtech2.initvalues(values =  np.sum(rowVals.T, axis = 1), interval = self.domain[:2])
            resolvedRows = chebpy.core.algorithms.happinessCheck(tech = rowTech, vals = rowVals.T, pref = prefx)
            isHappy = resolvedRows and resolvedCols

            if len(pivotVal) == 1 and pivotVal == 0:
                pivPos = np.array([[0,0]])
                isHappy = 1
            else:
                pivPos = np.array([[xx[0,pivotPos[j,1]], yy[pivotPos[j,0],0]] for j in range(len(pivotVal))])
                PP = pivotPos
            
            #### Phase 2:
            # Resolve along the column and row slices:
            n, m = grid[1], grid[0]

            while not isHappy and not failure:
                if not resolvedCols:
                    n, nesting = gridRefine(n, prefy)
                    xx, yy = np.meshgrid(pivPos[:,0], chebpy.chebpts(n, self.domain[2:], prefy)[0])
                    colVals = evaluate(g, xx, yy, vectorize)
                    PP[:,0] = nesting[PP[:,0]]
                else:
                    xx, yy = np.meshgrid(pivPos[:,0], chebpy.chebpts(n, self.domain[2:], prefy)[0])
                    colVals = evaluate(g, xx, yy, vectorize)
                
                if not resolvedRows:
                    m, nesting = gridRefine(m, prefx)
                    xx, yy = np.meshgrid(chebpy.chebpts(m, self.domain[:2], prefx)[0], pivPos[:,1])
                    rowVals = evaluate(g, xx, yy, vectorize)
                    PP[:,1] = nesting[PP[:,1]]
                else:
                    xx, yy = np.meshgrid(chebpy.chebpts(m, self.domain[:2], prefx)[0], pivPos[:,1])
                    rowVals = evaluate(g, xx, yy, vectorize)

                nn = len(pivotVal)

                for kk in range(nn):
                    colVals[:, kk+1:] = colVals[:, kk+1:] - colVals[:,kk] @ (rowVals[kk, PP[kk+1:nn,1]]/pivotVal[kk])
                    rowVals[kk+1:, :] = rowVals[kk+1:, :] - colVals[PP[kk+1:nn,0], kk] @ (rowVals[kk,:]/pivotVal[kk])

                # !!! Check if this is correct
                if nn == 1:
                    rowVals = rowVals.T

                # Are the columns and rows resolved now?
                if not resolvedCols:
                    colTech = chebpy.core.chebtech.Chebtech2.initvalues(values = np.sum(colVals,axis = 1), interval = self.domain[2:])
                    resolvedCols = chebpy.core.algorithms.happinessCheck(tech = colTech, vals = colVals, pref = prefy)

                if not resolvedRows:
                    rowTech = chebpy.core.chebtech.Chebtech2.initvalues(values =  np.sum(rowVals.T, axis = 1), interval = self.domain[:2])
                    resolvedRows = chebpy.core.algorithms.happinessCheck(tech = rowTech, vals = rowVals.T, pref = prefx)

                isHappy = resolvedRows and resolvedCols

                # Stop if degree is over maxLength
                sampleCheck = [m,n] < maxSample
                if not sampleCheck.all():
                    raise RuntimeWarning(f'CHEBFUN:CHEBFUN2:constructor:notResolved, Unresolved with maximum CHEBFUN length:{maxSample[np.argwhere(~sampleCheck)[0,0]]}')
                
        # !! Check if this is needed/correct
        if np.linalg.norm(colVals) == 0 or np.linalg.norm(rowVals) == 0:
            colVals = 0
            rowVals = 0
            pivotVal = np.inf
            pivotPos = np.array([0,0])
            isHappy = 1

        cols = Quasimatrix(data = chebpy.chebfun(colVals, domain = np.array(self.domain[2:]), prefs = prefy), transposed = False) 
        rows = Quasimatrix(data = chebpy.chebfun(rowVals.T, domain = np.array(self.domain[:2]), prefs = prefx), transposed = True)
        pivotValues = pivotVal
        pivotLocations = pivotPos

        # Write a Sample Test
        
        return cols, rows, pivotValues, pivotLocations
    
    # --------------------
    #  operator overloads
    # --------------------
    # This needs to be reimplemeneted bc there is a bug here
    # def __add__(self, g):
    #     if isinstance(g, int) or isinstance(g, float):
    #         g = Chebfun2(lambda x,y: float(g)*np.ones(x.shape), domain = self.domain, prefs = self.prefs)
        
    #     assert isinstance(g,Chebfun2), f"Addition/Subtraction between type {type(g) and type(self)} is not supported."

    #     # !!! Implement a check for a zero Chebfun2 (need to implement the same for a quasimatrix too)

    #     """
    #     Add Chebfun2 objects together by a compression algorithm:
    #     If A = XY^T and B = WZ^T, then A + B = [X W]*[Y Z]^T,
    #     [Qleft, Rleft] = qr([X W])
    #     [Qright, Rright] = qr([Y Z])
    #     A + B = Qleft * (Rleft * Rright') * Qright'
    #     [U, S, V] = svd( Rleft * Rright' )
    #     A + B = (Qleft * U) * S * (V' * Qright')     -> new low rank representation
    #     """
    #     f = self

    #     # Ensure that g has smaller pivot values.
    #     if np.min(np.abs(f.pivotValues)) < np.min(np.abs(g.pivotValues)):
    #         f,g = deepcopy(g),deepcopy(f)
        
    #     fScl = np.diag(1/f.pivotValues)
    #     gScl = np.diag(1/g.pivotValues)

    #     cols = Quasimatrix(deepcopy(np.hstack([f.cols.data, g.cols.data])), transposed = False)
    #     rows = Quasimatrix(deepcopy(np.hstack([f.rows.data, g.rows.data])), transposed = False)

    #     Z = np.zeros((len(fScl)+len(gScl), len(fScl)+len(gScl)))
    #     Z[:len(fScl),:len(fScl)] = fScl
    #     Z[-len(gScl):,-len(gScl):] = gScl

    #     Qcols, Rcols = cols.qr()
    #     Qrows, Rrows = rows.qr()

    #     U, S, Vt = np.linalg.svd(Rcols @ Z @ Rrows.T)

    #     Qcols = Qcols * U
    #     Qrows = Qrows * Vt
        
    #     # !!!! Check if this is okay
    #     # Compress the format if possible.
    #     idx = np.linalg.matrix_rank(np.diag((S)))
    #     Qcols = Qcols[:,:idx]
    #     Qrows = Qrows[:,:idx]
    #     S = S[:idx]

    #     return Chebfun2([Qcols, S, Qrows.T])

    # def __sub__(self, g):
    #     return deepcopy(self) + (-g)
    
    # def __rsub__(self, g):
    #     return g + (-deepcopy(self))
    
    # def __neg__(self):
    #     negative = deepcopy(self)
    #     negative.pivotValues = -negative.pivotValues
    #     return negative
    
    def __getitem__(self, key):
        """
        Evaluate a learned chebfun2 object on numeric, array, or meshgrid inputs
        """
        x, y = key
        if (isinstance(x,slice) and x == slice(None)) and (isinstance(y,slice) and y == slice(None)):
            return self
        
        C,D,R = self.cdr()

        if (isinstance(x,slice) and x == slice(None)):
            if (isinstance(y,int) or isinstance(y,float)):
                y = np.array([y])
            if y.dtype == np.int64 or y.dtype == np.float64:
                # Make evaluation points a vector.
                y = y.reshape(-1)
                return C[y,:] @ D * R
        
        if (isinstance(y,slice) and y == slice(None)):
            if (isinstance(x,int) or isinstance(x,float)):
                x = np.array([x])
            if x.dtype == np.int64 or x.dtype == np.float64:
                # Make evaluation points a vector.
                x = x.reshape(-1)
                return C * (D @ R[:,x])

        eps = np.finfo(np.float64).eps
        if isinstance(x,np.ndarray) and isinstance(y,np.ndarray) and min(x.shape) > 1 and x.shape == y.shape and len(x.shape) == 2:
            if np.max(x - x[0:1,:]) <= 10*eps and np.max(y - y[:,0:1]) < 10*eps:
                x, y = x[0,:], y[:,0]
                return C[y,:] @ D @ R[:,x]
            elif np.max(y - y[0:1,:]) <= 10*eps and np.max(x - x[:,0:1]) < 10*eps:
                x, y = x[:,0], y[0,:]
                return (C[y,:] @ D @ R[:,x]).T
            else:
                # Evaluate at matrices, but they are not from meshgrid:
                m, n = x.shape

                # Unroll the loop that is the longest
                if m > n:
                    out = np.array([np.sum(C[y[:,i],:] @ D * R[:,x[:,i]].T, axis = 1) for i in range(n)]).T
                else:
                    out = np.array([np.sum(C[y[j,:],:] @ D * R[:,x[j,:]].T, axis = 1) for j in range(m)])
                return out
        if (isinstance(x,int) or isinstance(x,float)) and (isinstance(y,int) or isinstance(y,float)):
            x, y = np.array([x]), np.array([y])
        if isinstance(x,np.ndarray) and isinstance(y,np.ndarray) and x.shape == y.shape and (len(x.shape) == 1 or min(x.shape) == 1):
            shape = x.shape
            x, y = x.reshape(-1), y.reshape(-1)
            return (C[y,:] * R[:,x].T @ np.diag(D)).reshape(shape)
        
        raise NotImplementedError('Cannot evaluate chebfun2 object with given inputs')
    
    # Plotting function for the Chebfun2 objects.
    def plot(self, fig = None, ax = None, **kwds):
        plt = import_plt()
        if plt:
            plt.ion()
            if fig is None:
                fig = plt.figure(figsize = kwds.pop("figsize",None))
            if ax is None:
                ax = plt.gca()

        xx = np.linspace(self.domain[0],self.domain[1],2000)
        yy = np.linspace(self.domain[2],self.domain[3],2000)
        x, y = np.meshgrid(xx,yy)
        G = self[x,y]
        cf = ax.contourf(x,y,G, 50, cmap = 'turbo', vmin = np.min(G), vmax = np.max(G))
        fig.colorbar(cf)


    # Return a CUR decomposition for the Chebfun2 object.
    def cdr(self):
        return self.cols, np.diag(1/self.pivotValues), self.rows
    
    # Return a Singular Value Expansion for the Chebfun2 object.
    def svd(self):
        C, D, R = self.cdr()
        
        # If the function is the zero function, then special care is required:
        if np.linalg.norm(np.diag(D)) == 0:
            width, height = np.diff(self.domain[2:]), np.diff(self.domain[:2])
            f = Chebfun2(lambda x,y: np.ones(x.shape), domain = self.domain, prefs = self.prefs)
            U = (1/np.sqrt(width)) * f.cols
            V = (1/np.sqrt(height)) * f.rows
            return U, 0, V
        
        Ql, Rl = C.qr()
        Qr, Rr = R.T.qr()
        U1, S, V1t = np.linalg.svd(Rl @ D @ Rr.T)
        U = Ql * U1
        V = (Qr * V1t.T).T
        
        return U, S, V
    
    # Truncate the Chebfun2 object to a rank K.
    def truncate(self, K):
        self.cols, self.pivotValues, self.rows = self.cols[:,:K], self.pivotValues[:K], self.rows[:K,:]
    
    def norm(self): # sqrt(integral of abs(F)^2)
        _, s, _ = self.svd()
        return np.sqrt(np.sum(s*s))
    
    def integralTransform(self, g):
        """
        Computes the integral transform on input function g (chebfun), considering the chebfun2, f,
        as a kernel.
        \\int f(x,y) g(x) dx
        """
        f = self
        cols = f.cols
        rows = f.rows
        fScl = np.diag(1/f.pivotValues)
        X = rows * Quasimatrix(data = [g], transposed = False)
        return (cols * (fScl @ X)).data[0]
    
    @property
    def T(self): # Transpose
        transpose = deepcopy(self)
        transpose.domain = np.array([transpose.domain[2],transpose.domain[3],transpose.domain[0],transpose.domain[1]])
        transpose.rows, transpose.cols = transpose.cols.T, transpose.rows.T
        return transpose
    
    @property
    def cornervalues(self): # Corner values of the Chebfun2 object
        xx, yy = np.meshgrid(self.domain[:2], self.domain[2:])  
        return self[xx,yy].reshape(-1)

    @property
    def rank(self): # Rank of the Chebfun2 object
        return len(self.pivotValues)

    @property
    def max(self): # Maximum value of the Chebfun2 object
        m, n = len(self.rows), len(self.cols)
        # Minmum samples = 9, Maximum samples = 2000
        m, n = max(min(m,9),2000), max(min(n,9),2000)
        prefx, prefy = self.prefs.prefx, self.prefs.prefy
        x, y = points2D(m, n, self.domain, prefx, prefy)
        return np.max(self[x,y])
    
    @property
    def min(self): # Minimum value of the Chebfun2 object
        m, n = len(self.rows), len(self.cols)
        # Minmum samples = 9, Maximum samples = 2000
        m, n = max(min(m,9),2000), max(min(n,9),2000)
        prefx, prefy = self.prefs.prefx, self.prefs.prefy
        x, y = points2D(m, n, self.domain, prefx, prefy)
        return np.min(self[x,y])
    
    @property
    def vscale(self): # Vertical scale of the Chebfun2 object
        m, n = len(self.rows), len(self.cols)
        # Minmum samples = 9, Maximum samples = 2000
        m, n = max(min(m,9),2000), max(min(n,9),2000)
        prefx, prefy = self.prefs.prefx, self.prefs.prefy
        x, y = points2D(m, n, self.domain, prefx, prefy)
        return np.max(np.abs(self[x,y]))

# Helper functions for the constructor
def Max(A):
    return np.max(A), np.argmax(A)

def points2D(m, n, dom, prefx, prefy):
    # Get the sample points that correspond to the right grid for a particular
    # technology.
    
    if prefy == None:
        prefy = prefx
        
    # What tech am I based on?
    techx, techy = prefx.tech, prefy.tech
    
    if techx == "Chebtech2":
        x = chebpy.chebpts(m, dom[:2])
    else:
        raise Exception('CHEBFUN:CHEBFUN2:constructor:points2D:tecType, Unrecognized technology')

    if techy == "Chebtech2":
        y = chebpy.chebpts(n, dom[2:])
    else:
        raise Exception('CHEBFUN:CHEBFUN2:constructor:points2D:tecType, Unrecognized technology')

    [xx, yy] = np.meshgrid(x,y)
    return xx, yy

def evaluate(op, xx, yy, vectorize = 0):
    # Wrap the function handle in a FOR loop if the vectorize flag is turned on.

    if(vectorize):
        vals = np.zeros((yy.shape[0], xx.shape[1]))
        for jj in range(yy.shape[0]):
            for kk in range(xx.shape[1]):
                vals[jj, kk] = op( xx(0,kk), yy(jj,0))
    else:
        vals = op(xx,yy)  # Matrix of values at cheb2 pts.
    return vals

def getTol(xx, yy, vals, dom, pseudoLevel):
    """
    Calculate a tolerance for the Chebfun2 constructor. This is the 2D analogue of the tolerance employed in the chebtech
    constructors. It is based on a finite difference approximation to the gradient, the size of the approximation domain,
    the internal working tolerance, and an arbitrary (2/3) exponent.
    """

    m, n = vals.shape
    grid = max(m,n)
    dfdx, dfdy = 0, 0

    if m > 1 and n > 1:
        # Remove some edge values so that df_dx and df_dy have the same size.
        dfdx = np.diff(vals[:m-1,:],1,1) / np.diff(xx[:m-1,:],1,1) # xx diffs column-wise.
        dfdy = np.diff(vals[:,:n-1],1,0) / np.diff(yy[:,:n-1],1,0) # yy diffs row-wise.
    elif m > 1 and n == 1:
        # Constant in x-direction
        dfdy = np.diff(vals,1,1) / np.diff(yy,1,1)
    elif m == 1 and n > 1:
        # Constant in y-direction
        dfdx = np.diff(vals,1,2) / np.diff(xx,1,2)

    # An approximation for the norm of the gradient over the whole domain.
    Jac_norm = max(np.max(np.abs(dfdx[:])), np.max(np.abs(dfdy[:])))
    vscale = np.max(np.abs(vals[:]))
    relTol = grid**(2/3) * pseudoLevel # This should be vscale and hscale invariant
    absTol = np.max(np.abs(dom[:])) * max(Jac_norm, vscale) * relTol

    return relTol, absTol

def gridRefine( grid, pref):
    # Hard code grid refinement strategy

    # What tech am I based on?:
    tech = pref.tech

    # What is the next grid size?
    if tech == "Chebtech2":
        # Double sampling on tensor grid:
        grid = np.power(2,np.floor(np.log2(grid))+1) + 1
        nesting = np.arange(1,grid+1,2)
    else:
        raise RuntimeError('CHEBFUN:CHEBFUN2:constructor:gridRefine:techType, Technology is unrecognized.')
    return grid, nesting

def completeACA(A, absTol, factor):
    """
    Adaptive Cross Approximation with complete pivoting. This command is the continuous analogue of Gaussian elimination with complete pivoting.
    Here, we aim to dynamically determine the numerical rank of the function.
    """

    # Set up output variables.
    nx, ny = A.shape
    width = min(nx, ny)        # Use to tell us how many pivots we can take.
    pivotValue = np.empty(int(np.ceil(width/factor)))         # Store an unknown number of Pivot values.
    pivotElement = np.empty((int(np.ceil(width/factor)),2), dtype = np.int64) # Store (j,k) entries of pivot location.
    pivotValue[:], pivotElement[:] = np.nan, -1
    ifail = 1                  # Assume we fail.

    # Main algorithm
    zRows = 0                  # count number of zero cols/rows.
    infNorm, ind = Max(np.abs(A))
    
    # Check here for errors bc python's default between row and column major is different.
    row, col = np.unravel_index(ind, A.shape) 
    # Error possiblity

    # Bias toward diagonal for square matrices (see reasoning below):
    if (nx == ny) and (np.max(np.abs(np.diag(A))) - infNorm) > -absTol:
        infNorm, ind = Max(np.abs(np.diag(A)))
        row = ind
        col = ind

    scl = infNorm

    # The function is the zero function.
    if scl == 0:
        """
        Return a zero matrix of the same size as A.
        
        This ensures that chebpy2(np.zeros(5)) returns a 5x5 zero coefficient matrix.
        """
        pivotValue = 0
        rows = np.zeros((1, A.shape[1]))
        cols = np.zeros((A.shape[0], 1))
        ifail = 0
    else:
        rows = np.zeros((1, A.shape[1]))
        cols = np.zeros((A.shape[0], 1))
    
    while (infNorm > absTol) and (zRows < width/factor) and (zRows < min(nx,ny)):
        # Check if zRows+1 is correct here
        if zRows == 0:
            rows = A[row,:].reshape((1,-1))
            cols = A[:,col].reshape((-1,1))
        else:
            rows = np.vstack([rows,A[row,:].reshape((1,-1))])
            cols = np.hstack([cols,A[:,col].reshape((-1,1))])            # Extract the columns.
        PivVal = A[row,col]
        A = A - cols[:,zRows].reshape(-1,1) @ (rows[zRows,:].reshape(1,-1)/PivVal) # One step of GE.

        # Keep track of progress.
        pivotValue[zRows] = PivVal                 #pivotValue[zRows] = PivVal             # Store pivot value.
        pivotElement[zRows,:] = [row, col]   #pivotElement[zRows,:]=[row col]        # Store pivot location.
        zRows = zRows + 1                         # One more row is zero.

        # Next pivot.
        infNorm , ind = Max(np.abs(A)) # Slightly faster.
        row , col = np.unravel_index(ind, A.shape) # Check for col/row major

        """
        Have a bias towards the diagonal of A, so that it can be used as a test for nonnegative definite functions.
        This bias is important as nonnegative definite functions have an absolute maximum on the diagonal. 
        Complete GE and Cholesky are the same in this context, but there is a possibility of a tie with an off-diagonal absolute maximum. 
        Biasing towards diagonal maxima helps prevent this scenario.
        """
        if (nx == ny) and ((np.max(np.abs(np.diag(A))) - infNorm) > -absTol):
            infNorm, ind = Max(np.abs(np.diag(A)))
            row = ind
            col = ind
    
    if infNorm <= absTol:
        ifail = 0                               # We didn't fail.

    if zRows >= (width/factor):
        ifail = 1                               # We did fail.
    
    return pivotValue[:zRows], pivotElement[:zRows,:], rows, cols, ifail