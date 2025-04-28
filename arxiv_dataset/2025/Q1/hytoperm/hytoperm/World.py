
# external imports
import warnings
import numpy as np
from typing import List, Dict, Set
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
import matplotlib.colors as colors
from scipy.spatial import ConvexHull

# internal imports
from .PyPlotHelpers.Plotters import *
from .PlotAttributes import PlotAttributes
_plotAttr = PlotAttributes()


class Domain:
    def __init__(self, xrange = [0,1.5], yrange = [0,1]):
        self._xmin = xrange[0]
        self._xmax = xrange[1]
        self._ymin = yrange[0]
        self._ymax = yrange[1]

    def xrange(self) -> List[float]:
        return [self._xmin, self._xmax]

    def yrange(self) -> List[float]:
        return [self._ymin, self._ymax]
    
    def xmin(self) -> float:
        return self._xmin
    
    def xmax(self) -> float:
        return self._xmax
    
    def ymin(self) -> float:
        return self._ymin
    
    def ymax(self) -> float:    
        return self._ymax


class Region:

    """
    Represents an abstract region that.
    """
    targetRegion : bool = False

    def region(self):
        return self
    
    def contains(self, x: np.ndarray, tol : float = 0) -> bool:
        """
        Checks if a point is contained within the region.
        
        Args:
            x: Point to be checked.
            tol: Tolerance for the check.

        Returns:
            True iff the point is contained within the region.
        """
        pass

    def violates(self, x: np.ndarray, tol : float = 0) -> List[int]:
        """
        Checks which constraints are violated at x.
        
        Args:
            x: Point to be checked.
            tol: Tolerance for the check.

        Returns:
            A list of violated constraints.
        """
        pass

    def distToBoundary(self, x: np.ndarray) -> float:
        """
        Computes the distance to the boundary of the region.

        Args:
            x: Point to be checked.

        Returns:
            The distance to the boundary.
        """
        pass

    def randomBoundaryPoint(self) -> np.ndarray:
        pass

    def projectToBoundary(self, x0, xf):
        '''
        This function projects a point onto the boundary of the region along the 
        ray xf - x0, where x0 is assumed to be inside the region.
        
        Args:
            x0: Initial point (within the region).
            xf: Final point provides the direction (xf - x0).
        '''
        pass

    def planPath(self, x0 : np.ndarray, xf : np.ndarray) -> List[np.ndarray]:
        '''
        This function plans a path between two points in the region.
        
        Args:
            x0: Initial point.
            xf: Final point.

        Returns:
            A list of waypoints.
        '''
        pass

    def travelCost(self, x0 : np.ndarray, xf : np.ndarray) -> float:
        """
        Computes the travel cost between two points in the region.

        Args:
            x0: Initial point.
            xf: Final point.
        """
        pass

    def plot(self, ax : plt.Axes = None, **kwargs) -> PlotObject:
        pass

    def fill(self, ax : plt.Axes = None, **kwargs) -> PlotObject:
        pass

    def getGrid(self, domain : Domain, dx = 0.05, dy=0.05, mdtb = 0.1):
        X = np.arange(domain.xmin(),domain.xmax(),dx)
        Y = np.arange(domain.ymin(),domain.ymax(),dy)
        V, W = np.meshgrid(X,Y)
        
        for i in range(V.shape[0]):
            for j in range(V.shape[1]):
                tester = np.array([V[i,j],W[i,j]])
                if not self.contains(tester):
                    V[i,j] = np.nan
                    W[i,j] = np.nan
                dist = self.distToBoundary(tester)
                if dist < mdtb:
                    V[i,j] = np.nan
                    W[i,j] = np.nan
                    
        return V, W

    def p(self) -> np.ndarray:
        pass

    def isObstacle(self) -> bool:
        return False
    
    def isTargetRegion(self) -> bool:
        return self.targetRegion

    def randomPoint(self, **kwargs) -> np.ndarray:
        pass


class Partition:
    def __init__(self, regions : Set[Region]) -> None:
        self._regions = regions

    def regions(self) -> Set[Region]:
        return self._regions

    def plot(self, ax : plt.Axes = None, **kwargs) -> PlotObject:
        ax = getAxes(ax)
        po = PlotObject()
        extended_kwargs = extendKeywordArgs(_plotAttr.partition.getAttributes())
        for r in self.regions():
            po.add(r.plot(ax, **extended_kwargs))
        return po
    

class CPRegion(Region):
    """
    Represents a convex polygon region defined by linear inequalities g*x <= b.

    Attributes:
        g:                      Linear constraint functions.
        b:                      Constraint bounds.
        p:                      A point in the region.
        domain:                 Domain of the region.
    """

    def __init__(
            self, 
            g,
            b,
            p : np.ndarray, 
            domain : Domain = Domain()
            ) -> None:
        """
        Initializes a Region object.

        Args:
            g (dict, optional): Dictionary of constraint gradients.
            b (dict, optional): Dictionary of constraint bounds.
            p (np.ndarray, optional): A point in the region.
        """
        
        self._g = {}
        self._b = {}
        self._ch : ConvexHull = None
        self._p : np.ndarray = None
        self._domain : Domain = None

        self.assignConstraints(g, b, domain)
        self.assignPoint(p)      

    def copy(self):
        return CPRegion(self.g(), self.b(), self.domain())

    def g(self) -> Dict[int, np.ndarray]:
        return self._g
    
    def b(self) -> Dict[int, float]:
        return self._b

    def p(self) -> np.ndarray:
        return self._p
    
    def domain(self) -> Domain:
        return self._domain

    def getConvexHull(self) -> ConvexHull:
        if self._ch is None:
            self.assignConvexHull()
        return self._ch
        
    def getOrthogonalConstraintNodes(self, dx):
        nodes = self.getConvexHull().vertices
        
        for i in range(len(nodes)):
            p = self.getConvexHull().points[nodes[i]]
            q = self.getConvexHull().points[nodes[i-1]]

            a = q - p
            normal = np.array([-a[1], a[0]])
            if abs(np.dot(a,dx)) < 1e-3 and np.dot(normal,dx) > 0:
                return p, q

        plt.plot(self.p()[0], self.p()[1], 'bo')            
        plt.quiver(self.p()[0],self.p()[1],dx[0],dx[1],angles='xy',color='blue')
        for i in range(len(nodes)):
            p = self.getConvexHull().points[nodes[i]]
            q = self.getConvexHull().points[nodes[i-1]]
            a = q - p

            plt.plot(p[0], p[1], 'bo')            
            plt.quiver(p[0], p[1], a[0], a[1], angles='xy', color='green')
        raise Exception("No orthogonal constraint found")
    
    def distToBoundary(self, p) -> float:
        # Project p onto g*x = b
        min_dist = np.inf
        for i in self.g().keys():
            g = self.g()[i]
            b = self.b()[i]
            dist = np.abs(np.dot(g,p) - b)/np.linalg.norm(g)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def randomBoundaryPoint(self) -> np.ndarray:        
        i = np.random.randint(0, len(self.getConvexHull().vertices))
        alpha = np.random.uniform(0, 1)
        vtx1 = self._ch.points[self._ch.vertices[i]]
        vtx2 = self._ch.points[self._ch.vertices[i-1]]
        return (1 - alpha) * vtx1 + alpha * vtx2

    def projectToBoundary(self, x0, xf):
        bdp = None

        if np.linalg.norm(xf - x0) < np.finfo(float).eps:
            return None
        
        # compute the normal vector of xf - x0
        n = np.array([-(xf[1] - x0[1]), xf[0] - x0[0]])/np.linalg.norm(xf - x0)
        h = np.dot(n, x0)
        closest = np.inf
        for i in self.g().keys():
            q = np.array([[self.g()[i][0], self.g()[i][1]], [n[0], n[1]]])
            r = np.array([self.b()[i], h])
            
            # Skip parallel constraints
            if (np.abs(np.linalg.det(q)) < np.finfo(float).eps):
                continue
            
            x = np.linalg.solve(q, r)

            # Skip non-forward directions
            if np.dot(x - x0, xf - x0) < np.finfo(float).eps:
                continue

            dist = np.linalg.norm(x - x0)
            if dist < closest:
                closest = dist
                bdp = x
        return bdp

    def assignConstraints(self, g : dict, b : dict, domain : Domain) -> None:
        """
        Assigns the constraint gradients and bounds of the region.

        Args:
            g: Constraint gradients.
            b: Constraint bounds.
        """
        if not isinstance(g, dict):
            raise ValueError("g must be a dictionary.")
        if not isinstance(b, dict):
            raise ValueError("b must be a dictionary.")
        if not isinstance(domain, Domain):
            raise ValueError("domain must be a Domain object.")
        if len(g) != len(b):
            raise ValueError("g and b must have the same length.")
        for i in g.keys():
            if i not in b.keys():
                raise ValueError("Keys of g and b must match.")
            if not isinstance(g[i], np.ndarray):
                raise ValueError("g must be a numpy array.")
            try:
                float(b[i])
            except:
                raise ValueError("b must be a float.")
        self._g = g.copy()
        self._b = b.copy()

        self.addConstraint(np.array([-1,0]), domain.xmin())
        self.addConstraint(np.array([1,0]), domain.xmax())
        self.addConstraint(np.array([0,-1]), domain.ymin())
        self.addConstraint(np.array([0,1]), domain.ymax())  

    def assignPoint(self, p : np.ndarray) -> None:
        if not isinstance(p, np.ndarray):
            raise ValueError("p must be a numpy array.")
        if not self.contains(p):
            raise ValueError("p must be contained in the region.")
        self._p = p

    def assignConvexHull(self) -> None:
        ip = self.computeIntersections()
        vertices = []
        for p in ip:
            if self.contains(p,tol=1e-10):
                vertices.append(p)
        if len(vertices) < 3:
            plt.plot(self.p()[0], self.p()[1], 'yo', markersize=5)
        self._ch = ConvexHull(vertices)

    def addConstraint(self, g : np.ndarray, b) -> None:
        """
        Adds a new constraint to the region.

        Args:
            g: Constraint gradient.
            b: Constraint bound.
        """
        self._g[len(self._g)+1] = g
        self._b[len(self._b)+1] = b

    def contains(self, x: np.ndarray, tol : float = 0) -> bool:
        for i in self.g().keys():
            g = self.g()[i]
            b = self.b()[i]
            if np.dot(g, x) > b + tol:
                return False
        return True
        
    def violates(self, x: np.ndarray, tol : float = 0) -> List[int]:
        violated = []
        for i in self.g().keys():
            g = self.g()[i]
            b = self.b()[i]
            viol = np.dot(g, x) - b + tol
            if viol > 0:
                violated.append([i, viol])
        return violated

    def distToBoundary(self, x: np.ndarray) -> float:
        dist = np.inf
        for i in self.g().keys():
            g = self.g()[i]
            b = self.b()[i]
            d = np.abs(np.dot(g, x)-b)/np.dot(g,g)
            if d < dist:
                dist = d
        return dist

    def computeIntersections(self) -> List[np.ndarray]:
        intersections = []
        for i in self.g().keys():
            for j in self.g().keys():
                if i >= j:
                    continue
                q = np.array([
                    [self.g()[i][0], self.g()[i][1]], 
                    [self.g()[j][0], self.g()[j][1]]
                    ])
                r = np.array([self.b()[i], self.b()[j]])
                
                # skip parallel constraints
                if (np.abs(np.linalg.det(q)) < 1e-14):
                    continue
                
                x = np.linalg.solve(q, r)
                intersections.append(x)
        return intersections
        
    def travelCost(self, x0 : np.ndarray, xf : np.ndarray) -> float:
        return np.linalg.norm(xf - x0)

    def planPath(self, x0 : np.ndarray, xf : np.ndarray) -> List[np.ndarray]:
        return [x0, xf]

    def plot(self, ax : plt.Axes = None, **kwargs) -> PlotObject:
        ax = getAxes(ax)
        chp = self.getConvexHull().points
        chv = self.getConvexHull().vertices
        xs : list = chp[chv,0].tolist()
        ys : list = chp[chv,1].tolist()
        xs.append(xs[0])
        ys.append(ys[0])
        if ax is None:
            ax = plt.gca()
        return PlotObject(ax.plot(xs, ys, **kwargs))
    
    def plotPoint(self, ax : plt.Axes = None, **kwargs) -> PlotObject:
        ax = getAxes(ax)
        return PlotObject(ax.plot(self.p()[0], self.p()[1], **kwargs))

    def fill(self, ax : plt.Axes = None, **kwargs) -> PlotObject:
        ax = getAxes(ax)
        chp = self.getConvexHull().points
        chv = self.getConvexHull().vertices

        if self.isObstacle():
            eka = extendKeywordArgs(
                _plotAttr.obstacle_background.getAttributes(), **kwargs
            )
        else:
            eka = extendKeywordArgs(
                _plotAttr.partition_background.getAttributes(), **kwargs
            )
        return PlotObject(ax.fill(chp[chv,0], chp[chv,1], **eka))

    def randomPoint(self, **kwargs) -> np.ndarray:
        nump = len(self.getConvexHull().points)
        alphas = []
        p = np.zeros(2)
        for i in range(nump-1):
            m = 1-sum(alphas)            
            if 'distribution' in kwargs and kwargs['distribution'] == 'uniform':
                alphas.append(np.random.uniform(0,m))
            else:
                m = 0.9*m
                alphas.append(max(0,min(np.random.normal(1/nump, 1/nump/10),m)))

            p += alphas[-1] * self.getConvexHull().points[i]
        alphas.append(1 - sum(alphas))
        p += alphas[-1] * self.getConvexHull().points[-1]
        return p

    def randomPointUniform(self) -> np.ndarray:
        nump = len(self.getConvexHull().points)
        alphas = []
        p = np.zeros(2)
        for i in range(nump-1):
            m = 0.9*(1-sum(alphas))
            alphas.append(np.random.uniform(0,m))
            p += alphas[-1] * self.getConvexHull().points[i]
        alphas.append(1 - sum(alphas))
        p += alphas[-1] * self.getConvexHull().points[-1]
        return p


class ObstacleCPRegion(CPRegion):

    def __init__(self, g, b, p : np.ndarray, domain : Domain = Domain()):
        super().__init__(g,b,p,domain)

    def travelCost(self, x0: np.ndarray, xf: np.ndarray) -> float:
        return np.inf

    def isObstacle(self) -> bool:
        return True


class Dynamics:
    def __init__(self, nx : int, nz : int, nu : int):
        self._nx : np.ndarray = None
        self._nz : np.ndarray = None
        self._nu : np.ndarray = None
        self.zx : np.ndarray = None
        self.zz : np.ndarray = None
        self.zu : np.ndarray = None

        self.setNX(nx)
        self.setNZ(nz)
        self.setNX(nu)

    def dynamics(self):
        return self

    def nx(self) -> int:
        return self._nx
    
    def nz(self) -> int:
        return self._nz
    
    def nu(self) -> int:
        return self._nu
    
    def setNX(self, nx : int) -> None:
        try:
            if int(nx) != nx or nx < 0:
                raise ValueError("Argument must be a nonnegative integer.")
        except:
            raise ValueError("Argument must be a nonnegative integer.")
        self._nx = int(nx)
        self.zx = self.zeroVec(nx)

    def setNZ(self, nz : int) -> None:
        try:
            if int(nz) != nz or nz < 0:
                raise ValueError("Argument must be a nonnegative integer.")
        except:
            raise ValueError("Argument must be a nonnegative integer.")
        self._nz = int(nz)
        self.zz = self.zeroVec(nz)

    def setNU(self, nu : int) -> None:
        try:
            if int(nu) != nu or nu < 0:
                raise ValueError("Argument must be a nonnegative integer.")
        except:
            raise ValueError("Argument must be a nonnegative integer.")
        self._nu = int(nu)
        self.zu = self.zeroVec(nu)

    def zeroVec(self, n):
        return np.zeros(n)

    def __call__(self, x, z, u):
        pass

    def eval(self,  x = None, z = None, u = None):
        if x is None:
            x = self.zx
        if z is None:
            z = self.zz
        if u is None:
            u = self.zu
        return self(x,z,u)
            
    def plotVectorField(
            self, 
            XY : list, 
            ax : plt.Axes, 
            scale = 1, 
            **kwargs) -> PlotObject:
        if len(XY) != 2:
            raise ValueError("XY must be a list of two numpy arrays.")
        if not all(isinstance(XY[i], np.ndarray) for i in range(2)):
            raise ValueError("XY must be a list of two numpy arrays.")
        X : np.ndarray = XY[0]; Y : np.ndarray= XY[1]
        
        DX = np.zeros(X.shape)
        DY = np.zeros(X.shape)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                x = X[i,j]
                y = Y[i,j]
                state = np.array([x,y])
                if x is np.nan or y is np.nan:
                    DX[i,j]= np.nan
                    DY[i,j]= np.nan
                    continue
                F = self.eval(state)
                DX[i,j]= F[0]
                DY[i,j]= F[1]
        vfAttr = _plotAttr.vector_field.getAttributes()
        eka = extendKeywordArgs(vfAttr, **kwargs)
        po = PlotObject(ax.quiver(X, Y, scale*DX, scale*DY, pivot='mid', **eka))
        return po


class ConstantDynamics(Dynamics):
    def __init__(self, nx : int, nz : int, nu : int, v : np.ndarray):
        super().__init__(nx,nz,nu)
        self._v : np.ndarray = None
        self.setV(v)

    def __call__(self, x, z, u):
        return self.v()

    def setV(self, v : np.ndarray) -> None:
        if not isinstance(v, np.ndarray):
            raise ValueError("Argument must be a numpy array.")
        self._v = v

    def v(self) -> np.ndarray:
        return self._v


class DynamicCPRegion(CPRegion):
    def __init__(self, g,b,p,domain,dynamics : Dynamics):
        super().__init__(g,b,p,domain)
        self._dynamics : Dynamics = None
        self.assignDynamics(dynamics)

    def assignDynamics(self, dynamics : Dynamics) -> None:
        if not isinstance(dynamics, Dynamics):
            raise ValueError("Argument must be a Dynamics object.")
        self._dynamics = dynamics

    def assignRegion(self, region : Region) -> None:
        if not isinstance(region, Region):
            raise ValueError("Argument must be a Region object.")
        self._region = region
    
    def dynamics(self) -> Dynamics:
        return self._dynamics


class ConstantDCPRegion(DynamicCPRegion):
    def __init__(self, g,b,p,domain,dynamics : ConstantDynamics):
        super().__init__(g,b,p,domain,dynamics)

    def dynamics(self) -> ConstantDynamics:
        return self._dynamics
    
    def travelCost(self, x0 : np.ndarray, xf : np.ndarray) -> float:
        '''
        This function determines the optimal travel time between two points in 
        the region. This is done by solving a quadratic equation obtained from 
        the root finding problem 
            u'*u = 1, u = (xf - x0)/t - v.

        Args:
            x0: Initial point.
            xf: Final point.

        Returns:
            float: The optimal travel cost between x0 and xf.
        '''
        v = self.dynamics().v()
        a = np.dot(v,v) - 1
        b = -2 * np.dot(xf - x0, v)
        c = np.dot(xf - x0, xf - x0)

        # final point is equal to initial point
        if c < np.finfo(float).eps:
            return 0
        
        # missing affine part (allows for factoring out t)
        if abs(a) < np.finfo(float).eps:
            if b >= 0:
                return np.inf
            else:
                return -c/b

        # general solution
        delta = pow(b,2) - 4 * a * c
        t_star = 0
        if (delta < 0):
            return np.inf
        
        t1 = (-b + np.sqrt(delta)) / (2 * a)
        t2 = (-b - np.sqrt(delta)) / (2 * a)
        if t1 < 0 and t2 < 0:
            return np.inf

        if t1 > 0 and t2 > 0:
            t_star = min(t1,t2)
        else:
            t_star = max(t1,t2)
        u_star = (xf - x0)/t_star - v
        return t_star

    def setV(self, v : np.ndarray) -> None:
        if not isinstance(v, np.ndarray):
            raise ValueError("Argument must be a numpy array.")
        self.dynamics().setV(v)

class Target:
    
    def __init__(
            self, 
            pos : np.ndarray, 
            region : Region = None, 
            phi0 : np.ndarray = None, 
            A : np.ndarray = None, 
            Q : np.ndarray = None
            ) -> None:
        
        self._r : Region = None                                                 # region
        self._p : np.ndarray = None                                             # position
        self._phi : np.ndarray = None                                           # internal state    
        self.A : np.ndarray = None                                              # internal state LTI term
        self.Q : np.ndarray = None                                              # internal state covariance of stochasticity
        
        self.name : str = None

        self.assignRegion(region)
        self.assignPosition(pos)
        self.assignStateMatrix(A)
        self.assignCovariance(Q)
        self.assignInternalState(phi0)
        
    def p(self) -> np.ndarray:
        return self._p

    def region(self) -> Region:
        return self._r
    
    def internalState(self) -> np.ndarray:
        return self._phi

    def getNumberOfStates(self) -> int:
        return self.A.shape[0]

    def assignPosition(self, p : np.ndarray) -> None:
        if not isinstance(p, np.ndarray):
            raise ValueError("Expected argument of type numpy.ndarray.")
        if self.region() is not None:
            if not self.region().contains(p):
                raise ValueError(
                    "Position must be contained in the target's region."
                    )
        self._p = p

    def assignRegion(self, r : Region) -> None:
        if not isinstance(r, Region) and r is not None:
            raise ValueError("Expected argument of type Region (or None).")
        self._r = r

    def assignInternalState(self, phi0 : np.ndarray) -> None:
        if phi0 is None:
            if self.A is not None:
                self._phi = np.zeros(self.A.shape[0])
                return
        if not isinstance(phi0, np.ndarray):
            raise ValueError("Expected argument of type numpy.ndarray.")
        self._phi = phi0

    def assignStateMatrix(self, A : np.ndarray) -> None:
        if not isinstance(A, np.ndarray):
            raise ValueError("Expected argument of type numpy.ndarray.")
        if (A.ndim == 1 and A.shape[0] == 1):
            self.A = A.reshape((1,1))
            return
        else:
            if A.ndim != 2:
                raise ValueError("Expected argument to have two dimensions.")
        self.A = A

    def assignCovariance(self, Q : np.ndarray) -> None:
        if not isinstance(Q, np.ndarray):
            raise ValueError("Expected argument of type numpy.ndarray.")
        if (Q.ndim == 1 and Q.shape[0] == 1):
            self.Q = Q.reshape((1,1))
            return
        else:
            if Q.ndim != 2:
                raise ValueError("Expected argument to have two dimensions.")
        self.Q = Q

    def plot(self, ax : plt.Axes = None, annotate=False, **kwargs) -> PlotObject:
        ax = getAxes(ax)
        eka = extendKeywordArgs(_plotAttr.target.getAttributes(), **kwargs)
        po = PlotObject(ax.plot(self._p[0], self._p[1], **eka))

        if annotate:
            dx = 0; dy = 0.05
            po.add(ax.text(self._p[0] + dx, self._p[1] + dy, self.name))

        return po


class World:
    
    def __init__(self, objs : List= [], domain : Domain = Domain()) -> None:
        
        self._regions : Set[Region] = []
        self._targets : List[Target] = []
        self._region_to_target : Dict[Region, Target] = {}
        self._target_to_region : Dict[Target, Region] = {}
        self._partition : Partition = None
        self._domain : Domain = domain

        self.setRegions(objs)
        self.setPartition()
        
    def setRegions(self, objs) -> None:
        for obj in objs:
            try:
                self.addRegion(obj.region())
            finally:
                pass
    
    def setTarget(self, objs) -> None:
        for obj in objs:
            try:
                self.addTarget(obj.target())
            finally:
                pass

    def addRegion(self, region : Region) -> None:
        if not isinstance(region, Region):
            raise ValueError("Expected argument of type Region.")
        if region not in self._regions:
            self._regions.append(region)

    def addTarget(self, target : Target) -> None:
        if not isinstance(target, Target):
            raise ValueError("Expected argument of type Target.")
        if target not in self._targets:
            self._targets.append(target)
            if target.region() is not None:
                target.region().targetRegion = True

    def setPartition(self) -> None:
        self._partition = Partition(self.regions()) 

    def addTargetRegion(self, target : Target, region : Region) -> None:
        self._targets.append(target)
        self._regions.append(region)
        self._target_to_region[target] = region
        self._region_to_target[region] = target
    
    # getters
    def regions(self) -> Set[Region]:
        return self._regions
    
    def targets(self) -> List[Target]:
        return self._targets
        
    def getTarget(self, idx : int) -> Target:
        if idx >= len(self._targets) or idx < 0:
            return None
        return self._targets[idx]
    
    def getTargetByName(self, name : str) -> Target:
        for t in self._targets:
            if t.name == name:
                return t
        return None

    def partition(self) -> Partition:
        return self._partition
    
    def target(self, i : int) -> Target:
        if i >= self.nTargets() or i < 0:
            raise IndexError("Index of target out of bounds.")
        return self._targets[i]
    
    def region(self, i : int) -> Region:
        if i >= self.nRegions() or i < 0:
            raise IndexError("Index of region out of bounds.")
        return self._regions[i]

    def nTargets(self) -> int:
        return len(self._targets)
    
    def nRegions(self) -> int:
        return len(self._regions)
    
    def nObstacles(self) -> int:
        n = 0
        for r in self._regions:
            if r.isObstacle():
                n+=1
        return n

    def domain(self) -> Domain:
        return self._domain
    
    def getRegion(self, p : np.ndarray, tol = 1e-10) -> Region:
        '''
        Get a region that contains the point p.

        Args:
            p: Point to be checked.
            tol: Tolerance for the check.

        Returns:
            None, if no region contains the point.
            Otherwise, a unique region that contains p. If p is contained in multiple regions, return one of those regions.
        '''
        regs = list(self.getRegions(p, tol))
        if len(regs) == 0:
            return None
        if len(regs) > 1:
            warnings.warn("Point is in multiple regions. Returning first region.")
            return regs[0]
        return regs[0]

    def getRegions(self, p : np.ndarray, tol = 1e-10) -> Set[Region]:
        regions = set()
        for r in self._regions:
            if r.contains(p, tol=tol):
                regions.add(r)
        return regions

    def getMeshgrid(self, dx = 0.005, dy = 0.005):
        xmin = self.domain().xmin()-0.5*dx
        xmax = self.domain().xmax()+0.5*dx
        ymin = self.domain().ymin()-0.5*dy
        ymax = self.domain().ymax()+0.5*dy
        x = np.arange(xmin, xmax, dx)
        y = np.arange(ymin, ymax, dy)
        X, Y = np.meshgrid(x, y)
        Z = np.nan*np.ones(X.shape)
        return X, Y, Z

    def hasTarget(self, region : Region) -> bool:
        for target in self._targets:
            if target.region() == region:
                return True
        return False

    # plotters
    def plotMissionSpace(
            self, 
            ax : plt.Axes = None, 
            add_target_labels=False, 
            fill_empty_regions=True,
            plot_partition=True,
            plot_targets=True,
            plot_vector_field=True
            ) -> PlotObject:
        ax = getAxes(ax)
        po = PlotObject()        

        if plot_partition:
            po.add(self.plotPartition(ax, fill_empty_regions))
        
        if plot_vector_field:
            po.add(self.plotVectorField(ax))
                
        if plot_targets:
            po.add(self.plotTargets(ax, add_target_labels))

        return po

    def plotTargets(self, ax : plt.Axes = None, add_target_labels=False) -> PlotObject:
        ax = getAxes(ax)
        po = PlotObject()
        for target in self._targets:
            po.add(target.plot(ax, annotate=add_target_labels))

        return po

    def plotVectorField(self, ax : plt.Axes = None) -> PlotObject:
        ax = getAxes(ax)
        po = PlotObject()
        for region in self.regions():
            if not hasattr(region, 'dynamics'):
                continue
            dynamics : Dynamics = region.dynamics()
            if not isinstance(dynamics, Dynamics):
                warnings.warn("Failed to extract Dynamics component.")
                continue
            d = 0.033
            po.add(
                dynamics.plotVectorField(
                    region.getGrid(self.domain(),dx=d,dy=d,mdtb=d), 
                    ax=ax, 
                    scale=0.6
                    )
                )
        return po

    def plotPartition(self, ax : plt.Axes = None, fill_empty_regions=True) -> PlotObject:
        ax = getAxes(ax)
        po = PlotObject()
        po.add(self.partition().plot(ax))
        for region in self.regions():
            if not self.hasTarget(region) and fill_empty_regions:
                po.add(region.fill(ax))
                continue
            if region.isObstacle():
                po.add(region.fill(ax))
        return po

    def plotDomain(self, ax : plt.Axes = None) -> PlotObject:
        ax = getAxes(ax)
        x = self.domain().xmin()
        y = self.domain().ymin()
        dx = self.domain().xmax() - self.domain().xmin()
        dy = self.domain().ymax() - self.domain().ymin()
        fc = colors.to_rgba(
            _plotAttr.partition_background.color, 
            _plotAttr.partition_background.alpha
            )
        ec = _plotAttr.partition.color
        lw = _plotAttr.partition.linewidth
        rect = patches.Rectangle((x,y), dx, dy, lw=lw, ec=ec, fc=fc)

        # Add the patch to the Axes
        return ax.add_patch(rect)
    
    def plotdistToBoundary(self, ax : plt.Axes = None) -> PlotObject:
        ax = getAxes(ax)
        X, Y, Z = self.getMeshgrid()
        for i in range(X.shape[0]):
            for j in range(Y.shape[1]):
                p = np.array((X[i,j], Y[i,j]))
                for region in self.regions():
                    if region.contains(p):
                        Z[i,j] = region.distToBoundary(p)
        return PlotObject(ax.contourf(X, Y, Z, antialiased=True, alpha=0.5))

    def plotTravelCostPerRegion(self, ax : plt.Axes = None) -> PlotObject:
        ax = getAxes(ax)
        X, Y, Z = self.getMeshgrid()
        for i in range(X.shape[0]):
            for j in range(Y.shape[1]):
                p = np.array((X[i,j], Y[i,j]))
                for region in self.regions():
                    if region.contains(p):
                        Z[i,j] = region.travelCost(p, region.p())
                        if Z[i,j] == np.inf:
                            Z[i,j] = -region.travelCost(region.p(), p)
        return PlotObject(ax.contourf(X, Y, Z, antialiased=True, alpha=0.5))
