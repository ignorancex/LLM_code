      
# external imports
import warnings

# internal imports
from .World import *
from .Dynamics import *
from .GlobalPlanning import *
from .Optimization import *
from .Sensor import *
from .PlotAttributes import PlotAttributes
_plotAttr = PlotAttributes()


'''
omegaDot: the ODE right hand side of the state estimator's covariance matrix
'''
def omegaDot(p, Omega, target : Target, sensor : Sensor, inTargetRegion=False):
    A = target.A
    Q = target.Q
    H = sensor.getMeasurementMatrix(target)
    R_inv = sensor.getMeasurementNoiseInverse(target)
    unmonitored = Q + A @ Omega + Omega @ A.T
    if inTargetRegion:
        mf = sensor.getQualityFunction(target)(p, target.p())
        return unmonitored  - mf*mf*Omega @ H.T @ R_inv @ H @ Omega
    return unmonitored


'''
unmonitoredOmegaSimulator: build a parameterized casadi function that can be 
    utilized to simulate the estimator's covariance matrix while the target
    is not being monitored.
'''
def unmonitoredOmegaSimulator(
        target : Target, 
        sensor : Sensor,
        N : int
        ) -> cad.Function:
    # states
    no = target.getNumberOfStates()
    Omega = cad.SX.sym('Omega', no*no)
    nx = Omega.shape[0]
    
    # parameters
    tf = cad.SX.sym('tf', 1)
    Omega0 = cad.SX.sym('Omega0', no*no)
    params = cad.vertcat(tf, Omega0)

    # Model equations
    oDot = omegaDot(None, Omega, target, sensor, False)

    # Objective term
    L = 0
    for i in range(no):
        L += Omega[i*no + i]

    # Fixed step Runge-Kutta 4 integrator
    M = 4 # RK4 steps per interval
    DT = tf/N/M
    f = cad.Function('f', [Omega, params], [oDot, L])
    X0 = cad.SX.sym('X0', no*no)
    X = X0
    Q = 0
    for j in range(M):
        k1, k1_q = f(X, params)
        k2, k2_q = f(X + DT/2 * k1, params)
        k3, k3_q = f(X + DT/2 * k2, params)
        k4, k4_q = f(X + DT * k3, params)
        X=X+DT/6*(k1 +2*k2 +2*k3 +k4)
        Q = Q + DT/6*(k1_q + 2*k2_q + 2*k3_q + k4_q)
    F = cad.Function('F', [X0, params], [X, Q], ['x0','p'], ['xf','qf'])

    # Start with an empty NLP
    OmegaTrajectory=[]
    J = 0
    mse = [sum([Omega0[i*no + i] for i in range(no)])]
    
    # "Lift" initial conditions
    Xk = Omega0
    OmegaTrajectory += [Omega0]
    
    # Formulate the integrator
    for k in range(N):

        # Integrate till the end of the interval
        Fk = F(x0=Xk, p=params)
        Xk = Fk['xf']
        OmegaTrajectory += [Xk]
        J=J+Fk['qf']

        # Add Xk to Omega vec
        mse += [sum([Xk[i*no + i] for i in range(no)])]

    omegaSim = cad.Function(
        'OmegaSim', 
        [params], 
        [J, cad.vertcat(*mse), cad.vertcat(*OmegaTrajectory)], 
        ['p'], 
        ['Ik', 'mse', 'OmegaTrajectory']
    )

    return omegaSim


'''
cadToNumpy: transform a casadi SX object to a numpy array
'''
def cadToNumpy(x : cad.SX, nrow=None, ncol=None) -> np.ndarray:
    if nrow is None:
        return x.full().flatten()
    if ncol is None:
        return x.full().flatten().reshape(-1,nrow)
    return np.reshape(x.full().flatten(), (nrow, ncol), order='F')


'''
simulateUnmonitoredOmega: simulate the estimator's covariance matrix while the
    target is not being monitored. 

@param fun: the casadi function that simulates the estimator's covariance matrix
    (see unmonitoredOmegaSimulator)
'''
def simulateUnmonitoredOmega(fun : cad.Function, tf, Omega0):
    
    # set params and evaluate
    params = cad.vertcat(tf, np.matrix(Omega0).getA1())
    sim = fun(params)
    
    # get dimensions
    no = int(np.sqrt(len(Omega0)))
    N = int(len(sim[1].full().flatten())/no) - 1

    # generate trajectories
    t_grid = np.array([tf/N*k for k in range(N+1)]).flatten()
    Ik = sim[0].full().flatten()[0]
    mseTrajectory = Trajectory(cadToNumpy(sim[1], no, N+1), t_grid)
    omegaTrajectory = Trajectory(cadToNumpy(sim[2], no*no, N+1), t_grid)
    
    # gradient: dIk_dtf = loss function evaluated at the terminal time
    dIk_dtf = mseTrajectory.getEndPoint()

    return mseTrajectory, omegaTrajectory, Ik, dIk_dtf


'''
CovarianceParameters: A class that captures the covariance parameters for a
    target and sensor pair.
'''
class CovarianceParameters:
    def __init__(self, target : Target, sensor : Sensor, Omega0 : np.ndarray):
        self.A : np.ndarray = target.A
        self.Q : np.ndarray = target.Q
        self.H : np.ndarray = sensor.getMeasurementMatrix(target)
        self.R_inv : np.ndarray = sensor.getMeasurementNoiseInverse(target)
        self.sqf : SensingQualityFunction = sensor.getQualityFunction(target)


'''
SwitchingPoint: a point in the state space where the target switches regions
'''
class SwitchingPoint:
    def __init__(self, p : np.ndarray):
        self._p : np.ndarray = p

    def p(self) -> np.ndarray:
        return self._p

    def plot(self, ax : plt.Axes = None, **kwargs) -> PlotObject:
        ax = getAxes(ax)
        if kwargs.get('marker') is None:
            kwargs['marker'] = 'o'
        return PlotObject(ax.plot(self._p[0], self._p[1], **kwargs))


'''
SwitchingParameters: the parameters that define a switching segment, i.e., it is
    made up of start and end SwitchingPoints, a duration, and the initial 
    covariance matrix for all targets'.
'''
class SwitchingParameters:
    def __init__(
            self, 
            phi : SwitchingPoint,                                               # entrance point
            psi : SwitchingPoint,                                               # departure point
            tf : float,                                                         # duration
            Omega0 : Dict[Target, np.ndarray] = {},                             # initial covariance matrices for all targets
            N = 100                                                             # number of discretization nodes
            ) -> None:
        self._phi : SwitchingPoint = phi
        self._psi : SwitchingPoint = psi
        self._Omega0 : Dict[Target, np.ndarray] = Omega0
        self._tf : float = tf
        self._N = N


'''
TrajectorySegment: an abstract description of a trajectory segment. 
'''
class TrajectorySegment:
    def __init__(
            self, 
            ucs : Dict[Target, cad.Function],                                   
            params : SwitchingParameters = None                                 
            ) -> None:
        self.pTrajectory : Trajectory = None                                    # the agent's trajectory for this segment    
        self.uTrajectory : Trajectory = None                                    # the agent's control for this segment    
        self.mseTrajectories : Dict[Target, Trajectory] = {}                    # the estimator's mean squared error trajectory for each target
        self.params : SwitchingParameters = params                              # switching parameters for the segment
        self._cost : float = None                                               # the cost of the segment           
        self._gradient_tau = None                                               # the cost gradient with respect to trajectory duration
        self._ucs : Dict[Target : cad.Function] = ucs                           # an unmmonitored covariance simulator for each target
        self._cov_f : Dict[Target, np.ndarray] = {}

    def update(self) -> None:
        pass

    # getters
    def getStartPoint(self) -> np.ndarray:
        return self.pTrajectory.x[:,0]
    
    def getEndPoint(self) -> np.ndarray:
        return self.pTrajectory.x[:,-1]
    
    def getDuration(self) -> float:
        return self.pTrajectory.t[-1] - self.pTrajectory.t[0]

    def getCost(self) -> float:
        return self._cost
    
    def getGradient(self):
        return self._gradient_tau

    def getTerminalCovarianceMatrices(self) -> Dict[Target, np.ndarray]:
        return self._cov_f
        
    # modifiers
    def updateInitialCovarianceMatrices(
            self, 
            omega0 : Dict[Target, np.ndarray]
            ) -> None:
        self.params._Omega0 = omega0

    def updateTerminalCovarianceMatrix(
            self, 
            target : Target, 
            omega_f : np.ndarray
            ) -> None:
        self._cov_f[target] = omega_f

    def updateMSETrajectory(
            self, 
            target : Target, 
            mseTrajectory : Trajectory
            ) -> None:
        self.mseTrajectories[target] = mseTrajectory

    def shiftTime(self, t0 : float) -> None:
        self.pTrajectory.shiftTime(t0)
        self.uTrajectory.shiftTime(t0)
        for target in self.mseTrajectories.keys():
            self.mseTrajectories[target].shiftTime(t0)

    # plotters
    def plotInMissionSpace(self, ax : plt.Axes = None, **kwargs) -> PlotObject:
        ax = getAxes(ax)
        return self.pTrajectory.plotStateVsState(0, 1, ax, **kwargs)


'''
MonitoringController: a class that encapsulates the optimal monitoring control
'''
class MonitoringController:
    def __init__(self, target : Target, sensor : Sensor, N : int = 100) -> None:
        self.solver : NLPSolver = None   
        self.target : Target = target                                           # target to be monitored
        self.N : int = N                                                        # number of control intervals
        self.nx : int = None                                                    # number of states (assigned by builder)
        self.nu : int = None                                                    # number of controls (assigned by builder)
        self.no : int = None                                                    # number of target states (assigned by builder)
        self.add_region_constraints = False                                     # whether or not to include all region boundaries as constraints
        self.buildOptimalMonitoringSolver(target, sensor)

    def buildOptimalMonitoringSolver(
            self, 
            target : Target, 
            sensor : Sensor
            ) -> None:
        
        # input assertions
        region = target.region()
        if not hasattr(region, 'dynamics'):
            raise Exception("Expected target region to have a dynamics model.")
        if not isinstance(region, CPRegion):
            raise Exception(
                "Not implemented for regions other than CPRegion."
                )
        dynamics : ConstantDynamics = region.dynamics()   
        if not isinstance(dynamics, ConstantDynamics):
            raise Exception(
                "Not implemented for dynamics other than constant dyanmics."
                )

        # states
        no = target.getNumberOfStates()
        p = cad.SX.sym('p', 2)
        Omega = cad.SX.sym('Omega', no*no)
        x = cad.vertcat(p, Omega)
        nx = x.shape[0]

        # controls
        u = cad.SX.sym('u', 2)       
        nu = u.shape[0]
        
        # parameters
        phi = cad.SX.sym('phi', 2)
        psi = cad.SX.sym('psi', 2)
        tf = cad.SX.sym('tf', 1)
        Omega0 = cad.SX.sym('Omega0', no*no)
        params = cad.vertcat(phi, psi, tf, Omega0)

        T = tf                      # Time horizon
        N = self.N                  # number of control intervals

        # Model equations     
        pDot = dynamics.v() + u
        oDot = omegaDot(p, Omega, target, sensor, True)
        xDot = cad.vertcat(pDot, oDot)

        # Objective term
        L = 0
        for i in range(no):
            L += Omega[i*no + i]
            
        # Fixed step Runge-Kutta 4 integrator
        M = 4 # RK4 steps per interval
        DT = T/N/M
        f = cad.Function('f', [x, u, params], [xDot, L])
        X0 = cad.SX.sym('X0', nx)
        U = cad.SX.sym('U', nu)
        X = X0
        Q = 0
        for j in range(M):
            k1, k1_q = f(X, U, params)
            k2, k2_q = f(X + DT/2 * k1, U, params)
            k3, k3_q = f(X + DT/2 * k2, U, params)
            k4, k4_q = f(X + DT * k3, U, params)
            X=X+DT/6*(k1 +2*k2 +2*k3 +k4)
            Q = Q + DT/6*(k1_q + 2*k2_q + 2*k3_q + k4_q)
        F = cad.Function('F',[X0,U,params],[X,Q],['x0','u0','p'],['xf','qf'])

        # Region constraints
        region = target.region()
        g_constr = region.g()
        b_constr = region.b()
        g_constr_term = []
        for i in g_constr.keys():
            g_constr_term.append(cad.dot(g_constr[i], X0[0:2]) - b_constr[i])
        R = cad.Function('r',[X0],[cad.vertcat(*g_constr_term)],['x0'],['r'])

        lbx = -np.inf*np.ones(nx)
        ubx = -lbx
        lbu = -np.inf*np.ones(nu)
        ubu = -lbu
        x0 = np.zeros(nx)
        u0 = np.zeros(nu)

        # Start with an empty NLP
        w=[]
        w0 = []
        lbw = []
        ubw = []
        J = 0
        g=[]
        lbg = []
        ubg = []

        # "Lift" initial conditions
        Xk = cad.SX.sym('X0', nx)
        w += [Xk]
        lbw += [lbx]
        ubw += [ubx]
        w0 += [x0]

        # Initial constraints
        g += [cad.vertcat(Xk[0:2]-phi, Xk[2:] - Omega0)]
        lbg += [np.zeros(nx)]
        ubg += [np.zeros(nx)]
        
        # Formulate the NLP
        for k in range(N):
            # New NLP variable for the control
            Uk = cad.SX.sym('U_' + str(k), nu)
            w   += [Uk]
            lbw += [lbu]
            ubw += [ubu]
            w0  += [u0]

            # Integrate till the end of the interval
            Fk = F(x0=Xk, u0=Uk, p=params)
            Xk_end = Fk['xf']
            J=J+Fk['qf']

            # New NLP variable for state at end of interval
            Xk = cad.SX.sym('X_' + str(k+1), nx)
            w   += [Xk]
            lbw += [lbx]
            ubw += [ubx]
            w0  += [x0]

            # Add equality constraint
            g   += [Xk_end-Xk]
            lbg += [np.zeros(nx)]
            ubg += [np.zeros(nx)]

            # Add control bound
            g   += [cad.dot(Uk, Uk)]
            lbg += [-np.inf]
            ubg += [1]

            # Add region constraints
            if (self.add_region_constraints):
                g   += [R(Xk)]
                lbg += [-np.inf*np.ones(len(g_constr_term))]
                ubg += [np.zeros(len(g_constr_term))]

        # Terminal constraint
        g   += [Xk[0:2]-psi]
        lbg += [np.zeros(2)]
        ubg += [np.zeros(2)]

        # Create an NLP solver
        prob = {'f': J, 'x': cad.vertcat(*w), 'g': cad.vertcat(*g), 'p': params}
        
        # Allocate an NLP solver
        self.solver = NLPSolver(prob, w0, lbw, ubw, lbg, ubg, quiet=True)
        self.nx = nx
        self.nu = nu
        self.no = no
        
    def optimalMonitoringControl(
            self, 
            params : SwitchingParameters
            ) -> Tuple[Trajectory,Trajectory,Trajectory,Trajectory,float,float]:
        self.solver.params = cad.vertcat(
            *[params._phi.p(), 
              params._psi.p(), 
              np.array(params._tf), 
              np.matrix(params._Omega0[self.target]).getA1()]
        )
        sol = self.solver.solve()

        if not self.solver.solver.stats()['success']:
            raise Exception("Optimal monitoring control failed to converge...")
        w_opt = sol['x'].full()

        # Plot the solution
        nx = self.nx; nu = self.nu; no = self.no; N = self.N 
        p = np.zeros((2, N+1))
        omega = np.zeros((no*no, N+1))
        mse = np.zeros((1, N+1))
        u = np.nan*np.zeros((nu, N))
        p[0,:] = w_opt[0::nx+nu].flatten()
        p[1,:] = w_opt[1::nx+nu].flatten()
        for i in range(no*no):
            omega[i,:] = w_opt[2+i::nx+nu].flatten()
            if i % no == 0:
                mse[0,:] = mse[0,:] + omega[i,:]

        u[0,:] = w_opt[nx::nx+nu].flatten()
        u[1,:] = w_opt[nx+1::nx+nu].flatten()
        t_grid = np.array([params._tf/N*k for k in range(N+1)])

        pTrajectory = Trajectory(p, t_grid)
        mseTrajectory = Trajectory(mse, t_grid)
        omegaTrajectory = Trajectory(omega, t_grid)
        uTrajectory = Trajectory(u, t_grid[0:-1])
        f = sol['f'].full().flatten()[0]
        lam = sol['lam_p']

        # extract trajectories
        return pTrajectory, mseTrajectory, omegaTrajectory, uTrajectory, f, lam


'''
MonitoringSegment: monitoring a single target region
'''
class MonitoringSegment(TrajectorySegment):
    def __init__(
            self, 
            target : Target, 
            sensor : Sensor, 
            ucs : Dict[Target, cad.Function], 
            params : SwitchingParameters = None
            ) -> None:
        self._target : Target = target
        self._monitoring_controller : MonitoringController = None
        self._monitoring_controller = MonitoringController(
            target=target, 
            sensor=sensor, 
            N=params._N
        )
        super().__init__(ucs, params)

    def update(self):
        mc = self._monitoring_controller
        p, m, omega, u, Jk, duals_p = mc.optimalMonitoringControl(self.params)
        self.pTrajectory = p
        self.updateMSETrajectory(self._target, m)
        self.uTrajectory = u
        self.updateTerminalCovarianceMatrix(self._target, omega.getEndPoint())
        
        self._cost = Jk
        self._gradient_tau = -duals_p[4].full().flatten()
        for target in self._ucs.keys():
            if target == self._target:
                continue
            mse, Omega, Ik, dIk_dt = simulateUnmonitoredOmega(
                self._ucs[target], 
                self.params._tf, 
                self.params._Omega0[target]
            )
            self.updateMSETrajectory(target, mse)
            self.updateTerminalCovarianceMatrix(target, Omega.getEndPoint())
            self._cost += Ik
            self._gradient_tau += dIk_dt       


'''
SwitchingSegment: switching from one target region to the next
'''
class SwitchingSegment(TrajectorySegment):
    def __init__(
            self, 
            ucs : Dict[Target, cad.Function], 
            params : SwitchingParameters = None
            ) -> None:
        super().__init__(ucs, params)

    def update(self) -> None:
        self._cost = 0
        self._gradient_tau = 0

        # Don't need to update the control or path, but reset to 0 relative time
        t0 = self.uTrajectory.t[0]
        self.uTrajectory.shiftTime(-t0)
        self.pTrajectory.shiftTime(-t0)

        # update mse trajectories
        for target in self._ucs.keys():
            mse, Omega, Ik, dIk_dtf = simulateUnmonitoredOmega(
                self._ucs[target], 
                self.params._tf, 
                self.params._Omega0[target]
            )
            self.updateMSETrajectory(target, mse)
            self.updateTerminalCovarianceMatrix(target, Omega.getEndPoint())
            self._cost += Ik
            self._gradient_tau += dIk_dtf


'''
Cycle: a sequence of trajectory segments that make up a complete cycle
'''
class Cycle:
    def __init__(
            self, 
            ts : List[TrajectorySegment], 
            omega0 : Dict[Target, np.ndarray] = {}, 
            t0 : float = 0.0, 
            counter : int = 0
            ) -> None:
        self.pTrajectory : Trajectory = None
        self.uTrajectory : Trajectory = None
        self.mseTrajectories : Dict[Target, Trajectory] = {}
        self._trajectorySegments : List[TrajectorySegment] = ts
        self._covAtCycleStart : Dict[Target, np.ndarray] = omega0
        self._covAtCycleEnd : Dict[Target, np.ndarray] = None

        # statistics
        self._cycle_start : float = t0
        self._counter : int = counter

        # plot controls
        self._switchColor = 'blue'
        self._monitorColor = 'red'

    # getters
    def getDuration(self) -> float:
        return sum([ts.getDuration() for ts in self._trajectorySegments])

    def getStartTime(self) -> float:
        return self._cycle_start
    
    def getEndTime(self) -> float:
        return self._cycle_start + self.getDuration()

    def getCost(self) -> float:
        return sum(ts.getCost() for ts in self._trajectorySegments)
    
    def getGradient(self) -> np.ndarray:
        monSeg : List[TrajectorySegment] = []
        for ts in self._trajectorySegments:
            if isinstance(ts, MonitoringSegment):
                monSeg.append(ts)
        return np.array([ts.getGradient() for ts in monSeg]).flatten()

    def getInitialCovarianceMatrices(self) -> Dict[Target, np.ndarray]:
        return self._covAtCycleStart
    
    def getTerminalCovarianceMatrices(self) -> Dict[Target, np.ndarray]:
        return self._covAtCycleEnd.copy()
    
    def steadyState(self, tol=1e-2) -> bool:
        for target in self._covAtCycleStart.keys():
            if self._covAtCycleEnd is None:
                return False
            
            if not np.allclose(
                self._covAtCycleStart[target], 
                self._covAtCycleEnd[target], 
                atol=tol):
                return False
            
        return True
    
    # modifiers
    def simulate(self) -> None:
        t0 = self._cycle_start
        omega0 = self._covAtCycleStart.copy()
        self.clearTrajectories()
        for ts in self._trajectorySegments:
            ts.updateInitialCovarianceMatrices(omega0)
            ts.update()
            omega0 = ts.getTerminalCovarianceMatrices()
            ts.shiftTime(t0)
            t0 += ts.getDuration()

            self.appendTrajectorySegment(ts)
            
        self._covAtCycleEnd = omega0

    def updateInitialCovarianceMatrices(
            self, 
            omega0 : Dict[Target, np.ndarray]
            ) -> None:
        self._covAtCycleStart = omega0.copy()
    
    def shiftTime(self, deltaT : float = None) -> None:
        if deltaT is None:
            deltaT = -self.pTrajectory.t[0]
        self._cycle_start += deltaT
        for ts in self._trajectorySegments:
            ts.shiftTime(deltaT)
        if self.pTrajectory is not None:
            self.pTrajectory.shiftTime(deltaT)
            self.uTrajectory.shiftTime(deltaT)
            for target in self.mseTrajectories.keys():
                self.mseTrajectories[target].shiftTime(deltaT)

    def clearTrajectories(self) -> None:
        self.pTrajectory = None
        self.uTrajectory = None
        self.mseTrajectories = {}

    def appendTrajectorySegment(self, ts : TrajectorySegment) -> None:
        if self.pTrajectory is None:
            self.pTrajectory = Trajectory(ts.pTrajectory.x, ts.pTrajectory.t)
            self.uTrajectory = Trajectory(ts.uTrajectory.x, ts.uTrajectory.t)
            for target in ts.mseTrajectories.keys():
                self.mseTrajectories[target] = Trajectory(
                    ts.mseTrajectories[target].x, 
                    ts.mseTrajectories[target].t
                )
        else:
            self.pTrajectory.appendTrajectory(ts.pTrajectory)
            self.uTrajectory.appendTrajectory(ts.uTrajectory)
        for target in ts.mseTrajectories.keys():
            self.mseTrajectories[target].appendTrajectory(
                ts.mseTrajectories[target]
            )

    # plotters
    def plot(self, ax : plt.Axes = None, **kwargs) -> PlotObject:
        ax = getAxes(ax)
        eka = extendKeywordArgs(_plotAttr.agent.getAttributes(), **kwargs)
        return self.pTrajectory.plotStateVsState(0, 1, ax, **eka)
    
    def plotControls(
            self, 
            ax : plt.Axes = None, 
            add_monitoring_labels=True, 
            **kwargs
            ) -> PlotObject:
        ax = getAxes(ax)
        po = PlotObject()
        po.add(self.plotMonitoringControls(
            ax, 
            add_monitoring_labels=add_monitoring_labels, 
            **kwargs)
        )
        po.add(self.plotSwitchingControls(ax, **kwargs))
        return po
    
    def plotSwitchingControls(
            self, 
            ax : plt.Axes = None, 
            **kwargs
            ) -> PlotObject:
        ax = getAxes(ax)
        po = PlotObject()
        u1_pA = _plotAttr.u1_switch
        u2_pA = _plotAttr.u2_switch
        un_pA = _plotAttr.u_norm_switch
        eka1 = extendKeywordArgs(u1_pA.getAttributes(), **kwargs)
        eka2 = extendKeywordArgs(u2_pA.getAttributes(), **kwargs)
        eka3 = extendKeywordArgs(un_pA.getAttributes(), **kwargs)
        for ts in self._trajectorySegments:
            if not isinstance(ts, SwitchingSegment):
                continue
            u1 = ts.uTrajectory.x[0,:]
            u2 = ts.uTrajectory.x[1,:]
            u_norm = np.sqrt(np.square(u1)+np.square(u2))
            po.add(ax.plot(ts.uTrajectory.t, u_norm, **eka3))
            po.add(ts.uTrajectory.plotStateVsTime(0, ax, **eka1))
            po.add(ts.uTrajectory.plotStateVsTime(1, ax, **eka2))
        return po
    
    def plotMonitoringControls(
            self, 
            ax : plt.Axes, 
            add_monitoring_labels=True, 
            **kwargs
            ) -> PlotObject:
        po = PlotObject()
        u1_pA = _plotAttr.u1_monitor
        u2_pA = _plotAttr.u2_monitor
        un_pA = _plotAttr.u_norm_monitor
        eka1 = extendKeywordArgs(u1_pA.getAttributes(), **kwargs)
        eka2 = extendKeywordArgs(u2_pA.getAttributes(), **kwargs)
        eka3 = extendKeywordArgs(un_pA.getAttributes(), **kwargs)
        for ts in self._trajectorySegments:
            if not isinstance(ts, MonitoringSegment):
                continue

            if ts == self._trajectorySegments[-1]:
                eka1['label'] = '$u_1$'
                eka2['label'] = '$u_2$'
                eka3['label'] = '$\|u\|$'

            u1 = ts.uTrajectory.x[0,:]
            u2 = ts.uTrajectory.x[1,:]
            u_norm = np.sqrt(np.square(u1)+np.square(u2))
            po.add(ax.plot(ts.uTrajectory.t, u_norm, **eka3))
            po.add(ts.uTrajectory.plotStateVsTime(0, ax, **eka1))
            po.add(ts.uTrajectory.plotStateVsTime(1, ax, **eka2))
   
            if add_monitoring_labels:
                tText = np.median(ts.uTrajectory.t)
                uText = np.nanmedian(u_norm) + 0.1
                po.add(ax.text(tText, uText, ts._target.name))

        return po

    def plotTargetMSE(
            self, 
            target : Target,
            add_label : bool = False,
            ax : plt.Axes = None, 
            **kwargs
            ) -> PlotObject:
        ax = getAxes(ax)
        po = PlotObject()
        eka = kwargs.copy()
        if add_label:
            eka['label'] = target.name
        ext = extendKeywordArgs(eka, **kwargs)
        po.add(self.mseTrajectories[target].plotStateVsTime(0, ax, **ext))
        mseStart = self.mseTrajectories[target].getInitialValue() 
        po.add(ax.hlines(
            mseStart, 
            self._cycle_start, 
            self._cycle_start + self.getDuration(), 
            alpha=0.2, 
            **ext
            ))
        return po

    def plotMSE(
            self, 
            add_labels=False, 
            ax : plt.Axes = None, 
            **kwargs
            ) -> PlotObject:
        ax = getAxes(ax)
        po = PlotObject()
        i = 0
        for target in self.mseTrajectories.keys():
            ext = {'color': _plotAttr.target_colors[-i]}
            eka = extendKeywordArgs(ext, **kwargs)
            po.add(self.plotTargetMSE(target, add_labels, ax, **eka))
            i += 1
        return po


'''
Agent: the main agent class
'''
class Agent:
    def __init__(
            self, 
            world : World, 
            sensor : Sensor,
            gpp : GlobalPathPlanner = None,
            name : str = ""
            ) -> None:
        self.name = name
        self._world : World = world                                             # world instance    
        self._sensor : Sensor = sensor                                          # utilized sensor        
        self._gpp : GlobalPathPlanner = None                                    # global path planner
        self._tvs : List[Target] = []                                           # target visiting sequence
        self._cycle : Cycle = None                                              # the world instance    

        # optimization parameters
        self._tau : Dict[int, float] = {}                                       # map a target visit index to a monitoring duration
        self._lambda : Dict[int, float] = {}                                    # map a target visit index to a monitoring duration dual
        self._tau_min : Dict[int, float] = {}                                   # map a target visit index to a minimum monitoring duration
        self.op : OptimizationParameters = OptimizationParameters()             # optimization parameters
        
        # optimization statistics
        self._kkt_residuals : Dict[int, float] = {}                             # map a target visit index to a KKT residual
        self._global_costs : List[float] = []                                   # global cost (per steady state cycle) 
        self._global_gradient_norms : List[float] = []                          # global gradient norm (per steady state cycle) 
        self._tau_vals : List[np.ndarray] = []                                  # monitoring durations (per steady state cycle)
        self._kkt_violations : List[np.ndarray] = []                            # KKT residuals (per steady state cycle)
        self._steady_state_iters : List[int] = []                               # number of iterations to reach steady state
        self._alphas : List[float] = []                                         # step sizes (per steady state cycle)
        self._isSteadyState : List[bool] = []                                   # steady state reached (per steady state cycle)

        # decomposition
        self._switchingSegments : List[SwitchingSegment] = []                   # switching segments
        self._monitoringSegments : List[MonitoringSegment] = []                 # monitoring segments
        self._N = 100                                                           # number of discretization nodes per region

        # simulation functions for forward integration of covariance matrices
        self._ucs : Dict[Target, cad.Function] = {}                             # unmonitored covariance simulators

        self.initialize()
        self.setGlobalPathPlanner(gpp)

    def initialize(self) -> None:
        for target in self.world().targets():
            self._ucs[target] = unmonitoredOmegaSimulator(
                target, 
                self.sensor(), 
                self._N
                )
        
    def setGlobalPathPlanner(self, gpp : GlobalPathPlanner = None) -> None:
        if gpp is None:
            gpp = GlobalPathPlanner(self.world())
        if not isinstance(gpp, GlobalPathPlanner):
            raise ValueError("Expected a GlobalPathPlanner instance. Ignoring input.")
        self._gpp = gpp
   
    def computeVisitingSequence(self) -> None:
        self.gpp().solveTSP()
        self._tvs = self.gpp().tsp().getTargetVisitingSequence()

    def setTargetVisitingSequence(self, tvs : List[Target]) -> None:
        self._tvs = tvs
    
    def simulateCycle(self) -> None:
        
        try: 
            self.initializeCycle()
        except Exception as e:
            print(e)
            return
        
        self._cycle.simulate()
        
    def optimizeCycle(self) -> None:
        
        try: 
            self.initializeCycle()
        except Exception as e:
            print(e)
            return
        
        it = 0
        while True:
            steady, ssc = self.simulateToSteadyState(self.op.steady_state_iters)
            self._steady_state_iters.append(ssc)
            self._isSteadyState.append(steady)

            dJ_dt = self.updateMonitoringDurations()

            # Compute first order stationarity condition
            for i in range(len(self._tvs)):
                self._kkt_residuals[i] = dJ_dt[i] - self._lambda[i]
            self._kkt_violations.append(
                np.array(list(self._kkt_residuals.values()))
                )
            
            self.printIteration(it)

            o = np.max(np.abs(self._kkt_violations[-1])) < self.op.kkt_tolerance
            if o and steady:
                print("Optimal cycle found!")
                break

            if it > self.op.optimization_iters:
                print("Max iterations reached...")
                break

            it += 1
    
    def initializeCycle(self) -> None:
        
        if (len(self._tvs) <= 1):
            warnings.warn("Expected at least two targets in the visiting sequence. Did you run 'computeVisitingSequence()'?")
            return
                    
        # create lists of trajectory segments
        self._switchingSegments, self._tvs = self.initializeSwitchingSegments()
        self._monitoringSegments = self.initializeMonitoringSegments()
        trajectorySegments : List[TrajectorySegment] = []

        # combine trajectory segments
        for i in range(len(self._tvs)):
            trajectorySegments.append(self._switchingSegments[i])
            trajectorySegments.append(self._monitoringSegments[i])

        # assign cycle
        self._cycle = Cycle(
            trajectorySegments, 
            self.getInitialCovarianceMatrices()
            )

    def initializeSwitchingSegments(
            self
            ) -> Tuple[List[SwitchingSegment], List[Target]]:
        segments : List[SwitchingSegment] = []
        refined_tvs = []

        for i in range(len(self._tvs)):
            ot = self._tvs[i-1]
            ct = self._tvs[i]
            swPath : Tree = self.gpp().targetPath(ot, ct).getParent()

            while swPath:
                swSeg, target, swPath = self.extractSwitchingSegment(swPath)
                segments.append(swSeg)
                refined_tvs.append(target)
        
        return segments, refined_tvs

    def extractSwitchingSegment(
            self, 
            path : Tree
            ) -> Tuple[SwitchingSegment, Target, Tree]:
        '''
        Move up the tree and extract the switching points until a target region 
        is reached. We then return the switching segment together with the 
        remaining tree beginning from the first node where the active region is
        not the reached target region.
        '''
        
        node : Tree = path
        if node is None:
            warnings.warn("Path is empty. Returning None")
            return None, None, None
        
        phi = node.getData().p().reshape(-1,1)
        tf = np.array(0).reshape(1)
        u = np.nan*np.zeros((2,1))
        pTrajectory = Trajectory(phi, tf)
        uTrajectory = Trajectory(u, tf)
        ep = SwitchingPoint(phi)
        dp = ep
        while node is not None and not node.isRoot():

            # check if we have reached a target region
            for target in self.world().targets():
                if node.getData().activeRegionToParent() == target.region():
                    sp = SwitchingParameters(ep,dp,tf,N=self._N)
                    ts = SwitchingSegment(self._ucs, sp)
                    ts.pTrajectory = pTrajectory
                    ts.uTrajectory = uTrajectory
                    n = None
                    if node.getParent().hasParent():
                        n = node.getParent() 
                    return ts, target, n

            # otherwise add the next switching point
            psi = node.getParent().getData().p().reshape(-1,1)
            dp = SwitchingPoint(psi)
            deltaT = node.getData().costToParent()
            
            # TODO(Jonas): Hacky solution currently in place
            # What I need here is the control from one node to the next
            # In my current setting (constant Dynamics on the regions, this is 
            # a constant control law). In general doesn't need to be... 
            # So really, should have a trajectory to parent stored in the node 
            # or even better in an edge between the two nodes...
            artp : DynamicCPRegion = node.getData().activeRegionToParent()
            dyn : ConstantDynamics = artp.dynamics()
            v = dyn.v().reshape(-1,1)            
            u = (psi - phi)/deltaT - v
            
            # update trajectories
            uTrajectory.extend(u, tf)
            uTrajectory.extend(u, tf + deltaT)
            pTrajectory.extend(psi, tf + deltaT)

            # update time
            tf = tf + deltaT
            phi = psi

            # get next node
            node = node.getParent()

        raise Exception("Should not be reached ...")

    def initializeMonitoringSegments(self) -> List[MonitoringSegment]:
        segments : List[MonitoringSegment] = []
        for i in range(len(self._tvs)):
            target = self._tvs[i]
            phi = self._switchingSegments[i].getEndPoint()
            nextIdx = (i+1) % len(self._switchingSegments)
            psi = self._switchingSegments[nextIdx].getStartPoint()
            tf = max(0.1, 1.1*target.region().travelCost(phi, psi))
            self._tau[i] = tf
            self._tau_min[i] = target.region().travelCost(phi, psi)

            params = SwitchingParameters(
                SwitchingPoint(phi), 
                SwitchingPoint(psi), 
                tf, 
                N=self._N
                )
            segment = MonitoringSegment(target,self.sensor(),self._ucs,params)
            segments.append(segment)
        return segments

    def simulateToSteadyState(self, maxIter = 100) -> Tuple[bool, int]:
        it = 0
        omega_f = self._cycle.getInitialCovarianceMatrices()
        while True:
            it += 1

            self._cycle.simulate()
            omega_f = self._cycle.getTerminalCovarianceMatrices()
            
            cycle_average_cost = self._cycle.getCost()/self._cycle.getDuration()
            self._global_costs.append(cycle_average_cost)
            
            isSteady = self._cycle.steadyState(tol=self.op.sim_to_steady_state_tol)
            self._cycle.updateInitialCovarianceMatrices(omega_f)

            if isSteady:
                return True, it

            if it >= maxIter:
                return False, it
            
            self._cycle._cycle_start += self._cycle.getDuration()
    
    def globalCostGradient(self):
        T = self._cycle.getDuration()
        J = self._cycle.getCost()
        dJ_dt = self._cycle.getGradient()
        
        # global average cost gradient
        return (dJ_dt * T - J * np.ones(len(dJ_dt)))/T**2

    def updateMonitoringDurations(self) -> None:
        '''
        Simple projected gradient descend
        '''
        dJ_dt = self.globalCostGradient()
        self._global_gradient_norms.append(np.linalg.norm(dJ_dt, ord=np.inf))
        if self._global_gradient_norms[-1] > self.op.tr:
            dJ_dt = dJ_dt * self.op.tr / self._global_gradient_norms[-1]
        
        self._tau_vals.append(
            np.array([self._tau[i] for i in range(len(self._tvs))])
            )
        for i in range(len(self._tvs)):
            if self._tau[i] == self._tau_min[i] and dJ_dt[i] > 0:
                self._lambda[i] = dJ_dt[i]
            else:
                self._lambda[i] = 0
                self._tau[i] = max(
                    self._tau_min[i] + self.op.sigma, 
                    self._tau[i] - self.op.alpha * dJ_dt[i]
                    )
            self._monitoringSegments[i].params._tf = self._tau[i]
        self.op.alpha *= self.op.beta
        self._alphas.append(self.op.alpha)
        return dJ_dt

    # Getters
    def gpp(self) -> GlobalPathPlanner:
        return self._gpp   

    def world(self) -> World:
        return self._world

    def sensor(self) -> Sensor:
        return self._sensor
    
    def getInitialCovarianceMatrices(self) -> Dict[Target, np.ndarray]:
        Omegas = {}
        for target in self.world().targets():
            Omegas[target] = np.eye((target.getNumberOfStates()))
        return Omegas        
    
    # Plotters
    def plotMSE(
            self, 
            ax : plt.Axes = None, 
            add_labels = False, 
            **kwargs
            ) -> PlotObject:
        ax = getAxes(ax)
        return self._cycle.plotMSE(add_labels=add_labels, ax=ax, **kwargs)
    
    def plotControls(
            self, 
            ax : plt.Axes = None, 
            add_monitoring_labels = False, 
            **kwargs
            ) -> PlotObject:
        ax = getAxes(ax)
        return self._cycle.plotControls(
            ax, 
            add_monitoring_labels=add_monitoring_labels, 
            **kwargs
            )

    def plotCycle(self, ax : plt.Axes = None, **kwargs) -> PlotObject:
        ax = getAxes(ax)
        po = PlotObject()
        self._cycle.plot(ax, **kwargs)
       
        for ms in self._monitoringSegments:
            p = ms.pTrajectory.x[:,-2]
            q = ms.pTrajectory.x[:,-1]
            po.add(self.plotArrow(p, q, ax, **kwargs))

        return po

    def plotArrow(self, p : np.array, q : np.array, ax : plt.Axes = None, **kwargs) -> PlotObject:
        ax = getAxes(ax)
        po = PlotObject(ax.arrow(
            p[0], 
            p[1], 
            q[0] - p[0], 
            q[1] - p[1], 
            width=0, 
            head_width = 0.02, 
            head_length=0.033, 
            overhang=0.3, 
            length_includes_head=True,**kwargs
            ))
        return po
        
    def plotMonitoringSegments(
            self,
            ax : plt.Axes = None, 
            **kwargs
            ) -> PlotObject:
        ax = getAxes(ax)
        po = PlotObject()
        for ms in self._monitoringSegments:
            po.add(ms.pTrajectory.plotStateVsState(0, 1, ax, **kwargs))
        return po
    
    def plotEntryPoints(self, ax : plt.Axes = None, **kwargs) -> PlotObject:
        ax = getAxes(ax)
        po = PlotObject()
        eka = extendKeywordArgs(_plotAttr.phi.getAttributes(), **kwargs)
        for ms in self._monitoringSegments:
            po.add(ms.params._phi.plot(ax, **eka))
        return po
    
    def plotDeparturePoints(self, ax : plt.Axes = None, **kwargs) -> PlotObject:
        ax = getAxes(ax)
        po = PlotObject()
        eka = extendKeywordArgs(_plotAttr.psi.getAttributes(), **kwargs)
        for ms in self._monitoringSegments:
            po.add(ms.params._psi.plot(ax, **eka))
        return po

    def plotSwitchingPoints(self, ax : plt.Axes = None, **kwargs) -> PlotObject:
        ax = getAxes(ax)
        po = PlotObject()
        for ms in self._monitoringSegments:
            eka = extendKeywordArgs(_plotAttr.phi.getAttributes(),**kwargs)
            po.add(ms.params._phi.plot(ax, **eka))
            
            eka = extendKeywordArgs(_plotAttr.psi.getAttributes(),**kwargs)
            po.add(ms.params._psi.plot(ax, **eka))
        return po

    def plotSwitchingSegments(
            self, 
            ax : plt.Axes = None, 
            **kwargs
            ) -> PlotObject:
        ax = getAxes(ax)
        po = PlotObject()
        for ss in self._switchingSegments:
            po.add(ss.pTrajectory.plotStateVsState(0, 1, ax, **kwargs))
        return po

    def plotGlobalCosts(
            self, 
            ax : plt.Axes = None, 
            **kwargs
            ) -> PlotObject:
        ax = getAxes(ax)
        return PlotObject(ax.plot(self._global_costs, **kwargs))
    
    def plotGlobalGradientNorms(
            self, 
            ax : plt.Axes = None, 
            **kwargs
            ) -> PlotObject:
        ax = getAxes(ax)
        return PlotObject(ax.plot(self._global_gradient_norms, **kwargs))
    
    def plotTauVals(
            self, 
            ax : plt.Axes = None, 
            add_lower_bounds = True, 
            **kwargs
            ) -> PlotObject:
        ax = getAxes(ax)
        po = PlotObject()
        tv = np.array(self._tau_vals)
        for i in range(tv.shape[1]):
            eka = extendKeywordArgs(
                {'color': _plotAttr.target_colors[-i]}, 
                **kwargs
                )
            po.add(ax.plot(tv[:,i], **eka))

            if add_lower_bounds:
                eka = extendKeywordArgs(
                    {'alpha' : 0.75, 'linestyle' : '--'}, 
                    **eka
                    )
                po.add(ax.hlines(self._tau_min[i], 0, len(tv[:,i])-1, **eka))
        return po

    def plotKKTViolations(self, ax : plt.Axes = None, **kwargs) -> PlotObject:
        ax = getAxes(ax)
        po = PlotObject()
        po.add(PlotObject(ax.plot(self._kkt_violations, **kwargs)))
        po.add(PlotObject(ax.axhspan(
            -self.op.kkt_tolerance, 
            self.op.kkt_tolerance, 
            alpha=0.2, 
            color='green'))
            )
        return po

    def plotAlphas(self, ax : plt.Axes = None, **kwargs) -> PlotObject:
        ax = getAxes(ax)
        return PlotObject(ax.plot(self._alphas, **kwargs))

    def plotSensorQuality(
            self, 
            grid_size = 0.005, 
            ax : plt.Axes = None, 
            **kwargs
            ) -> PlotObject:
        ax = getAxes(ax)
        X, Y, Z = self._world.getMeshgrid(dx=grid_size, dy=grid_size)
        sensor = self.sensor()
        for i in range(X.shape[0]):
            for j in range(Y.shape[1]):
                p = np.array((X[i,j], Y[i,j]))
                sensor.setPosition(p)
                for target in self._world.targets():
                    region = target.region()
                    if region.contains(p):
                        Z[i,j] = sensor.getSensingQuality(target)
        sqAttr = _plotAttr.sensor_quality.getAttributes()
        eka = extendKeywordArgs(sqAttr, **kwargs)
        cf = ax.contourf(X, Y, Z, **eka)
        # plt.colorbar(res)
        return PlotObject(cf)
    
    def addSteadyStateLines(self, ax : plt.Axes = None, **kwargs) -> PlotObject:
        ax = getAxes(ax)
        cumsum = np.cumsum(self._steady_state_iters)
        po = PlotObject()
        for i in range(len(cumsum)):
            po.add(ax.axvline(cumsum[i], **kwargs))
        return po

    # printers
    def printHeader(self) -> None:
        print("----|-----------|-----------|-----------|--------|--------")
        print(" it | avrg cost | grad. nrm | step size | it std | is std ")
        print("----|-----------|-----------|-----------|--------|--------")
              
    def printIteration(self, it) -> None:
        if it % 10 == 0:
            self.printHeader()
        print("{:3d} | {:9.2e} | {:9.2e} | {:9.2e} | {:6d} | {:>6s}".format(
            it, 
            self._global_costs[-1], 
            self._global_gradient_norms[-1], 
            self._alphas[-1], 
            self._steady_state_iters[-1],
            'T' if self._isSteadyState[-1] else 'F'
        ))
