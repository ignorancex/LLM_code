
# external imports
import casadi as cad


class NLPSolver:
    def __init__(
            self, 
            prob = None, 
            w0 = None, 
            lbw = None, 
            ubw = None, 
            lbg = None, 
            ubg = None, 
            quiet = True
            ) -> None:
        self.solver = None
        self.lbw = None
        self.ubw = None
        self.lbg = None
        self.ubg = None
        self.w0 = None
        self.params = None
        self.print_level = 0 if quiet else 4
        self.initialize(prob, w0, lbw, ubw, lbg, ubg)

    def initialize(
            self, 
            prob = None, 
            w0 = None, 
            lbw = None, 
            ubw = None, 
            lbg = None, 
            ubg = None
            ):
        opts = {}
        opts['ipopt.print_level'] = self.print_level 
        opts['print_time'] = self.print_level
        self.solver = cad.nlpsol('solver', 'ipopt', prob, opts)
        self.lbw = cad.vertcat(*lbw)
        self.ubw = cad.vertcat(*ubw)
        self.lbg = cad.vertcat(*lbg)
        self.ubg = cad.vertcat(*ubg)
        self.w0 = cad.vertcat(*w0)

    def solve(self):
        return self.solver(
            x0=self.w0, 
            lbx=self.lbw, 
            ubx=self.ubw, 
            lbg=self.lbg, 
            ubg=self.ubg, 
            p=self.params
            )


class OptimizationParameters:
    def __init__(self) -> None:
        self.kkt_tolerance : float = 1e-1                                       # KKT tolerance
        self.alpha : float = 1.0                                                # step size for gradient descent    
        self.sigma : float = 1e-1                                               # constraint regularization parameter
        self.beta : float = 0.95                                                # step size reduction factor for gradient descent
        self.tr : float = 0.5                                                   # trust region radius
        self.sim_to_steady_state_tol : float = 1e-1                             # tolerance for simulation to steady state
        self.steady_state_iters : int = 1                                       # maximum number of iterations for steady state simulation
        self.optimization_iters : int = 100                                     # maximum number of iterations for optimization

    def copy(self):
        op = OptimizationParameters()
        op.kkt_tolerance = self.kkt_tolerance
        op.alpha = self.alpha
        op.sigma = self.sigma
        op.beta = self.beta
        op.tr = self.tr
        op.sim_to_steady_state_tol = self.sim_to_steady_state_tol
        op.steady_state_iters = self.steady_state_iters
        op.optimization_iters = self.optimization_iters
        return op
