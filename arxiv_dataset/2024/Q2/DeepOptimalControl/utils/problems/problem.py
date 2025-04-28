import numpy as np
from scipy.integrate import cumtrapz, solve_ivp


class ControlProblem:
    def __init__(self):
        self.N_states = None
        self.N_controls = None

        self.time_march_initial_tol = 1e-1
        self.data_tol = 1e-5
        self.max_nodes = 5000
        self.ode_solver = "BDF"
        self.device = "cpu"
        self.optimizer = "Adam"

        self.v_layers = 3
        self.v_units = 64
        self.learning_rate = 1e-3
        self.batch_size = 128

    def U_star(self, X_aug):
        """ Optimal controls as a function of augmented states. """
        raise NotImplementedError

    def make_U_NN(self, A):
        """ Build the graph of controls with gradient of Value function. """
        raise NotImplementedError

    def make_LQR(self, F, G, Q, Rinv, P1):
        """ Solves the Riccati ODE for this OCP. """
        GRG = G @ Rinv @ G.T
        def riccati_ODE(t, p):
            P = p.reshape(F.shape)
            PF = - P @ F
            dPdt = PF.T + PF - Q + P @ GRG @ P
            return dPdt.flatten()
        SOL = solve_ivp(riccati_ODE, [self.tT, 0.], P1.flatten(),
            dense_output=True, method='LSODA', rtol=1e-04)
        return SOL.sol

    def U_LQR(self, t, X):
        """ Evaluates the LQR controller for this OCP. """
        t = np.reshape(t, (1,))
        P = self.P(t).reshape(self.N_states, self.N_states)
        return self.RG @ P @ X

    def make_bc(self, X0):
        """ Return callable boundary condition. """
        raise NotImplementedError

    def running_cost(self, X, U, tf=None):
        """ Return Running cost. """
        raise NotImplementedError

    def terminal_cost(self, X):
        """ Return Terminal cost. """
        raise NotImplementedError

    def compute_cost(self, t, X, U, tf=None):
        """ Computes the accumulated cost of a state-control trajectory as
        an approximation of V(t). """
        L = self.running_cost(X, U)
        J = cumtrapz(L, t, initial=0.)
        return J[0,-1] - J + self.terminal_cost(X[:,-1])
    
    def Hamiltonian(self, t, X_aug, tf=None):
        """ Evaluates the Hamiltonian for this OCP. """
        U = self.U_star(X_aug)
        L = self.running_cost(X_aug[:self.N_states], U, tf)

        F = self.aug_dynamics(t, X_aug, np.array([tf]))
        H = L + np.sum(X_aug[self.N_states:2*self.N_states] * F[:self.N_states],
            axis=0, keepdims=True)

        return H

    def dynamics(self, t, X, U_fun):
        raise NotImplementedError

    def aug_dynamics(self, t, x):
        """ Augmented dynamics, dynamics of PMP. """
        raise NotImplementedError


