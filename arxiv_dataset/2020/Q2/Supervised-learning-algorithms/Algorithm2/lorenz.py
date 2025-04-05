import numpy as np
from numpy import linalg as LA

class Lorenz:
    """
    Create a lorenz object by specifying it's parameters as input at initialization of the object.
    For the considered parameters, lorenz system is a bistable dynamical system with 2 stable steady states
    and one unstable steady state
    """

    def __init__(self, para1, para2, para3):
        self.sigma = para1  # lorenz parameter
        self.r = para2  # lorenz parameter
        self.b = para3  # lorenz parameter
        self.tau = 10  # parameter for bin_classifier()
        self.dt = 0.001  # time step
        self.control = 0  # initialize control as 0
        self.max_control = 10
        self.seed = np.random.seed(1)
        self.state = None
        self.desired_state = [0.0, 0.0, 0.0]  # desired state an unstable fixed point
        self.X = None
        self.U = None

    def reset(self):
        """
        :return: randomly initialized state of the lorenz object
        """
        self.state = np.random.uniform(low=-5, high=5, size=(3,))
        return self.state

    def step(self, u):
        """
        takes input u as the action/control to be applied
        calculates next state by calling 4th order Runge-Kutta solver
        returns state at the next time step
        """
        y = self.state
        self.control = u
        new_y = self.rk4(y)
        self.state = new_y
        return self.state

    def reward(self):
        """
        :return: returns reward as the negative of the 2 norm between current state and the desired state
        """
        return -LA.norm(self.state - self.desired_state, axis=0)

    def bin_classifier(self):
        """
        :return: binary control based on the locally weighted binary classifier
        """
        w = np.exp(-((LA.norm(self.state - self.X, axis=1))**2)/(2*self.tau))
        w /= np.sum(w)
        if np.dot(w, self.U) > 0:
            return self.max_control
        else:
            return -self.max_control

    def lyapunov_control(self):
        """
        :return: lyapunov control output
        """
        return -(self.sigma + self.r) * self.state[1]

    def dydt(self, y):
        """
        :param y: current state of the lorenz object
        :return: right hand side of lorenz ODEs
        """
        dy0 = self.sigma*(y[1] - y[0]) + self.control
        dy1 = self.r*y[0] - y[1] - y[0]*y[2]
        dy2 = y[0]*y[1] - self.b*y[2]
        return dy0, dy1, dy2

    def rk4(self, y0):
        """
        :param y0: current state of the lorenz object
        :return: state y of the lorenz object at next time step using 4th order Runge-Kutta method
        """
        h = self.dt
        f = self.dydt
        k1 = h * np.asarray(f(y0))
        k2 = h * np.asarray(f(y0 + k1 / 2))
        k3 = h * np.asarray(f(y0 + k2 / 2))
        k4 = h * np.asarray(f(y0 + k3))
        y = y0 + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return y

    def trajectory(self, n, c):
        """
        :param n: number of time steps in trajectory
        :param c: integer parameter to determine if control is Lyapunov, or learning based
        :return: trajectory x at time steps t and control u
        """
        y, u, t = np.zeros((n, 3)), np.zeros((n, 1)), np.zeros((n, 1))
        y[0, :] = self.state
        if c == 0:
            f = self.bin_classifier
        else:
            f = self.lyapunov_control
        u[0, 0] = f()
        for i in range(1, n):
            y[i, :] = self.step(u[i - 1, 0])
            t[i, 0] = i * self.dt
            u[i, 0] = f()
        return y, u, t


    def trajectory_no_control(self, n):
        """
        :param n: number of time steps in trajectory
        :return: trajectory x at time steps t
        """
        y, t = np.zeros((n, 3)), np.zeros((n, 1))
        y[0, :] = self.state
        for i in range(1, n):
            y[i, :] = self.step(0)
            t[i, 0] = i * self.dt
        return y, t



