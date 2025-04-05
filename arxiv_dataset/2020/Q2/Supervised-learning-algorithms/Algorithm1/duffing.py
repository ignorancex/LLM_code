import numpy as np
from numpy import linalg as LA

class Duffing:
    """
    Create a duffing object by specifying it's parameter delta as input at initialization of the object
    It is a bistable dynamical system with 2 stable steady states and one unstable steady state
    """

    def __init__(self, dl):
        self.delta = dl  # duffing parameter
        self.tau = 0.1  # parameter for bin_classifier()
        self.dt = 0.001  # time step
        self.control = 0  # initialize control as 0
        self.max_control = 4
        self.seed = np.random.seed(0)
        self.state = None
        self.desired_state = [1.0, 0.0]  # desired state, also a stable fixed point
        self.fp2 = [0.0, 0.0]  # unstable fixed point
        self.fp3 = [-1.0, 0.0]  # stable fixed point
        self.X = None
        self.U = None

    def reset(self):
        """
        :return: randomly initialized state of the duffing object
        """
        self.state = np.random.uniform(low=-4, high=4, size=(2,))
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
        :return: reward as the negative of the 2 norm between current state and the desired state
        """
        return -LA.norm(self.state - self.desired_state, axis=0)

    def bin_classifier(self):
        """
        :return: binary control (0 or 1) based on the locally weighted binary classifier
        """
        w = np.exp(-(LA.norm(self.state - self.X, axis=1)**2)/(2*self.tau))
        w /= np.sum(w)
        if np.dot(w, self.U) > 0.5:
            return 1
        else:
            return 0

    def dydt(self, y):
        """
        :param y: current state of the duffing oscillator
        :return: right hand side of duffing ODEs
        """
        dy0 = y[1] + self.control
        dy1 = y[0] - y[0]**3 - self.delta*y[1]
        return dy0, dy1

    def rk4(self, y0):
        """
        :param y0: current state of the duffing object
        :return: state y of the duffing object at next time step using 4th order Runge-Kutta method
        """
        h = self.dt
        f = self.dydt
        k1 = h * np.asarray(f(y0))
        k2 = h * np.asarray(f(y0 + k1 / 2))
        k3 = h * np.asarray(f(y0 + k2 / 2))
        k4 = h * np.asarray(f(y0 + k3))
        y = y0 + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return y

    def trajectory(self, n):
        """
        :param n: number of time steps in trajectory
        :return: trajectory y at time steps t and control u
        """
        y, u, t = np.zeros((n, 2)), np.zeros((n, 1)), np.zeros((n, 1))
        y[0, :] = self.state
        u[0, 0] = self.max_control * self.bin_classifier()
        for i in range(1, n):
            y[i, :] = self.step(u[i - 1, 0])
            t[i, 0] = i * self.dt
            u[i, 0] = self.max_control * self.bin_classifier()
        return y, u, t


    def trajectory_no_control(self, n):
        """
        :param n: number of time steps in trajectory
        :return: trajectory y at time steps t
        """
        y, t = np.zeros((n, 2)), np.zeros((n, 1))
        y[0, :] = self.state
        for i in range(1, n):
            y[i, :] = self.step(0)
            t[i, 0] = i * self.dt
        return y, t



