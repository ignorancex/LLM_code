import numpy as np
from scipy.integrate import solve_ivp
#Base class for plant models

class Dynamics:
    def __init__(self, initial_state):
        self.state = initial_state

    def dynamics(self, x, u):
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def output(self, x):
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def update(self, u):
        x = self.state
        f = self.dynamics
        g = self.output
        if self.type == 'CT':
            dt = self.dt
            if False:
                #CT integration
                xdot = lambda t,x: self.dynamics(x,u)
                sol = solve_ivp(xdot, [0, dt], x, method='RK45',rtol=1.e-8, atol=1.e-8)
                self.state = sol.y[:,-1]
            if True:
                #Simpler R4 implementation
                k1 = f(x, u)
                k2 = f(x + dt/2.0 * k1, u)
                k3 = f(x + dt/2.0 * k2, u)
                k4 = f(x + dt * k3, u)
                self.state = x + dt/6.0 * (k1 + 2*k2 + 2*k3 + k4)
            y = g(self.state, u)
        elif self.type == 'DT':
            self.state = f(x, u)
            y = g(self.state, u)
        return y

    def simulate(self, inputs):
        y = np.array([self.output(self.state, inputs[0])])
        for u in inputs:
            y = np.vstack((y, self.update(u)))
        return y

    