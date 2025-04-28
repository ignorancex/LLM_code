import numpy as np
from plant_models.plant_model import Dynamics

class trigonometric(Dynamics):
    def __init__(self, initial_state, 
                qx = 0.01, qy = 0.01, 
                u_lim = 0.5, type='DT'):
        
        super().__init__(initial_state)
        self.qx = qx
        self.qy = qy
        self.u_lim = u_lim
        self.a = np.array([0.5,0.6,0.4]) #Could be random
        self.b = np.array([1.7,0.4,0.9])
        self.c = np.array([2.2,1.8,-1.])
        self.type = type
        self.nx_plant = 3
        self.nu_plant = 1

    def dynamics(self, x, u):
        a = self.a
        b = self.b
        qx = self.qx
        x0 = a[0]*np.sin(x[0]) + b[0]*u[0]*np.cos(0.5*x[1]) + qx*np.random.randn(1)
        x1 = a[1]*np.sin(x[0]+x[2]) + b[1]*u[0]*np.arctan(x[0]+x[1]) + qx*np.random.randn(1)
        x2 = a[2]*np.exp(-x[1]) + b[2]*u[0]*np.sin(-x[0]/2.) + qx*np.random.randn(1)
        return np.array([x0[0], x1[0], x2[0]])
    
    def output(self, x, u):
        c = self.c
        qy = self.qy
        y = np.arctan(c[0]*x[0]**3) + np.arctan(c[1]*x[1]**3) + np.arctan(c[2]*x[2]**3) + qy*np.random.randn(1)
        return np.array([y[0]])
    
    