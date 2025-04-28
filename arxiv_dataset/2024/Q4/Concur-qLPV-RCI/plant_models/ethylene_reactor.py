import numpy as np
from plant_models.plant_model import Dynamics
from scipy.integrate import solve_ivp

class ethylene_reactor(Dynamics):
    def __init__(self, initial_state, 
                 gam1 = -8.13, gam2 = -7.12, gam3 = -11.07,
                 A1 = 92.80, A2 = 12.66, A3 = 2412.71,
                    B1 = 7.32, B2 = 10.39, B3 = 2170.57, B4 = 7.02, dt = 5.0, type='DT'):

        super().__init__(initial_state)
        self.gam1 = gam1
        self.gam2 = gam2
        self.gam3 = gam3
        self.A1 = A1
        self.A2 = A2
        self.A3 = A3
        self.B1 = B1
        self.B2 = B2
        self.B3 = B3
        self.B4 = B4
        self.Tc = 1.

        self.dt = dt
        self.type = type #Doing integration inside
        self.nx_plant = 4
        self.nu_plant = 2

        self.u1_minimum = 0.0704
        self.u1_maximum = 0.7042
        self.u1_mean = 0.5*(self.u1_minimum + self.u1_maximum)
        self.u1_lim = 0.5*(self.u1_maximum - self.u1_minimum)

        self.u2_minimum = 0.35
        self.u2_maximum = 0.65
        self.u2_mean = 0.5*(self.u2_minimum + self.u2_maximum)
        self.u2_lim = 0.5*(self.u2_maximum - self.u2_minimum)

        self.u_mean = np.array([self.u1_mean, self.u2_mean])
        self.u_lim = np.array([self.u1_lim, self.u2_lim])

        self.eq_state = np.array([0.9980647, 0.42912848, 0.03026791, 1.00193906])

        self.params = [self.gam1, self.gam2, self.gam3,
                       self.A1, self.A2, self.A3,
                       self.B1, self.B2, self.B3, self.B4, self.Tc,
                       self.u_mean]

    def dynamics(self, x, u):
        x_corrected = x + self.eq_state
        f = lambda t, x_corrected: standalone_dynamics(x_corrected, u, self.params)
        sol = solve_ivp(f, [0, self.dt], x_corrected, method='RK45', rtol=1.e-8, atol=1.e-8)
        return sol.y[:, -1]-self.eq_state
    
    def output(self, x, u):
        return np.array([x[2]])
    
def standalone_dynamics(x, u, params):
    gam1, gam2, gam3 = params[:3]
    A1, A2, A3 = params[3:6]
    B1, B2, B3, B4, Tc = params[6:11]
    u_mean = params[-1]

    u_0 = u[0] + u_mean[0]
    u_1 = u[1] + u_mean[1]

    r1 = np.exp(gam1 / x[3]) * (x[1] * x[3])**0.5
    r2 = np.exp(gam2 / x[3]) * (x[1] * x[3])**0.25
    r3 = np.exp(gam3 / x[3]) * (x[2] * x[3])**0.5

    dx_1 = u_0 * (1. - x[0] * x[3])
    dx_2 = u_0 * (u_1 - x[1] * x[3]) - A1 * r1 - A2 * r2
    dx_3 = -u_0 * x[2] * x[3] + A1 * r1 - A3 * r3
    dx_4 = (u_0 * (1. - x[3]) + B1 * r1 + B2 * r2 + B3 * r3 - B4 * (x[3] - Tc)) / x[0]

    return np.array([dx_1, dx_2, dx_3, dx_4])
    
    