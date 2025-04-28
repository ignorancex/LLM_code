import numpy as np
from plant_models.plant_model import Dynamics
from scipy.integrate import solve_ivp

class nonlinear_MSD(Dynamics):
    def __init__(self, initial_state, 
                        mass=1., k1 = 0.5, k2 = 1., b1=1., b2 = 0.,
                        dt=0.1, u_lim = 2., 
                        sigma_u = 0.0001, sigma_y = 0.0001, 
                        type='DT'):
        super().__init__(initial_state)

        num_mass = len(initial_state)//2
        self.num_mass = num_mass
        self.mass = mass
        self.k1 = k1
        self.k2 = k2
        self.b1 = b1
        self.b2 = b2
        self.dt = dt
        self.type = type
        self.params = [num_mass, mass, k1, k2, b1, b2]
        self.nx_plant = 2*num_mass
        self.nu_plant = 1
        self.u_lim = u_lim

        self.sigma_u = sigma_u
        self.sigma_y = sigma_y
    
    def dynamics(self, x, u):
        u_noise = u+ self.sigma_u*np.random.randn(self.nu_plant)[0]
        f = lambda t, x: chain_dynamics(x, u_noise, self.params)
        sol = solve_ivp(f, [0, self.dt], x, method='RK45', rtol=1.e-8, atol=1.e-8)
        return sol.y[:, -1]
    
    def output(self, x, u):
        #Position of last mass
        return np.array([x[-2]+self.sigma_y*np.random.randn(1)[0]])

def chain_dynamics(x, u, params):
    #Input applied only to the last mass
    num_mass, mass, k1, k2, b1, b2 = params

    def K(dx):
        return k1*dx + k2*dx**3
    
    def b(d_dx):
        return b1*d_dx + b2*d_dx**3
    
    dx_vec = np.zeros(2*num_mass)

    #First mass
    dx_vec[0] = x[1]
    if num_mass == 1:
        dx_vec[1] = (u - K(x[0]) - b(x[1]))/mass
    else:
        dx_vec[1] = (-(K(x[0])+b(x[1]))+(K(x[2]-x[0])+b(x[3]-x[1])))/mass

    for i in range(1,num_mass-1):
        #Middle masses
        prev_pos = x[2*i-2]
        prev_vel = x[2*i-1]
        curr_pos = x[2*i]
        curr_vel = x[2*i+1]
        next_pos = x[2*i+2]
        next_vel = x[2*i+3]
        dx_vec[2*i] = curr_vel
        dx_vec[2*i+1] = ((K(prev_pos-curr_pos)+b(prev_vel-curr_vel))+(K(next_pos-curr_pos)+b(next_vel-curr_vel)))/mass
    
    #Last mass
    if num_mass>1:
        prev_pos = x[-4]
        prev_vel = x[-3]
        curr_pos = x[-2]
        curr_vel = x[-1]
        dx_vec[-2] = curr_vel
        dx_vec[-1] = (u+(K(prev_pos-curr_pos)+b(prev_vel-curr_vel)))/mass
    
    return dx_vec
    



        




 