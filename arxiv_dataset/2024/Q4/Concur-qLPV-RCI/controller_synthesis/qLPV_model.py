from imports import *
"""
Model class for qLPV systems for control design : plant input to plant output. Scaling handled inside
"""

def swish(x):
    return x/(1+np.exp(-x))

def swish_casadi(x):
    return x / (1 + ca.exp(-x))

class qLPV_model:
    def __init__(self, parameters):
        self.sizes = parameters['sizes']
    
        self.A = parameters['A']
        self.B = parameters['B']
        self.C = parameters['C']
        self.L = parameters['L']
        self.ny = self.C.shape[0]
        self.nu = self.B[0].shape[1]
        self.nx = self.sizes[0]
        self.nq = self.sizes[1]
        self.nth = self.sizes[2]
        self.nH = self.sizes[3]

        self.Win = parameters['Win']
        self.bin = parameters['bin']
        self.Whid = parameters['Whid']
        self.bhid = parameters['bhid']
        self.Wout = parameters['Wout']
        self.bout = parameters['bout']
        self.W = parameters['W']
        self.u_scaler = parameters['u_scaler']
        self.y_scaler = parameters['y_scaler']

        self.HU = parameters['HU']
        self.hU = parameters['hU']
        self.HY = parameters['HY']
        self.hY = parameters['hY']
        self.only_px = parameters['only_px']

    def scale_model(self, y_plant, u_plant):
        y_scaler = self.y_scaler
        u_scaler = self.u_scaler
        y_plant_scale = (y_plant-y_scaler[0])*y_scaler[1]
        u_plant_scale = (u_plant-u_scaler[0])*u_scaler[1]
        return y_plant_scale, u_plant_scale

    def parameter(self, x, u):
        if self.only_px == False:
            z = np.hstack((x,u))
        else:
            z = x
        Win = self.Win
        bin = self.bin
        Whid = self.Whid
        bhid = self.bhid
        Wout = self.Wout
        bout = self.bout
        nq = self.nq
        nH = self.nH
        p = np.array([[1.]])
        for i in range(nq-1):
            post_linear = Win[i]@z+bin[i]
            post_activation = swish(post_linear)
            for j in range(nH-1): 
                post_linear = Whid[i][j]@post_activation+bhid[i][j]
                post_activation = swish(post_linear)
            post_linear = Wout[i]@post_activation+bout[i]
            p = np.vstack((p,np.exp(post_linear)))
        return p/np.sum(p)

    def parameter_casadi(self, x, u):
        if self.only_px == False:
            z = ca.vertcat(x, u)
        else:
            z = x
        Win = self.Win
        bin = self.bin
        Whid = self.Whid
        bhid = self.bhid
        Wout = self.Wout
        bout = self.bout
        nq = self.nq
        nH = self.nH
        p = ca.DM.ones(1)
        for i in range(nq-1):
            post_linear = Win[i] @ z + bin[i]
            post_activation = swish_casadi(post_linear)
            for j in range(nH-1): 
                post_linear = Whid[i][j] @ post_activation + bhid[i][j]
                post_activation = swish_casadi(post_linear)
            post_linear = ca.reshape(Wout[i], (1, -1)) @ post_activation + bout[i]
            p = ca.vertcat(p, ca.exp(post_linear))  # Append the exponential of the output
        p = p / ca.sum1(p)
        return p
    
    def observer(self, x, p, u_plant, y_plant):
        y, u = self.scale_model(y_plant, u_plant)
        C = self.C
        L = self.L
        nq = self.nq
        L_tot = L[0]*p[0]
        for i in range(1,nq):
            L_tot += L[i]*p[i]  
        return L_tot@(y-C@x)

    def dynamics(self, x, u_plant, y_plant):
        y, u = self.scale_model(y_plant, u_plant)
        if isinstance(u_plant, ca.MX):
            #When using in optimization problems
            p = self.parameter_casadi(x, u)
        else:
            p = self.parameter(x, u)
        A = self.A
        B = self.B
        nq = self.nq
        A_tot = A[0]*p[0]
        B_tot = B[0]*p[0]
        for i in range(1,nq):
            A_tot += A[i]*p[i]
            B_tot += B[i]*p[i]
        return A_tot@x+B_tot@u+self.observer(x, p, u_plant, y_plant)
    
    def output(self, x):
        C = self.C
        y_scaler = self.y_scaler
        return ((C@x)/y_scaler[1])+y_scaler[0]

