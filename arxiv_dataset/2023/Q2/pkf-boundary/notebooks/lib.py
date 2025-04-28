import os
import numpy as np
import scipy as sp
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import tqdm.notebook as tqdm
from scipy.interpolate import interp1d

class Model(object):
    ''' Template for numerical model prediction of evolution dynamics

    Note
    ====

        Single-time-step / multiple-time-step schemes can be considered here:

        It is possible to consider **coupled time step** (leapfrog) as
        well as **decoupled time step** (euler, rk4,..)

    '''

    def __init__(self, *args, time_scheme='rk4', **kwargs):
        self.set_time_scheme(time_scheme)
        self._trend_matrix = None
        self._backward_matrix = None
        self._forward_matrix = None

    def set_time_scheme(self, time_scheme:str):
        """ Selection of the time scheme """
        # Harmonise la chaine de caractère fournie par l'utilisateur
        selected_time_scheme = '_'+time_scheme.lower()

        # Selectionne le schéma temporelle -- si la méthode appelée existe
        if hasattr(self, selected_time_scheme):
            self.time_scheme = getattr(self, selected_time_scheme)
        else:
            # Lance une "exception" quand la méthode d'intégration temporelle n'est pas connue.
            raise ValueError(f'The time scheme {selected_time_scheme} is not implemented')


    def trend(self, t, state):
        """
        Trend of the dynamics called during the time integration
        :param t:
        :param state:
        :return:

        """
        raise NotImplementedError()

    @staticmethod
    def _forward_euler(trend, t, state, dt):
        """
        Euler scheme

        :param t:
        :param dt:
        :param state:
        :return: updated state
        """
        return state + dt * trend(t, state)

    @staticmethod
    def _rk2(trend, t, state, dt):
        """
        Second order Ruge Kuta scheme

        :param t:
        :param dt:
        :param state:
        :return: updated state
        """
        state_demi = state + trend(t, state) * (dt*0.5)
        return state + dt * trend(t+0.5*dt, state_demi)

    @staticmethod
    def _rk4(trend, t, state, dt):
        """
        Fourth order Runge-Kuta scheme

        :param t:
        :param dt:
        :param state:
        :return: updated state
        """
        k1 = trend(t, state)
        k2 = trend(t+dt*0.5, state+dt*0.5*k1)
        k3 = trend(t+dt*0.5, state+dt*0.5*k2)
        k4 = trend(t+dt, state+dt*k3)
        return state + (k1+k2*2+k3*2+k4)*(dt/6)
    
    def _backward_euler(self, trend, t, state, dt):
        n = state.size
        if self._backward_matrix is None:
            bc_backup = self._bc
            self._bc = 'matrix_computation'
            self._trend_matrix = self._compute_trend_matrix(n, trend, t)
            self._bc = bc_backup
            self._backward_matrix = (sp.sparse.eye(n, format="csc") - dt*self._trend_matrix)
            self._forward_matrix = sp.sparse.linalg.inv(self._backward_matrix)
        return self._forward_matrix@self._backward_euler_apply_boundary(t, state, dt)

    @staticmethod
    def _compute_trend_matrix(n, trend, t):
        rows=[]
        cols=[]
        coefs=[]
        e = np.zeros(n)
        for j in range(n):
            e[j] = 1
            for i,coef in zip(range(n),trend(t, e)):
                if coef!=0:
                    rows.append(i)
                    cols.append(j)
                    coefs.append(coef)
            e[j] = 0
        return sp.sparse.csc_matrix((coefs,(rows,cols)))


    def forecast(self, window, u0, saved_times=None):
        """ Time integrates a single state over a time window

        :param window: time window
        :param u0: initial state of the integration
        :param saved_times: optional saved times, default is the given time window
        :return: a dictionary of computed time steps and at saved times.
        """

        # Return a dictionary of computed time steps saved time steps over the time forecast_window.
        # update saved_times
        saved_times = self._check_saved_times(window, saved_times)
        traj = {}
        for time, next_time in zip(window[:-1], window[1:]):
            if time in saved_times: traj[time] = u0
            #
            # True for separated time steps..
            #
            dt = next_time - time
            #try:
            u1 = self.time_scheme(self.trend, time, u0, dt)
            u0 = u1
            #except:
            #    print(f"An exception occurs in Model.forecast at time {time}")
            #    return traj

        time = window[-1]
        if time in saved_times: traj[time] = u0
        return traj

    def _forecast(self, args):
        """ Internal forecast method for parallel computation """
        return self.forecast(*args)

    def ensemble_forecast(self, window, states, saved_times=None, parallel=True, nb_pool=8):
        """
        Ensemble forecasting of a list of given state at a given time.
        :param window:
        :param states: list of input state
        :param saved_times:
        :param parallel:
        :return:
        """

        # Return a dictionnary of computed time steps saved time steps over the time forecast_window.
        # update saved_times
        saved_times = self._check_saved_times(window, saved_times)
        forecasts = {time:[] for time in saved_times}

        if parallel:

            parallel = Parallel(self._forecast, nb_pool)
            tmp_forecasts = parallel.run([ [window, state, saved_times] for state in states])

            while tmp_forecasts:
                forecast = tmp_forecasts.pop(0)
                for time in forecast:
                    forecasts[time].append( forecast[time] )
        else:

            for state in states:
                forecast = self.forecast(window, state, saved_times)
                for time in forecast:
                    forecasts[time].append( forecast[time] )
        return forecasts


    def window(self, dt, end, start=0.):
        return np.arange(start, end, dt)
    
    @staticmethod
    def derivative(f, dx, order=1, bc=None):
        """
        Compute the spatial derivative at the first and second order with a 2nd order of consistancy
        """
        dfdx = np.zeros(f.size)

        if order==1:
            if bc is None or bc[0] is None:
                dfdx[0] = (-0.5*f[2] + 2*f[1] - 1.5*f[0])/dx
            else:
                dfdx[0] = (f[1] - bc[0])/(2*dx)

            if bc is None or bc[1] is None:
                dfdx[-1] = (0.5*f[-3] + -2*f[-2] + 1.5*f[-1])/dx
            else:
                dfdx[-1] = (bc[1] - f[-2])/(2*dx)

            dfdx[1:-1] = (f[2:] - f[:-2])/(2*dx)

        elif order==2:
            if bc is None or bc[0] is None:
                dfdx[0] = (-1*f[3] + 4*f[2] - 5*f[1] + 2*f[0])/dx**2
            else:
                dfdx[0] = (f[1] - 2*f[0] + bc[0])/dx**2

            if bc is None or bc[1] is None:
                dfdx[-1] = (-1*f[-4] + 4*f[-3] - 5*f[-2] + 2.*f[-1])/dx**2
            else:
                dfdx[-1] = (bc[1] -2*f[-1] + f[-2])/dx**2

            dfdx[1:-1] = (f[2:] - 2*f[1:-1] + f[:-2])/(dx**2)

        return dfdx

    @staticmethod
    def _check_saved_times(window, saved_times):
        if saved_times is None:
            saved_times = window
        elif type(saved_times)==float or type(saved_times)==np.float64:
            saved_times = [saved_times]
        return saved_times




class Diffusion(Model):
    def __init__(self, *args, x, D, bc=None, bx0=None, bxL=None, **kwargs):
        super().__init__(*args, **kwargs)

        self._bc = bc
        self._bx0 = bx0 # Boundary condition as x=0
        self._bxL = bxL # Boundary condition as x=0

        self._n = x.size
        self._x = x
        self._dx = self._x[1]
        self._D = D

    @property
    def n(self):
        return self._n

    @property
    def x(self):
        return self._x

    @property
    def dx(self):
        return self._dx

    @property
    def D(self):
        return self._D

    def _backward_euler_apply_boundary(self, t, c, dt):
        bc = np.zeros(c.size)
        matrix_coef_x0 = (dt/(4*self._dx**2))*(self._D[1]-4*self._D[0]-self._bx0(t)[1])
        matrix_coef_xL = (dt/(4*self._dx**2))*(self._D[-2]-4*self._D[-1]-self._bxL(t)[1])
        if self._bc == 'dirichlet':
            bc[0] = matrix_coef_x0*self._bx0(t)[0]
            bc[-1] = matrix_coef_xL*self._bxL(t)[0]
        elif self._bc == 'neumann':
            bc[0] = matrix_coef_x0*(c[1]-2*self._dx*self._bx0(t)[0])
            bc[-1] = matrix_coef_xL*(c[-2]+2*self._dx*self._bxL(t)[0])
        return c - bc
    
    def trend(self, t, c):
        bc_D = (self._bx0(t)[1], self._bxL(t)[1])
        if self._bc == 'matrix_computation':
            bc_c = (0,0)
        elif self._bc == 'dirichlet':
            bc_c = (self._bx0(t)[0],self._bxL(t)[0])
        elif self._bc == 'neumann':
            bc_c = (c[1]-2*self._dx*self._bx0(t)[0], c[-2]+2*self._dx*self._bxL(t)[0])
        else:
            bc_c = None

        dcdt = self.derivative(self._D, self._dx, bc=bc_D)*self.derivative(c, self._dx, bc=bc_c) + self._D*self.derivative(c, self._dx, order=2, bc=bc_c)
        return dcdt





class DiffusionPKF(Diffusion):

    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs) # Time scheme is set from Model.__init__()
    
    def trend(self, t, state):
        """ Trend of the dynamics """

        dstate = np.zeros(state.shape)
        dc = dstate[0]
        dV_c = dstate[1]
        ds_c_xx = dstate[2]

        # Load physical functions from state
        c = state[0]
        V_c = state[1]
        s_c_xx = state[2]

        # Alias for constant functions
        D = self._D

        # Format boundary conditions
        bc_D = (self._bx0(t)[3], self._bxL(t)[3])
        if self._bc == 'dirichlet':
            bc_c = (self._bx0(t)[0],self._bxL(t)[0])
            bc_V_c = (self._bx0(t)[1],self._bxL(t)[1])
            bc_s_c_xx = (self._bx0(t)[2],self._bxL(t)[2])
        elif self._bc == 'neumann':
            bc_c = (c[1]-2*self._dx*self._bx0(t)[0], c[-2]+2*self._dx*self._bxL(t)[0])
            bc_V_c = (V_c[1]-2*self._dx*self._bx0(t)[1], V_c[-2]+2*self._dx*self._bxL(t)[1])
            bc_s_c_xx = (s_c_xx[1]-2*self._dx*self._bx0(t)[2], s_c_xx[-2]+2*self._dx*self._bxL(t)[2])
        else:
            bc_c = None
            bc_V_c = None
            bc_s_c_xx = None

        # Compute derivative
        DD_x_o1 = self.derivative(D, self._dx, bc=bc_D)
        DD_x_o2 = self.derivative(D, self._dx, order=2, bc=bc_D)
        Dc_x_o1 = self.derivative(c, self._dx, bc=bc_c)
        Dc_x_o2 = self.derivative(c, self._dx, order=2, bc=bc_c)
        DV_c_x_o1 = self.derivative(V_c, self._dx, bc=bc_V_c)
        DV_c_x_o2 = self.derivative(V_c, self._dx, order=2, bc=bc_V_c)
        Ds_c_xx_x_o1 = self.derivative(s_c_xx, self._dx, bc=bc_s_c_xx)
        Ds_c_xx_x_o2 = self.derivative(s_c_xx, self._dx, order=2, bc=bc_s_c_xx)

        # Implementation of the trend
        dc[:] = D*Dc_x_o2 + DD_x_o1*Dc_x_o1
        dV_c[:] = -D*DV_c_x_o1**2/(2*V_c) + D*DV_c_x_o2 - 2*D*V_c/s_c_xx + DD_x_o1*DV_c_x_o1
        ds_c_xx[:] = 2*D*DV_c_x_o1**2*s_c_xx/V_c**2 + D*DV_c_x_o1*Ds_c_xx_x_o1/V_c - 2*D*DV_c_x_o2*s_c_xx/V_c - 2*D*Ds_c_xx_x_o1**2/s_c_xx + D*Ds_c_xx_x_o2 + 4*D - 2*DD_x_o1*DV_c_x_o1*s_c_xx/V_c + 2*DD_x_o1*Ds_c_xx_x_o1 - 2*DD_x_o2*s_c_xx

        return dstate





class Advection(Model):
    def __init__(self, *args, x, u, bc=None, bx0=None, bxL=None, **kwargs):
        super().__init__(*args, **kwargs)

        self._bc = bc
        self._bx0 = bx0 # Boundary condition as x=0
        self._bxL = bxL # Boundary condition as x=0

        self._n = x.size
        self._x = x
        self._dx = self._x[1]
        self._u = u

    @property
    def n(self):
        return self._n

    @property
    def x(self):
        return self._x

    @property
    def dx(self):
        return self._dx

    @property
    def u(self):
        return self._u

    def trend(self, t, c):
        bc_u = (self._bx0(t)[1], self._bxL(t)[1])
        if self._bc == 'dirichlet':
            bc_c = (self._bx0(t)[0], self._bxL(t)[0])
        elif self._bc == 'neumann':
            bc_c = (c[1]-2*self._dx*self._bx0(t)[0], c[-2]+2*self._dx*self._bxL(t)[0])
        else:
            bc_c = None

        dcdt = -self._u*self.derivative(c, self._dx, bc=bc_c)
        return dcdt





class AdvectionPKF(Advection):

    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs) # Time scheme is set from Model.__init__()
    
    def trend(self, t, state):
        """ Trend of the dynamics """

        dstate = np.zeros(state.shape)
        dc = dstate[0]
        dV_c = dstate[1]
        ds_c_xx = dstate[2]

        # Load physical functions from state
        c = state[0]
        V_c = state[1]
        s_c_xx = state[2]

        # Alias for constant functions
        u = self._u

        # Format boundary conditions
        bc_u = (self._bx0(t)[3], self._bxL(t)[3])
        if self._bc == 'dirichlet':
            bc_c = (self._bx0(t)[0],self._bxL(t)[0])
            bc_V_c = (self._bx0(t)[1],self._bxL(t)[1])
            bc_s_c_xx = (self._bx0(t)[2],self._bxL(t)[2])
        elif self._bc == 'neumann':
            bc_c = (c[1]-2*self._dx*self._bx0(t)[0], c[-2]+2*self._dx*self._bxL(t)[0])
            bc_V_c = (V_c[1]-2*self._dx*self._bx0(t)[1], V_c[-2]+2*self._dx*self._bxL(t)[1])
            bc_s_c_xx = (s_c_xx[1]-2*self._dx*self._bx0(t)[2], s_c_xx[-2]+2*self._dx*self._bxL(t)[2])
        else:
            bc_c = None
            bc_V_c = None
            bc_s_c_xx = None

        # Compute derivative
        Du_x_o1 = self.derivative(u, self._dx, bc=bc_u)
        Dc_x_o1 = self.derivative(c, self._dx, bc=bc_c)
        DV_c_x_o1 = self.derivative(V_c, self._dx, bc=bc_V_c)
        Ds_c_xx_x_o1 = self.derivative(s_c_xx, self._dx, bc=bc_s_c_xx)

        # Implementation of the trend
        dc[:] = -u*Dc_x_o1
        dV_c[:] = -u*DV_c_x_o1
        ds_c_xx[:] = -u*Ds_c_xx_x_o1 + 2*s_c_xx*Du_x_o1

        return dstate




    
class BoundaryTimeSerie(object):
    def __init__(self, model, times,
                 spatial_variances,
                 spatial_scales,
                 temporal_variances,
                 temporal_scales,
                 transition_scale=1,
                 offset=None
                ):

        self._model = model
        self._times = times

        self._transition_scale = transition_scale
        spatial_times = self.model._x/self._transition_scale

        self._spatial_variances = spatial_variances
        self._spatial_aspect_tensors = spatial_scales**2
        self._temporal_variances = temporal_variances
        self._temporal_aspect_tensors = temporal_scales**2

        if self._temporal_variances.ndim == 1:
            self._nbc = 1
            self._Q = self.model._n + self._times.size

            self._extended_times = np.concatenate((self._times, spatial_times+self._times[-1]+self.times[1]))

            self._temporal_variances = np.flip(self._temporal_variances)
            self._temporal_aspect_tensors = np.flip(self._temporal_aspect_tensors)

            self._variances = np.concatenate((self._temporal_variances, self._spatial_variances))
            self._aspect_tensors = np.concatenate((self._temporal_aspect_tensors, (1/self._transition_scale**2)*self._spatial_aspect_tensors))

        elif self._temporal_variances.ndim == 2:
            self._nbc = 2
            self._Q = self.model._n + 2*self._times.size

            self._extended_times = np.concatenate((self._times, spatial_times+self._times[-1]+self.times[1], self._times+spatial_times[-1]+self._times[-1]+self.times[1]))

            self._temporal_variances[:,0] = np.flip(self._temporal_variances[:,0])
            self._temporal_aspect_tensors[:,0] = np.flip(self._temporal_aspect_tensors[:,0])

            self._variances = np.concatenate((self._temporal_variances[:,0], self._spatial_variances, self._temporal_variances[:,1]))
            self._aspect_tensors = np.concatenate((self._temporal_aspect_tensors[:,0], (1/self._transition_scale**2)*self._spatial_aspect_tensors, self._temporal_aspect_tensors[:,1]))

        if offset is None:
            self._offset = self._Q
        else:
            self._offset = offset

        self._init_covariance()

    @property
    def times(self):
        return self._times
    @property
    def variances(self):
        return self._temporal_variances
    @property
    def aspect_tensors(self):
        return self._temporal_aspect_tensors
    @property
    def model(self):
        return self._model

    @staticmethod
    def ornstein_uhlenbeck(times, mu=None, sigma=None, l=None, bc_constraint=None):
        if mu is None: mu = np.zeros(times.size)
        if sigma is None: sigma = np.ones(times.size)
        if l is None: l = np.ones(times.size)

        def wiener(times):
            w = np.zeros(times.size)
            for i in range(1,times.size):
                w[i] = w[i-1] + np.random.randn()/np.sqrt(n)
            return w

        theta = 1/l
        x = np.zeros(times.size)
        if bc_constraint is None:
            x[0] = mu[0] + sigma[0]*np.random.randn()
        else:
            x[0] = bc_constraint
        w = wiener(times)
        dw = w[1:]-w[:-1]
        dt = times[1:]-times[:-1]
        for i in range(1,times.size):
            x[i] = x[i-1]*(1-theta[i]*dt[i-1]) + mu[i]*theta[i]*dt[i-1] + sigma[i]*dw[i-1]
        return x

    def _init_covariance(self):
        print('Warning: generate the covariance + its SVD -- this takes some time')

        self._P = np.zeros((self._Q,self._Q))
        for k in tqdm.tqdm(range(self._Q)):
            for o in range(-self._offset,self._offset+1):
                i = k
                j = k + o
                if j<0 or j>=self._Q:
                    continue
                else:
                    # Extract parameters
                    t1 = self._extended_times[i]
                    t2 = self._extended_times[j]
                    v1 = self._variances[i]
                    v2 = self._variances[j]
                    s1 = self._aspect_tensors[i]
                    s2 = self._aspect_tensors[j]

                    # Compute the covariance
                    self._P[i,j] = np.sqrt(v1*v2)*np.sqrt(np.sqrt(s1*s2)) / np.sqrt(0.5*(s1+s2)) * np.exp(-0.5*(t1-t2)**2 / (0.5*(s1+s2)))

        # 2. Set the square-root of P
        U,D,VT = np.linalg.svd(self._P)
        self._sqrt_P = U@np.diag(np.sqrt(D))@VT

    def sample_time_serie(self):
        """ Engender a sample of the time serie """
        zeta = np.random.normal(size=(self._Q))
        sample = self._sqrt_P @ zeta
        sample_bc0 = np.flip(sample[:self.times.size])
        if self._nbc == 1:
            initial_condition = sample[self.times.size:]
            sample_bcL = np.zeros(self.times.size)
        elif self._nbc == 2:
            initial_condition = sample[self.times.size:self.times.size+self.model._n]
            sample_bcL = sample[self.times.size+self.model._n:]
        return initial_condition, sample_bc0, sample_bcL

    def boundary_condition_factory(self):
        """ Return bondary condition functions """

        initial_condition, sample_bc0, sample_bcL = self.sample_time_serie()
        bc0 = interp1d(self._times, sample_bc0)
        bcL = interp1d(self._times, sample_bcL)
        return initial_condition, bc0, bcL

    def plot_length_scale(self):
        plt.plot(self._times, np.sqrt(self._aspect_tensors), label='Time correlation length-scale')
        plt.xlabel("time")
        plt.ylabel("Length-scale")

    def plot_variance(self):
        plt.plot(self._times, self._variances, label='Variance')
        plt.xlabel("time")
        plt.ylabel("Variance")

    def plot_aspect_tensor(self):
        plt.plot(self._times, self._aspect_tensors, label='Time aspect tensort')
        plt.xlabel("time")
        plt.ylabel("Aspect tensor")

    def plot_boundary_condition(self, bc):
        ltimes = [t for t in self._times if t>=0]
        plt.plot(ltimes, [bc(t) for t in ltimes], label='Boundary condition')
        plt.xlabel("time")
        plt.ylabel("Boundary condition")
    
    def show_covariances(self):
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        ax.matshow(self._P)



class EnKFExperiment(object):

    def __init__(self, model, boundary_generator):

        self._model = model
        self._boundary = boundary_generator

        self._ensemble_forecasts = None
        self._Ne = None
        self._mean = None
        self._correlations = None
        self._covariances = None
        self._variances = None
        self._length_scales = None
        self._normalized_length_scales = None
        self._metrics = None
        self._aspect_tensors = None
        self._errors = None
        self._normalized_errors = None

    @property
    def Ne(self):
        return self._Ne

    @property
    def model(self):
        return self._model

    @property
    def mean(self):
        if self._mean is None:
            self._mean = {}
            for t,ensemble in self._ensemble_forecasts.items():
                data = np.asarray(ensemble)
                self._mean[t] = data.mean(axis=0)
        return self._mean

    @property
    def covariances(self):
        if self._covariances is None:
            self._covariances = {t:self.estimate_covariance(ensemble) for t,ensemble in self._ensemble_forecasts.items()}
        return self._covariances

    @property
    def correlations(self):
        if self._correlations is None:
            self._correlations = {t:self.make_C(P) for t,P in self.covariances.items()}
        return self._correlations

    @property
    def variances(self):
        if self._variances is None:
            self._variances = {t:np.diag(P) for t,P in self.covariances.items()}
        return self._variances

    @property
    def metrics(self):
        """ Ensemble estimation of the local metric tensor """
        if self._metrics is None:
            self._metrics = {}
            for t,ensemble in self._ensemble_forecasts.items():

                data = np.asarray(ensemble)

                # 1. Compute the mean
                mean = data.mean(axis=0)

                # 2. Compute the error
                data -= mean

                # 3. Computation of the normalized errors
                std = np.sqrt(self.variances[t])
                istd = 1./std
                data *= istd

                # 4. Compute the spatial derivatives
                data = np.asarray([self._model.derivative(error, self._model.dx) for error in data])

                # 5. Compute the metric g_ij = E[(Di \epsilon) (Dj \epsilon)]
                self._metrics[t] = (data**2).mean(axis=0)

        return self._metrics

    @property
    def aspect_tensors(self):
        if self._aspect_tensors is None:
            self._aspect_tensors = {t:1/g for t,g in self.metrics.items()}
        return self._aspect_tensors

    @property
    def length_scales(self):
        if self._length_scales is None:
            self._length_scales = {t:np.sqrt(s) for t,s in self.aspect_tensors.items()}
        return self._length_scales

    @property
    def normalized_length_scales(self):
        if self._normalized_length_scales is None:
            self._normalized_length_scales = {t:field/self.model.dx for t,field in self.length_scales.items()}
        return self._normalized_length_scales

    @property
    def errors(self):
        if self._errors is None:
            self._errors = {}
            for t,ensemble in self._ensemble_forecasts.items():
                data = np.asarray(ensemble)
                self._errors[t] = data - data.mean(axis=0)
        return self._errors

    @property
    def normalized_errors(self):
        if self._normalized_errors is None:
            self._normalized_errors = {}
            for t,ensemble in self._ensemble_forecasts.items():
                data = np.asarray(ensemble)
                data -= data.mean(axis=0)
                self._normalized_errors[t] = data/np.sqrt(self.variances[t])
        return self._normalized_errors

    def run_ensemble(self, Ne, times, saved_times, initial_condition, bx0, bxL):
        ensemble_forecasts = {t : [] for t in saved_times}
        self._Ne = Ne

        for k in tqdm.tqdm(range(Ne)):
            # 1. Create a boundary condition
            random_init_cond, random_bc0, random_bcL = self._boundary.boundary_condition_factory()
            init_cond = initial_condition + random_init_cond
            if bx0(0)[0] is None:
                self._model._bx0 = lambda t : (None, bx0(t)[1])
            else:
                self._model._bx0 = lambda t : (bx0(t)[0] + random_bc0(t), bx0(t)[1])
            if bxL(0)[0] is None:
                self._model._bxL = lambda t : (None, bxL(t)[1])
            else:
                self._model._bxL = lambda t : (bxL(t)[0] + random_bcL(t), bxL(t)[1])
            # 2. Forecast with the boundary condition
            traj = self._model.forecast(times, init_cond, saved_times=saved_times) 
            # 3. Feads the ensemble of traj
            for t in saved_times:
                ensemble_forecasts[t].append(traj[t])
            
        self._ensemble_forecasts = ensemble_forecasts

    def estimate_covariance(self,data):
        if isinstance(data, list):
            data = np.asarray(data)
        mean = data.mean(axis=0)
        pert = (data - mean)
        pert = pert.T
        cov = pert @ pert.T
        cov *= 1/data.shape[0]
        return cov

    def make_C(self,P):
        istd = np.diag(1/np.sqrt(np.diag(P)))
        return istd @P @istd