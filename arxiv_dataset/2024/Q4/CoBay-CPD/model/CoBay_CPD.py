import numpy as np
import pandas as pd
import copy
from scipy.misc import derivative
from scipy.special import logsumexp
import math
from scipy.stats import beta
from scipy.special import expit
from scipy.special import psi
from scipy.stats import expon
from scipy.stats import gamma
from scipy.stats import poisson
from scipy.stats import bernoulli
from scipy.stats import uniform
from scipy.stats import multinomial
from scipy.stats import multivariate_normal
from scipy.stats import invgauss
from scipy.stats import dirichlet
from numpy.polynomial import legendre
import matplotlib.pyplot as plt
import scipy.integrate as si
from scipy import stats, optimize
import matplotlib.pyplot as plt
from scipy import stats, optimize
from IPython.display import display, Math, Latex, clear_output
import multiprocessing
from functools import partial
from pypolyagamma import PyPolyaGamma
import time

class Gibbs:
    PG = PyPolyaGamma()
    def __init__(self, number_of_basis):
        """
        Initialize Model.
        \bar{\lambda} and w are parameters, beta densities are hyperparameters.
        """
        # self.number_of_basis = number_of_basis
        self.number_of_basis = 5
        self.beta_ab = np.empty((number_of_basis, 3))
        self.T_phi = None

        self.lamda_ub = None
        self.lamda_ub_estimated = None
        self.base_activation = None
        self.base_activation_estimated = None
        self.weight = np.empty(number_of_basis)
        self.weight_estimated = np.empty(number_of_basis)
    
    def set_hawkes_hyperparameters(self, beta_ab, T_phi):
        r"""
        Fix the hyperparameters : parameters a, b, shift and scale for basis functions (Beta densities). 

        :type beta_ab: numpy array
        :param beta_ab: [[a,b,shift],[a,b,shift]...] for basis functions.
        :type T_phi: float
        :param T_phi: the support of influence functions (the scale of basis functions)
        """
        # Raise ValueError if the given parameters do not have the right shape
        if np.shape(beta_ab) != (self.number_of_basis, 3):
            raise ValueError('given basis functions have incorrect shape')
        if np.shape(T_phi) != ():
            raise ValueError('given scale parameter has incorrect shape')
        self.beta_ab = copy.copy(beta_ab)
        self.T_phi = copy.copy(T_phi)
    
    def set_hawkes_parameters(self, lamda_ub, base_activation, weight):
        r"""
        Fix the parameters: intensity upperbound, base activation and influence weight. 
        It is used in the simulation.

        :type lamda_ub: 1D numpy array
        :param lamda_ub: :math:`\bar{\lambda}`.
        :type base_activation: 1D numpy array
        :param base_activation: :math:`\mu`.
        :type weight: number_of_dimensions*number_of_dimensions*number_of_basis numpy array
        :param weight: :math:`w_{ijb}`.
        """
        # Raise ValueError if the given parameters do not have the right shape
        if np.shape(lamda_ub) != ():
            raise ValueError('given intensity upperbounds have incorrect shape')
        if np.shape(base_activation) != ():
            raise ValueError('given base activations have incorrect shape')
        if np.shape(weight) != (self.number_of_basis,):
            raise ValueError('given weight have incorrect shape')
        self.lamda_ub = copy.copy(lamda_ub)
        self.base_activation = copy.copy(base_activation)
        self.weight = copy.copy(weight)
    
    def set_hawkes_parameters_estimated(self, lamda_ub_estimated, W_estimated):
        r"""
        Set the estimated intensity upperbound, base activation and influence weight. 
        It is used in the visualization.  

        :type lamda_ub_estimated: 1D numpy array
        :param lamda_ub_estimated: :math:`\hat\bar{\lamda}`.
        :type W_estimated: number_of_dimensions * (number_of_dimensions * number_of_basis + 1) numpy array
        :param W_estimated: `W[:,0]` is the estimated base activation, `W[:,1:]` is the estimated influence weight
        """
        # Raise ValueError if the given parameters do not have the right shape
#         if np.shape(lamda_ub_estimated) != (self.number_of_dimensions,):
#             raise ValueError('given estimated intensity upperbounds have incorrect shape')
        if np.shape(W_estimated) != (self.number_of_basis + 1,):
            raise ValueError('given estimated W have incorrect shape')
        self.lamda_ub_estimated = copy.copy(lamda_ub_estimated)
        self.base_activation_estimated = copy.copy(W_estimated[0])
        self.weight_estimated = copy.copy(W_estimated[1:])
    
    def intensity(self, t, timestamps_history, estimation = False):
        """
        Given the historical timestamps, evaluate the conditional intensity at t on the target dimension.
        It is used in the simulation and visualization. If `estimation` is False, the intensity function is using 
        the ground truth parameters; if `estimation` is True, the intensity function is using the estimated parameters. 

        :type t: float
        :param t: the target time
        :type target_dimension: int
        :param target_dimension: the target dimension
        :type timestamps_history: list
        :param timestamps_history: [[t_1,t_2,...,t_N_1],[t_1,t_2,...,t_N_2],...], the historical timestamps before t
        :type estimation: bool
        :param estimation: indicate to use whether the ground-truth or estimated parameters

        :rtype: float
        :return: the conditional intensity at t
        """
        # Raise ValueError if the given historical timestamps do not have the right shape
#         if len(timestamps_history) != self.number_of_dimensions:
#             raise ValueError('given historical timestamps have incorrect shape')
        if estimation == False:
            lamda_ub = self.lamda_ub
            base_activation = self.base_activation
            weight = self.weight
        else:
            lamda_ub = self.lamda_ub_estimated
            base_activation = self.base_activation_estimated
            weight = self.weight_estimated
        intensity = 0
        # print("t_m_history",timestamps_history)
        for i in range(len(timestamps_history)):
                if timestamps_history[i] >= t:
                        break
                elif t - timestamps_history[i] > self.T_phi: 
                        continue
                for b in range(4):
                        intensity += weight[b] * beta.pdf(t - timestamps_history[i], a = self.beta_ab[b][0], b = self.beta_ab[b][1], \
                                              loc = self.beta_ab[b][2], scale = self.T_phi)
        return lamda_ub * expit(base_activation + intensity)
        
    def simulation(self, T):
        r"""
        Simulate a sample path of the sigmoid nonlinear multivariate Hawkes processes with Beta densities as basis functions.

        :type T: float
        :param T: time at which the simulation ends.
        :rtype: list
        :return: the timestamps when events occur on each dimension.
        """
        t = 0
        points_hawkes = []
        intensity_sup = self.lamda_ub
        while(t < T):
            r = expon.rvs(scale = 1 / intensity_sup)
            t += r
            intensity_t = self.intensity(t,points_hawkes)
            assert intensity_t <= intensity_sup, "intensity exceeds the upper bound"
            D = uniform.rvs(loc = 0,scale = 1)
            if D * intensity_sup <= intensity_t:
                points_hawkes.append(t)
        if points_hawkes[-1] > T:
            del points_hawkes[-1]
        return points_hawkes
    
    def simulation_one(self):
        r"""
        Simulate a sample path of the sigmoid nonlinear multivariate Hawkes processes with Beta densities as basis functions.

        :type T: float
        :param T: time at which the simulation ends.
        :rtype: list
        :return: the timestamps when events occur on each dimension.
        """
        t = 0
        points_hawkes = []
        m = 0
        intensity_sup = self.lamda_ub
        while(m < 1):
            r = expon.rvs(scale = 1 / intensity_sup)
            t += r
            intensity_t = self.intensity(t,points_hawkes)
            assert intensity_t <= intensity_sup, "intensity exceeds the upper bound"
            D = uniform.rvs(loc = 0,scale = 1)
            if D * intensity_sup <= intensity_t:
                points_hawkes.append(t)
                m = 1
        return t
    
    def Phi_t(self, t_in, points_hawkes): 
        Phi_t = [1]
        for j in range(self.number_of_basis):
            index = (np.array(points_hawkes) < t_in) & ((t_in - np.array(points_hawkes)) <= self.T_phi)
            Phi_t.append(sum(beta.pdf(t_in - np.array(points_hawkes)[index], a=self.beta_ab[j][0], \
                                      b=self.beta_ab[j][1], loc=self.beta_ab[j][2], scale=self.T_phi)))

        return np.array(Phi_t)
    
    def Phi_n_g(self, points_hawkes, points_g):
        r"""
        Evaluate \Phi(t) on all observed points and grid nodes (Gaussian quadrature nodes). 

        :type points_hawkes: list
        :param points_hawkes: the timestamps when events occur on each dimension
        :type points_g: list
        :param points_g: the timestamps of grid nodes or Gaussian quadrature nodes on [0,T]
        :rtype: number_of_dimensions*N_i*(number_of_dimensions*number_of_basis+1), num_g*(number_of_dimensions*number_of_basis+1)
        :return: list of \Phi(t_n), \Phi(t_g)
        """
        N = len(points_hawkes)
        num_g = len(points_g)
        Phi_n = np.zeros((N,self.number_of_basis+1))
        for n in range(N):
            Phi_n[n] = self.Phi_t(points_hawkes[n],points_hawkes)
        Phi_g = np.zeros((num_g, self.number_of_basis+1))
        for g in range(num_g):
            Phi_g[g] = self.Phi_t(points_g[g],points_hawkes)
        return Phi_n, Phi_g
    
    def inhomo_simulation(self, intensity, T):
        r"""
        Simulate an inhomogeneous Poisson process with a discrete intensity function (vector)

        :type intensity: 1-D numpy array
        :param intensity: the discrete intensity function (vector)
        :type T: float
        :param T: the observation windown [0,T]

        :rtype: 1D list
        :return: [t_1, t_2, ..., t_N]
        """
        delta_t = T/len(intensity)
        Lambda = np.max(intensity)*T
        r = poisson.rvs(Lambda)
        x = uniform.rvs(size = r)*T
        measures = intensity[(x/delta_t).astype(int)]
        ar = measures/np.max(intensity)
        index = bernoulli.rvs(ar)
        index = np.array(index)
        points = x[index.astype(bool)]
        if points.size == 0:
            points = [0.5]
        return points
    
    def Gibbs_inference(self, data, points_hawkes, t_index, t_star, T_star, T_end, num_grid, num_iter, initial_lamb, initial_weight):
        T = T_end - T_star
        N = len(points_hawkes)
        if initial_weight is None:
            W = np.random.uniform(-1,1,size=(self.number_of_basis+1))
        else:
            W = copy.copy(initial_weight)
        lamda = N / T
        lamda_list=[]
        W_list=[]
        logl=[]
        sig_w = [100] * (1+self.number_of_basis)
        sig_w = np.array(sig_w)
        K_sigma = 1
        K=np.eye(self.number_of_basis+1)*K_sigma
        K_inv = np.linalg.inv(K)
        rlp = np.zeros(t_index + 1)
        for i in range(t_star, t_index + 1):
            rlp[i] = 1/N

        for ite in range(num_iter):
            T = T_end - T_star
            N = len(points_hawkes)
            w_n = np.zeros(N)  
            t_m = []
            w_m = []
            Phi_m = []
            Phi_n, Phi_g = self.Phi_n_g(points_hawkes, np.linspace(T_star,T_end,num_grid))
            intensity_g = np.zeros(num_grid)

            # sample w_n
            for n in range(N):
                w_n[n]=self.PG.pgdraw(1,W.dot(Phi_n[n]))

            # sample t_m and w_m
            for g in range(num_grid):
                intensity_g[g]=lamda*expit(-W.dot(Phi_g[g]))
            intensity_g = np.maximum(intensity_g, 1e-2)
            t_m = self.inhomo_simulation(intensity_g,T)
            Phi_m=np.array([self.Phi_t(t_, points_hawkes) for t_ in t_m])
            w_m=np.array([self.PG.pgdraw(1,W.dot(Phi_m[m])) for m in range(len(t_m))])
            # print("t_m",len(t_m))

            # sample lambda
            T = np.maximum(T, 1e-6)
            lamda = gamma(a=N+len(t_m),scale=1/T).rvs()

            # sample w
            v = np.array([0.5]*N+[-0.5]*len(t_m))
            Sigma_inv = np.diag(list(w_n) + list(w_m))
            Phi=np.concatenate((Phi_n,Phi_m))
            Sigma_W=np.linalg.inv((Phi.T).dot(Sigma_inv).dot(Phi)+K_inv)
            mean_W = Sigma_W.dot(Phi.T).dot(v)
            W = multivariate_normal(mean=mean_W,cov=Sigma_W).rvs()

            lamda_list.append(lamda.copy())
            W_list.append(W.copy())
               
        return lamda_list,W_list
    
class CoBay_CPD:
    def __init__(self, data):
        number_of_basis=4
        self.toy_model = Gibbs(number_of_basis)
        self.beta_ab=np.array([[50,50,-2],[50,50,-1],[50,50,0],[50,50,1],[1,6,1]])
        # self.beta_ab=np.array([[1,2,-2],[50,50,-1],[50,50,0],[1,6,1]])
        self.T_phi=6.
        self.T = 1
        self.lamda_ub=5.
        self.base_activation=0.
        self.weight=np.array([0.5,0.5,0.5,0.5,0.5])
        self.W = np.array([0.,0.5,0.5,0.5,0.5,0.5])
        self.number_points = 100
        self.predictprob = []
        self.theta = []
        # Import data
        self.data = data
        self.nData = len(self.data)
        self.t = 0
        #self.t0 is the start points of last process
        self.t0 = [0]
        
        # Setup computational effort
        self.rmax = 20     # Max run length 
        self.npts = 100    # Number of posterior samples
        self.nrls = 100    # Number of run length samples
        
        # Setup credible interval
        self.flag_lci = 1      # Test left tail -> fire up drastic drecrease
        self.flag_rci = 1      # Test right tail -> fire up drastic increase
        self.risk_level_l =9.5      # Left percentage of probability risk     
        self.risk_level_r =0.5   # Right percentage of probability risk
        self.pred_mean = np.zeros(self.nData)                         # Predictive mean
        if self.flag_lci: self.percentile_l = np.zeros(self.nData)    # Left percentile (if flagged up)
        if self.flag_rci: self.percentile_r = np.zeros(self.nData)    # Right percentile (if flagged up)

        # Risk tolerance
        self.riskTol_l = 0.1
        self.riskTol_r = 0.1
        
        # Initialize changepoints
        self.changepoints = {}
    
    def apply(self):            
        for t_ in range(1,self.nData):
            self.T = t_
            self.num_iter = 100
            print('Time:', t_)
            self.toy_model.set_hawkes_hyperparameters(self.beta_ab, self.T_phi)
            self.toy_model.set_hawkes_parameters(self.lamda_ub, self.base_activation, self.weight)

            #update parameters
            T_end = self.data[t_ - 1]  
            t0 = self.t0[t_ - 1]
            T_star = self.data[t_ - t0 - 1]
            points_hawkes = self.data[t_ - t0 - 1:t_]
            t_star = t_ - t0 - 1
            t_end = t_ - 1
            if len(points_hawkes) > self.rmax:
                points_hawkes = self.data[t_ - self.rmax: t_]
                T_star =self.data[t_ - self.rmax]
                t_star = t_ - self.rmax
            if t0 == 0:
                bool_test = 0
            else:
                lamb_gibbs, W_gibbs= self.toy_model.Gibbs_inference(self.data[:t_], points_hawkes, \
                    t_end, t_star, T_star, T_end, num_grid = math.ceil(5*(T_end - T_star)), num_iter = self.num_iter, \
                        initial_lamb = self.lamda_ub, initial_weight = self.W)

                lamb_mean = np.sum(lamb_gibbs[-11:-1])/10
                w_meanpar = np.ones((10,1))*(1/10)    
                W_mean = np.array(W_gibbs[-11:-1]).T.dot(w_meanpar)
                weight=W_mean.T
                
                self.lamda_ub=lamb_mean
                self.base_activation=weight[0][0]
                self.weight=weight[0][1:]
                self.W = weight[0]
            self.toy_model.set_hawkes_hyperparameters(self.beta_ab, self.T_phi)
            self.toy_model.set_hawkes_parameters(self.lamda_ub, self.base_activation, self.weight)
            theta = np.append(self.lamda_ub, self.W)
            self.theta.append(theta)
            print("theta=",theta)

            #generate new points
            points_hawkes_new = np.zeros(self.number_points)
            for i in range(self.number_points):
                points_hawkes_new[i] = self.toy_model.simulation_one()
            self.data_new = points_hawkes_new + self.data[t_ - 1]

            #check t_th point whether if CP
            datat = self.data[t_ ]
            self.pred_mean[t_] = np.mean(self.data_new)
            if self.flag_lci == 1:
                self.percentile_l[t_] = np.percentile(self.data_new, self.risk_level_l)        
            if self.flag_rci == 1:
                self.percentile_r[t_] = np.percentile(self.data_new, 100 - self.risk_level_r) 
            bool_test = 0
            if self.t0[t_ - 1]>20 and t_ > 30:
                if self.flag_lci:  
                    if datat > self.percentile_r[t_]: 
                        risk = np.abs( ( datat - self.percentile_r[t_] ) / ( self.pred_mean[t_] - self.percentile_r[t_] ) )
                        print("risk_r=",risk)
                        if risk > self.riskTol_r:
                            self.changepoints.update( {t_: risk} )   
                            print('Changepoint at time', t_, 'for drastic increase')
                            print("inrisk:",risk,"value",  datat - self.percentile_r[t_])
                            bool_test = 1
                if self.flag_lci:           
                    if datat < self.percentile_l[t_]: 
                        risk = np.abs( ( datat - self.percentile_l[t_] ) / ( self.pred_mean[t_] - self.percentile_l[t_] ) )
                        print("risk_l=",risk)
                        if risk > self.riskTol_l:
                            self.changepoints.update( {t_ : risk} )   
                            print('Changepoint at time', t_ , 'for drastic decrease')
                            print("derisk:",risk,"value",  datat - self.percentile_l[t_+1])
                            bool_test = 1

            
            if bool_test == 1:
                if t_ == 0:
                    self.t0 = [0]
                else:
                    self.t0.append(0)
            else:
                if t_ == 0:
                    self.t0 = [0]
                else:
                    self.t0.append(self.t0[-1] + 1) 