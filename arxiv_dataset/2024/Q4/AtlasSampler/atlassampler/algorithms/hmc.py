import os
import sys
import numpy as np
from scipy.stats import multivariate_normal

#parent_dir_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
#sys.path.append(parent_dir_name)
from ..util import Sampler, DualAveragingStepSize, PrintException


__all__ = ["HMC"]


class HMC():

    def __init__(self, D, log_prob, grad_log_prob, mass_matrix=None, dtype=np.float64):

        self.D = D
        self.log_prob, self.grad_log_prob = log_prob, grad_log_prob
        self.V = lambda x : self.log_prob(x)*-1.

        if mass_matrix is None: self.mass_matrix = np.eye(D).astype(dtype)
        else: self.mass_matrix = mass_matrix.astype(dtype)
        self.inv_mass_matrix = np.linalg.inv(self.mass_matrix).astype(dtype)
        self.KE =  lambda p : 0.5*np.dot(p, np.dot(self.mass_matrix, p)).astype(dtype)
        self.KE_g =  lambda p : np.dot(self.mass_matrix, p).astype(dtype)
        
        self.leapcount = 0
        self.Vgcount = 0
        self.Hcount = 0


    def adapt_stepsize(self, q, n_stepsize_adapt, target_accept=0.65):
        '''Adapt stepsize to meet the target_accept rate with dual averaging.
        '''
        print("Adapting step size for %d iterations"%n_stepsize_adapt)
        step_size = self.step_size
        n_stepsize_adapt_kernel = DualAveragingStepSize(step_size, 
                                                        target_accept=target_accept)

        for i in range(n_stepsize_adapt+1):
            qprev = q.copy()
            q, p, acc, Hs = self.hmc_step(q, self.n_leapfrog, step_size)
            if (qprev == q).all():
                prob = 0 
            else:
                prob = np.exp(Hs[0] - Hs[1])

            if i < n_stepsize_adapt:
                if np.isnan(prob) or np.isinf(prob): 
                    prob = 0.
                    continue
                if prob > 1: prob = 1.
                step_size, avgstepsize = n_stepsize_adapt_kernel.update(prob)
            elif i == n_stepsize_adapt:
                _, step_size = n_stepsize_adapt_kernel.update(prob)
                print("Step size fixed to : ", step_size)
                self.step_size = step_size
        return q
        

    def V_g(self, x):
        '''Gradient of negative log probability
        '''
        self.Vgcount += 1
        v_g = self.grad_log_prob(x)
        return v_g *-1.
    
    def H(self, q, p, M=None):
        self.Hcount += 1
        Vq = self.V(q)
        Kq = self.KE(p)
        return Vq + Kq

    def leapfrog(self, q, p, N, step_size, M=None, g=None):        
        self.leapcount += 1        
        qvec, gvec = [], []
        q0, p0 = q, p
        g0 = g
        try:
            if g0 is not None:
                g = g0
                g0 = None
            else:
                g =  self.V_g(q)
            p = p - 0.5*step_size * g
            qvec.append(q)
            gvec.append(g)
            for i in range(N-1):
                q = q + step_size * self.KE_g(p)
                g = self.V_g(q)
                p = p - step_size * g
                qvec.append(q)
                gvec.append(g)
            q = q + step_size * self.KE_g(p)
            g = self.V_g(q)
            p = p - 0.5*step_size * g
            qvec.append(q)
            gvec.append(g)            
            return q, p, qvec, gvec

        except Exception as e:  # Sometimes nans happen. 
            if self.verbose: PrintException()
            return q0, p0, qvec, gvec


    def accept_log_prob(self, qp0, qp1, return_H=False):
        '''Evaluate the Hamiltonian and acceptance log_prob
        '''
        try:
            q0, p0 = qp0
            q1, p1 = qp1
            H0 = self.H(q0, p0)
            H1 = self.H(q1, p1)
            log_prob = H0 - H1
            if np.isnan(log_prob)  or (q0-q1).sum()==0:
                log_prob = -np.inf
            log_prob = min(0., log_prob)
            if return_H is False: return log_prob
            else: return log_prob, H0, H1
        except Exception as e:
            PrintException()
            log_prob, H0, H1 = np.NaN, np.NaN, np.NaN
            if return_H is False: return log_prob
            else: return log_prob, H0, H1
            

    def metropolis(self, qp0, qp1, M=None):
        '''Accept/reject the sample based on Metropolis criterion
        '''
        try:
            log_prob, H0, H1 = self.accept_log_prob(qp0, qp1, return_H=True)
            q0, p0 = qp0
            q1, p1 = qp1
            u =  np.random.uniform(0., 1., size=1)
            if  np.log(u) > min(0., log_prob):
                return q0, p0, -1., [H0, H1]
            else:
                return q1, p1, 1., [H0, H1]
        except Exception as e:
            PrintException()
            q0, p0 = qp0
            return q0, p0, -1, [np.NaN, np.NaN]

    # Quality of life function to inherit for standard stepping
    def hmc_step(self, q, n_leapfrog=None, step_size=None): 
        '''One iteration of HMC
        '''
        if n_leapfrog is None: n_leapfrog = self.n_leapfrog
        if step_size is None: step_size = self.step_size
        self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0 # reset counts
        
        p =  multivariate_normal.rvs(mean=np.zeros(self.D), cov=self.inv_mass_matrix, size=1)
        q1, p1, qvec, gvec = self.leapfrog(q, p, N=n_leapfrog, step_size=step_size)
        qf, pf, accepted, Hs = self.metropolis([q, p], [q1, p1])
        return qf, pf, accepted, Hs

    
    def step(self, q, n_leapfrog=None, step_size=None):
        '''One step of the algorithm'''        
        return self.hmc_step(q, n_leapfrog, step_size)

    
    def sample(self, q, p=None,
               seed=99,
               n_samples=100, n_burnin=0, step_size=0.1, n_leapfrog=10,
               n_stepsize_adapt=0, target_accept=0.65, jitter_n_leapfrog=True, 
               verbose=False):

        state = Sampler()
        np.random.seed(seed)
        self.step_size = step_size
        self.n_leapfrog = n_leapfrog
        self.verbose = verbose

        if jitter_n_leapfrog:           # Setup function for leapfrog steps
            self.n_leapfrog_dist = lambda : np.random.randint(1, self.n_leapfrog)
        else:
            self.n_leapfrog_dist = lambda : self.n_leapfrog

        if n_stepsize_adapt:            # Adapt stepsize       
           q = self.adapt_stepsize(q, n_stepsize_adapt, target_accept=target_accept) 

        for i in range(n_burnin):       # Burnin
            q, p, accepted, Hs = self.step(q, n_leapfrog=self.n_leapfrog_dist(), step_size=self.step_size) 

        for i in range(n_samples):     # Sample
            q, p, accepted, Hs = self.step(q, n_leapfrog=self.n_leapfrog_dist(), step_size=self.step_size) 
            state.appends(q=q, accepted=accepted, Hs=Hs, gradcount=self.Vgcount, energycount=self.Hcount)
            state.i += 1
            
        state.to_array()
        return state
