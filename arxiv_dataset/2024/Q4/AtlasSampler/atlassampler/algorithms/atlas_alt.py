import sys
import numpy as np
from scipy.stats import multivariate_normal

from ..util import Sampler, PrintException
from .stepadapt_samplers import DRHMC_AdaptiveStepsize

# Setup MPI environment
from mpi4py import MPI
comm = MPI.COMM_WORLD
wrank = comm.Get_rank()
wsize = comm.Get_size()

__all__ = ["Atlas_Uturn", "Atlas_HMC"]

class Atlas_Uturn(DRHMC_AdaptiveStepsize):
    """Atlas is "Adaptive Trajectory Length And Stepsize" sampler. 
    """
    def __init__(self, D, log_prob, grad_log_prob, mass_matrix=None, 
                 offset=None,
                 offset_delayed=None,
                 constant_trajectory=0,
                 probabilistic=0,
                 low_nleap_percentile=10, 
                 high_nleap_percentile=90, 
                 **kwargs):
        """Initialize an instance of Atlas.
        
        Args:
            D (int): dimensionality of the parameter space
            log_prob (function): function that takes in parameter value and returns log_probability
            grad_log_prob (function):  function that takes in parameter value 
                and returns the gradient of the log_probability
            offset (float in range 0-1, default=None): sample the proposal from [offset, 1) fraction
                of the no-U turn trajectory. If None, offset is shuffled and randomly picked
                between [0.33, 0.66] for every iteration.      
            constant_trajectory (0, 1, or 2, default=1): determine if number of leapfrog steps for
                delayed proposals are scaled by stepsize ratio. 0: no scaling for any delayed proposal. 
                1: only delayed proposal on failure are scaled. 2: both delayed proposals are sacled 
            probabilistic (0 or 1, default=0): wether to make delayed proposal without failure
                0: always make second delayed proposal. 1: make delayed proposal only the first proposal
                is rejected not due to a sub u-turn  
        
        Keyword args:
            min_nleapfrog (int, default=3): minimum number of leapfrog steps.
                use delayed_rejection_with_failure if number of steps in first proposal is less that it.
            max_nleapfrog (int, default=1024): maximum number of leapfrog steps in any iteration.
            n_hessian_samples (int, default=10): minimum number of points needed to evaluate Hessian
            n_hessian_attempts (int, default=10): maximum number of attempts to evaluate Hessian
                by reducing step-size with a factor of 2 on each attempt. 
            low_nleap_percentile (int, default=10): percentile below which u-turn trajectories
                in the empirical trajectory length distribution of warmup are discarded.
            high_nleap_percentile (int, default=50): percentile above which u-turn trajectories
                in the empirical trajectory length distribution of warmup are discarded.
            nleap_factor (float, default=1.): scale the empirical distribution of trajectory lengths.
            max_stepsize_reduction (float, default=500): minimum possible step-size is set to be the
                baseline stepsize/max_stepsize_reduction
        """

        super(Atlas_Uturn, self).__init__(D=D, log_prob=log_prob, grad_log_prob=grad_log_prob, 
                                    mass_matrix=mass_matrix, offset=offset,
                                    low_nleap_percentile=low_nleap_percentile,
                                    high_nleap_percentile=high_nleap_percentile, 
                                    **kwargs)
        if self.min_nleapfrog <= 2:
            print("min nleapfrog should at least be 2 for delayed proposals")
            raise
        self.probabilistic = probabilistic
        self.constant_trajectory = constant_trajectory
        self.offset_delayed = offset_delayed
        
    def preliminary_step(self, q, p, step_size, offset, n_leapfrog=None):
        """
        First step is making a no-uturn proposal.
        """
        try:
            # Go forward
            Nuturn, qlist, plist, glist, success = self.nuts_criterion(q, p, step_size)
            if Nuturn < self.min_nleapfrog :
                # if self.verbose: 
                log_prob_list = [-np.inf, -np.inf, -np.inf]
                Hs, n_leapfrog = [0, 0], 0.
                return [q, p], [qlist, plist, glist], log_prob_list, Hs, n_leapfrog

            else:
                if (n_leapfrog is not None):
                    if (Nuturn <= n_leapfrog): 
                        # If ghost trajectory can never reach n_leapfrog proposals
                        # Hence this should never make a DR proposal. Thus set log_prob_accept=0
                        log_prob_list, Hs = [0., 0., 0.], [0., 0.]
                        return [q, p], [qlist, plist, glist], log_prob_list, Hs, n_leapfrog

                n_leapfrog, lp_N = self.nleapfrog_sample_and_lp(Nuturn,  offset, nleapfrog=n_leapfrog)
                q1, p1 = qlist[n_leapfrog], plist[n_leapfrog]
                
                # Go backward
                Nuturn_rev, qlist_rev, plist_rev, glist_rev, success = self.nuts_criterion(q1, -p1, step_size)
                n_leapfrog_rev, lp_N_rev = self.nleapfrog_sample_and_lp(Nuturn_rev, offset, nleapfrog=n_leapfrog)
                assert n_leapfrog_rev == n_leapfrog
                self.Vgcount -= min(Nuturn_rev, n_leapfrog) #adjust for number of common steps when going backward
                
                # Hastings
                log_prob_H, H0, H1 = self.accept_log_prob([q, p], [q1, p1], return_H=True)
                log_prob_N = lp_N_rev - lp_N 
                log_prob_accept = log_prob_H + lp_N_rev - lp_N 
                log_prob_accept = min(0, log_prob_accept)
                log_prob_list = [log_prob_accept, log_prob_H, log_prob_N]
                
                return [q1, p1], [qlist, plist, glist], log_prob_list, [H0, H1], n_leapfrog

        except Exception as e:
            PrintException()
            return [q, p], [[], [], []], [-np.inf, 0., 0.], [0, 0], 0.

        
    def delayed_step(self, q0, p0, qlist, glist, step_size, offset, n_leapfrog, log_prob_accept_first):
        ##
        try:
            # Estimate the Hessian given the rejected trajectory, use it to estimate step-size
            eps1, epsf1 = self.get_stepsize_distribution(q0, p0, qlist, glist, step_size)
            step_size_new = epsf1.rvs(size=1)[0]
            
            # Make the second proposal
            if self.constant_trajectory == 2:
                n_leapfrog_new = int(min(self.max_nleapfrog, n_leapfrog*step_size / step_size_new))
            else:
                n_leapfrog_new = n_leapfrog                
            q1, p1, _, _ = self.leapfrog(q0, p0, n_leapfrog_new, step_size_new)
            log_prob_H, H0, H1 = self.accept_log_prob([q0, p0], [q1, p1], return_H=True)
            
            # Ghost trajectory for the second proposal
            qp_ghost, qpg_ghost_list, log_prob_ghost_list, Hs_ghost, n_leapfrog_ghost = \
                                        self.preliminary_step(q1, -p1, step_size, offset=offset, n_leapfrog=n_leapfrog)

            if n_leapfrog_ghost == 0: #in this case, ghost stage cannot lead to the current delayed propsal
                qf, pf, accepted = q0, p0, -14

            else:
                log_prob_accept_ghost, log_prob_H_ghost, log_prob_N_ghost = log_prob_ghost_list
                qlist_ghost, plist_ghost, glist_ghost = qpg_ghost_list
                log_prob_delayed = np.log((1-np.exp(log_prob_accept_ghost))) - np.log((1- np.exp(log_prob_accept_first)))
                if self.probabilistic == 1:
                    if log_prob_N_ghost == -np.inf: 
                        log_prob_delayed = -np.inf

                # Hastings
                eps2, epsf2 = self.get_stepsize_distribution(q1, -p1, qlist_ghost, glist_ghost, step_size)
                log_prob_eps = epsf2.logpdf(step_size_new) - epsf1.logpdf(step_size_new)
                log_prob_accept = log_prob_H + log_prob_delayed + log_prob_eps

                u =  np.random.uniform(0., 1., size=1)
                if np.isnan(log_prob_accept) or (q0-q1).sum()==0:
                    qf, pf, accepted = q0, p0, -99
                elif  np.log(u) > min(0., log_prob_accept):
                    qf, pf = q0, p0
                    if log_prob_delayed == -np.inf:
                        accepted = -12
                    elif log_prob_eps == -np.inf:
                        accepted = -13
                    else:
                        accepted = -11
                else: 
                    qf, pf, accepted = q1, p1, 2

            return qf, pf, accepted, [H0, H1]
            
        except Exception as e:
            PrintException()
            return q0, p0, -199, [0, 0]
        

    def delayed_step_upon_failure(self, q0, p0, qlist, glist, step_size, offset, n_leapfrog, log_prob_accept_first):
        """Delayed step upon failure is executed when Nuturn of first step < min_nleapfrog.
        """
        try:
            # Estimate the Hessian and the stepsize given the rejected trajectory
            eps1, epsf1 = self.get_stepsize_distribution(q0, p0, qlist, glist, step_size)
            step_size_new = epsf1.rvs(size=1)[0]
            
            # Make the second proposal
            if self.offset_delayed is None:
                offset_new = self.offset_function()
            elif self.offset_delayed == 1:
                offset_new = self.offset_function()
            else:
                offset_new = self.offset_delayed
                
            qp, qpg_list, log_prob_list, Hs, n_leapfrog = self.preliminary_step(q0, p0, step_size_new, offset=offset_new)
            log_prob_accept_uturn, log_prob_H, log_prob_N = log_prob_list
            q1, p1 = qp
            qlist, plist, glist = qpg_list
            H0, H1 = Hs

            # Ghost trajectory for the second proposal
            qp_ghost, qpg_list_ghost, log_prob_ghost_list, Hs_ghost, n_leapfrog_ghost = \
                                            self.preliminary_step(q1, -p1, step_size, offset=offset, n_leapfrog=n_leapfrog)
            
            if n_leapfrog_ghost != 0: #in this case, ghost stage cannot lead to the current delayed propsal
                qf, pf, accepted = q0, p0, -24
                
            else :
                log_prob_accept_ghost, log_prob_H_ghost, log_prob_N_ghost = log_prob_ghost_list
                qlist_ghost, plist_ghost, glist_ghost = qpg_list_ghost
                eps2, epsf2 = self.get_stepsize_distribution(q1, -p1, qlist_ghost, glist_ghost, step_size)

                # Hastings
                log_prob_delayed = np.log((1-np.exp(log_prob_accept_ghost))) - np.log((1- np.exp(log_prob_accept_first)))
                log_prob_eps = epsf2.logpdf(step_size_new) - epsf1.logpdf(step_size_new)
                log_prob_accept = log_prob_accept_uturn + log_prob_delayed + log_prob_eps

                u =  np.random.uniform(0., 1., size=1)
                if np.isnan(log_prob_accept) or (q0-q1).sum()==0:
                    qf, pf, accepted = q0, p0, -99
                elif  np.log(u) > min(0., log_prob_accept):
                    qf, pf = q0, p0
                    if log_prob_delayed == -np.inf:
                        accepted = -22
                    elif log_prob_eps == -np.inf:
                        accepted = -23
                    else:
                        accepted = -21
                else: 
                    qf, pf, accepted = q1, p1, 3
                    
            return qf, pf, accepted, [H0, H1]
            
        except Exception as e:
            PrintException()
            print("exception : ", e)
            return q0, p0, -299, [0, 0]
        

    def step(self, q, n_leapfrog=None, step_size=None):

        if step_size is None: step_size = self.step_size
        self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0
        
        p =  multivariate_normal.rvs(mean=np.zeros(self.D), cov=self.inv_mass_matrix, size=1)
        offset = self.offset_function()
        qp, qpg_list, log_prob_list, Hs, n_leapfrog = self.preliminary_step(q, p, step_size, offset=offset)
        log_prob_accept, log_prob_H, log_prob_N = log_prob_list
        q1, p1 = qp
        qlist, plist, glist = qpg_list
        
        # Hastings
        if n_leapfrog == 0 :
            return self.delayed_step_upon_failure(q, p, qlist, glist,
                                                step_size=step_size, offset=offset, n_leapfrog=n_leapfrog,
                                                log_prob_accept_first=log_prob_accept)
        else:
            u =  np.random.uniform(0., 1., size=1)
            if  np.log(u) > min(0., log_prob_accept):
                if self.probabilistic == 1:
                    if log_prob_N == -np.inf:
                        return q, p, -1, Hs
                    else:
                        return self.delayed_step(q, p, qlist, glist,
                                                step_size=step_size, offset=offset, n_leapfrog=n_leapfrog,
                                                log_prob_accept_first=log_prob_accept)
                else:
                    return self.delayed_step(q, p, qlist, glist, 
                                            step_size=step_size, offset=offset, n_leapfrog=n_leapfrog,
                                            log_prob_accept_first=log_prob_accept)
            else:
                qf, pf = q1, p1
                accepted = 1
                return qf, pf, accepted, Hs
        
            
    def sample(self, q, p=None,
               n_samples=100, n_burnin=0, step_size=0.1, n_leapfrog=10,
               n_stepsize_adapt=0, n_leapfrog_adapt=100,
               target_accept=0.65, 
               seed=99,
               verbose=False):
    
        state = Sampler()
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)
        self.step_size = step_size
        self.n_leapfrog = n_leapfrog
        self.verbose = verbose

        if n_stepsize_adapt:
            q = self.adapt_stepsize(q, n_stepsize_adapt, target_accept=target_accept) 

        if n_leapfrog_adapt:  # Construct a distribution of trajectory lengths
            print("No leapfrog steps to adapt")
        else:
            self.nleapfrog_jitter_dist = lambda x:  self.n_leapfrog
            
        for i in range(n_burnin):  # Burnin
            q, p, accepted, Hs = self.step(q) 

        for i in range(n_samples):  # Sample
            state.i += 1
            q, p, accepted, Hs = self.step(q) 
            state.appends(q=q, accepted=accepted, Hs=Hs, gradcount=self.Vgcount, energycount=self.Hcount)
            if (i%(n_samples//10) == 0):
                print(f"In rank {wrank}: iteration {i} of {n_samples}")

        state.to_array()
        return state

    
######################
class Atlas_HMC(DRHMC_AdaptiveStepsize):
    """Atlas is "Adaptive Trajectory Length And Stepsize" sampler. 
    """
    def __init__(self, D, log_prob, grad_log_prob, mass_matrix=None, 
                 offset=None,
                 stepinit_factor=1.,
                 **kwargs):
        """Initialize an instance of Atlas.
        """
        super(Atlas_HMC, self).__init__(D=D, log_prob=log_prob, grad_log_prob=grad_log_prob, 
                                    mass_matrix=mass_matrix, offset=offset,
                                    **kwargs)
        if self.min_nleapfrog <= 2:
            print("min nleapfrog should at least be 2 for delayed proposals")
            raise
        self.stepinit_factor = stepinit_factor

        
    def preliminary_step(self, q, p, step_size, offset, n_leapfrog=None):
        """
        First step is making a no-uturn proposal.
        """
        try:
            # Go forward
            Nuturn, qlist, plist, glist, success = self.nuts_criterion(q, p, step_size)
            if Nuturn < self.min_nleapfrog :
                # if self.verbose: 
                log_prob_list = [-np.inf, -np.inf, -np.inf]
                Hs, n_leapfrog = [0, 0], 0.
                return [q, p], [qlist, plist, glist], log_prob_list, Hs, n_leapfrog

            else:
                if (n_leapfrog is not None):
                    if (Nuturn <= n_leapfrog): 
                        # If ghost trajectory can never reach n_leapfrog proposals
                        # Hence this should never make a DR proposal. Thus set log_prob_accept=0
                        log_prob_list, Hs = [0., 0., 0.], [0., 0.]
                        return [q, p], [qlist, plist, glist], log_prob_list, Hs, n_leapfrog

                n_leapfrog, lp_N = self.nleapfrog_sample_and_lp(Nuturn,  offset, nleapfrog=n_leapfrog)
                q1, p1 = qlist[n_leapfrog], plist[n_leapfrog]
                
                # Go backward
                Nuturn_rev, qlist_rev, plist_rev, glist_rev, success = self.nuts_criterion(q1, -p1, step_size)
                n_leapfrog_rev, lp_N_rev = self.nleapfrog_sample_and_lp(Nuturn_rev, offset, nleapfrog=n_leapfrog)
                assert n_leapfrog_rev == n_leapfrog
                self.Vgcount -= min(Nuturn_rev, n_leapfrog) #adjust for number of common steps when going backward
                
                # Hastings
                log_prob_H, H0, H1 = self.accept_log_prob([q, p], [q1, p1], return_H=True)
                log_prob_N = lp_N_rev - lp_N 
                log_prob_accept = log_prob_H + lp_N_rev - lp_N 
                log_prob_accept = min(0, log_prob_accept)
                log_prob_list = [log_prob_accept, log_prob_H, log_prob_N]
                
                return [q1, p1], [qlist, plist, glist], log_prob_list, [H0, H1], n_leapfrog

        except Exception as e:
            PrintException()
            return [q, p], [[], [], []], [-np.inf, 0., 0.], [0, 0], 0.        

        
    def step(self, q, n_leapfrog=None):

        self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0
        
        p =  multivariate_normal.rvs(mean=np.zeros(self.D), cov=self.inv_mass_matrix, size=1)
        try:
            eps1, epsf1 = self.get_stepsize_distribution(q, p, [q], [q], self.step_size*self.stepinit_factor)
        except:
            PrintException()
            return q, p, -1, [0, 0]
        step_size = epsf1.rvs(size=1)[0]
        
        offset = self.offset_function()
        qp, qpg_list, log_prob_list, Hs, n_leapfrog = self.preliminary_step(q, p, step_size, offset=offset)
        log_prob_accept_uturn, log_prob_H, log_prob_N = log_prob_list
        q1, p1 = qp
        qlist, plist, glist = qpg_list
        
        try:
            eps2, epsf2 = self.get_stepsize_distribution(q1, -p1, [q1], [q1], self.step_size*self.stepinit_factor)
        except:
            PrintException()
            return q, p, -1, [0, 0]

        log_prob_eps = epsf2.logpdf(step_size) - epsf1.logpdf(step_size)
        log_prob_accept = log_prob_accept_uturn + log_prob_eps
        
        # Hastings
        u =  np.random.uniform(0., 1., size=1)
        if  np.log(u) > min(0., log_prob_accept):
            if log_prob_N == -np.inf:
                return q, p, -2, Hs
            else:
                return q, p, 0, Hs
        else:
            qf, pf = q1, p1
            accepted = 1
            return qf, pf, accepted, Hs
        
            
    def sample(self, q, p=None,
               n_samples=100, n_burnin=0, step_size=0.1, n_leapfrog=10,
               n_stepsize_adapt=0, n_leapfrog_adapt=100,
               target_accept=0.65, 
               seed=99,
               verbose=False):
    
        state = Sampler()
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)
        self.step_size = step_size
        self.n_leapfrog = n_leapfrog
        self.verbose = verbose

        if n_stepsize_adapt:
            q = self.adapt_stepsize(q, n_stepsize_adapt, target_accept=target_accept) 

        if n_leapfrog_adapt:  # Construct a distribution of trajectory lengths
            print("No leapfrog stpes to adapt")
            
        for i in range(n_burnin):  # Burnin
            q, p, accepted, Hs = self.step(q) 

        for i in range(n_samples):  # Sample
            state.i += 1
            q, p, accepted, Hs = self.step(q) 
            state.appends(q=q, accepted=accepted, Hs=Hs, gradcount=self.Vgcount, energycount=self.Hcount)
            if (i%(n_samples//10) == 0):
                print(f"In rank {wrank}: iteration {i} of {n_samples}")

        state.to_array()
        return state
