import sys
import numpy as np
from scipy.stats import multivariate_normal

from .uturn_samplers import HMC_Uturn_Jitter
from ..util import Sampler, PrintException, power_iteration
from ..hessians import Hessian_approx
from ..distributions import setup_stepsize_distribution

# Setup MPI environment
# from mpi4py import MPI
# comm = MPI.COMM_WORLD
# wrank = comm.Get_rank()
# wsize = comm.Get_size()


class DRHMC_AdaptiveStepsize(HMC_Uturn_Jitter):
    """Adapting stepsize in delayed rejection framework.
    
    Warmup: 
    In the warmup phase, the algorithm first adapts the stepsize and the trajectory length as follows:
        1. Baseline stepsize is adapted with dual averaging to target an acceptance rate of ``target_accept``.
        2. If ``n_leapfrog_adapt`` is 0, then it uses the `n_leapfrog` as the number of leapfrog steps. 
        Else it constructs an empirical distribution of trajectory lengths (edist_traj).
        It run ``n_leapfrog_adapt`` iterations upto U-turn in every chain and stores the trajectory length.
        Then it combines this information from all chains, removes the lenghts that are too short or too long
        (outside ``low_nleap_percentile`` and ``high_nleap_percentile``) and saves the remaining lengths
        to construct edist_traj.

    Sampling:
    This is a 2-step delayed rejection version of HMC with adaptive stepsize. The algorithm has two stages:

        1. In the first stage, the trajectory length is sampled from the empirical distribution (edist_traj)
        constructed in warmup and integration is done with baseline stepsize. If this is accepted,
        algorithm moves to next iteration. If not, then it moves to delayed stage.

        2. Delayed stage: This step locally adapts stepsize by constructing an approximate local Hessian
        using the trajectory from the previous stage. If ``constant_trajectory`` = 0, the number of leapfrog steps
        is the same as the first stage, otherwise  it scaled by the ratio of stepsize to baseline stepsize.
    """
    def __init__(self, D, log_prob, grad_log_prob, mass_matrix=None, 
                 n_hessian_samples=10, n_hessian_attempts=10, 
                 max_stepsize_reduction=1000,
                 constant_trajectory=False,
                 hessian_mode='bfgs',
                 stepsize_distribution='beta',
                 stepsize_sigma=2.,
                 **kwargs):
        super(DRHMC_AdaptiveStepsize, self).__init__(D=D, log_prob=log_prob, grad_log_prob=grad_log_prob,
                                                     mass_matrix=mass_matrix, **kwargs)        
        self.n_hessian_samples=n_hessian_samples
        self.n_hessian_attempts = n_hessian_attempts
        self.max_stepsize_reduction = max_stepsize_reduction
        self.constant_trajectory = constant_trajectory
        self.hessian_mode = hessian_mode
        self.stepsize_distribution = stepsize_distribution
        self.stepsize_sigma = stepsize_sigma
        
    def get_stepsize_distribution(self, q0, p0, qlist, glist, step_size):
        """
        Construct a distribution for stepsize at the current point (q0, p0).
        This is done in three steps-
            1. Approximate the Hessian using the list of points (qlist) and gradients (glist).
            2. Estimate the largest eigenvalue with power iteration and approximate the largest stable stepsize.
            3. Use this to construct a suitable proposal distribution for stepsize.
        """
        est_hessian = True
        i = 0
        while (est_hessian) & (i < self.n_hessian_attempts):
            if i > 0:                       # something went wrong, reduce step size and try again
                step_size /= 2.
                if self.verbose: print(f'{i}, halve stepsize', step_size)
                q1, p1, qlist, glist = self.leapfrog(q0, p0, N=self.n_hessian_samples + 1, step_size=step_size)

            qs, gs = [], []
            for ig, g in enumerate(glist):  # discard any nans from the trajectory
                if np.isnan(g).any():
                    pass
                else:
                    qs.append(qlist[ig])
                    gs.append(glist[ig])
            if len(qs) < self.n_hessian_samples:
                if self.verbose: print('nans in g')
                i+= 1
                continue

            # Estimate Hessian now
            h_est, points_used = Hessian_approx(positions = np.array(qs[::-1]), 
                                                gradients = np.array(gs[::-1]), 
                                                H = None, 
                                                mode = self.hessian_mode)
            if (points_used < self.n_hessian_samples) :
                if self.verbose: print('skipped too many')
                i += 1
                continue
            elif  np.isnan(h_est).any():
                if self.verbose: print("nans in H")
                i+=1
                continue
            else:
                est_hessian = False

        if est_hessian:
            print(f"step size reduced to {step_size} from {self.step_size}")
            print("Exceeded max attempts to estimate Hessian")
            eps_mean = 4 * step_size/self.max_stepsize_reduction
            # raise RecursionError
        else:
            eigv = power_iteration(h_est + np.eye(self.D)*1e-6)[0]
            if eigv < 0:
                print("negative eigenvalue : ", eigv)
                eps_mean = 4 * step_size/self.max_stepsize_reduction
                # raise ArithmeticError            
            else: 
                eps_mean = min(0.5*step_size, 0.5*np.sqrt(1/ eigv))
                
        if eps_mean < step_size/self.max_stepsize_reduction:
            eps_mean = 4 * step_size/self.max_stepsize_reduction
        epsf = setup_stepsize_distribution(epsmean = eps_mean, 
                                            epsmax = step_size, 
                                            epsmin = step_size/self.max_stepsize_reduction, 
                                            distribution = self.stepsize_distribution,
                                            stepsize_sigma = self.stepsize_sigma)
        return eps_mean, epsf
        
        

    def delayed_step(self, q0, p0, qlist, glist, n_leapfrog, step_size, log_prob_accept1):
        """Adapt stepsize for the second proposal using the rejected trajectory."""       
        try:
            # Estimate the Hessian given the rejected trajectory, use it to estimate step-size
            eps1, epsf1 = self.get_stepsize_distribution(q0, p0, qlist, glist, step_size)
            step_size_new = epsf1.rvs(size=1)[0]
            
            # Make the second proposal
            if self.constant_trajectory:
                nleap_new = int(min(n_leapfrog*step_size/step_size_new, self.max_nleapfrog))
            else:
                nleap_new = int(n_leapfrog)
            q1, p1, _, _ = self.leapfrog(q0, p0, nleap_new, step_size_new)
            log_prob_H, H0, H1 = self.accept_log_prob([q0, p0], [q1, p1], return_H=True)
            
            # Ghost trajectory and corresponding stepsize distribution
            q1_ghost, p1_ghost, qlist_ghost, glist_ghost = self.leapfrog(q1, -p1, n_leapfrog, step_size)
            log_prob_accept2 = self.accept_log_prob([q1, -p1], [q1_ghost, p1_ghost])
            eps2, epsf2 = self.get_stepsize_distribution(q1, -p1, qlist_ghost, glist_ghost, step_size)
            steplist = [eps1, eps2, step_size_new]

            # Calcualte different Hastings corrections
            if log_prob_accept2 == 0: # ghost proposal is always accepted
                qf, pf = q0, p0
                accepted = -1
            else:            
                log_prob_delayed = np.log((1-np.exp(log_prob_accept2))) - np.log((1- np.exp(log_prob_accept1)))
                log_prob_eps = epsf2.logpdf(step_size_new) - epsf1.logpdf(step_size_new)
                log_prob = log_prob_H + log_prob_eps + log_prob_delayed

                # Accept/reject and return           
                u =  np.random.uniform(0., 1., size=1)
                if np.isnan(log_prob) or (q0-q1).sum()==0:
                    qf, pf = q0, p0
                    accepted = -99
                elif  np.log(u) > min(0., log_prob):
                    qf, pf = q0, p0
                    accepted = -1
                else: 
                    qf, pf = q1, p1
                    accepted = 2
               
            return qf, pf, accepted, [H0, H1], steplist
            
        except Exception as e:
            PrintException()
            qf, pf = q0, p0
            accepted = -99
            Hs, steplist = [0, 0],[0, 0, 0]
            return qf, pf, accepted, Hs, steplist
        
        
    def step(self, q, n_leapfrog, step_size=None):

        if step_size is None: step_size = self.step_size
        self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0
        
        p =  multivariate_normal.rvs(mean=np.zeros(self.D), cov=self.inv_mass_matrix, size=1)
        q1, p1, qlist, glist = self.leapfrog(q, p, N=n_leapfrog, step_size=step_size)
        qf, pf, accepted, Hs = self.metropolis([q, p], [q1, p1])
                            
        if (accepted <= 0):
            log_prob_accept1 = (Hs[0] - Hs[1])
            if np.isnan(log_prob_accept1) or (q-q1).sum() == 0 :
                log_prob_accept1 = -np.inf
            log_prob_accept1 = min(0, log_prob_accept1)
            qf, pf, accepted, Hs, steplist = self.delayed_step(q, p, qlist, glist, 
                                                            n_leapfrog=n_leapfrog, step_size=step_size,
                                                            log_prob_accept1=log_prob_accept1)
        else:
            steplist = [0, 0, step_size]
        return qf, pf, accepted, Hs, steplist



    def sample(self, q, p=None,
               n_samples=100, n_burnin=0, step_size=0.1, n_leapfrog=10,
               n_stepsize_adapt=0, n_leapfrog_adapt=100,
               target_accept=0.65, 
               seed=99,
               comm=None,
               verbose=False):
    
        state = Sampler()
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)
        self.step_size = step_size
        self.n_leapfrog = n_leapfrog
        self.verbose = verbose

        # More state variables to keep track of
        state.steplist = []
               
        if n_stepsize_adapt:
            q = self.adapt_stepsize(q, n_stepsize_adapt, target_accept=target_accept)

        if n_leapfrog_adapt:  # Construct a distribution of trajectory lengths
            q = self.adapt_trajectory_length(q, n_leapfrog_adapt)
            self.combine_trajectories_from_chains(comm)
            state.trajectories = self.traj_array
            print(f"Shape of trajectories after bcast : ", self.traj_array.shape)
            self.nleapfrog_jitter()
        else:
            self.nleapfrog_jitter_dist = lambda x:  self.n_leapfrog
                      
        for i in range(n_burnin):  # Burnin
            n_leapfrog = self.nleapfrog_jitter_dist(step_size)
            q, p, accepted, Hs, steplist = self.step(q, n_leapfrog=n_leapfrog, step_size=self.step_size) 

        for i in range(n_samples):  # Sample
            state.i += 1
            n_leapfrog = self.nleapfrog_jitter_dist(step_size)
            q, p, accepted, Hs, steplist = self.step(q, n_leapfrog=n_leapfrog, step_size=self.step_size) 
            state.appends(q=q, accepted=accepted, Hs=Hs, gradcount=self.Vgcount, energycount=self.Hcount)
            state.steplist.append(steplist)
            if (i%(n_samples//10) == 0):
                print(f"Iteration {i} of {n_samples}")

        state.to_array()
        return state
