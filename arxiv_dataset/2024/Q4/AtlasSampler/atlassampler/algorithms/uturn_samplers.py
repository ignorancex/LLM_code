import sys
import numpy as np
from scipy.stats import multivariate_normal, uniform

from .hmc import HMC
from ..util import Sampler, DualAveragingStepSize, PrintException

# # Setup MPI environment
# from mpi4py import MPI
# comm = MPI.COMM_WORLD
# wrank = comm.Get_rank()
# wsize = comm.Get_size()


__all__ = ["HMC_Uturn", "HMC_Uturn_Jitter"]


class HMC_Uturn(HMC):
    """GIST sampler with U-turn criterion.
    
    Warmup: 
    In the warmup phase, the algorithm first adapts the baseline stepsize with dual averaging to
    target an acceptance rate of ``target_accept``.

    Sampling:
    In every iteration, the algorithm runs a trajectory with baseline step_size uptil U-turn.
    Then in makes a proposal in the [``offset``, 1) fraction of the trajectory.
    When ``offset`` = 1, it randomly chooses the offset for every iteration in the range [0.33, 0.66).
    <<Need a better way of handling this argument>>
    """
    def __init__(self, D, log_prob, grad_log_prob, mass_matrix=None, 
                offset=None, min_offset=0.33, max_offset=0.66,
                min_nleapfrog=3, max_nleapfrog=1024, **kwargs):
        super(HMC_Uturn, self).__init__(D=D, 
                                                log_prob=log_prob, 
                                                grad_log_prob=grad_log_prob, 
                                                mass_matrix=mass_matrix)        
        self.min_nleapfrog = min_nleapfrog
        self.max_nleapfrog = max_nleapfrog
        self.exceed_max_steps = 0
        if offset == 1:
            self.offset_function = lambda : np.random.uniform(min_offset, max_offset)
            offset = None
        else:
            self.offset_function = lambda : offset
        # # parse remaining kwargs
        # for key, val in kwargs.items():
        #     setattr(self, key, val)


    def nuts_criterion(self, q, p, step_size):
        """Run a leaprfrog trajectory until U-turn, divergence or max_leapfrog_steps"""
        qs, ps, gs = [], [], []
        log_joint_qp = self.H(q, p)
        q_next = q
        p_next = p
        g_next =  self.V_g(q)
        old_distance = 0
        success = True

        for n in range(self.max_nleapfrog):
            qs.append(q_next)
            ps.append(p_next)
            gs.append(g_next)
            q_next, p_next, qlist, glist = self.leapfrog(q_next, p_next, 1, step_size, g=g_next)
            g_next = glist[-1]
            log_joint_next = self.H(q_next, p_next)
            if np.abs(log_joint_qp - log_joint_next) > 10:
                success = False
                break
            distance = np.sum((q_next - q) ** 2)
            if distance <= old_distance:
                break
            old_distance = distance

        return n+1, qs, ps, gs, success

                
    def nleapfrog_sample_and_lp(self, L, offset, nleapfrog=None):
        """
        Uniform distribution for sampling a proposal from trajectory
        """        
        LB = np.max([1, int(np.floor(offset * L))])
        if LB == L:
            if L != 1: raise
            if nleapfrog is None : nleapfrog, lp = L-1, 0.
            else:
                if nleapfrog == L-1 : lp = 0
                else : lp = -np.inf
        if nleapfrog is None:
            nleapfrog = self.rng.integers(LB, L)
        lp = - np.log(L - LB)
        if (nleapfrog < LB) or (nleapfrog >= L) :
            lp = -np.inf
        return int(nleapfrog), lp

    
    def step(self, q, step_size=None):
        """Run a single step/iteration of the algorithm"""
        if step_size is None: step_size = self.step_size
        self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0
        
        p =  multivariate_normal.rvs(mean=np.zeros(self.D), cov=self.inv_mass_matrix, size=1)
        offset = self.offset_function()

        try:
            # Go forward
            Nuturn, qs, ps, gs, success = self.nuts_criterion(q, p, step_size)
            nleapfrog, lp1 = self.nleapfrog_sample_and_lp(Nuturn, offset=offset)
            q1, p1, qlist, glist = qs[nleapfrog], ps[nleapfrog], qs, gs

            # Go backward
            Nuturn_rev = self.nuts_criterion(q1, -p1, step_size)[0]
            self.Vgcount -= min(Nuturn_rev, nleapfrog) #adjust for number of common steps when going backward

            nleapfrog2, lp2 = self.nleapfrog_sample_and_lp(Nuturn_rev, offset=offset, nleapfrog=nleapfrog)
            assert nleapfrog2 == nleapfrog
            stepcount = [Nuturn, Nuturn_rev, nleapfrog]

            # evaluate accept/reject probability
            log_prob_H, H0, H1 = self.accept_log_prob([q, p], [q1, p1], return_H=True)
            log_prob_N = lp2 - lp1
            mh_factor = np.exp(log_prob_N)
            Hs = [H0, H1]
            log_prob_total = log_prob_H + log_prob_N 

            # accept/reject sample and return
            u =  np.random.uniform(0., 1., size=1)
            if np.isnan(log_prob_total) or (q-q1).sum()==0:
                qf, pf = q, p
                accepted = -99
            elif  np.log(u) > min(0., log_prob_total):
                qf, pf = q, p
                accepted = -1
            else:
                qf, pf = q1, p1
                accepted = 1
                
            return qf, pf, accepted, Hs, stepcount, mh_factor
            
        except Exception as e:
            PrintException()
            qf, pf = q, p
            accepted = -99
            Hs, stepcount, mh_factor = [0, 0], [0, 0, 0], 0
            return qf, pf, accepted, Hs, stepcount, mh_factor

    
    def sample(self, q, p=None,
            n_samples=100, n_burnin=0, step_size=0.1, n_leapfrog=10,
            n_stepsize_adapt=0, target_accept=0.65, 
            seed=99,
               verbose=False, **kwargs):

        state = Sampler()
        self.step_size = step_size
        self.n_leapfrog = n_leapfrog
        self.verbose = verbose
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)
        if (n_leapfrog is not None) & (n_stepsize_adapt == 0):
            print("n_leapfrog argument is only used to adapt stepsize in U-turn sampler")
        # parse remaining kwargs
        # for key, val in kwargs.items():
        #     setattr(self, key, val)
        
        # additional quantities to keep track of
        state.stepcount = []
        state.mh_factor = []
        
        if n_stepsize_adapt:        # Adapt stepsize       
            q = self.adapt_stepsize(q, n_stepsize_adapt, target_accept=target_accept) 

        for i in range(n_burnin):   # Burnin
            q, p, accepted, Hs, stepcount, mh_factor = self.step(q, step_size=self.step_size) 
    
        for i in range(n_samples):  # Burnin
            q, p, accepted, Hs, stepcount, mh_factor = self.step(q, self.step_size)
            state.i += 1
            if (i%(n_samples//10) == 0):
                print(f"Iteration {i} of {n_samples}")
            state.appends(q=q, accepted=accepted, Hs=Hs, gradcount=self.Vgcount, energycount=self.Hcount)
            state.stepcount.append(stepcount)
            state.mh_factor.append(mh_factor)

        state.to_array()
        
        return state



###################################################
class HMC_Uturn_Jitter(HMC_Uturn):
    """GIST sampler with pre-constructed U-turn distribution.
    
    
    Warmup: 
    In the warmup phase, it adapts the stepsize and the trajectory lenght as follows:
        1. Baseline stepsize is adapted with dual averaging to target an acceptance rate of ``target_accept``.
        2. Next, Atlas constructs an empirical distribution of trajectory lengths (edist_traj).
        It run ``n_leapfrog_adapt`` iterations upto U-turn in every chain and stores the trajectory length.
        Then it combines this information from all chains, removes the lenghts that are too short or too long
        (outside ``low_nleap_percentile`` and ``high_nleap_percentile``) and saves the remaining lengths
        to construct edist_traj.

    Sampling:
    In every iteration, the trajecotry length is sampled from edist_traj and the leapfrog integration is done
    using the baseline stpesize.
    """
    
    def __init__(self, D, log_prob, grad_log_prob, mass_matrix=None, offset=0.5, 
                 low_nleap_percentile=10, high_nleap_percentile=90, nleap_factor=1.,
                 nleap_distribution='empirical',
                 **kwargs):
        super(HMC_Uturn_Jitter, self).__init__(D=D, log_prob=log_prob, grad_log_prob=grad_log_prob, 
                                            mass_matrix=mass_matrix, offset=offset,
                                            **kwargs)
        self.low_nleap_percentile = low_nleap_percentile
        self.high_nleap_percentile = high_nleap_percentile
        self.nleap_factor = nleap_factor
        self.nleap_distribution = nleap_distribution


    def adapt_trajectory_length(self, q, n_leapfrog_adapt):
        """Run prelimiary iterations to construct an empirical distribution of u-turn lengths"""        
        print("Adapting trajectory length for %d iterations"%n_leapfrog_adapt)
        self.traj_array = [] 
        nleapfrogs, traj = [], []
        step_size = self.step_size

        for i in range(n_leapfrog_adapt):            
            p =  multivariate_normal.rvs(mean=np.zeros(self.D), cov=self.inv_mass_matrix, size=1)
            step_size = self.step_size
            success = 0 
            for i in range(10): #sometimes the stepsize is too large
                N, qlist, plist, glist, success = self.nuts_criterion(q, p, step_size)
                if success: break
                step_size = step_size * 0.5
            if success:
                self.traj_array.append(N * step_size)
                q = qlist[-1]
            else:
                self.traj_array.append(0.)

        self.traj_array = np.array(self.traj_array) * 1.
        return q

    
    def nleapfrog_jitter(self):
        """
        Construct the jitter function to randomly sample a trajectory length 
        from the empirical distribution.
        """
        if not hasattr(self, "trajectories"):
            l = np.percentile(self.traj_array, self.low_nleap_percentile)
            h = np.percentile(self.traj_array, self.high_nleap_percentile)
            if h == l:
                h += 1
            trajectories = self.traj_array.copy()
            trajectories = trajectories[trajectories >= l]
            trajectories = trajectories[trajectories < h]
            self.trajectories = trajectories * self.nleap_factor # E.g. nleap_factor= 2/3 factor to not make a full U-turn
            print("average number of steps  : ", (self.trajectories/self.step_size).mean())
            if self.trajectories.size == 0 :
                print("Set of viable trajectories are empty")
                raise

        if self.nleap_distribution == 'empirical':
            self.nleapfrog_jitter_dist = \
                lambda step_size : min(max(int(np.random.choice(self.trajectories, 1) / step_size), self.min_nleapfrog), self.max_nleapfrog)
        elif self.nleap_distribution == 'uniform':
            self.nleapfrog_jitter_dist = \
                lambda step_size : min(max(int(np.random.uniform(self.trajectories.min(), self.trajectories.max()) / step_size), self.min_nleapfrog), self.max_nleapfrog)
        elif self.nleap_distribution == 'constant':
            self.nleapfrog_jitter_dist = \
                lambda step_size : self.n_leapfrog
            

    def combine_trajectories_from_chains(self, comm):
        if comm is not None:
            comm.Barrier()
            all_traj_array = np.zeros(len(self.traj_array) * comm.size)
            all_traj_array_tmp = comm.gather(self.traj_array, root=0)
            if comm.rank == 0 :
                all_traj_array = np.concatenate(all_traj_array_tmp)
            comm.Bcast(all_traj_array, root=0)
            self.traj_array = all_traj_array*1.
            comm.Barrier()            
        else:
            self.traj_array = np.array(self.traj_array)
        self.traj_array = self.traj_array[ self.traj_array!=0]

    def sample(self, q, p=None,
               n_samples=100, n_burnin=0, step_size=0.1, n_leapfrog=10,
               n_stepsize_adapt=0, n_leapfrog_adapt=100,
               target_accept=0.65, 
               seed=99,
               comm=None,
               verbose=False, **kwargs):
        
        state = Sampler()
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)
        self.step_size = step_size
        self.n_leapfrog = n_leapfrog
        self.verbose = verbose
        # parse remaining kwargs
        # for key, val in kwargs.items():
        #     setattr(self, key, val)
               
        if n_stepsize_adapt:
            q = self.adapt_stepsize(q, n_stepsize_adapt, target_accept=target_accept) 

        if n_leapfrog_adapt:  # Construct a distribution of trajectory lengths
            q = self.adapt_trajectory_length(q, n_leapfrog_adapt)
            comm.Barrier()
            self.combine_trajectories_from_chains(comm)
            state.trajectories = self.traj_array
            comm.Barrier()
            print(f"Shape of trajectories : ", self.traj_array.shape)
            self.nleapfrog_jitter()
                      
        for i in range(n_burnin):  # Burnin
            n_leapfrog = self.nleapfrog_jitter_dist(step_size)
            q, p, accepted, Hs = self.hmc_step(q, n_leapfrog=n_leapfrog, step_size=self.step_size) 

        for i in range(n_samples):  # Sample
            n_leapfrog = self.nleapfrog_jitter_dist(step_size)
            q, p, accepted, Hs = self.hmc_step(q, n_leapfrog=n_leapfrog, step_size=self.step_size) 
            state.i += 1
            state.appends(q=q, accepted=accepted, Hs=Hs, gradcount=self.Vgcount, energycount=self.Hcount)
            if (i%(n_samples//10) == 0):
                print(f"Iteration {i} of {n_samples}")

        state.to_array()
        return state
