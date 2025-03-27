#import gym
import gymnasium as gym
import numpy as np
import pdb
from utils import powerset, symlog, symsqrt, sigmoid, tanh
import math, random
import copy

TIME_SMOOTHER = 10000

# assuming 1-1 between node-queue
# SA - server allocation
class SAQueue:
    def __init__(self, name, queue_info, random_starts = False, gridworld = False):
        self.name = name
        self.queue_info = queue_info
        self.random_starts = random_starts
        self.queue_init_high = 100
        self.gridworld = gridworld
        self.reset()

    def reset(self):
        if self.random_starts:
            if self.gridworld:
                self.num_jobs = np.random.randint(low = -self.queue_init_high, high = self.queue_init_high)
            else:
                self.num_jobs = np.random.randint(low = 0, high = self.queue_init_high)
        else:
            self.num_jobs = 0

    def get_arrival_prob(self, t):
        stationary = self.queue_info['arrival']['is_stationary']
        if stationary:
            return self.queue_info['arrival']['prob']
        else:
            return self._get_periodic_value('arrival', t)

    def get_service_prob(self, t):
        stationary = self.queue_info['service']['is_stationary']
        if stationary:
            return self.queue_info['service']['prob']
        else:
            return self._get_periodic_value('service', t)
    def get_connection_prob(self, t):
        stationary = self.queue_info['connection']['is_stationary']
        if stationary:
            return self.queue_info['connection']['prob']
        else:
            return self._get_periodic_value('connection', t)
    
    def _get_periodic_value(self, metric, t):
        typ = self.queue_info[metric]['type']
        if typ == 'piecewise':
            probs = self.queue_info[metric]['probs']
            pieces = len(self.queue_info[metric]['probs'])
            phase_length = self.queue_info[metric]['length']
            trans_length = self.queue_info[metric]['trans_length']
            period = pieces * (phase_length + trans_length)

            # hard-coded for two phases (excluding two transitions)
            t = (t % period)
            p_trans = phase_length + trans_length
            pr = 0
            if 0 <= t and t < phase_length:
                pr = probs[0]
            elif phase_length <= t and t < p_trans:
                slope = (probs[1] - probs[0]) / (trans_length)
                pr = probs[0] + slope * (t - phase_length)
            elif p_trans <= t and t < phase_length + p_trans:
                pr = probs[1]
            elif phase_length + p_trans <= t and t < 2 * p_trans:
                slope = (probs[0] - probs[1]) / (trans_length)
                pr = probs[1] + slope * (t - phase_length - p_trans)
            return pr
        elif typ == 'sine':
            period = self.queue_info[metric]['period']
            offset = self.queue_info[metric]['offset']
            multiplier = self.queue_info['arrival']['multiplier']
            periodic_offset = self.queue_info['arrival']['periodic_offset']
            offset = self.queue_info['arrival']['offset']
            return self._sigmoid_sine(t, period, multiplier, periodic_offset, offset)
    
    def _sigmoid_sine(self, t, period, multiplier = 1, periodic_offset = 0, offset = 0):
        t = t / TIME_SMOOTHER # simulating so form of continuous changes along curve
        val = multiplier * np.sin((2 * np.pi * t / period) + periodic_offset)
        return 1. / (1. + np.exp(-(val + offset))) 

class SANetwork(gym.Env):
    def __init__(self, queues, network_name = 'server_alloc', reward_func = 'opt',\
        state_trans = 'id', reward_transformation = 'id', use_mask = False,\
        gridworld = False, state_bound = np.inf, lyp_power = 1.):
        self.qs = queues
        self.reward_func = reward_func
        self.action_space = gym.spaces.Discrete(len(self.qs))
        # number of job counts (one per queue) + (0/1) 2D one-hot per queue
        dim = len(self.qs) + len(self.qs)
        if gridworld:
            self.lower_state_bound = -state_bound
            self.upper_state_bound = state_bound
            self.observation_space = gym.spaces.Box(low = self.lower_state_bound, high = self.upper_state_bound, shape = (dim,), dtype = float)
        else:
            self.lower_state_bound = 0
            self.upper_state_bound = state_bound
            self.observation_space = gym.spaces.Box(low = self.lower_state_bound, high = self.upper_state_bound, shape = (dim,), dtype = float)
        self.horizon = -1
        self.state_trans = state_trans
        self.reward_transformation = reward_transformation
        self.use_mask = use_mask
        self.t = 0
        self.gridworld = gridworld
        self.network_name = network_name
        self.lyp_power = lyp_power

    def set_reset_eps(self, eps):
        self.reset_eps = eps

    def set_horizon(self, horizon):
        self.horizon = horizon
    
    def set_state_transformation(self, trans):
        self.state_trans = trans

    def _get_period(self, q, param_name):
        if q.queue_info[param_name]['is_stationary']:
            period = 1
        else:
            if q.queue_info[param_name]['type'] == 'piecewise':
                pieces = len(q.queue_info[param_name]['probs'])
                phase_length = q.queue_info[param_name]['length']
                trans_length = q.queue_info[param_name]['trans_length']
                period = pieces * (phase_length + trans_length)
            else:
                period = q.queue_info[param_name]['period']
                if period < 1:
                    num_dec = str(period)[::-1].find('.')
                    period *= (10 ** num_dec)
                    period = int(period)
        return period

    # common multiple of periods considering for different parameters and queues
    def get_period_cm(self, q_sub):
        lam_prod = 1
        p_prod = 1
        c_prod = 1
        for q_idx in q_sub:
            q = self.qs[q_idx]
            lam_prod *= self._get_period(q, 'arrival')
            p_prod *= self._get_period(q, 'service')
            c_prod *= self._get_period(q, 'connection')
        return math.lcm(lam_prod, p_prod, c_prod)
        #return lam_prod * p_prod * c_prod * TIME_SMOOTHER # due to smoothing / 10 above

    def is_stable(self):

        ps = list(powerset(np.arange(len(self.qs))))
        min_gap = float('inf')
        # check every subset of queues
        for sub in ps:
            T = self.get_period_cm(sub)
            for t in np.arange(T):
                lam_p = 0
                c_prod = 1
                # for each subset apply stability criterion
                for q_idx in sub:
                    lam = self.qs[q_idx].get_arrival_prob(t)
                    p = self.qs[q_idx].get_service_prob(t)
                    c = self.qs[q_idx].get_connection_prob(t)
                    lam_p += (lam / p)
                    c_prod *= (1. - c)
                # if violated, break
                if lam_p >= 1. - c_prod:
                    pdb.set_trace()
                    return False
                min_gap = min(min_gap, (1. - c_prod) - lam_p)
        print ('distance from decision boundary {}'.format(min_gap))
        return True

    def transform_state(self, state):
        lens = state[:len(self.qs)]
        connects = state[len(self.qs):]
        if self.state_trans == 'symloge':
            adj_lens = symlog(lens, base = 'e')
            adj_state = np.concatenate((adj_lens, connects))
            return adj_state
        elif self.state_trans == 'symlog10':
            adj_lens = symlog(lens, base = '10')
            adj_state = np.concatenate((adj_lens, connects))
            return adj_state
        elif self.state_trans == 'symsqrt':
            adj_lens = symsqrt(lens)
            adj_state = np.concatenate((adj_lens, connects))
            return adj_state
        elif self.state_trans == 'sigmoid':
            adj_lens = sigmoid(lens)
            adj_state = np.concatenate((adj_lens, connects))
            return adj_state
        elif self.state_trans == 'tanh':
            adj_lens = tanh(lens)
            adj_state = np.concatenate((adj_lens, connects))
            return adj_state
        elif self.state_trans == 'id':
            return np.array(state)

    def reset(self, reset_time = True):
        lens = []
        if reset_time:
            self.t = 0
        for q in self.qs:
            q.reset()
            lens.append(q.num_jobs)
        lens = np.array(lens)
        #lens = np.clip(lens, 0, 60)

        # add connectivity into state
        connect_probs = [self.qs[idx].get_connection_prob(0) for idx in range(len(self.qs))]
        self.connects = np.random.binomial(n = 1, p = connect_probs)

        service_probs = [self.qs[idx].get_service_prob(self.t) for idx in range(len(self.qs))]
        service_success = np.random.binomial(n = 1, p = service_probs)

        arrival_probs = [self.qs[idx].get_arrival_prob(self.t) for idx in range(len(self.qs))]
        arrive_success = np.random.binomial(n = 1, p = arrival_probs)

        self.pre_arrival_lens = np.concatenate((lens, self.connects))
        self.native_state = np.concatenate((lens, self.connects))
        self.native_prev_state = np.concatenate((lens, self.connects))
        self.state = self.transform_state(self.native_state)
        self.prev_state = self.transform_state(self.native_prev_state)
        self.w_temp = np.tanh(4e-6 * np.maximum(self.t - 1000000, 0.01))
        #if self.gridworld:
        #    self.goal = np.random.randint(low = -100, high = 100, size = len(self.qs))
        #else:
        self.goal = np.array([0 for _ in range(len(self.qs))])
        print ('init state: {}, goal state: {}'.format(self.native_state, self.goal))
        return self.state, {}

    def _overall_avg_backlog(self, state):
        return np.mean(np.abs(state[:len(self.qs)] - self.goal))

    def _avg_backlog(self, state, next_state, weighted = False):
        lens = np.abs(next_state[:len(self.qs)] - self.goal)
        if weighted:
            su = np.sum(lens)
            if su == 0:
                su = 1.
            weights = lens / su
            met = np.sum(weights * lens)
        else:
            met = np.mean(lens)
        return met

    def _backlog_change(self, state, next_state):
        prev_lengths = np.mean(np.abs(state[:len(self.qs)] - self.goal))
        curr_lengths = np.mean(np.abs(next_state[:len(self.qs)] - self.goal)) # Manhatten distance
        metric = curr_lengths - prev_lengths
        return metric 

    def reward_function(self, state = None, action = None, next_state = None, pre_arrial_state = None):

        avg_q_len = self._avg_backlog(state, next_state)
        change_avg_q_len = self._backlog_change(state, next_state)

        if self.reward_func == 'opt':
            reward = -1 * avg_q_len
        elif 'stab' in self.reward_func:
            opt_rew = -avg_q_len
            if np.abs(self.lyp_power - 1) <= 1e-5:
                opt_rew = 1. / (avg_q_len + 1)
            prev_lens = np.mean(np.power(state[:len(self.qs)] - self.goal, self.lyp_power))
            curr_lens = np.mean(np.power(next_state[:len(self.qs)] - self.goal, self.lyp_power))            
            reward = -1 * (curr_lens - prev_lens) + opt_rew
        return reward

    def step(self, a):
        assert self.action_space.contains(a)
        #assert self.observation_space.contains(self.native_state)

        # policy takes an action based on the state
        # now the next step is dependent on the arrival time and service time
        # for example, if we have two queues, there are 8 possible next states given
        # the current state and action
        # new jobs in each queue may or may not arrive, and selected job may or may not
        # be serviced

        # record previous state
        self.prev_state = self.state
        self.native_prev_state = self.native_state

        # service rates of queues
        service_probs = [self.qs[idx].get_service_prob(self.t) for idx in range(len(self.qs))]
        service_success = np.random.binomial(n = 1, p = service_probs)

        # connectivity to queue
        is_connected = self.connects[a]
        #is_connected = np.split(self.connects, len(self.connects) / 2)[a][1]
        # service job
        if is_connected:
            serve_success = service_success[a]
        else:
            # job will not be served if server cannot connect to queue
            serve_success = 0
        curr_num_jobs = self.qs[a].num_jobs

        if serve_success:
            current_pos = self.qs[a].num_jobs
            # if at goal, nothing happens
            if current_pos > self.goal[a]:
                self.qs[a].num_jobs = current_pos - 1
            elif current_pos < self.goal[a]:
                self.qs[a].num_jobs = current_pos + 1
            
            # trim if negatives not allowed
            if not self.gridworld:
                self.qs[a].num_jobs = max(self.qs[a].num_jobs, 0)

        pre_arrival_lens = np.array([q.num_jobs for q in self.qs])
        pre_arrival_lens = np.clip(pre_arrival_lens, self.lower_state_bound, self.upper_state_bound)

        # arrival of new jobs
        new_lengths = []
        arrival_probs = [self.qs[idx].get_arrival_prob(self.t) for idx in range(len(self.qs))]
        arrive_success = np.random.binomial(n = 1, p = arrival_probs)
        for idx, q in enumerate(self.qs):
            current_pos = q.num_jobs
            # always holds true if above trimming done
            if current_pos >= self.goal[a]:
                q.num_jobs = current_pos + arrive_success[idx]
            else:
                q.num_jobs = current_pos - arrive_success[idx]
            
            if not self.gridworld:
                q.num_jobs = max(q.num_jobs, 0)
            new_lengths.append(q.num_jobs)
        new_lengths = np.array(new_lengths)
        new_lengths = np.clip(new_lengths, self.lower_state_bound, self.upper_state_bound)

        # connectivity of the new state
        connect_probs = [self.qs[idx].get_connection_prob(self.t) for idx in range(len(self.qs))]
        self.connects = np.random.binomial(n = 1, p = connect_probs)
        
        self.pre_arrival_lens = np.concatenate((pre_arrival_lens, self.connects))
        self.native_state = np.concatenate((new_lengths, self.connects))
        self.state = self.transform_state(self.native_state)

        backlog = self._overall_avg_backlog(state = self.native_state)

        # if based on change, then need updated state to compute reward
        reward = self.reward_function(self.native_prev_state, a, self.native_state, self.pre_arrival_lens)

        self.t += 1
        done = False
        if self.horizon != -1:    
            done = True if self.t >= self.horizon else False

        # cost may be either of metrics
        info = {
            'backlog': backlog,
            # 'backlog_change': backlog_change,
            'time': self.t,
            'stoch_state': np.concatenate((arrive_success, service_success)),
            'services': service_success,
            'arrivals': arrive_success,
            'native_state': self.native_prev_state,
            'next_native_state': self.native_state,
            'connects': self.connects,
            'action': a
        }
        #assert not done
        return self.state, reward, done, False, info

    def set_use_mask(self, flag):
        self.use_mask = flag

    def set_state_bound(self, bound):
        if self.gridworld:
            self.lower_state_bound = -bound
            self.upper_state_bound = bound
        else:
            self.lower_state_bound = 0
            self.upper_state_bound = bound
    
    def mask_extractor(self, obs):
        # iterate for each queue but vecotrized across examples
        obs_dim = self.observation_space.shape[0]
        masks = np.zeros((obs.shape[0], self.action_space.n)).astype(bool)
        if True:#self.use_mask:
            obs = obs[:, -obs_dim:]
            lens = obs[:, :len(self.qs)]
            cons = obs[:, len(self.qs): 2 * len(self.qs)]
            for i in range(len(self.qs)):
                masks[:, i] = np.logical_or(lens[:, i] == 0, cons[:, i] == 0) # if either empty or disconnected set mask on
                #masks[:, i] = (np.logical_or(lens[:, i] == 0, cons[:, 2 * i + 1] == 0)) # if either empty or disconnected set mask on
        return masks
    
    def critical_state_check(self, obs, action):
        obs_dim = self.observation_space.shape[0]
        obs = obs[:, -obs_dim:]
        lens = obs[:, :len(self.qs)]
        cons = obs[:, len(self.qs): 2 * len(self.qs)]
        critical = np.zeros((obs.shape[0], self.action_space.n)).astype(bool)
        
        for i in range(len(self.qs)):
            critical[:, i] = np.logical_or(lens[:, i] == 0, cons[:, i] == 0) # if either empty or disconnected set mask on
        
        # at least one queue empty, but excluding all queues empty
        check = not np.all(critical, axis = 1) and np.any(critical, axis = 1)
        
        def indices_for_true_values(row):
            return np.where(row)[0]
        unstable_action_taken = False
        if check:
            empty_qs = np.apply_along_axis(indices_for_true_values, axis=1, arr=critical)
            unstable_action_taken = np.isin(action, empty_qs, assume_unique=True)
            #print (obs, empty_qs, action, unstable_action_taken)
        return check, unstable_action_taken
            
