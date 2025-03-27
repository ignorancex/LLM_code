import gymnasium as gym
import numpy as np
import pdb
from utils import powerset, symlog, symsqrt, sigmoid, tanh
import math, random
import copy

TIME_SMOOTHER = 10000

class NModelNetwork(gym.Env):
    def __init__(self, arrivals, mus, holding_costs, network_name = 'nmodel', reward_func = 'opt',\
        state_trans = 'id', use_mask = False, state_bound = np.inf, lyp_power = 1.):
        self.arrivals = arrivals
        self.mus = mus

        self.uniform_rate = np.sum(arrivals)+np.sum(mus)  # uniform rate for uniformization
        self.p_arriving = np.divide(self.arrivals, self.uniform_rate)# normalized arrival rates
        self.p_compl = np.divide(self.mus, self.uniform_rate)#normalized service rates
        self.cumsum_rates = np.unique(np.cumsum(np.concatenate([self.p_arriving, self.p_compl])))

        self.holding_costs = np.array(holding_costs)
        self.reward_func = reward_func
        self.action_space = gym.spaces.Discrete(len(arrivals))
        # number of job counts (one per queue) + (0/1) 2D one-hot per queue
        dim = len(arrivals)
        self.dim = dim

        self.lower_state_bound = 0
        self.upper_state_bound = state_bound
        self.observation_space = gym.spaces.Box(low = self.lower_state_bound, high = self.upper_state_bound, shape = (dim,), dtype = float)
        self.horizon = -1
        self.state_trans = state_trans
        self.use_mask = use_mask
        self.t = 0
        self.network_name = network_name
        self.lyp_power = lyp_power

    def set_horizon(self, horizon):
        self.horizon = horizon
    
    def set_state_transformation(self, trans):
        self.state_trans = trans

    def is_stable(self):
        return True

    def transform_state(self, state):
        lens = state[:self.dim]
        if self.state_trans == 'symloge':
            adj_lens = symlog(lens, base = 'e')
            adj_state = np.array(adj_lens)
            return adj_state
        elif self.state_trans == 'symlog10':
            adj_lens = symlog(lens, base = '10')
            adj_state = np.array(adj_lens)
            return adj_state
        elif self.state_trans == 'symsqrt':
            adj_lens = symsqrt(lens)
            adj_state = np.array(adj_lens)
            return adj_state
        elif self.state_trans == 'sigmoid':
            adj_lens = sigmoid(lens)
            adj_state = np.array(adj_lens)
            return adj_state
        elif self.state_trans == 'tanh':
            adj_lens = tanh(lens)
            adj_state = np.array(adj_lens)
            return adj_state
        elif self.state_trans == 'id':
            return np.array(state)

    def reset(self, reset_time = True):
        lens = []
        if reset_time:
            self.t = 0
        for q in range(self.dim):
            lens.append(0)
        lens = np.array(lens)

        self.native_state = np.array(lens)
        self.native_prev_state = np.array(lens)
        self.state = self.transform_state(self.native_state)
        self.prev_state = self.transform_state(self.native_prev_state)
        self.goal = np.array([0 for _ in range(self.dim)])
        print ('init state: {}, goal state: {}'.format(self.native_state, self.goal))
        return self.state, {}

    def _overall_avg_backlog(self, state):
        return np.mean(np.abs(state[:self.dim] * self.holding_costs - self.goal))

    def _avg_backlog(self, state, next_state, weighted = False):
        lens = np.abs(next_state[:self.dim] * self.holding_costs - self.goal)
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
        prev_lengths = np.mean(np.abs(state[:self.dim] * self.holding_costs - self.goal))
        curr_lengths = np.mean(np.abs(next_state[:self.dim] * self.holding_costs - self.goal)) # Manhatten distance
        metric = curr_lengths - prev_lengths
        return metric 

    def reward_function(self, state = None, action = None, next_state = None):

        avg_q_len = self._avg_backlog(state, next_state)
        change_avg_q_len = self._backlog_change(state, next_state)

        if self.reward_func == 'opt':
            reward = -1 * avg_q_len
        elif 'stab' in self.reward_func:
            opt_rew = -avg_q_len
            if np.abs(self.lyp_power - 1) <= 1e-5:
                opt_rew = 1. / (avg_q_len + 1)
            prev_lens = np.mean(np.power(state[:self.dim] * self.holding_costs - self.goal, self.lyp_power))
            curr_lens = np.mean(np.power(next_state[:self.dim] * self.holding_costs - self.goal, self.lyp_power))
            reward = -1 * (curr_lens - prev_lens) + opt_rew
        return reward

    def next_state_N1(self, state, action):
        """
        :param state: current state
        :param action: action
        :return: next state
        """

        w = np.random.random()
        wi = 0
        while w > self.cumsum_rates[wi]:
            wi += 1
        if wi == 0:
            state_next = state + np.asarray([1, 0])
        elif wi == 1:
            state_next = state + np.asarray([0, 1])
        elif wi == 2 and (state[0] > 0):
            state_next = state - np.asarray([1, 0])
        elif wi == 3 and ((action == 0 or state[1] == 0) and state[0] > 1):
            state_next = state - np.asarray([1, 0])
        elif wi == 4 and ((action == 1 or state[0] < 2) and state[1] > 0):
            state_next = state - np.asarray([0, 1])
        else:
            state_next = state
        return state_next

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

        next_state = self.next_state_N1(self.native_state, a)

        next_state = np.array(next_state)
        next_state = np.clip(next_state, self.lower_state_bound, self.upper_state_bound) 

        # for idx, q in enumerate(self.qs):
        #     q.num_jobs = next_state[idx]            

        self.native_state = next_state
        self.state = self.transform_state(self.native_state)

        backlog = self._overall_avg_backlog(state = self.native_state)

        # if based on change, then need updated state to compute reward
        reward = self.reward_function(self.native_prev_state, a, self.native_state)

        self.t += 1
        done = False
        if self.horizon != -1:    
            done = True if self.t >= self.horizon else False

        # cost may be either of metrics
        info = {
            'backlog': backlog,
            # 'backlog_change': backlog_change,
            'time': self.t,
            'native_state': self.native_prev_state,
            'next_native_state': self.native_state,
            'action': a
        }
        #assert not done
        return self.state, reward, done, False, info

    def set_use_mask(self, flag):
        self.use_mask = flag

    def set_state_bound(self, bound):
        self.lower_state_bound = 0
        self.upper_state_bound = bound
    
    def mask_extractor(self, obs):
        # iterate for each queue but vecotrized across examples
        obs_dim = self.observation_space.shape[0]
        masks = np.zeros((obs.shape[0], self.action_space.n)).astype(bool)
        if self.use_mask:
            obs = obs[:, -obs_dim:]
            lens = obs[:, :self.dim]
            for i in range(self.dim):
                masks[:, i] = lens[:, i] == 0 # if either empty or disconnected set mask on
                #masks[:, i] = (np.logical_or(lens[:, i] == 0, cons[:, 2 * i + 1] == 0)) # if either empty or disconnected set mask on
        return masks
    
    def critical_state_check(self, obs, action):
        return False, False
        
