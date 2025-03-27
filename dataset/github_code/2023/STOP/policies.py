from cleanrl_algo.arppo import ARPPO
import torch
from torch import nn
import pdb
import numpy as np

# longest queue
class LQ:
    def __init__(self, env):
        self.env = env

    def __call__(self, s):
        a = 0 # if all are empty, just choose the first 
        max_job_q = 0 
        for q_idx, q in enumerate(self.env.qs):
            q_num_jobs = q.num_jobs
            if q_num_jobs > max_job_q:
                max_job_q = q_num_jobs
                a = q_idx
        # if still -1, then no-op. may mean all queues are empty
        return a

# largest service rate
class LSR:
    def __init__(self, env):
        self.env = env

    def __call__(self, s):
        a = 0 # if nothing else selected, select first
        max_service_rate = -1

        for q_idx, q in enumerate(self.env.qs):
            q_num_jobs = q.num_jobs
            if q_num_jobs > 0 and q.p > max_service_rate:
                max_service_rate = q.p
                a = q_idx
        return a

# longest connected queue 
class LCQ:
    def __init__(self, env):
        self.env = env
    def __call__(self, obs, t = None):
        return self._get_action(obs)
    
    def _get_action(self, s):
        a = 0 # if all are empty, just choose the first 
        max_job_q = 0
        if isinstance(self.env, VecNormalize):
            qs = self.env.get_attr('qs')[0]
            lens = s[:len(qs)]
            cons = s[len(qs):]
        else:
            qs = self.env.qs
            lens = np.abs(s[:len(self.env.qs)])
            cons = s[len(self.env.qs):2*len(self.env.qs)]
        #connections = np.split(cons, len(cons) / 2)
        connections = s[len(self.env.qs):2*len(self.env.qs)]
        for q_idx, q in enumerate(qs):
            q_num_jobs = lens[q_idx]
            if q_num_jobs > max_job_q and connections[q_idx]: # check is_connected flag in one-hot vector for queue
            #if q_num_jobs > max_job_q and connections[q_idx][1]: # check is_connected flag in one-hot vector for queue
                max_job_q = q_num_jobs
                a = q_idx
        return a

    def get_action(self, obs):
        actions = []
        # TODO vectorized?
        for o in obs:
            actions.append(self._get_action(o))
        return np.array(actions)

# maxweight
class MaxWeight:
    def __init__(self, env):
        self.env = env
    def __call__(self, obs, t = None):
        return self._get_action(obs, t)
    
    def _get_action(self, s, t):
        a = 0 # if all are empty, just choose the first 
        max_qp = -1
        lens = np.abs(s[:len(self.env.qs)])
        cons = s[len(self.env.qs):2*len(self.env.qs)]
        #connections = np.split(cons, len(cons) / 2)
        connections = s[len(self.env.qs):2*len(self.env.qs)]
        for q_idx, q in enumerate(self.env.qs):
            qp = lens[q_idx] * q.get_service_prob(t)
            if qp > max_qp and connections[q_idx]: # check is_connected flag in one-hot vector for queue
            #if qp > max_qp and connections[q_idx][1]: # check is_connected flag in one-hot vector for queue
                max_qp = qp
                a = q_idx
        return a

# largest service connected queue 
class LSCQ:
    def __init__(self, env):
        self.env = env
    def __call__(self, s, t):
        a = 0 # if all are empty, just choose the first
        lens = np.abs(s[:len(self.env.qs)])
        cons = s[len(self.env.qs):2*len(self.env.qs)]
        #connections = np.split(cons, len(cons) / 2)
        connections = s[len(self.env.qs):2*len(self.env.qs)]
        max_service_rate = -1
        for q_idx, q in enumerate(self.env.qs):
            q_num_jobs = lens[q_idx]
            #if q_num_jobs > 0 and connections[q_idx][1] and q.get_service_prob(t) > max_service_rate:
            if q_num_jobs > 0 and connections[q_idx] and q.get_service_prob(t) > max_service_rate:
                max_service_rate = q.get_service_prob(t)
                a = q_idx
        return a

# largest arrival * service 
class LASQ:
    def __init__(self, env):
        self.env = env

    def __call__(self, obs, t = None):
        return self._get_action(obs, t)

    def _get_action(self, s, t):
        a = 0 # if all are empty, just choose the first 
        max_ap = -1
        lens = s[:len(self.env.qs)]
        cons = s[len(self.env.qs):2*len(self.env.qs)]
        connections = np.split(cons, len(cons) / 2)
        for q_idx, q in enumerate(self.env.qs):
            qp = (1. - q.get_arrival_prob(t)) * q.get_service_prob(t)
            if qp > max_ap and connections[q_idx][1] and lens[q_idx] > 0: # check is_connected flag in one-hot vector for queue
                max_ap = qp
                a = q_idx
        return a

# largest success queue
class LSQ:
    def __init__(self, env):
        self.env = env

    def __call__(self, obs, t = None):
        return self._get_action(obs, t)

    def _get_action(self, s, t):
        a = 0 # if all are empty, just choose the first 
        max_ap = -1
        lens = s[:len(self.env.qs)]
        cons = s[len(self.env.qs):2*len(self.env.qs)]
        connections = np.split(cons, len(cons) / 2)
        for q_idx, q in enumerate(self.env.qs):
            qp = lens[q_idx] * q.get_service_prob(t) * q.get_connection_prob(t)
            if qp > max_ap and connections[q_idx][1] and lens[q_idx] > 0: # check is_connected flag in one-hot vector for queue
                max_ap = qp
                a = q_idx
        return a

# random
class Random:
    def __init__(self, env, smart = False):
        self.env = env
        self.smart = smart

    def __call__(self, obs, t = None):
        return self._get_action(obs, t)
    
    def _get_action(self, s, t):
        a = 0 # if all are empty, just choose the first 
        lens = s[:len(self.env.qs)]
        connections = s[len(self.env.qs):2*len(self.env.qs)]
        if self.smart:
            potential = (lens > 0) & (connections == 1)
            pot_queues = np.arange(len(self.env.qs))[potential]
        else:
            pot_queues = np.arange(len(self.env.qs))
        #if len(np.where((connections == 1) ==  True)[0]) != len(self.env.qs):
        #    pdb.set_trace()
        if len(pot_queues):
            a = np.random.choice(pot_queues)
        return a

class Threshold:
    def __init__(self, env, T = 11):
        self.env = env
        self.T = T

    def __call__(self, obs, t = None):
        return self._get_action(obs, t)
    
    def _get_action(self, s, t):
        a = 0 # if all are empty, just choose the first 
        lens = s
        # if jobs #1 are greater than threshold
        if s[0] >= self.T:
            return 0
        else:
            return 1

class MWNModel:
    def __init__(self, env):
        self.env = env
    def __call__(self, obs, t = None):
        return self._get_action(obs, t)
    
    def _get_action(self, s, t):
        a = 0 # if all are empty, just choose the first 
        max_qp = -1
        lens = np.abs(s)
        mus = self.env.mus[1:]
        cost = [3, 1]
        assert len(lens) == len(mus)
        for q_idx in range(self.env.dim):
            qp = cost[q_idx] * lens[q_idx] * mus[q_idx]
            if qp > max_qp and lens[q_idx] > 0:
            #if qp > max_qp and connections[q_idx][1]: # check is_connected flag in one-hot vector for queue
                max_qp = qp
                a = q_idx
        return a

class LSQNModel:
    def __init__(self, env):
        self.env = env
    def __call__(self, obs, t = None):
        return self._get_action(obs, t)
    
    def _get_action(self, s, t):
        a = 0 # if all are empty, just choose the first 
        max_p = -1
        lens = np.abs(s)
        mus = self.env.mus[1:]
        assert len(lens) == len(mus)
        for q_idx in range(self.env.dim):
            p = mus[q_idx]
            if p > max_p and lens[q_idx] > 0:
                max_p = p
                a = q_idx
        return a

class LQNModel:
    def __init__(self, env):
        self.env = env
    def __call__(self, obs, t = None):
        return self._get_action(obs, t)
    
    def _get_action(self, s, t):
        a = 0 # if all are empty, just choose the first 
        max_p = -1
        lens = np.abs(s)
        for q_idx in range(self.env.dim):
            p = lens[q_idx]
            if p > max_p and lens[q_idx] > 0:
                max_p = p
                a = q_idx
        return a

class CleanRLPolicy:
    def __init__(self, env,
            num_minibatches = 4,
            num_steps = 256,
            learning_rate = 3e-4,
            anneal_lr = False,
            total_timesteps = 100_000,
            update_epochs = 10,
            clip_coef = 0.2,
            clip_vloss = False,
            ent_coef = 0.0, 
            vf_coef = 0.5,
            max_grad_norm = 0.5,
            target_kl = None,
            variant = 'zhang',
            gamma = 0.99,
            gae_lambda = 0.95,
            use_action_mask = False,
            adam_beta = 0.9):

        self.env = env
        self.pi = ARPPO(self.env, 
            gamma = gamma, 
            learning_rate = learning_rate,
            num_steps = num_steps, 
            update_epochs = update_epochs,
            variant = variant,
            use_action_mask = use_action_mask,
            adam_beta = adam_beta)

    def learn(self, total_timesteps):
        self.pi.train(total_timesteps = total_timesteps)

    def __call__(self, state, t = None):
        pdb.set_trace()
        if self.use_lcq:
            ent = self.pi.get_entropy(state).item()
            if ent >= 0.5:
                return self.lcq_pi(state)
        return self.pi.predict(state, deterministic = True)[0]
    
    def get_stats(self, skip_time):
        stats = self.pi.get_stats()
        trimmed_stats = {}
        for key in stats.keys():
            trimmed_stats[key] = stats[key][:len(stats[key]):skip_time]
        return trimmed_stats

def linear_schedule(initial_value: float, use_schedule = False,\
    min_value_limit = 0, progress_thresh = 1):
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        if use_schedule:
            if progress_remaining <= progress_thresh:
                return max(progress_remaining * initial_value, min_value_limit)
            else:
                return initial_value
        else:
            return initial_value
    return func

class PositivityActivation(nn.Module):
    def __init__(self):
        super().__init__()
    
    def _positivity_activation(self, input):
        return torch.square(input)
        #return torch.log(1 + torch.exp(input))

    def forward(self, input):
        return self._positivity_activation(input)

class NeuralNetwork(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dim = 16, hidden_layers = 1,
                    activation = 'tanh',
                    batch_norm = False,
                    final_activation = None,
                    layer_norm = False):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.tensor = torch.as_tensor
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.activation = self._get_act_fn(activation)
        self.final_activation = self._get_act_fn(final_activation)
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.penultimate, self.output = self._create_network()
        self._initialize()

    def _get_act_fn(self, activation):
        th_act = None
        if activation == 'tanh':
            th_act = nn.Tanh()
        elif activation == 'relu':
            th_act = nn.ReLU()
        elif activation == 'silu':
            th_act = nn.SiLU()
        elif activation == 'lrelu':
            th_act = nn.LeakyReLU()
        elif activation == 'sigmoid':
            th_act = nn.Sigmoid()
        elif activation == 'positivity':
            th_act = PositivityActivation()
        return th_act

    def __call__(self, s):
        if not torch.is_tensor(s):
            s = torch.from_numpy(s)
        s = s.float()
        net_out = self.output(s)
        net_out = net_out.detach().numpy()
        return net_out

    def forward(self, s, requires_grad = True):
        if not torch.is_tensor(s):
            s = torch.from_numpy(s)
        s = s.float()
        net_out = self.output(s)
        return net_out

    def get_penultimate(self, s, requires_grad = True):
        s = torch.from_numpy(s).float()
        pen = self.penultimate(s)
        return pen

    def _create_network(self):
        net_arch = []
        curr_dims = self.input_dims
        next_dims = self.hidden_dim

        for l in range(self.hidden_layers):
            net_arch.append(nn.Linear(curr_dims, next_dims))
            if self.batch_norm:
                net_arch.append(nn.BatchNorm1d(next_dims))
            net_arch.append(self.activation)
            curr_dims = next_dims
        
        penultimate = nn.Sequential(*net_arch).float()
        net_arch.append(nn.Linear(curr_dims, self.output_dims))
        if self.final_activation:
            net_arch.append(self.final_activation)
        if self.layer_norm:
            net_arch.append(nn.LayerNorm(self.output_dims))
            net_arch.append(nn.Tanh())
        output = nn.Sequential(*net_arch).float()
        return penultimate, output

    def _initialize(self):
        for m in self.output.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.orthogonal_(m.weight.data)
                if hasattr(m.bias, "data"):
                    m.bias.data.fill_(0.0)

