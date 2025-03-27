import gym
import numpy as np
import torch
from gym import spaces

# the following should be in the env_param_dict
# from lending_experiment.config import EP_TIMESTEPS, ZETA_0, ZETA_1


class PPOEnvWrapper_fair(gym.Wrapper):
  def __init__(self, env, reward_fn, env_param_dict):
    '''
    ep_timesteps: done=True if rollout for more than ep_timesteps steps
    the reward (after applying this wrapper) will be of the form:
    [main_reward, [r_u_0,r_u_1,r_u_2], [r_b_0, r_b_1, r_b_2]] (assuming there are 3 groups)
    '''
    super(PPOEnvWrapper_fair, self).__init__(env)

    self.include_delta =  env_param_dict['include_delta'] # if true, include ratio "delta" in the observation space as in APPO's paper
    self.ep_timesteps = env_param_dict['ep_timesteps']
    self.zeta_0 = env_param_dict['zeta_0']
    self.zeta_1 = env_param_dict['zeta_1']

    self.num_groups = 2

    if self.include_delta:
      self.observation_space = spaces.Box(
      low=np.inf,
      high=np.inf,
      # (7) OHE of credit score + (2) group +  (2) TPRs of each group
      shape=(env.observation_space['applicant_features'].shape[0] + 2 * env.state.params.num_groups,),
    )
    else:
      self.observation_space = spaces.Box(
        low=np.inf,
        high=np.inf,
        # (7) OHE of credit score + (2) group 
        shape=(env.observation_space['applicant_features'].shape[0] + 1 * env.state.params.num_groups,),
      )

    self.action_space = spaces.Discrete(n=2)

    self.env = env
    self.reward_fn = reward_fn()

    self.timestep = 0

    self.tp = np.zeros(self.env.state.params.num_groups,)
    self.fp = np.zeros(self.env.state.params.num_groups,)
    self.tn = np.zeros(self.env.state.params.num_groups,)
    self.fn = np.zeros(self.env.state.params.num_groups,)
    self.tpr = np.zeros(self.env.state.params.num_groups,)
    self.delta = np.zeros(1, )
    self.old_bank_cash = 0
    self.delta_delta = 0

  def process_observation(self, obs):
    '''
    Input obs contains all "self.observable_state_vars" of the env in lending.py
    Including: 'bank_cash', 'applicant_features', 'group'
    '''
    credit_score = obs['applicant_features']
    group = obs['group']

    if self.include_delta:
      return np.concatenate(
      (credit_score,
       group,
       self.tpr,
       ),
      axis=0
    )
    else:
      return np.concatenate(
        (credit_score,
        group,
        ),
        axis=0
      )

  def compute_tpr(self, tp, fn):
    # tp: true positive, 2-dimensional for 2 groups
    # fn: false negative, 2-dimensional for 2 groups
    return np.divide(
      tp,
      tp + fn,
      out=np.zeros_like(tp),
      where=(tp + fn) != 0)

  def reset(self):
    self.timestep = 0
    self.tp = np.zeros(self.env.state.params.num_groups,)
    self.fp = np.zeros(self.env.state.params.num_groups,)
    self.tn = np.zeros(self.env.state.params.num_groups,)
    self.fn = np.zeros(self.env.state.params.num_groups,)
    self.tpr = np.zeros(self.env.state.params.num_groups,)
    self.delta = np.zeros(1, )
    self.old_bank_cash = 0
    self.delta_delta = 0

    return self.process_observation(self.env.reset())

  def step(self, action):
    '''
    modify the reward to 5 dimension: r = (agent reward, r_U_0, r_B_0, r_U_1, r_B_1)
    r_U_g = 1 if group = g and not default and action = 1, = 0 otherwise
    r_B_g = 1 if group = g and not default               , = 0 otherwise
    '''

    # Fairness reward
    r_U_0 = 0
    r_B_0 = 0
    r_U_1 = 0
    r_B_1 = 0

    old_delta = self.delta

    # Update fairness reward before we step the environment
    group_id = np.argmax(self.env.state.group)

    if group_id == 0:
      if not self.env.state.will_default:
        r_B_0 = 1
        if action == 1:
          r_U_0 = 1
    elif group_id == 1:
      if not self.env.state.will_default:
        r_B_1 = 1
        if action == 1:
          r_U_1 = 1
    else:
      raise ValueError('invalid group_id')
    
    # only for test, can be removed
    assert r_B_0 * r_B_1 == 0, 'r_B for both group are non-zero'
    assert (r_B_0 >= r_U_0) and (r_B_1 >= r_U_1), 'r_B < r_U error!'  
    
    # only for APPO
    if action == 1:
      # Check if agent would default
      if self.env.state.will_default:
        self.fp[group_id] += 1
      else:
        self.tp[group_id] += 1
    elif action == 0:
      if self.env.state.will_default:
        self.tn[group_id] += 1
      else:
        self.fn[group_id] += 1
    self.tpr = self.compute_tpr(tp=self.tp,
                                fn=self.fn)
    
    self.old_bank_cash = self.env.state.bank_cash

    # Update delta terms
    self.delta = np.abs(self.tpr[0] - self.tpr[1])
    self.delta_delta = self.delta - old_delta


    obs, _, done, info = self.env.step(action)

    r = self.reward_fn(old_bank_cash=self.old_bank_cash,
                       bank_cash=self.env.state.bank_cash,
                       tpr=self.tpr,
                       zeta0=self.zeta_0,
                       zeta1=self.zeta_1)

    self.timestep += 1
    if self.timestep >= self.ep_timesteps:
      done = True  

    return self.process_observation(obs), [r, [r_U_0, r_U_1], [r_B_0, r_B_1]], done, info
  
