from abc import ABC, abstractmethod
import gymnasium as gym
from gymnasium.spaces import Dict
import numpy as np
import os
import numpy as np
from typing import Callable, Optional

import gymnasium as gym
from gymnasium import logger
from morl_generalization.wrappers import ObsToNumpy, HistoryWrapper

class DREnv(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def reset_random(self):
        pass

    @abstractmethod
    def get_task(self):
        pass

class DRWrapper(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """Wrapper for DR environment."""

    def __init__(self, env: gym.Env, history_len=3, state_history=False, action_history=False):
        """Initialize the :class:`DRWrapper` wrapper with an environment and a transform function :attr:`f`.

        Args:
            env: The environment to apply the wrapper
            f: A function that transforms the observation
        """
        super(DRWrapper, self).__init__(env)
        if state_history or action_history:
            env = HistoryWrapper(env, history_len, state_history=state_history, action_history=action_history)
        self.env = env

    def reset(self, *, seed=None, options=None):
        self.env.unwrapped.reset_random() # domain randomization
        
        return self.env.reset(seed=seed, options=options) 
    
class DynamicsInObs(DRWrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env: gym.Env, dynamics_mask=None):
        """
            Stack the current env dynamics to the env observation vector

            dynamics_mask: list of int
                           indices of dynamics to randomize, i.e. to condition the network on
        """
        if not isinstance(env.unwrapped, DREnv):
            raise TypeError("The environment must implement be a DREnv, i.e. implement `get_task()`, before applying DynamicsInObs.")
        gym.utils.RecordConstructorArgs.__init__(self, dynamics_mask=dynamics_mask)
        DRWrapper.__init__(self, env)

        if dynamics_mask is not None:
            self.dynamics_mask = np.array(dynamics_mask)
            task_dim = env.get_task()[self.dynamics_mask].shape[0]
        else:  # All dynamics are used
            task_dim = env.get_task().shape[0]
            self.dynamics_mask = np.arange(task_dim)

        obs_space = env.observation_space
        low = np.concatenate([obs_space.low.flatten(), np.repeat(-np.inf, task_dim)], axis=0)
        high = np.concatenate([obs_space.high.flatten(), np.repeat(np.inf, task_dim)], axis=0)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=obs_space.dtype)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        dynamics = self.env.get_task()[self.dynamics_mask]
        obs = np.concatenate([obs.flatten(), dynamics], axis=0)
        return obs, reward, terminated, truncated, info
    
    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        dynamics = self.env.get_task()[self.dynamics_mask]
        obs = np.concatenate([obs.flatten(), dynamics], axis=0)
        return obs, info

    def get_actor_critic_masks(self):
        actor_obs_mask = list(range(0, self.observation_space.shape[0] - self.dynamics_mask.shape[0]))
        critic_obs_mask = list(range(self.observation_space.shape[0]))
        return actor_obs_mask, critic_obs_mask
    
class AsymmetricDRWrapper(gym.Wrapper):
    def __init__(self, env, history_len=3, state_history=False, action_history=False):
        super(AsymmetricDRWrapper, self).__init__(env)
        if state_history or action_history:
            env = HistoryWrapper(env, history_len, state_history=state_history, action_history=action_history)
        env = DynamicsInObs(env)
        self.env = env