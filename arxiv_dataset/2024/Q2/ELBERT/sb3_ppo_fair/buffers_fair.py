'''
The buffer in the file is a standard RL buffer with two additional components
1. Supply (U) and demand (B) "rewards". The buffer.reward is now [r, [r_U_0,..],[r_B_0,..]]
2. (Only for APPO) deltas and delta_deltas. They will NOT be used by other algorithms
'''
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Union

import numpy as np
import torch as th
from gym import spaces

from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.vec_env import VecNormalize

from .type_aliases import RolloutBufferSamples_fair

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None


class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)
    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
    ):
        super(BaseBuffer, self).__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)

        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.device = device
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)
        :param arr:
        :return:
        """
        raise ValueError('We should not use this function in the fairness setting. Use swap_and_flatten_fair() instead')
        shape = arr.shape
        if len(shape) < 3:
            shape = shape + (1,)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None):
        """
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    @abstractmethod
    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> RolloutBufferSamples_fair:
        """
        :param batch_inds:
        :param env:
        :return:
        """
        raise NotImplementedError()

    def to_torch(self, array: List[Union[np.ndarray,List[np.ndarray]]], copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default
        :param array:
        :param copy: Whether to copy or not the data
            (may be useful to avoid changing things be reference)
        :return:
        """
        if copy:
            if isinstance(array, np.ndarray):
                return th.tensor(array).to(self.device)
            else:
                # In this case, arr could be of the form [r, [r_U_0,..],[r_B_0,..]]
                assert len(array) == 3, 'array should be of the form [r, [r_U_0,..],[r_B_0,..]]'
                array_new = [] 
                # push every element to device
                array_new.append(th.tensor(array[0]).to(self.device))
                array_new.append( [th.tensor(arr).to(self.device) for arr in array[1]] )
                array_new.append( [th.tensor(arr).to(self.device) for arr in array[2]] )

                return array_new
            
        return th.as_tensor(array).to(self.device)

    @staticmethod
    def _normalize_obs(
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        env: Optional[VecNormalize] = None,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if env is not None:
            return env.normalize_obs(obs)
        return obs

    @staticmethod
    def _normalize_reward(reward: np.ndarray, env: Optional[VecNormalize] = None) -> np.ndarray:
        if env is not None:
            return env.normalize_reward(reward).astype(np.float32)
        return reward

class RolloutBuffer_fair(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.
    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.
    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments

    modification from the previous version:
    1. dealing with 2*M + 1 reward, which involves return, advantage. The order is: r_main, r_U: List, r_B: List
    We keep the "deltas = tpr difference" and delta_deltas in APPO's paper
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
        num_groups: int = 2,
    ):

        super(RolloutBuffer_fair, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.observations, self.actions = None, None
        self.episode_starts, self.log_probs = None, None

        self.num_groups = num_groups

        # 1 + 2*M rewards: main reward, r_U, r_B
        self.rewards = [None, [None for i in range(self.num_groups)], [None for i in range(self.num_groups)]]
        self.returns = [None, [None for i in range(self.num_groups)], [None for i in range(self.num_groups)]]
        self.values = [None, [None for i in range(self.num_groups)], [None for i in range(self.num_groups)]]
        self.advantages = [None, [None for i in range(self.num_groups)], [None for i in range(self.num_groups)]]

        self.generator_ready = False

        # Only for APPO
        self.deltas = None
        self.delta_deltas = None
        
        self.reset()

    def reset(self) -> None:

        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)   
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        for i in range(3):
            if i == 0:
                self.rewards[i] = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
                self.returns[i] = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
                self.values[i] = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
                self.advantages[i] = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
            else:
                self.rewards[i] = [np.zeros((self.buffer_size, self.n_envs), dtype=np.float32) for g in range(self.num_groups)]
                self.returns[i] = [np.zeros((self.buffer_size, self.n_envs), dtype=np.float32) for g in range(self.num_groups)]
                self.values[i] = [np.zeros((self.buffer_size, self.n_envs), dtype=np.float32) for g in range(self.num_groups)]
                self.advantages[i] = [np.zeros((self.buffer_size, self.n_envs), dtype=np.float32) for g in range(self.num_groups)]
        
        self.generator_ready = False
        
        # Only for APPO
        self.deltas = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.delta_deltas = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        super(RolloutBuffer_fair, self).reset()

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.
        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.
        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))
        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.
        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).

        Note that one buffer might contain several episodes, and "self.episode_starts" takes care of it. 

        modification: deal with 1 + 2*M reward functions
        :param last_values: state value estimation for the last step (one for each env), for 1 + 2*M rewards
        structure: [r, [r_U_0,...], [r_B_0,...]]
        """
        # Convert to numpy
        assert len(last_values) == 3, 'Incorrect length of last_values, should be 3: [v,[v_U_0,...],[v_B_0,...]]'
        last_values[0] = last_values[0].clone().cpu().numpy().flatten()
        last_values[1] = [i.clone().cpu().numpy().flatten() for i in last_values[1]]
        last_values[2] = [i.clone().cpu().numpy().flatten() for i in last_values[2]]

        last_gae_lam = [0, [0 for i in range(self.num_groups)], [0 for i in range(self.num_groups)]]
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = [self.values[0][step + 1], [self.values[1][g][step + 1] for g in range(self.num_groups)], [self.values[2][g][step + 1] for g in range(self.num_groups)]]  
            
            for i in range(3):
                if i == 0:
                    delta = self.rewards[i][step] + self.gamma * next_values[i] * next_non_terminal - self.values[i][step]
                    last_gae_lam[i] = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam[i]
                else:
                    delta = [(self.rewards[i][g][step] + self.gamma * next_values[i][g] * next_non_terminal - self.values[i][g][step]) for g in range(self.num_groups)]
                    last_gae_lam[i] = [(delta[g] + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam[i][g]) for g in range(self.num_groups)]
            
            self.advantages[0][step] = last_gae_lam[0]
            for g in range(self.num_groups):
                self.advantages[1][g][step] = last_gae_lam[1][g]
                self.advantages[2][g][step] = last_gae_lam[2][g]

        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        self.returns[0] = self.advantages[0] + self.values[0]        
        for g in range(self.num_groups):
            self.returns[1][g] = self.advantages[1][g] + self.values[1][g]   
            self.returns[2][g] = self.advantages[2][g] + self.values[2][g]   

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        deltas: th.Tensor,
        delta_deltas: th.Tensor

    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)
        
        self.rewards[0][self.pos] = np.array(reward[0]).copy()
        self.values[0][self.pos] = value[0].clone().cpu().numpy().flatten()
        for g in range(self.num_groups):
            self.rewards[1][g][self.pos] = np.array(reward[1][g]).copy()
            self.rewards[2][g][self.pos] = np.array(reward[2][g]).copy()

            self.values[1][g][self.pos] = value[1][g].clone().cpu().numpy().flatten()
            self.values[2][g][self.pos] = value[2][g].clone().cpu().numpy().flatten()

        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()        
        # only for APPO
        self.deltas[self.pos] = deltas.clone().cpu().numpy()
        self.delta_deltas[self.pos] = delta_deltas.clone().cpu().numpy()

        self.pos += 1

        if self.pos == self.buffer_size:
            self.full = True       

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples_fair, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:

            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",

                "deltas",
                "delta_deltas",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten_fair(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> RolloutBufferSamples_fair:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            [self.values[0][batch_inds].flatten(), [self.values[1][g][batch_inds].flatten() for g in range(self.num_groups)], [self.values[2][g][batch_inds].flatten() for g in range(self.num_groups)]],
            self.log_probs[batch_inds].flatten(),
            [self.advantages[0][batch_inds].flatten(), [self.advantages[1][g][batch_inds].flatten() for g in range(self.num_groups)], [self.advantages[2][g][batch_inds].flatten() for g in range(self.num_groups)]],
            [self.returns[0][batch_inds].flatten(), [self.returns[1][g][batch_inds].flatten() for g in range(self.num_groups)], [self.returns[2][g][batch_inds].flatten() for g in range(self.num_groups)]],
            # only for APPO
            self.deltas[batch_inds].flatten(),
            self.delta_deltas[batch_inds].flatten()
        )
        return RolloutBufferSamples_fair(*tuple(map(self.to_torch, data)))
    
    @staticmethod
    # def swap_and_flatten_fair(arr: Union[np.ndarray, List[np.ndarray]] ) -> np.ndarray: 
    def swap_and_flatten_fair(arr: Union[np.ndarray, List[Union[np.ndarray,List[np.ndarray]]]] ) -> np.ndarray:    
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)
        :param arr:
        :return:

        modification: the input could be a "fairness list" (of the form [r, [r_U_0,..],[r_B_0,..]]), where r, r_U_0 are np.ndarray
        """
        if isinstance(arr, np.ndarray):
            shape = arr.shape
            if len(shape) < 3:
                shape = shape + (1,)
            return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])
        else:
            assert len(arr) == 3, 'the input list should have length 3, and is of the form [r, [r_U_0,..],[r_B_0,..]]'
            assert len(arr[1]) == len(arr[2]), 'length error! should be of the form [r, [r_U_0,..],[r_B_0,..]]'
            num_groups = len(arr[1])
            arr_return = [None, [None for i in range(num_groups)], [None for i in range(num_groups)]]

            for i in range(3):
                if i == 0:
                    shape = arr[i].shape
                    if len(shape) < 3:
                        shape = shape + (1,)
                    arr_return[i] = arr[i].swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])
                else:
                    for g in range(num_groups):
                        shape = arr[i][g].shape
                        if len(shape) < 3:
                            shape = shape + (1,)
                        arr_return[i][g] = arr[i][g].swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

            return arr_return
