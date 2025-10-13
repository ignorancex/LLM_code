"""
Modified from Stable Baselines replay buffer to add support for tabular argmax amortization.

Original:
https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/common/buffers.py
"""

from typing import List, Union

import numpy as np

from third_party.stable_baselines_buffers import MinSegmentTree, ReplayBuffer, SumSegmentTree


class ReplayBufferWithArgmax(ReplayBuffer):
    def add(self, obs_t, action, reward, obs_tp1, done, argmax):
        data = (obs_t, action, reward, obs_tp1, done, argmax)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def extend(self, obs_t, action, reward, obs_tp1, done, argmax):
        for data in zip(obs_t, action, reward, obs_tp1, done, argmax):
            if self._next_idx >= len(self._storage):
                self._storage.append(data)
            else:
                self._storage[self._next_idx] = data
            self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes: Union[List[int], np.ndarray], env=None):
        obses_t, actions, rewards, obses_tp1, dones, argmaxes = [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done, argmax = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
            argmaxes.append(np.array(argmax, copy=False))
        return (
            self._normalize_obs(np.array(obses_t), env),
            np.array(actions),
            self._normalize_reward(np.array(rewards), env),
            self._normalize_obs(np.array(obses_tp1), env),
            np.array(dones),
            np.array(argmaxes),
        )

    def update_argmaxes(self, idxes, new_argmax):
        """Update the stored argmax approximations at given indices."""
        assert len(idxes) == len(new_argmax), "Index and argmax length mismatch"
        for i, argmax in zip(idxes, new_argmax):
            assert 0 <= i < len(self._storage), f"Index {i} out of bounds"
            obs_t, action, reward, obs_tp1, done, _ = self._storage[i]
            reshaped_argmax = np.expand_dims(argmax, axis=0)
            self._storage[i] = (obs_t, action, reward, obs_tp1, done, reshaped_argmax)


class PrioritizedReplayBufferWithArgmax(ReplayBufferWithArgmax):
    def __init__(self, size, alpha):
        super(PrioritizedReplayBufferWithArgmax, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, obs_t, action, reward, obs_tp1, done, argmax):
        idx = self._next_idx
        super().add(obs_t, action, reward, obs_tp1, done, argmax)
        self._it_sum[idx] = self._max_priority**self._alpha
        self._it_min[idx] = self._max_priority**self._alpha

    def extend(self, obs_t, action, reward, obs_tp1, done, argmax):
        idx = self._next_idx
        super().extend(obs_t, action, reward, obs_tp1, done, argmax)
        while idx != self._next_idx:
            self._it_sum[idx] = self._max_priority**self._alpha
            self._it_min[idx] = self._max_priority**self._alpha
            idx = (idx + 1) % self._maxsize

    def _sample_proportional(self, batch_size):
        mass = []
        total = self._it_sum.sum(0, len(self._storage) - 1)
        mass = np.random.random(size=batch_size) * total
        idx = self._it_sum.find_prefixsum_idx(mass)
        return idx

    def sample(self, batch_size: int, beta: float = 0, env=None):
        assert beta > 0

        idxes = self._sample_proportional(batch_size)
        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)
        p_sample = self._it_sum[idxes] / self._it_sum.sum()
        weights = (p_sample * len(self._storage)) ** (-beta) / max_weight
        encoded_sample = self._encode_sample(idxes, env=env)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities)
        assert np.min(priorities) > 0
        assert np.min(idxes) >= 0
        assert np.max(idxes) < len(self.storage)
        self._it_sum[idxes] = priorities**self._alpha
        self._it_min[idxes] = priorities**self._alpha

        self._max_priority = max(self._max_priority, np.max(priorities))
