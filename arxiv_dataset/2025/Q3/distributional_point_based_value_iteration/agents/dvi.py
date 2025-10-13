import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple, Literal, Optional, List


class DVI:
    def __init__(self, transition_fn: np.ndarray[float], reward_fn: np.ndarray[float], terminal_states: np.ndarray[bool],
                 gamma: float, dist_support: Tuple[int, int], representation: Literal['categorical', 'quantile'] = 'categorical',
                 num_particles: int = 51, cache_etas: bool = False):
        """
        Distributional Value Iteration (DVI) Algorithm
        :param transition_fn: transition function: state x action x next state
        :param reward_fn: reward function: state x action
        :param terminal_states: terminal states: state
        :param gamma: discount factor
        :param dist_support: support of the distribution
        :param representation: distributional representation type
        :param num_particles: number of particles
        :param cache_etas: cache eta at each iteration
        """
        match representation:
            case s if s.lower() == 'categorical':
                pass
            case _:
                raise ValueError(f'Invalid representation type: {representation}')
        assert transition_fn.ndim == 3, 'Transition function must be 3D'
        assert transition_fn.shape[0] == transition_fn.shape[2], 'Transition function must be square'
        assert reward_fn.ndim == 2, 'Reward function must be 2D'
        assert reward_fn.shape[0] == transition_fn.shape[0], 'Reward function must have same number of states as transition function'
        assert reward_fn.shape[1] == transition_fn.shape[1], 'Reward function must have same number of actions as transition function'
        assert 0.0 <= gamma < 1.0, 'Discount factor must be in [0, 1)'
        assert dist_support[0] < dist_support[1], 'Distributional support must be increasing'
        self._transition_function = transition_fn # state x action x next state
        self._reward_fn = reward_fn # state x action
        self._non_terminal_states = np.logical_not(terminal_states) # state
        self._gamma = gamma
        self._dist_support = dist_support
        self._n_particles = num_particles
        self._particles = np.linspace(self._dist_support[0], self._dist_support[1], self._n_particles)
        self._cache_etas = cache_etas
        self._num_states = transition_fn.shape[0]
        self._num_actions = transition_fn.shape[1]
        self._eta = self._converge_itrs = self._policy = self._eta_cache = None

    def plan(self, max_itrs: int, convergence_thresh: float = 1e-3) -> np.ndarray[int]:
        particle_step_size = (self._dist_support[1] - self._dist_support[0]) / (self._n_particles - 1)
        self._eta = np.full((self._num_states, self._num_actions, self._n_particles), 1.0 / self._n_particles, dtype=np.float32) # state x action x particle
        if self._cache_etas:
            self._eta_cache = [self._eta.copy()]
        for itr in tqdm(range(max_itrs)):
            # apply Distributional Bellman Optimality Operator to returns function
            p_primes = np.expand_dims(self._eta, 0).repeat(self._num_states, axis=0) # next state x state x action x particle
            p_primes = np.moveaxis(p_primes, 0, 2) # state x action x next state x particle
            # scale p_primes by alpha - trans prob, reward prob, and action prob
            p_primes *= np.expand_dims(self._transition_function, -1).repeat(self._n_particles, axis=-1) # state x action x next state x particle
            q_vals = (self._eta @ self._particles).max(-1) # state
            next_vals = np.expand_dims(self._reward_fn, -1).repeat(self._num_states, -1) # state x action x next state
            next_vals[:, :, self._non_terminal_states] += self._gamma * q_vals[self._non_terminal_states]
            next_vals = np.expand_dims(next_vals, -1).repeat(self._n_particles, axis=-1) # state x action x next state x particle
            zs = (next_vals - self._particles) / particle_step_size # state x action x next state x particle
            # apply Projection Operator - half triangular kernel
            triangular_kernel = np.zeros_like(p_primes) # state x action x next state x particle
            # Three cases: left edge, right edge, and middle
            # Middle
            zs_slice = zs[:, :, :, 1:-1] # state x action x next state x particle
            mask = (np.abs(zs_slice) < 1.0) | np.isclose(np.abs(zs_slice), 1.0)
            if mask.any():
                triangular_kernel[:, :, :, 1:-1] = np.where(mask, np.maximum(0, 1 - np.abs(zs_slice)), 0)
            # Left edge
            zs_slice = zs[:, :, :, 0] # state x action x next state
            mask = (zs_slice < 1.0) | np.isclose(zs_slice, 1.0)
            if mask.any():
                triangular_kernel[:, :, :, 0] = np.where(mask, np.where((zs_slice <= 0.0) | np.isclose(0.0, zs_slice), 1.0, np.maximum(0, 1 - np.abs(zs_slice))), 0)
            # Right edge
            zs_slice = zs[:, :, :, -1]
            mask = zs_slice > -1.0
            if mask.any():
                triangular_kernel[:, :, :, -1] = np.where(mask, np.where(zs_slice > 0.0, 1.0, np.maximum(0, 1 - np.abs(zs_slice))), 0)
            #eta_prime = (p_primes * triangular_kernel).sum(2) # state x action x particle: sum over next state
            # NOTE: we dont use p_primes because we need to move the weight of the particles - it's more efficient to just scale by the probability of the transition
            eta_prime = (np.expand_dims(self._transition_function, -1).repeat(self._n_particles, axis=-1) * triangular_kernel).sum(2) # state x action x particle
            # eta_prime /= eta_prime.sum(-1, keepdims=True) # normalize
            assert np.isclose(eta_prime.sum(-1), 1.0).all()
            # compute delta
            delta = np.abs(self._eta - eta_prime).max()
            # update eta
            self._eta = eta_prime
            if self._cache_etas:
                self._eta_cache.append(self._eta.copy())
            self._converge_itrs = itr
            if delta < convergence_thresh:
                break
        print(f'Converged after {self._converge_itrs} iterations')
        self._policy = (self._eta @ self._particles).argmax(-1) # state
        return self._policy

    def plot_distribution(self, state: int, action: Optional[int]):
        if action is None:
            action = self._eta[state].dot(self._particles).argmax(-1)
        x_ticks = [i for i, p in enumerate(self._particles) if i % 5 == 0 or i == self._n_particles - 1]
        x_tick_lbls = [f'{int(p)}' for i, p in enumerate(self._particles) if i % 5 == 0 or i == self._n_particles - 1]
        f, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(self._eta[state, action], ax=ax)
        ax.set_xticks(x_ticks, x_tick_lbls)
        ax.set_title(f'State {state} Action {action}')
        ax.set_ylim(0.0, 1.0)
        plt.show()


    def plot_distributions(self):
        f, axes = plt.subplots(4, 12, figsize=(40, 10))
        x_ticks = [i for i, p in enumerate(self._particles) if i % 5 == 0 or i == self._n_particles - 1]
        x_tick_lbls = [f'{int(p)}' for i, p in enumerate(self._particles) if i % 5 == 0 or i == self._n_particles - 1]
        for r in range(4):
            for c in range(12):
                ax = axes[r][c]
                s = r * 12 + c
                optimal_action = self._eta[s].dot(self._particles).argmax(-1)
                sns.barplot(self._eta[s, optimal_action], ax=ax)
                ax.set_xticks(x_ticks, x_tick_lbls)
                ax.set_title(f'State {s}')
                ax.set_ylim(0.0, 1.0)
        f.suptitle('Greedy Policy Value Distributions by State')
        plt.tight_layout()
        plt.show()

    def plot_value_function(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(self._eta.dot(self._particles).max(-1).reshape(4, 12), annot=True, fmt='.1f', cmap='Blues_r', linewidths=1.0, ax=ax)
        ax.set_title(f'Value function converged after {self._converge_itrs} iterations')
        plt.show()

    @property
    def eta(self) -> Optional[np.ndarray[float]]:
        """
        Get the distributional value function
        :return: state x action x particle
        """
        return self._eta

    @property
    def policy(self) -> Optional[np.ndarray[int]]:
        """
        Get the optimal policy
        :return: state
        """
        return self._policy

    @property
    def converge_itrs(self) -> Optional[int]:
        """
        Get the number of iterations to convergence
        :return: number of iterations
        """
        return self._converge_itrs

    @property
    def eta_cache(self) -> Optional[List[np.ndarray[float]]]:
        """
        Get the cache of distributional value functions
        :return: list of state x action x particle
        """
        return self._eta_cache
