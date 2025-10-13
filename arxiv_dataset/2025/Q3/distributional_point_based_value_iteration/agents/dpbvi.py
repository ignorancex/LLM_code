import numpy as np
from pyparsing import alphas
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Literal


class DPBVI:
    def __init__(self, transition_fn: np.ndarray[float], sensor_model: np.ndarray[float], reward_fn: np.ndarray[float], terminal_states: np.ndarray[bool],
                 gamma: float, dist_support: Tuple[int, int], representation: Literal['categorical'] = 'categorical', num_particles: int = 51):
        """
            Distributional Point-Based Value Iteration (DPBVI) Algorithm
            :param transition_fn: transition function: state x action x next state
            :param sensor_model: sensor model: state x observation
            :param reward_fn: reward function: state x action
            :param terminal_states: terminal states: state
            :param gamma: discount factor
            :param dist_support: support of the distribution
            :param representation: distributional representation type
            :param num_particles: number of particles
            """
        match representation:
            case s if s.lower() == 'categorical':
                pass
            case _:
                raise ValueError(f'Invalid representation type: {representation}')
        assert transition_fn.ndim == 3, 'Transition function must be 3D'
        assert transition_fn.shape[0] == transition_fn.shape[2], 'Transition function must be square'
        assert sensor_model.ndim == 2, 'Sensor model must be 2D'
        assert sensor_model.shape[0] == transition_fn.shape[0], 'Sensor model must have same number of states as transition function'
        assert reward_fn.ndim == 2, 'Reward function must be 2D'
        assert reward_fn.shape[0] == transition_fn.shape[0], 'Reward function must have same number of states as transition function'
        assert reward_fn.shape[1] == transition_fn.shape[1], 'Reward function must have same number of actions as transition function'
        assert 0.0 <= gamma < 1.0, 'Discount factor must be in [0, 1)'
        assert dist_support[0] < dist_support[1], 'Distributional support must be increasing'
        self._transition_function = transition_fn
        self._sensor_model = sensor_model
        self._reward_fn = reward_fn
        self._non_terminal_states = np.logical_not(terminal_states)  # state
        self._gamma = gamma
        self._dist_support = dist_support
        self._n_particles = num_particles
        self._particles = np.linspace(self._dist_support[0], self._dist_support[1], self._n_particles)
        self._particle_step_size = (self._dist_support[1] - self._dist_support[0]) / (self._n_particles - 1)
        self._num_states = transition_fn.shape[0]
        self._num_actions = transition_fn.shape[1]
        self._num_obs = sensor_model.shape[1]
        self._etas = self._policy = self._values = self._beliefs = self._cached_values = self._cached_etas = None

    # region Public Methods/Functions

    def plan(self, beliefs: np.ndarray[float], max_itrs: int, steps_before_belief_expansion: Optional[int] = None, convergence_eps: float = 1e-3,
             init_dist_type: Literal['uniform', 'zero'] = 'zero', backup_type: Literal['og', 'efficient'] = 'efficient') -> np.ndarray[int]:
        """
        Perform DPBVI planning
        :param beliefs: belief x state: P(s | b) - probability of being in state s given belief b
        :param max_itrs: maximum number of iterations to run
        :param steps_before_belief_expansion: number of steps to run before expanding the belief set
        :param convergence_eps: epsilon for convergence
        :param init_dist_type: type of distribution to use for initialization
        :param backup_type: type of backup operation to use
        :return: a plan conditioned on the belief and sequence of observations
        """
        assert np.isclose(beliefs.sum(-1), 1.0).all(), 'Belief must sum to 1'
        assert beliefs.ndim == 2, 'Belief must be 2D'
        assert beliefs.shape[1] == self._num_states, 'Belief must be defined for all states'
        can_continue = True
        self._etas: np.ndarray[float] = np.full((1, self._num_states, self._n_particles), 1 / self._n_particles, dtype=np.float32)
        if init_dist_type == 'zero':
            self._etas = np.zeros((1, self._num_states, self._n_particles), dtype=np.float64)
            zero_idx = np.isclose(self._particles, 0.0).nonzero()[0][0]
            self._etas[:, :, zero_idx] = 1.0
        self._values: np.ndarray[float] = np.zeros((beliefs.shape[0]), dtype=np.float32)
        self._cached_values = [self._values.copy()]
        self._cached_etas = [self._etas.copy()]
        self._policy: np.ndarray[int] = np.zeros(beliefs.shape[0], dtype=np.int64)
        self._beliefs = beliefs
        itr = 0
        with tqdm(desc='DPBVI Iterations', unit=' iteration', total=max_itrs) as pbar:
            while can_continue:
                match backup_type:
                    case 'og':
                        etas, values, self._policy = self._do_backup(self._etas, self._beliefs)
                    case 'efficient':
                        etas, values, self._policy = self._do_efficient_backup(self._etas, self._beliefs)
                    case _:
                        raise ValueError(f'Invalid backup type: {backup_type}')
                itr += 1
                can_continue = (max_itrs is None or itr < max_itrs)
                if values.shape[0] == self._values.shape[0]:
                    can_continue &= not np.isclose(values, self._values, atol=convergence_eps, rtol=0.0).all()
                if can_continue and (steps_before_belief_expansion is None or itr >= steps_before_belief_expansion):
                    self._beliefs = self._expand_beliefs(self._beliefs)
                self._values = values
                self._etas = etas
                self._cached_values.append(self._values.copy())
                self._cached_etas.append(self._etas.copy())
                pbar.update(1)
            pbar.close()
        self._cached_values = np.stack(self._cached_values, axis=0)
        print('DPBVI converged in', itr, 'iterations')
        return self._policy, self._beliefs
    # endregion

    # region Private Methods/Functions

    def _do_efficient_backup(self, prev_etas: np.ndarray[float], beliefs: np.ndarray[float]) -> Tuple[np.ndarray[float], np.ndarray[float], np.ndarray[int]]:
        """
        Perform the efficient backup operation
        :param prev_etas: eta x state x particle
        :param beliefs: belief x state
        :return:
        """
        # Bellman Operator Projections for each Action and Observation, and Distribution Projection
        next_vals = prev_etas @ self._particles # eta x next state
        next_vals = next_vals[:, np.newaxis, np.newaxis, np.newaxis, :] # eta x action x observation x state x next state
        next_vals = next_vals.repeat(self._num_actions, 1)
        next_vals = next_vals.repeat(self._num_obs, 2)
        next_vals = next_vals.repeat(self._num_states, 3)
        rewards = self._reward_fn[np.newaxis, :, :, np.newaxis, np.newaxis] # eta x state x action x observation x next state
        rewards = rewards.repeat(prev_etas.shape[0], 0)
        rewards = rewards.repeat(self._num_obs, 3)
        rewards = rewards.repeat(self._num_states, 4)
        rewards = np.moveaxis(rewards, 1, 3) # eta x action x observation x state x next state
        values = rewards
        values[:, :, :, :, self._non_terminal_states] += self._gamma * next_vals[:, :, :, :, self._non_terminal_states]
        # values += self._gamma * next_vals # This is used if we dont want to account for terminal states - it's really here to compare this output with that of PBVI because it doesnt consider terminal states
        values = values[:, :, :, :, :, np.newaxis].repeat(self._n_particles, -1) # eta x action x observation x state x next state x particle
        zs = (values - self._particles) / self._particle_step_size # eta x action x observation x state x next state x particle
        # apply Projection Operator - triangular (middle atoms) and half-triangular (edge atoms) kernels
        triangular_kernel = np.zeros_like(zs) # eta x action x observation x state x next state x particle
        # Three cases: left edge, right edge, and middle
        # Middle
        zs_slice = zs[:, :, :, :, :, 1:-1] # eta x action x observation x state x next state x particle
        mask = (np.abs(zs_slice) < 1.0) | np.isclose(np.abs(zs_slice), 1.0)
        if mask.any():
            triangular_kernel[:, :, :, :, :, 1:-1] = np.where(mask, np.maximum(0, 1 - np.abs(zs_slice)), 0)
        # Left edge
        zs_slice = zs[:, :, :, :, :, 0] # eta x action x observation x state x next state
        mask = (zs_slice < 1.0) | np.isclose(zs_slice, 1.0)
        if mask.any():
            triangular_kernel[:, :, :, :, :, 0] = np.where(mask, np.where((zs_slice <= 0.0) | np.isclose(0.0, zs_slice), 1.0, np.maximum(0, 1 - np.abs(zs_slice))), 0)
        # Right edge
        zs_slice = zs[:, :, :, :, :, -1]
        mask = zs_slice > -1.0
        if mask.any():
            triangular_kernel[:, :, :, :, :, -1] = np.where(mask, np.where(zs_slice > 0.0, 1.0, np.maximum(0, 1 - np.abs(zs_slice))), 0)
        trans_fn_rep = self._transition_function[np.newaxis, :, :, np.newaxis, :, np.newaxis] # eta x state x action x observation x next state x particle
        trans_fn_rep = trans_fn_rep.repeat(prev_etas.shape[0], 0)
        trans_fn_rep = trans_fn_rep.repeat(self._num_obs, 3)
        trans_fn_rep = trans_fn_rep.repeat(self._n_particles, -1)
        trans_fn_rep = np.moveaxis(trans_fn_rep, 1, 3) # eta x action x observation x state x next state x particle
        sensor_fn_rep = self._sensor_model[np.newaxis, np.newaxis, :, :, np.newaxis, np.newaxis] # eta x action x next state x observation x state x particle
        sensor_fn_rep = sensor_fn_rep.repeat(prev_etas.shape[0], 0)
        sensor_fn_rep = sensor_fn_rep.repeat(self._num_actions, 1)
        sensor_fn_rep = sensor_fn_rep.repeat(self._num_states, 4)
        sensor_fn_rep = sensor_fn_rep.repeat(self._n_particles, -1)
        sensor_fn_rep = np.moveaxis(sensor_fn_rep, 2, 4) # eta x action x observation x state x next state x particle
        eta_a_o = (trans_fn_rep * sensor_fn_rep * triangular_kernel).sum(4) # eta x action x observation x state x particle
        # NOTE: eta_a_o does not contain true probability distributions because the observations are split up. Summing across the observations will yield a distribution.
        # Compute Cross-Sums and Extract Best Etas for Each Belief
        eta_a_max_o_b = (eta_a_o @ self._particles) @ beliefs.transpose() # eta x action x observation x belief
        eta_indices = eta_a_max_o_b.argmax(axis=0) # select the max eta for each action-observation pair
        action_indices, observation_indices, belief_indices = np.meshgrid(np.arange(self._num_actions), np.arange(self._num_obs), np.arange(beliefs.shape[0]), indexing='ij')
        etas_a_max_o_b = eta_a_o[eta_indices, action_indices, observation_indices] # action x observation x belief x state x particle
        etas_a_max_o_b = etas_a_max_o_b.sum(axis=1) # action x belief x state x particle: sum over observation
        etas_a_max_o_b = np.swapaxes(etas_a_max_o_b, 0, 1) # belief x action x state x particle
        # normalize etas_a_max_o_b
        # etas_a_max_o_b /= etas_a_max_o_b.sum(-1, keepdims=True) + 1e-6 # add some epsilon to avoid division by zero
        # assert np.allclose(etas_a_max_o_b.sum(-1), 1.0), 'Etas are not normalized' # This seems to occur for minigrid example - it's because our transition function isn't a true probability distribution for some states and actions - i.e., some state actions have zero transition probability
        q_values = (beliefs[:, np.newaxis].repeat(self._num_actions, 1) * (etas_a_max_o_b @ self._particles)).sum(-1) # belief x action
        values = q_values.max(-1) # belief
        policy = q_values.argmax(-1) # belief
        etas = etas_a_max_o_b[np.arange(beliefs.shape[0]), policy] # belief x state x particle
        etas = np.unique(etas, axis=0) # filter out duplicates
        return etas, values, policy

    def _do_backup(self, prev_etas: np.ndarray[float], beliefs: np.ndarray[float]) -> Tuple[np.ndarray[float], np.ndarray[float], np.ndarray[int]]:
        """
        Perform the backup operation
        :param prev_etas: eta x state x particle
        :param beliefs: belief x state
        :return:
        """
        # etas_a = self._reward_fn.transpose()  # action x state
        etas_a_o = np.zeros((prev_etas.shape[0], self._num_actions, self._num_obs, self._num_states, self._n_particles)) # eta x action x observation x state x particle
        # Bellman Operator Projections for each Action and Observation, and Distribution Projection
        for eta_idx in range(prev_etas.shape[0]):
            prev_eta = prev_etas[eta_idx]
            for state in range(self._num_states):
                for action in range(self._num_actions):
                    for obs in range(self._num_obs):
                        for next_state in range(self._num_states):
                            next_val = prev_eta[next_state].dot(self._particles) # particles -> float
                            alpha = self._transition_function[state, action, next_state] * self._sensor_model[next_state, obs]
                            if self._non_terminal_states[next_state]:
                                self._project_value_onto(etas_a_o[eta_idx, action, obs, state], alpha, self._reward_fn[state, action] + self._gamma * next_val)
                            else:
                                self._project_value_onto(etas_a_o[eta_idx, action, obs, state], alpha, self._reward_fn[state, action])
        # NOTE: eta_a_o does not contain true probability distributions because the observations are split up. Summing across the observations will yield a distribution.
        etas_a_o_expected_val = (etas_a_o @ self._particles) # eta x action x observation x state
        # Compute Cross-Sums and Extract Best Etas for Each Belief
        values = np.zeros((beliefs.shape[0]), dtype=np.float32)
        policy = np.zeros((beliefs.shape[0]), dtype=np.int64)
        etas = list()
        for b in range(beliefs.shape[0]):
            belief = beliefs[b]
            alphas_a_o_b = etas_a_o_expected_val @ belief # eta x action x observation
            alphas_a_max_o_b = np.zeros((self._num_actions, self._num_obs, self._num_states, self._n_particles), dtype=np.float32)
            for action in range(self._num_actions):
                for obs in range(self._num_obs):
                    max_alpha = alphas_a_o_b[:, action, obs].argmax() # get max alpha for a given action-observation pair
                    alphas_a_max_o_b[action, obs] = etas_a_o[max_alpha, action, obs]
            alphas_a_max_o_b = alphas_a_max_o_b.sum(1) # action x state x particle: sum over observations
            q_values = ((alphas_a_max_o_b @ self._particles) @ belief) # action
            values[b] = q_values.max() # max over actions
            best_action = q_values.argmax() # argmax over actions
            policy[b] = best_action
            etas.append(alphas_a_max_o_b[best_action])
        etas = np.stack(etas, axis=0)
        etas = np.unique(etas, axis=0) # filter out duplicates
        # assert np.allclose(etas.sum(-1), 1.0), 'Etas are not normalized' # This seems to occur for minigrid example - i believe it's because our transition function isn't a true probability distribution for some states and actions - i.e., some state actions have zero transition probability
        return etas, values, policy

    def _project_value_onto(self, eta_s: np.ndarray[float], alpha: float, value: float):
        """
        Adds the weighted projection of the value to eta. Projection is accomplished using half triangular kernels.
        :param eta_s: particle: state distribution
        :param alpha: value weight (i.e., prob of the value)
        :param value: value to project onto the support of eta
        """
        for p in range(self._n_particles):
            z = (value - self._particles[p]) / self._particle_step_size
            match p:
                case 0:
                    mask = (z < 1.0) | np.isclose(z, 1.0)
                    if mask:
                        masked_z = z
                        triangular_kernel = np.where((masked_z <= 0.0) | np.isclose(0.0, masked_z), 1.0, np.maximum(0, 1 - np.abs(masked_z)))
                case n if n == self._n_particles - 1:
                    mask = z > -1.0
                    if mask:
                        masked_z = z
                        triangular_kernel = np.where(masked_z > 0.0, 1.0, np.maximum(0, 1 - np.abs(masked_z)))
                case _:
                    mask = ((np.abs(z) < 1.0) & (np.abs(z) > -1.0)) | np.isclose(np.abs(z), 1.0) | np.isclose(np.abs(z), -1.0)
                    if mask:
                        masked_z = z
                        triangular_kernel = np.maximum(0, 1 - np.abs(masked_z))
            if mask:
                eta_s[p] += alpha * triangular_kernel # prob of the value scaled by the kernel

    def _expand_beliefs(self, beliefs: np.ndarray[float]) -> np.ndarray[float]: # TODO: move this to a base class or make DPBVI inherit from PBVI
        """
        Expand the set of beliefs by PBVI method
        :param beliefs: belief x state: P(s | b) - probability of being in state s given belief b
        :return: a slightly larger set of beliefs to consider
        """
        new_beliefs = list()
        for b in range(beliefs.shape[0]):
            belief = beliefs[b]
            # compute the set of all possible one-step look ahead beliefs
            next_beliefs = np.zeros((self._num_actions, self._num_states), dtype=np.float32)
            for action in range(self._num_actions):
                for next_state in range(self._num_states):
                    next_beliefs[action, next_state] = self._sensor_model[next_state, action]
                    for state in range(self._num_states):
                        next_beliefs[action, next_state] += self._transition_function[state, action, next_state] * belief[state]
                # normalize
                next_beliefs[action] /= next_beliefs[action].sum()
            # compute distances
            distances = np.linalg.norm(belief - next_beliefs, axis=-1, ord=1)
            furthest_belief = next_beliefs[distances.argmax()]
            # check if it is already in the set
            b_in_old_beliefs = np.isclose(beliefs, furthest_belief).all(-1).any()
            b_in_new_beliefs = np.isclose(np.array(new_beliefs, dtype=np.float32), furthest_belief).all(-1).any() if new_beliefs else False
            if not (b_in_old_beliefs or b_in_new_beliefs):
                new_beliefs.append(furthest_belief)
        if new_beliefs:
            new_beliefs = np.concatenate([beliefs, np.array(new_beliefs, dtype=np.float32)], axis=0)
            return new_beliefs
        else:
            return beliefs

    # endregion

    # region Properties

    @property
    def etas(self) -> np.ndarray[float]:
        return self._etas

    @property
    def policy(self) -> np.ndarray[int]:
        return self._policy

    @property
    def values(self) -> np.ndarray[float]:
        return self._values

    @property
    def beliefs(self) -> np.ndarray[float]:
        return self._beliefs

    @property
    def cached_values(self) -> np.ndarray[float]:
        return self._cached_values

    @property
    def cached_etas(self) -> np.ndarray[float]:
        return self._cached_etas

    # endregion
