import  warnings
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Literal, List


class PBVI:
    def __init__(self, transition_fn: np.ndarray[float], sensor_model: np.ndarray[float], reward_fn: np.ndarray[float], terminal_states: np.ndarray[bool], gamma: float):
        """
        Point-Based Value Iteration (PBVI) Algorithm
        :param transition_fn: state x action x state: P(s' | s, a) - probability of transitioning to s' given s and a
        :param sensor_model: state x action x observation: P(o | s) - probability of observing o given s
        :param reward_fn: state x action: R(s, a) - reward for taking action a in state s
        :param terminal_states: terminal states: state
        :param gamma: discount factor
        :param epsilon: convergence threshold
        """
        assert transition_fn.ndim == 3, 'Transition function must be 3D'
        assert transition_fn.shape[0] == transition_fn.shape[2], 'Transition function must be square'
        assert sensor_model.ndim == 2, 'Sensor model must be 2D'
        assert sensor_model.shape[0] == transition_fn.shape[0], 'Sensor model must have same number of states as transition function'
        assert reward_fn.ndim == 2, 'Reward function must be 2D'
        assert reward_fn.shape[0] == transition_fn.shape[0], 'Reward function must have same number of states as transition function'
        assert reward_fn.shape[1] == transition_fn.shape[1], 'Reward function must have same number of actions as transition function'
        assert 0.0 <= gamma <= 1.0, 'Discount factor must be in [0, 1]'
        self._transition_function = transition_fn
        self._sensor_model = sensor_model
        self._reward_fn = reward_fn
        self._terminal_states = terminal_states
        self._gamma = gamma
        self._num_states = transition_fn.shape[0]
        self._num_actions = transition_fn.shape[1]
        self._num_obs = sensor_model.shape[1]
        self._alphas = self._policy = self._values = self._beliefs = self._cached_values = self._cached_alphas = None

    # region Public Methods/Functions

    def plan(self, beliefs: np.ndarray[float], steps_before_belief_expansion: Optional[int] = None, max_itrs: Optional[int] = None, convergence_eps: float = 1e-3,
             backup_type: Literal['og', 'efficient'] = 'efficient') -> Tuple[np.ndarray[int], np.ndarray[float]]:
        """

        :param beliefs: belief x state: P(s | b) - probability of being in state s given belief b
        :param steps_before_belief_expansion: number of steps to run before expanding the belief set
        :param max_itrs: maximum number of iterations to run
        :param convergence_eps: convergence threshold
        :param backup_type: backup method to use
        :return: a plan conditioned on the belief and sequence of observations
        """
        warnings.warn('This implementation does not account for terminal states like DBVI does', UserWarning)
        assert np.isclose(beliefs.sum(-1), 1.0).all(), 'Belief must sum to 1'
        assert beliefs.ndim == 2, 'Belief must be 2D'
        assert beliefs.shape[1] == self._num_states, 'Belief must be defined for all states'
        can_continue = True
        # PBVI only maintains one alpha-vector per belief
        self._alphas: np.ndarray[float] = np.zeros((1, self._num_states), dtype=np.float32)
        self._values: np.ndarray[float] = np.zeros((beliefs.shape[0]), dtype=np.float32)
        self._policy: np.ndarray[int] = np.zeros(beliefs.shape[0], dtype=np.int64)
        self._beliefs = beliefs
        self._cached_values = [self._values.copy()]
        self._cached_alphas = [self._alphas.copy()]
        itr = 0
        with tqdm(desc='PBVI Iterations', unit=' iteration', total=max_itrs) as pbar:
            while can_continue:
                match backup_type:
                    case 'og':
                        alphas, values, self._policy = self._do_backup(self._alphas, self._beliefs)
                    case 'efficient':
                        alphas, values, self._policy = self._do_efficient_backup(self._alphas, self._beliefs)
                    case _:
                        raise ValueError(f'Invalid backup type: {backup_type}')
                itr += 1
                can_continue = (max_itrs is None or itr < max_itrs)
                if values.shape[0] == self._values.shape[0]:
                    can_continue &= not np.isclose(values, self._values, atol=convergence_eps, rtol=0.0).all()
                if can_continue and (steps_before_belief_expansion is None or itr >= steps_before_belief_expansion):
                    self._beliefs = self._expand_beliefs(self._beliefs)
                self._values = values
                self._alphas = alphas
                self._cached_values.append(self._values.copy())
                self._cached_alphas.append(self._alphas.copy())
                pbar.update(1)
            pbar.close()
        self._cached_values = np.stack(self._cached_values, axis=0)
        print('PBVI converged in', itr, 'iterations')
        return self._policy, self._beliefs

    def plot_alphas(self):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        for a_idx in range(self._alphas.shape[0]):
            sns.lineplot(x=np.arange(self._num_states), y=self._alphas[a_idx], ax=ax)
        plt.show()

    # endregion Public Methods/Functions

    # region Private Methods/Functions

    def _do_efficient_backup(self, prev_alphas: np.ndarray[float], beliefs: np.ndarray[float]) -> Tuple[np.ndarray[float], np.ndarray[float], np.ndarray[int]]:
        """

        :param prev_alphas: alphas x states
        :param beliefs: beliefs x states
        :return:
        """
        # Create Projections
        alphas_a: np.ndarray[float] = self._reward_fn.transpose() # action x state
        # expand transition function to include observations
        alphas_a_o = np.expand_dims(self._transition_function, 2).repeat(self._num_obs, 2) # state x action x observation x next state
        # handle terminal states
        alphas_a_o[:, :, :, self._terminal_states] = 0.0
        # scale by sensor model
        alphas_a_o[:, :] *= self._sensor_model.transpose()
        # cross-product with alphas
        alphas_a_o = alphas_a_o @ prev_alphas.transpose() # state x action x observation x alpha
        # scale by gamma
        alphas_a_o *= self._gamma
        alphas_a_o = np.swapaxes(alphas_a_o, 0, -1) # alpha x action x observation x state
        # Compute Cross-Sums and Extract Best Alphas for Each Belief
        alphas_a_max_o_b = alphas_a_o @ beliefs.transpose() # alpha x action x observation x belief
        alphas_indices = alphas_a_max_o_b.argmax(axis=0) # select the max alpha for each action-observation pair
        action_indices, observation_indices, belief_indices = np.meshgrid(np.arange(self._num_actions), np.arange(self._num_obs), np.arange(beliefs.shape[0]), indexing='ij')
        alphas_a_max_o_b = alphas_a_o[alphas_indices, action_indices, observation_indices] # action x observation x belief x state
        alphas_a_max_o_b = alphas_a_max_o_b.sum(axis=1) # action x belief x state: sum over observations
        alphas_a_max_o_b = np.swapaxes(alphas_a_max_o_b, 0, 1) # belief x action x state
        alphas_a_b = np.expand_dims(alphas_a, 0).repeat(beliefs.shape[0], 0) + alphas_a_max_o_b # belief x action x state
        q_values = (np.expand_dims(beliefs, 1).repeat(self._num_actions, 1) * alphas_a_b).sum(-1) # belief x action
        values = q_values.max(-1) # belief
        policy = q_values.argmax(-1) # belief
        alphas = alphas_a_b[np.arange(beliefs.shape[0]), policy] # belief x state
        alphas = np.unique(alphas, axis=0) # filter out duplicates
        return alphas, values, policy

    def _do_backup(self, prev_alphas: np.ndarray[float], beliefs: np.ndarray[float]) -> Tuple[np.ndarray[float], np.ndarray[float], np.ndarray[int]]:
        """

        :param prev_alphas: alphas x states
        :param beliefs: beliefs x states
        :return:
        """
        # Create Projections
        alphas_a: np.ndarray[float] = np.zeros((self._num_actions, self._num_states)) # action x state
        alphas_a_o: np.ndarray[float] = np.zeros((prev_alphas.shape[0], self._num_actions, self._num_obs, self._num_states)) # alpha x action x observation x state
        for prev_alpha_idx in range(prev_alphas.shape[0]): # update each alpha-vector
            prev_alpha = prev_alphas[prev_alpha_idx]
            for state in range(self._num_states):
                for action in range(self._num_actions):
                    alphas_a[action, state] = self._reward_fn[state, action]
                    for obs in range(self._num_obs):
                        value = 0.0
                        for state_prime in range(self._num_states):
                            if not self._terminal_states[state_prime]:
                                value += self._transition_function[state, action, state_prime] * self._sensor_model[state_prime, obs] * prev_alpha[state_prime]
                        value = self._gamma * value
                        alphas_a_o[prev_alpha_idx, action, obs, state] = value
        # Compute Cross-Sums and Extract Best Alphas for Each Belief
        values = np.zeros(beliefs.shape[0], dtype=np.float32)
        policy = np.zeros(beliefs.shape[0], dtype=np.int64)
        alphas = list()
        for b in range(beliefs.shape[0]):
            belief = beliefs[b]
            alphas_a_max_o_b = (alphas_a_o * belief).sum(-1) # alpha x action x observation
            alphas_a_max_o_b_new = np.zeros((self._num_actions, self._num_obs, self._num_states), dtype=np.float32)
            # Well this is really inefficient but it seems to work
            for action in range(self._num_actions):
                for obs in range(self._num_obs):
                    max_alpha = alphas_a_max_o_b[:, action, obs].argmax() # get max alpha for a given action-observation pair
                    alphas_a_max_o_b_new[action, obs] = alphas_a_o[max_alpha, action, obs]
            alphas_a_max_o_b = alphas_a_max_o_b_new.sum(1) # action x state: sum over observations
            alpha_a_b = alphas_a + alphas_a_max_o_b
            q_values = (alpha_a_b * belief).sum(-1) # sum over states
            value = q_values.max() # max over actions
            values[b] = value
            best_action = q_values.argmax(-1)
            policy[b] = best_action
            alphas.append(alpha_a_b[best_action])
        alphas = np.stack(alphas, axis=0)
        alphas = np.unique(alphas, axis=0)
        return alphas, values, policy

    def _expand_beliefs(self, beliefs: np.ndarray[float]) -> np.ndarray[float]:
        """

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

    # endregion Private Methods/Functions

    # region Properties

    @property
    def policy(self) -> np.ndarray[int]:
        return self._policy

    @property
    def values(self) -> np.ndarray[float]:
        return self._values

    @property
    def alphas(self) -> np.ndarray[float]:
        return self._alphas

    @property
    def beliefs(self) -> np.ndarray[float]:
        return self._beliefs

    @property
    def cached_values(self) -> np.ndarray[float]:
        return self._cached_values

    @property
    def cached_alphas(self) -> List[np.ndarray[float]]:
        return self._cached_alphas

    # endregion Properties
