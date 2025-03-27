from collections import defaultdict
from copy import deepcopy
from time import time

import numpy as np
import torch as t
from torch.distributions import Categorical
from torch.nn.functional import cross_entropy

from ..abstract_solver import AbstractSolver
from ...configs import SIMULATOR


class ExploratoryPolicyGradientSearch(AbstractSolver):
    def __init__(self,
                 max_time_per_step=1e9,
                 n_trials=1e9,
                 min_policy_confidence=1.0,
                 learning_rate=0.01,
                 l2_penalty=0.0,
                 entropy_regularisation=0.0,
                 exploration_constant=0.0):
        super().__init__()

        self.max_time_per_step = max_time_per_step
        self.n_trials = int(n_trials)
        self.min_policy_confidence = min_policy_confidence
        self.learning_rate = learning_rate
        self.l2_penalty = l2_penalty
        self.entropy_regularisation = entropy_regularisation
        self.exploration_constant = exploration_constant

        self.states_cache = {}
        self.branch_reward = {}
        self.terminal = {}
        self.features_cache = {}
        self.values = defaultdict(float)
        self.visitations = defaultdict(int)
        self.plan = {}
        self.keys_cache = {}

        self.actions = np.arange(SIMULATOR.n_actions)
        self.optimiser = None

    def _initialise_caches(self, state) -> None:
        state_key = SIMULATOR.hash(state)
        self.states_cache = {'root': state}
        self.branch_reward = {}
        self.terminal = {}
        self.keys_cache = {'root': state_key}
        self.values = defaultdict(float)
        self.visitations = defaultdict(int)

    def _initialise_branch_if_not_exists(self, node: str, action: int) -> None:
        branch = f'{node}->{action}'
        if branch not in self.states_cache:
            next_state, reward, terminal = SIMULATOR.simulate(self.states_cache[node], action)

            self.states_cache[branch] = next_state
            self.branch_reward[branch] = reward
            self.terminal[branch] = terminal
            self.keys_cache[branch] = SIMULATOR.hash(next_state)

    def _get_branch_visitations(self, node):
        visitations = np.zeros(SIMULATOR.n_actions)
        for action in self.actions:
            self._initialise_branch_if_not_exists(node, action)
            visitations[action] = self.visitations[self.keys_cache[f'{node}->{action}']]

        return visitations

    def _get_policy(self, node: str, exploration=False) -> t.Tensor:
        state_key = self.keys_cache[node]
        features = self.features_cache.get(state_key)
        if features is None:
            features = self.features_for_policy(SIMULATOR.to_tensor(self.states_cache[node]))
            self.features_cache[state_key] = features
        policy = self.policy_network(features, is_features=True).reshape(-1).cpu().numpy()
        if exploration:
            policy += self.exploration_constant / (1 + self._get_branch_visitations(node))
            policy += (SIMULATOR.legal_actions(self.states_cache[node]) == 0) * -1e20

        policy = np.exp(policy - np.max(policy))
        return policy / np.sum(policy)

    def _perform_rollout(self, depth: int = SIMULATOR.max_steps) -> tuple:
        state_keys, actions, prob_actions, rewards = [], [], [], []
        is_solved = False
        node = 'root'
        for i in range(depth):
            policy = self._get_policy(node, exploration=True)
            action = np.random.choice(self.actions, p=policy)

            self._initialise_branch_if_not_exists(node, action)

            state_keys.append(self.keys_cache[node])
            actions.append(action)
            prob_actions.append(policy[action])
            rewards.append(self.branch_reward[f'{node}->{action}'])
            self.visitations[self.keys_cache[f'{node}->{action}']] += 1

            node = f'{node}->{action}'

            if self.terminal[node]:
                is_solved = SIMULATOR.is_solved(self.states_cache[node])
                break

        if is_solved:
            for state_key, action in zip(reversed(state_keys), reversed(actions)):
                if state_key not in self.plan:
                    self.plan[state_key] = action

        return state_keys, actions, prob_actions, rewards

    def _backup(self, state_keys, actions, prob_actions, rewards) -> None:
        if state_keys[-1] not in self.values or self.values[state_keys[-1]] < rewards[-1]:
            self.values[state_keys[-1]] = rewards[-1]

        for i in reversed(range(len(state_keys) - 1)):
            # If the value has not been initialised or if the newer value is better than the previous value, update the value.
            if state_keys[i] not in self.values or self.values[state_keys[i]] < rewards[i] + self.values[state_keys[i + 1]]:
                self.values[state_keys[i]] = rewards[i] + self.values[state_keys[i + 1]]

        features = [self.features_cache[key] for key in state_keys]
        values = [self.values[key] for key in state_keys]
        rewards = list(reversed(np.cumsum(list(reversed(rewards)))))

        features = t.cat(features, dim=0)
        actions = t.tensor(actions, dtype=t.long, device=features.device)
        prob_actions = t.tensor(prob_actions, dtype=t.float32, device=features.device)
        values = t.tensor(values, dtype=t.float32, device=features.device)
        rewards = t.tensor(rewards, dtype=t.float32, device=features.device)

        advantage = rewards - values
        normalised_advantage = (advantage - t.min(advantage)) / (t.max(advantage) - t.min(advantage) + 1e-10) + 1e-2
        with t.enable_grad():
            logits = self.f_policy(features)
            importance_weights = t.cumprod(t.gather(t.softmax(logits, dim=-1).detach().clone(),
                                                    dim=1,
                                                    index=actions.reshape(-1, 1)).reshape(-1)
                                           / prob_actions,
                                           dim=-1)
            nll = cross_entropy(logits, actions, reduction='none')

            loss = (nll * importance_weights * normalised_advantage).mean() - self.entropy_regularisation * Categorical(logits=logits).entropy().mean()

            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

    def get_action(self, state) -> int:
        # if the plan is already available, use it
        state_key = SIMULATOR.hash(state)
        if state_key in self.plan:
            return self.plan[state_key]
        else:
            # Initialise the caches
            self._initialise_caches(state)

            prior_policy = self._get_policy('root')
            if np.max(prior_policy) > self.min_policy_confidence:
                return int(np.argmax(prior_policy))
            else:
                original_policy_weights = deepcopy(self.f_policy.state_dict())
                self.optimiser = t.optim.SGD(self.f_policy.parameters(), lr=self.learning_rate, weight_decay=self.l2_penalty)

                # start the trials
                start_time = time()
                for trial in range(self.n_trials):
                    if time() - start_time > self.max_time_per_step:
                        break

                    # perform a rollout
                    state_keys, actions, prob_actions, rewards = self._perform_rollout()

                    # if a solution was found, break the loop
                    if self.keys_cache['root'] in self.plan:
                        return self.plan[self.keys_cache['root']]

                    # update the information
                    self._backup(state_keys, actions, prob_actions, rewards)

                action = np.random.choice(self.actions, p=self._get_policy('root'))
                self.f_policy.load_state_dict(original_policy_weights)
                return action

    def reset(self) -> None:
        self.states_cache = {}
        self.branch_reward = {}
        self.terminal = {}
        self.features_cache = {}
        self.values = defaultdict(float)
        self.visitations = defaultdict(int)
        self.plan = {}
        self.keys_cache = {}

    def update(self, action: int) -> None:
        return

    def eval(self, trt=True):
        f_policy = deepcopy(self.f_policy)
        super().eval(trt)
        self.f_policy = f_policy

    def __str__(self):
        return f'Solver Name:----------------> {self.__class__.__name__}\n' \
               f'Max Search Time / Step -----> {self.max_time_per_step}\n' \
               f'Number of Trials -----------> {self.n_trials}\n' \
               f'Min Policy Confidence ------> {self.min_policy_confidence}\n' \
               f'Learning Rate --------------> {self.learning_rate}\n' \
               f'L2 Penalty -----------------> {self.l2_penalty}\n' \
               f'Entropy Regularisation -----> {self.entropy_regularisation}\n' \
               f'Exploration Constant--------> {self.exploration_constant}\n'
