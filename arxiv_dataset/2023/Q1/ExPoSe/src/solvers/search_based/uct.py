from collections import defaultdict
from time import time

import numpy as np
import torch as t
from torch.distributions import Categorical

from ..abstract_solver import AbstractSolver
from ...configs import SIMULATOR


# UCT for Deterministic Problems
class UCT(AbstractSolver):
    def __init__(self,
                 max_time_per_step=1e9,
                 n_trials=1e9,
                 min_policy_confidence=1.0,
                 exploration_constant=0):

        super().__init__()

        self.max_time_per_step = max_time_per_step
        self.n_trials = int(n_trials)
        self.min_policy_confidence = min_policy_confidence
        self.exploration_constant = exploration_constant

        self.states = {}
        self.branch_reward = {}
        self.terminal = {}
        self.features_for_policy_cache = {}
        self.features_for_value_cache = {}
        self.policy = {}
        self.values = {}
        self.visitations = defaultdict(int)
        self.plan = {}
        self.hashes = {}
        self.tensors = {}

        self.visited = set()
        self.visited_during_rollout = set()

        self.actions = np.arange(SIMULATOR.n_actions)

    def initialise(self, state) -> None:
        self.states = {'root': state}
        self.branch_reward = {'root': 0}
        self.terminal = {'root': False}
        self.hashes = {}

        self.visited = {'root'}
        self.visited_during_rollout = set()

    def get_features_for_policy(self, node: str) -> t.FloatTensor:
        key = self.hash(node)
        if key not in self.features_for_policy_cache:
            self.features_for_policy_cache[key] = self.features_for_policy(self.get_tensor(node))

        return self.features_for_policy_cache[key]

    def get_features_for_value(self, node: str) -> t.FloatTensor:
        key = self.hash(node)
        if key not in self.features_for_value_cache:
            self.features_for_value_cache[key] = self.features_for_value(self.get_tensor(node))

        return self.features_for_value_cache[key]

    def get_policy(self, node: str) -> t.FloatTensor:
        key = self.hash(node)
        if key not in self.policy:
            self.policy[key] = t.softmax(self.policy_network(self.get_features_for_policy(node), is_features=True), dim=-1).reshape(-1).cpu().numpy()

        return self.policy[key]

    def get_value(self, node: str) -> float:
        if node not in self.values:
            self.values[node] = self.value_network(self.get_features_for_value(node), is_features=True).item()
            self.visitations[node] = 1
        return self.values[node]

    def get_visitations(self, node: str) -> int:
        if node not in self.visitations:
            self.visitations[node] = 0
        return self.visitations[node]

    def get_tensor(self, node: str) -> t.FloatTensor:
        key = self.hash(node)
        if key not in self.tensors:
            self.tensors[key] = SIMULATOR.to_tensor(self.states[node])

        return self.tensors[key]

    def hash(self, state) -> int:
        if isinstance(state, str):
            if state not in self.hashes:
                self.hashes[state] = SIMULATOR.hash(self.states[state])
            return self.hashes[state]
        else:
            return SIMULATOR.hash(state)

    def rollout(self, node: str, depth: int = SIMULATOR.max_steps) -> float:
        # if state is terminal then return 0
        if self.terminal[node]:
            self.visited_during_rollout.add(node)
            return 0

        state = self.states[node]
        states, actions, value, is_solved = [], [], 0, False
        path = node
        for i in range(depth):
            action = np.random.choice(self.actions, p=self.get_policy(path))

            states.append(state)
            actions.append(action)
            path = f'{path}->{action}'

            state, reward, terminal = SIMULATOR.simulate(state, action)

            value += reward
            self.states[path] = state
            self.branch_reward[path] = reward
            self.terminal[path] = terminal

            if terminal:
                # if successfully reached the goal then set the solved flag
                is_solved = SIMULATOR.is_solved(state)
                break

        self.visited_during_rollout.add(path)

        if is_solved:
            for state, action in zip(reversed(states), reversed(actions)):
                key = self.hash(state)
                if key not in self.plan:
                    self.plan[key] = action

        return value

    def q_value(self, node: str, action: int) -> float:
        # check if that action is allowed
        if SIMULATOR.legal_actions(self.states[node])[action] == 0:
            return -np.inf

        child = '{}->{}'.format(node, action)
        # if state has not been initialised in the cache, then initialise it first.
        if child not in self.states:
            next_state, reward, terminal = SIMULATOR.simulate(self.states[node], action)

            self.states[child] = next_state
            self.branch_reward[child] = reward
            self.terminal[child] = terminal

        # return the value of next node + the branch reward
        return self.branch_reward[child] + self.get_value(child) * (not self.terminal[child])

    def q_values(self, node: str) -> np.ndarray:
        return np.array([self.q_value(node, a) for a in range(SIMULATOR.n_actions)])

    def branch_visitations(self, node: str) -> np.ndarray:
        return np.asarray([self.get_visitations(f'{node}->{a}') for a in range(SIMULATOR.n_actions)])

    def select_branch_to_explore(self, node: str) -> int:
        q_values = self.q_values(node)
        visitations = self.branch_visitations(node)
        exploration_bonus = self.exploration_constant * np.sqrt(np.log1p(np.sum(visitations)) / (1 + visitations))

        return int(np.argmax(q_values + exploration_bonus))

    def backup(self, node: str, value=None) -> None:
        if value is None:
            value = self.rollout(node)

        is_current_leaf_solved = self.hash(node) in self.plan
        action = -1
        while True:
            current_node_value, current_node_visitations = self.get_value(node), self.get_visitations(node)

            self.values[node] = (current_node_value * current_node_visitations + value) / (current_node_visitations + 1)
            self.visitations[node] = current_node_visitations + 1

            if is_current_leaf_solved:
                key = self.hash(node)
                if key not in self.plan:
                    self.plan[key] = action

            if node == 'root':
                break

            value += self.branch_reward[node]

            idx = node.rindex('>')
            action = int(node[idx + 1:])
            node = node[:idx - 1]

    def get_action(self, state) -> int:
        # if the plan is already available, use it
        if self.hash(state) in self.plan:
            return self.plan[self.hash(state)]

        distribution = Categorical(logits=self.policy_network(SIMULATOR.to_tensor(state)))
        if distribution.probs.max().item() > self.min_policy_confidence:
            return t.argmax(distribution.probs).item()
        else:
            # initialise all the caches
            self.initialise(state)

            # start the trials
            start_time = time()
            for trial in range(self.n_trials):
                if time() - start_time > self.max_time_per_step:
                    break

                node = 'root'
                while not self.terminal[node]:
                    action = self.select_branch_to_explore(node)
                    if action == -1:
                        break

                    node = f'{node}->{action}'

                    if node not in self.visited:
                        break

                # add the new leaf node to the tree
                self.visited.add(node)

                # backup from this leaf node
                self.backup(node)

                if self.hash('root') in self.plan:
                    return self.plan[self.hash('root')]

            return self.plan.get('root', np.random.choice(self.actions, p=self.get_policy('root')))

    def update(self, action: int) -> None:
        child = f'root->{action}'
        self.values = {k.replace(child, 'root'): v for k, v in self.values.items() if child in k}
        self.visitations = {k.replace(child, 'root'): v for k, v in self.visitations.items() if child in k}

    def reset(self) -> None:
        self.states = {}
        self.branch_reward = {}
        self.terminal = {}
        self.features_for_policy_cache = {}
        self.features_for_value_cache = {}
        self.policy = {}
        self.values = {}
        self.visitations = defaultdict(int)
        self.plan = {}
        self.hashes = {}
        self.tensors = {}

        self.visited = set()
        self.visited_during_rollout = set()

    def __str__(self):
        return f'Solver Name:----------------> {self.__class__.__name__}\n' \
               f'Max Search Time / Step -----> {self.max_time_per_step}\n' \
               f'Number of Trials -----------> {self.n_trials}\n' \
               f'Min Policy Confidence ------> {self.min_policy_confidence}\n' \
               f'Exploration Constant -------> {self.exploration_constant}\n'
