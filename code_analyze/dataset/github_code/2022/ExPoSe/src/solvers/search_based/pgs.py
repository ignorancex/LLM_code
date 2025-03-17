from collections import defaultdict
from copy import deepcopy
from time import time

import numpy as np
import torch as t
from torch.distributions import Categorical
from torch.nn.functional import cross_entropy

from .uct import UCT
from ...configs import SIMULATOR


class PolicyGradientSearch(UCT):
    def __init__(self,
                 max_time_per_step=1e9,
                 n_trials=1e9,
                 min_policy_confidence=1.0,
                 learning_rate=0.01,
                 l2_penalty=0.0,
                 entropy_regularisation=0.0,
                 exploration_constant=0.0):
        super(PolicyGradientSearch, self).__init__(max_time_per_step, n_trials, min_policy_confidence, exploration_constant)
        self.learning_rate = learning_rate
        self.l2_penalty = l2_penalty
        self.entropy_regularisation = entropy_regularisation

    def initialise(self, state) -> None:
        super().initialise(state)
        self.values = {}
        self.visitations = defaultdict(int)

    def get_policy(self, node: str) -> t.Tensor:
        return t.softmax(self.policy_network(self.get_features_for_policy(node), is_features=True), dim=-1).reshape(-1).cpu().numpy()

    def rollout(self, node: str, depth: int = SIMULATOR.max_steps) -> tuple:
        if self.terminal[node]:
            self.visited_during_rollout.add(node)
            return [], [], t.tensor([0])

        state = self.states[node]
        state_keys, features, actions, rewards, is_solved = [], [], [], [], False
        path = node
        for i in range(depth):
            action = np.random.choice(self.actions, p=self.get_policy(path))

            state_keys.append(self.hash(state))
            features.append(self.get_features_for_policy(path))
            actions.append(action)

            path = f'{path}->{action}'

            state, reward, terminal = SIMULATOR.simulate(state, action)

            rewards.append(reward)

            self.states[path] = state
            self.branch_reward[path] = reward
            self.terminal[path] = terminal

            if terminal:
                is_solved = SIMULATOR.is_solved(state)
                break

        self.visited_during_rollout.add(path)

        if is_solved:
            for key, action in zip(reversed(state_keys), reversed(actions)):
                if key not in self.plan:
                    self.plan[key] = action

        rewards = list(reversed(np.cumsum(list(reversed(rewards)))))

        return (t.cat(features, dim=0),
                t.tensor(actions, dtype=t.long, device=features[0].device),
                t.tensor(rewards, dtype=t.float32, device=features[0].device))

    def backup(self, node: str, value=None) -> None:
        features, actions, rewards = self.rollout(node)

        # Update the policy using rewards
        if len(features) != 0:
            optimiser = t.optim.SGD(self.f_policy.parameters(), lr=self.learning_rate, weight_decay=self.l2_penalty)
            advantage = (rewards - t.min(rewards)) / (t.max(rewards) - t.min(rewards) + 1e-10) + 1e-2
            with t.enable_grad():
                logits = self.f_policy(features)
                nll = cross_entropy(logits, actions, reduction='none')

                loss = (nll * advantage).mean() - self.entropy_regularisation * Categorical(logits=logits).entropy().mean()

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

        super(PolicyGradientSearch, self).backup(node, rewards[0].item())

    def get_action(self, state) -> int:
        # if the plan is already available, use it
        if self.hash(state) in self.plan:
            return self.plan[self.hash(state)]

        prior_policy = t.softmax(self.policy_network(SIMULATOR.to_tensor(state)), dim=-1).reshape(-1).cpu().numpy()
        if np.max(prior_policy) > self.min_policy_confidence:
            return int(np.argmax(prior_policy))
        else:
            original_policy_weights = deepcopy(self.f_policy.state_dict())

            # initialise all the caches
            self.initialise(state)

            # start the trials
            start_time = time()
            for trial in range(self.n_trials):
                if time() - start_time > self.max_time_per_step:
                    break

                node = 'root'
                action = self.select_branch_to_explore(node)
                if action != -1:
                    node = f'{node}->{action}'

                self.backup(node)

                if self.hash('root') in self.plan:
                    break

            action = np.random.choice(self.actions, p=self.get_policy('root'))
            self.f_policy.load_state_dict(original_policy_weights)
            return self.plan.get(self.hash('root'), action)

    def update(self, action: int) -> None:
        return

    def eval(self, trt=True):
        f_policy = deepcopy(self.f_policy)
        super(PolicyGradientSearch, self).eval(trt)
        self.f_policy = f_policy

    def __str__(self):
        return super().__str__() + \
               f'Learning Rate --------------> {self.learning_rate}\n' \
               f'L2 Penalty -----------------> {self.l2_penalty}\n' \
               f'Entropy Regularisation -----> {self.entropy_regularisation}\n'
