from torch.distributions import Categorical

from ..abstract_solver import AbstractSolver
from ...configs import SIMULATOR


class ModelFreePolicy(AbstractSolver):
    def __init__(self, **kwargs):
        super().__init__()

    def get_action(self, state):
        logits = self.policy_network(SIMULATOR.to_tensor(state))
        return Categorical(logits=logits).sample().item()

    def reset(self):
        return

    def update(self, action):
        return

    def __str__(self):
        return self.__class__.__name__
