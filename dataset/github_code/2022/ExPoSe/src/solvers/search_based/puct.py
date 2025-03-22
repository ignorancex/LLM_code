import numpy as np

from .uct import UCT


class PUCT(UCT):
    def select_branch_to_explore(self, node: str) -> int:
        q_values = self.q_values(node)
        exploration_bonus = self.exploration_constant * self.get_policy(node) / (1 + self.branch_visitations(node))
        return int(np.argmax(q_values + exploration_bonus))
