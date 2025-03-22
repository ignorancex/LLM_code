import time
from copy import deepcopy

import gymnasium as gym
import numpy as np
from typing import Iterator

class Multi2SingleObjectiveWrapper(gym.Wrapper):
    """This wrapper will convert a multi-objective environment into a single-objective environment by selecting a single objective from the multi-objective reward space.

    Args:
        env (Env): The environment to convert
        obj_idx (int): The index of the objective to select
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def step(self, action):
        """Steps through the environment, recording the episode statistics."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        assert "original_scalar_reward" in info, "Multi-objective environment must provide the original scalar reward in the info dictionary in order to convert it to a single-objective environment."
        info["multi_objective_reward"] = reward
        return obs, info['original_scalar_reward'], terminated, truncated, info

