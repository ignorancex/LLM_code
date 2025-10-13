import gymnasium as gym
import numpy as np
from typing import Optional
import copy
import re
from typing_extensions import override
import pandas as pd

from ragen.env.base import BaseLanguageBasedEnv
from ragen.env.countdown.reward_function import compute_score


class CountdownEnv(BaseLanguageBasedEnv, gym.Env):
    """
    Countdown environment
    Given a list of numbers and a target number, the agent needs to input a mathematical expression that evaluates to the target number.
    
    ## Action Space
    For Countdown (text-based), the action space is text input

    ## Rewards
    - Format reward (if correct, not penalty): 0.1
    - Correct answer: 1

    ## Observation / Feedback
    No observation/feedback: Empty string

    NOTE only one step
    """

    INVALID_ACTION = ""
    PENALTY_FOR_INVALID = 0 # -0.1
    
    def __init__(self, parquet_path: str):
        """
        Initialize the environment for Countdown Problem

        Args:
            train_path: Path to the train parquet file
            test_path: Path to the test parquet file
        """
        BaseLanguageBasedEnv.__init__(self)
        self.data, self.seed_to_index = self._get_data_from_parquet(parquet_path)
        self.parquet_path = parquet_path
        self.last_action = None
        self._success = False
        self._finished = False

        self.index = None # index of the data

    @staticmethod
    def _get_data_from_parquet(path: str):
        """
        Get data from parquet file and create mapping.

        Args:
            path: Path to the parquet file containing the data

        Returns:
            tuple: (data, mapping) where
                data: List of dicts containing target and numbers for each problem
                mapping: Dict mapping original indices to new sequential indices

        The function:
        1. Reads the parquet file
        2. Extracts target numbers and available numbers for each problem
        3. Creates an index mapping to maintain reference to original data
        """
        df = pd.read_parquet(path)
        
        # Extract target and numbers for each problem
        data = [
            {
                'target': item['ground_truth']['target'],
                'numbers': item['ground_truth']['numbers'].tolist()
            }
            for item in df.reward_model.values
        ]

        # Create mapping from original indices to sequential indices
        original_indices = [item['index'] for item in df.extra_info.values]
        seed_to_index = {orig_idx: new_idx for new_idx, orig_idx in enumerate(original_indices)}
        
        return data, seed_to_index


    def extract_action(self, text):
        """
        Extract action from text, all text-based input is valid
        """
        if not isinstance(text, str):
            return self.INVALID_ACTION
        return text

    def reset(self, seed: int = None, mode: str = 'text'):
        """Reset the environment and reward distributions"""
        gym.Env.reset(self, seed=seed)
        self._finished = False
        self._success = False
            
        # Reset tracking variables
        self._reset_tracking_variables()
        self.last_action = None
        self.index = self.seed_to_index[seed]
        return self.render(mode)

    def step(self, action: str):
        """
        Take text-based input and calculate the reward
        """
        assert isinstance(action, str)
        self._finished = True  # only one step
        if action == self.INVALID_ACTION:
            return self.render(), 0, True, {"action_is_effective": False}
        self._success = True

        self.last_action = action


        # format reward is set 0.1 for correct format
        # correct answer reward is set as 1
        data = self.data[self.index]
        reward = compute_score(action, data, format_score=0.1, score=1.)
        return self.render(), reward, True, {"action_is_effective": True}  # only one step

    def render(self, mode='text'):
        """Render the current state"""
        assert mode in ['text', 'rgb_array', 'tiny_rgb_array']
        if mode in ['text', 'tiny_rgb_array']:
            return ""
        
        if mode == 'rgb_array':
            return np.ones((100, 100, 3), dtype=np.uint8) * 255


    @staticmethod
    @override
    def formulate_output(env_feedback: str, done: bool = False):
        """
        No observation for countdown environment
        """
        return ""

    @staticmethod
    @override
    def parse_update_info_to_obs(update_info, action_is_valid):
        return ""
        

    def copy(self):
        """Create a copy of the environment"""
        new_env = CountdownEnv(parquet_path=self.parquet_path)
        new_env.last_action = self.last_action
        new_env._copy_tracking_variables(self)
        new_env._finished = self._finished
        new_env._success = self._success
        return new_env
    
    def finished(self):
        return self._finished

    def success(self):
        return self._success
    
    def get_last_action(self) -> int:
        if self.last_action is None:
            return self.INVALID_ACTION
        return self.last_action
