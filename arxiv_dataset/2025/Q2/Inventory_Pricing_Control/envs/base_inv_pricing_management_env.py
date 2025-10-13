# envs/base_inv_pricing_management_env.py

import gym
import numpy as np
from scipy.stats import bernoulli, norm, truncnorm
from or_gym.utils import assign_env_config


class BaseInvManagementEnv(gym.Env):
    """
    Base Inventory Management Environment containing shared logic.
    """

    def __init__(self, *args, **kwargs):
        # Default parameters
        self.num_periods = 50
        self.init_inv = 15
        self.unit_price = 2.0
        self.unit_cost_ordering = 3.0
        self.demand_cost = 2.0
        self.holding_cost = 1.0
        self.supply_capacity = 15
        self.L = [0]
        self.num_stages = 2
        self.dist = 1
        self.dist_param = {'mu': [5, 5]}
        self.discount = 0.99
        self.seed_int = 0
        self.allowed_censoring = 1
        self.custom_mapping = {0: 2, 1: 4, 2: 6}
        self._max_rewards = 2000

        # Add environment configuration dictionary and keyword arguments
        assign_env_config(self, kwargs)

        # Check inputs
        self._validate_inputs()

        # Initialize
        self.reset()

    def _validate_inputs(self):
        assert self.init_inv >= 0, "Initial inventory cannot be negative."
        assert self.num_periods > 0, f"Number of periods must be positive. Given: {self.num_periods}"
        assert self.unit_price >= 0, "Unit price cannot be negative."
        assert self.unit_cost_ordering >= 0, "Ordering cost cannot be negative."
        assert self.demand_cost >= 0, "Unfulfilled demand cost cannot be negative."
        assert self.holding_cost >= 0, "Holding cost cannot be negative."
        assert self.supply_capacity > 0, "Supply capacity must be positive."
        assert 0 < self.discount <= 1, "Discount factor must be in the range (0, 1]."

    def reset(self):
        # To be implemented by subclasses
        pass

    def step(self, action):
        # To be implemented by subclasses
        pass

    def sample_action(self):
        x = self.action_space.sample().astype(float)
        x[1] = self.custom_mapping.get(int(x[1]), self.unit_price)
        return x

    # Include any additional shared methods
