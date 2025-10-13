# envs/inv_pricing_management_env_ppo.py

from .base_inv_pricing_management_env import BaseInvManagementEnv
import numpy as np
from gym import spaces
from scipy.stats import bernoulli, norm, truncnorm

class InvPricingManagementMasterEnvPPO(BaseInvManagementEnv):
    """
    Alternative Inventory Management Environment for PPO Policy.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Define action and observation spaces specific to this environment
        self.action_space = spaces.MultiDiscrete([16, 3])  # Ordering quantity (0-15), Pricing level (0-2)
        self.observation_space = spaces.Dict({
            "inventory": spaces.Discrete(300),        # 0 to 299
            "censor": spaces.Discrete(2),             # 0 or 1
            "sales_demand": spaces.Discrete(300),     # 0 to 299
            "state_of_econ": spaces.Discrete(2),      # 0 or 1
            "interest": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        })

        # Initialize environment
        self.reset()

    def reset(self):
        # Initialize variables
        periods = self.num_periods
        self.period = 0
        self.I = np.zeros(periods + 1)
        self.I[0] = self.init_inv
        self.state_of_economy = np.zeros(periods + 1)
        self.interest_rate = np.zeros(periods + 1)
        self.R = np.zeros(periods)
        self.D = np.zeros(periods)
        self.Sales = np.zeros(periods)
        self.LS = np.zeros(periods)
        self.P = np.zeros(periods)
        self.P_censored = np.zeros(periods)
        self.Indicator_censor = np.zeros(periods, dtype=int)
        self.state_log = [np.nan for _ in range(periods + 1)]
        self.ordering_log = np.zeros(periods)
        self.price_log = np.zeros(periods)
        self.state = {}

        # Set initial state
        self._update_state()

        return self.state

    def _update_state(self):
        t = self.period

        if t == 0 or self.state_of_economy[t - 1] == 1:
            self.state_of_economy[t] = int(bernoulli.rvs(0.8))
        else:
            self.state_of_economy[t] = int(bernoulli.rvs(0.3))

        if t == 0:
            self.interest_rate[t] = norm.rvs(0, 0.0001)
        else:
            self.interest_rate[t] = 0.25 * self.interest_rate[t - 1] + norm.rvs(0, 0.0001)

        if t == 0:
            self.state['inventory'] = np.array([self.init_inv], dtype=int)
        else:
            self.state['inventory'] = np.array([self.I[t]], dtype=int)

        if t == 0:
            self.state['censor'] = np.array([1], dtype=int)
        else:
            self.state['censor'] = np.array([int(self.Indicator_censor[t - 1])], dtype=int)

        if t == 0 or self.Indicator_censor[t - 1] == 0:
            if t == 0:
                self.state['sales_demand'] = np.array([5], dtype=int)
            else:
                self.state['sales_demand'] = np.array([self.D[t - 1]], dtype=int)
        else:
            self.state['sales_demand'] = np.array([self.D[t - 1]], dtype=int)

        self.state['state_of_econ'] = np.array([self.state_of_economy[t]], dtype=int)
        self.state['interest'] = np.array([self.interest_rate[t]], dtype=np.float32)
        self.state_log[t] = self.state

    def step(self, action):
        # Transform the pricing action using the custom mapping
        price_action = action[1]
        price = self.custom_mapping.get(price_action, self.unit_price)

        n = self.period
        order = max(action[0], 0)
        self.ordering_log[n] = order

        CurrentInventory = self.I[n] + order

        self.dist_param['mu'] = [3.5, 1]

        price_elasticity = -1.5
        economic_impact = 5
        interest_rate_impact = -5

        adjusted_price_effect = price_elasticity * price
        adjusted_economy_effect = economic_impact * self.state_of_economy[n]
        adjusted_interest_effect = interest_rate_impact * self.interest_rate[n]
        lag_demand = 1.2 * self.D[n - 1]
        random_disturbance = truncnorm.rvs(0, 10, self.dist_param['mu'][0], self.dist_param['mu'][1])

        if n == 0:
            Demand = max(round(
                5 + adjusted_price_effect + adjusted_economy_effect + adjusted_interest_effect + random_disturbance
            ), 0)
        else:
            Demand = max(round(
                lag_demand + adjusted_price_effect + adjusted_economy_effect + adjusted_interest_effect + random_disturbance
            ), 0)

        # Censoring logic
        if Demand > CurrentInventory and n >= self.allowed_censoring and \
           (self.Indicator_censor[n - self.allowed_censoring:n] == 0).all():
            Demand = CurrentInventory

        self.D[n] = Demand
        Sales = min(CurrentInventory, Demand)
        self.Sales[n] = Sales
        self.I[n + 1] = CurrentInventory - Sales
        LostSales = max(Demand - Sales, 0)
        self.LS[n] = LostSales

        # Calculate true profit
        Profit = (
            price * Sales
            - self.unit_cost_ordering * min(order, self.supply_capacity)
            - self.demand_cost * max(Demand - CurrentInventory, 0)
            - self.holding_cost * max(CurrentInventory - Demand, 0)
        )

        # Calculate observed profit
        if Demand > CurrentInventory:
            Profit_censored = (
                price * Sales
                - self.unit_cost_ordering * min(order, self.supply_capacity)
                - self.holding_cost * max(CurrentInventory - Demand, 0)
            )
            self.Indicator_censor[n] = 0
        else:
            Profit_censored = Profit
            self.Indicator_censor[n] = 1

        self.P[n] = Profit
        self.P_censored[n] = Profit_censored

        # Update period and state
        self.period += 1
        self._update_state()

        # Determine if simulation should terminate
        done = self.period >= self.num_periods

        return self.state, Profit_censored, done, {}

    def sample_action(self):
        return super().sample_action()
