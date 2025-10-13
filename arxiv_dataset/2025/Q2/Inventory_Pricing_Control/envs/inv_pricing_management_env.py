# envs/inv_pricing_management_env.py

from .base_inv_pricing_management_env import BaseInvManagementEnv
import numpy as np
from gym import spaces
from typing import Tuple, Dict
from scipy.stats import bernoulli, norm, truncnorm
from sksurv.nonparametric import kaplan_meier_estimator
import pandas as pd

class InvPricingManagementMasterEnv(BaseInvManagementEnv):
    """
    Original Inventory Management Environment for Reinforcement Learning.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Define action and observation spaces
        self.action_space = spaces.MultiDiscrete([16, 3])  # Ordering quantity (0-15), Pricing level (0-2)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, -np.inf]),
            high=np.array([np.inf, 1, np.inf, 1, np.inf]),
            dtype=np.float32
        )

        # Initialize environment
        self.reset()

    def reset(self) -> np.ndarray:
        # Initialize variables
        self.period = 0
        self.I = np.zeros(self.num_periods + 1)
        self.I[0] = self.init_inv
        self.state_of_economy = np.zeros(self.num_periods + 1)
        self.interest_rate = np.zeros(self.num_periods + 1)
        self.ordering_log = np.zeros(self.num_periods)
        self.price_log = np.zeros(self.num_periods)
        self.D = np.zeros(self.num_periods)
        self.Sales = np.zeros(self.num_periods)
        self.LS = np.zeros(self.num_periods)
        self.P = np.zeros(self.num_periods)
        self.P_censored = np.zeros(self.num_periods)
        self.Indicator_censor = np.zeros(self.num_periods)
        self.state_log = [np.nan] * (self.num_periods + 1)

        # Set initial state
        self._update_state()
        return self.state

    def _update_state(self):
        t = self.period
        state = np.zeros(5)

        # Update the state of the economy
        if t == 0 or self.state_of_economy[t - 1] == 1:
            self.state_of_economy[t] = bernoulli.rvs(0.8)
        else:
            self.state_of_economy[t] = bernoulli.rvs(0.3)

        # Update the interest rate
        if t == 0:
            self.interest_rate[t] = norm.rvs(0, 0.0001)
        else:
            self.interest_rate[t] = 0.25 * self.interest_rate[t - 1] + norm.rvs(0, 0.0001)

        # Inventory level
        state[0] = self.I[t]

        # Censoring indicator
        if t == 0:
            state[1] = 1
        else:
            state[1] = self.Indicator_censor[t - 1]

        # Previous sales or average demand
        if t == 0:
            state[2] = 5  # Historical average
        else:
            state[2] = self.Sales[t - 1] if self.Indicator_censor[t - 1] == 0 else self.D[t - 1]

        # State of the economy
        state[3] = self.state_of_economy[t]

        # Interest rate
        state[4] = self.interest_rate[t]

        self.state = state.copy()
        self.state_log[t] = state

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        # Transform the pricing action using the custom mapping
        action = action.astype(int)
        action[1] = self.custom_mapping.get(action[1], self.unit_price)

        n = self.period
        order_qty = action[0]
        price = action[1]

        # Log actions
        self.ordering_log[n] = order_qty
        self.price_log[n] = price

        # Current inventory after ordering
        current_inventory = self.I[n] + order_qty

        # Update demand parameters
        self.dist_param['mu'] = [3.5, 1]

        # Calculate demand
        price_elasticity = -1.5
        economic_impact = 5
        interest_rate_impact = -5

        adjusted_price_effect = price_elasticity * price
        adjusted_economy_effect = economic_impact * self.state_of_economy[n]
        adjusted_interest_effect = interest_rate_impact * self.interest_rate[n]
        random_disturbance = truncnorm.rvs(0, 10, self.dist_param['mu'][0], self.dist_param['mu'][1])

        if n == 0:
            demand = max(round(
                5 + adjusted_price_effect + adjusted_economy_effect + adjusted_interest_effect + random_disturbance
            ), 0)
        else:
            lag_demand = 1.2 * self.D[n - 1]
            demand = max(round(
                lag_demand + adjusted_price_effect + adjusted_economy_effect + adjusted_interest_effect + random_disturbance
            ), 0)

        # Censoring logic
        if demand > current_inventory and n >= self.allowed_censoring and \
           (self.Indicator_censor[n - self.allowed_censoring:n] == 0).all():
            demand = current_inventory

        self.D[n] = demand

        # Sales and inventory updates
        sales = min(current_inventory, demand)
        self.Sales[n] = sales
        self.I[n + 1] = current_inventory - sales
        lost_sales = max(demand - sales, 0)
        self.LS[n] = lost_sales

        # Profit calculations
        profit = (
            price * sales
            - self.unit_cost_ordering * min(order_qty, self.supply_capacity)
            - self.demand_cost * max(demand - current_inventory, 0)
            - self.holding_cost * max(current_inventory - demand, 0)
        )

        if demand > current_inventory:
            observed_profit = (
                price * sales
                - self.unit_cost_ordering * min(order_qty, self.supply_capacity)
                - self.holding_cost * max(current_inventory - demand, 0)
            )
            self.Indicator_censor[n] = 0
        else:
            observed_profit = profit
            self.Indicator_censor[n] = 1

        self.P[n] = profit
        self.P_censored[n] = observed_profit

        # Update period and state
        self.period += 1
        self._update_state()

        # Check if the simulation is done
        done = self.period >= self.num_periods

        # Additional information
        info = {
            'profit': profit,
            'observed_profit': observed_profit,
            'demand': demand,
            'sales': sales,
            'lost_sales': lost_sales,
            'inventory': self.I[n + 1]
        }

        return self.state, observed_profit, done, info

    def sample_action(self) -> np.ndarray:
        """Generates a random action by sampling from the action space."""
        return super().sample_action()

    # Include any additional methods like impute_reward, data_generator, etc., as needed.

    def impute_reward(self, df):
        """
        Impute rewards for censored data using Kaplan-Meier estimates.
        """
        df_1 = df.copy()
        df_1["Estimated Demand"] = df_1["Demand"]
        df_1["Imputed Profit"] = df_1["Observed Profit"]
        indexes = df_1.loc[df_1["Censor_1"] == 0].index.to_list()
        df_1.loc[indexes, "Estimated Demand"] = df_1.loc[indexes].apply(
            lambda row: self.estimate_demand(row["Sales_1"], df), axis=1)
        df_1.loc[indexes, "Imputed Profit"] = df_1.loc[indexes].apply(
            lambda row: row["Observed Profit"] - 2 * (row["Estimated Demand"] - row["Sales_1"]), axis=1)
        return df_1

    # Add methods like estimate_demand, prepare_data, calculate_survival, etc., as needed.
    def prepare_data(self,df):
        df = df.copy()
        df['Censor_1'] = df['Censor_1'].apply(lambda x: True if x == 1 else False)
        return df

    def calculate_survival(self,df, x):
        event_occurred = df["Censor_1"]
        durations = df["Sales_1"]
        level, survival_prob = kaplan_meier_estimator(event_occurred, durations)
        survival_df = pd.DataFrame({'level': level, 'survival_prob': survival_prob})
        con_survival = survival_df.loc[survival_df['level'] == x, 'survival_prob'].values[0]
        integ_survival = survival_df.loc[survival_df['level'] >= x, 'survival_prob'].sum()
        return con_survival, integ_survival

    def estimate_demand(self,x, df):
        prepared_df = self.prepare_data(df[['Censor_1', 'Sales_1']])
        con_survival, integ_survival = self.calculate_survival(prepared_df, int(x))
        estimated_demand = int(x) + (1 / con_survival) * integ_survival
        return estimated_demand

