# data/generate_datasets.py

import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from envs.inv_pricing_management_env_ppo import InvPricingManagementMasterEnvPPO


def data_generator(policy, episodes, env):
    """
    Generates data by running the environment with the given policy.

    Parameters:
        policy: The policy to use for action selection.
        episodes (int): Number of episodes to generate data for.
        env (gym.Env): The environment instance.

    Returns:
        pd.DataFrame: Generated dataset.
    """
    dic_data = {}
    for i in range(episodes):
        env.reset()
        data = np.empty((env.num_periods, 15))
        for j in range(env.num_periods):
            if policy is None:
                action = env.sample_action()
            else:
                action, _ = policy.predict(env.state)
                state_values = np.concatenate(list(env.state.values()))
            state, _, done, _ = env.step(action.squeeze())
            next_state_values = np.concatenate(list(state.values()))
            if j != 0:
                state_values[2] = env.Sales[j - 1]
            action_ = action.squeeze()
            action_[1] = env.custom_mapping.get(action_[1], action_[1])
            data[j] = np.concatenate([
                state_values,
                action_,
                np.array([env.P_censored[j]]),
                np.array([env.P[j]]),
                np.array([env.D[j]]),
                next_state_values
            ]).reshape(1, -1)
            if done:
                break
        columns = [
            "Inv_0", "Censor_0", "Sales_0", "State_0", "IRate_0",
            "Ordering_0", "Pricing_0", "Observed Profit", "Real Profit",
            "Demand", "Inv_1", "Censor_1", "Sales_1", "State_1", "IRate_1"
        ]
        dic_data[i] = pd.DataFrame(data, columns=columns)
    return pd.concat([dic_data[i] for i in range(episodes)], axis=0).reset_index(drop=True)



def main():
    env = InvPricingManagementMasterEnvPPO(allowed_censoring=3)
    model = PPO.load("models/ppo_multi_input_policy", env=env)

    for episodes in range(10, 60, 10):
        for run_index in range(50):
            data = data_generator(model, episodes, env)
            data.to_excel(f'data/data_ind_{run_index}_episode_{episodes}.xlsx', index=False)

if __name__ == "__main__":
    main()
