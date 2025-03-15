# File: markovianess/experiments/baseline.py
"""
baseline.py
-----------
Running "clean" PPO experiments (no modifications to the environment) and collecting
Markovian-violation measures via PCMCI.

Main Classes/Functions:
-----------------------
BaselineExperiments:
    A class to:
    1) Train a PPO agent on the specified environment.
    2) Collect rollouts (trained or random).
    3) Run PCMCI to measure Markov violation scores.

    Usage Example:
        config = json.load(open("config.json"))
        env_config = config["environments"][0]
        baseline = BaselineExperiments(environment=env_config["name"], config=env_config)
        baseline.train()
        baseline.run_ci_tests_fisher(n_times=5, policy='random', label='random_rollouts')

"""

import argparse
import json
import os
import random
import sys
import time
import uuid

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

# Import from our package
from markovianess.ci.conditional_independence_test import (
    ConditionalIndependenceTest,
    get_markov_violation_score
)
# If you have a shared logger in markovianess/utils.py:
# from markovianess.utils import logging
# For now, we'll just use Python's built-in logging:
import logging

# Configure logging (if not already configured globally)
logging.basicConfig(level=logging.INFO)


class RewardTrackingCallback(BaseCallback):
    """
    Records total reward per episode for a single-environment vectorized scenario.
    Useful for plotting learning curves afterward.
    """
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.current_episode_reward = 0.0

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        done = self.locals["dones"][0]
        self.current_episode_reward += reward

        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0.0

        return True

    def get_rewards(self):
        return self.episode_rewards


class BaselineExperiments:
    """
    Baseline RL experiments for a single environment, single training run.
    Also runs PCMCI-based conditional independence tests on random or trained rollouts.

    Attributes
    ----------
    environment : str
        Name of the gymnasium environment (e.g. "CartPole-v1").
    config : dict
        Configuration for the environment (read from config.json).
    storage : dict
        Dictionary for storing observations or other artifacts in memory.
    results_path : str
        Base path for results => "results/<ENV>".
    csv_path : str
        Path => "results/<ENV>/csv".
    plots_path : str
        Path => "results/<ENV>/plots/baseline".
    models_path : str
        Path => "results/<ENV>/models".
    pcmci_path : str
        Path => "results/<ENV>/pcmci".
    """

    def __init__(self, environment, config):
        """
        :param environment: (str) Environment name, e.g., "CartPole-v1".
        :param config: (dict) Configuration dictionary from config.json (for the specific environment).
        """
        self.environment = environment
        self.config = config
        self.last_seed = None
        self.baseline_seeds = []

        # Training parameters
        self.time_steps = self.config["time_steps"]
        self.learning_rate = self.config.get("learning_rate", 3e-4)
        self.num_envs = self.config["n_envs"]

        # PCMCI config
        self.pcmci_config = self.config.get("pcmci", {})
        self.tau_min = self.pcmci_config.get("tau_min", 1)
        self.tau_max = self.pcmci_config.get("tau_max", 5)
        self.alpha_level = self.pcmci_config.get("significance_level", 0.05)
        self.pc_alpha = self.pcmci_config.get("pc_alpha", None)

        # For storing observations or models, if needed
        self.loaded_models = {}
        self.storage = {
            "observations": {}
        }

        # Callback for RL training to track rewards
        self.reward_callback = RewardTrackingCallback()

        # Directory structure
        self.results_path = os.path.join("results", self.environment)
        self.csv_path = os.path.join(self.results_path, "csv")
        self.plots_path = os.path.join(self.results_path, "plots", "baseline")
        self.models_path = os.path.join(self.results_path, "models")
        self.pcmci_path = os.path.join(self.results_path, "pcmci")

        # Create directories if needed
        os.makedirs(self.csv_path, exist_ok=True)
        os.makedirs(self.plots_path, exist_ok=True)
        os.makedirs(self.models_path, exist_ok=True)
        os.makedirs(self.pcmci_path, exist_ok=True)

    def gather_observations(self, env_id, num_steps=1000, policy='random'):
        """
        Gather observations from a single environment instance, returning a numpy array.

        Parameters
        ----------
        env_id : str
            Gym environment name.
        num_steps : int
            Number of timesteps to gather.
        policy : str
            'random' => random actions, 'trained' => use loaded PPO model, or fallback to random.

        Returns
        -------
        np.ndarray
            Array of shape (num_steps, obs_dim).
        """
        import gymnasium as gym
        env = gym.make(env_id)

        observations = []
        obs, _ = env.reset()
        for _ in range(num_steps):
            if policy == 'random':
                action = env.action_space.sample()
            elif policy == 'trained':
                model = self.loaded_models.get(env_id, None)
                if model is None:
                    model = self._load_model_from_disk(env_id)
                if model is None:
                    logging.warning(f"No trained model found for {env_id}. Using random actions.")
                    action = env.action_space.sample()
                else:
                    action, _ = model.predict(obs, deterministic=True)
            else:
                logging.warning(f"Unrecognized policy='{policy}'. Using random actions.")
                action = env.action_space.sample()

            obs, reward, done, truncated, info = env.step(action)
            observations.append(obs)

            if done or truncated:
                obs, _ = env.reset()

        env.close()
        observations = np.array(observations)

        # Optionally store in class-level storage
        key = f"{env_id}_{policy}"
        self.storage.setdefault('observations', {})[key] = observations
        logging.info(f"Gathered {len(observations)} observations for env={env_id}, policy={policy}")
        return observations

    def _load_model_from_disk(self, env_id):
        """
        Attempt to load a .zip model from disk (the first one found in self.models_path).
        If none found, return None.
        """
        candidate_files = [
            f for f in os.listdir(self.models_path) if f.endswith(".zip")
        ]
        if not candidate_files:
            return None

        model_path = os.path.join(self.models_path, candidate_files[0])
        logging.info(f"Loading model from {model_path} for env_id={env_id}")
        model = PPO.load(model_path)
        self.loaded_models[env_id] = model
        return model

    def fishers_method(self, pvals, epsilon=1e-15):
        """
        Combine multiple p-values using Fisher's method.

        Parameters
        ----------
        pvals : array-like
            1D array or list of p-values from independent tests.
        epsilon : float
            A small constant to prevent log(0).

        Returns
        -------
        float
            Combined p-value from Fisher's method.
        """
        pvals = np.array(pvals, dtype=float)
        pvals = np.clip(pvals, epsilon, 1 - epsilon)
        statistic = -2.0 * np.sum(np.log(pvals))
        df = 2 * len(pvals)
        combined_pval = 1.0 - chi2.cdf(statistic, df)
        return combined_pval

    def run_ci_tests_fisher(self, n_times=5, policy='random', label=None):
        """
        Repeatedly gather observations and run PCMCI, combining the partial-corr & p-matrices
        across runs via:
          - Average val_matrix
          - Fisher's method for p_matrix
        Then logs the Markov violation score.

        Parameters
        ----------
        n_times : int
            How many rollouts to gather for averaging/fusing.
        policy : str
            'random' or 'trained'.
        label : str
            Label for file naming (e.g. 'random' or 'trained').

        Returns
        -------
        None
        """
        if label is None:
            label = policy

        run_dir = os.path.join(self.pcmci_path, label)
        os.makedirs(run_dir, exist_ok=True)

        cit = ConditionalIndependenceTest()
        val_matrices = []
        p_matrices = []

        # Gather multiple runs
        for i in range(n_times):
            obs = self.gather_observations(
                env_id=self.environment,
                num_steps=1000,
                policy=policy
            )
            # Run PCMCI
            results_dict = cit.run_pcmci(
                observations=obs,
                tau_max=self.tau_max,
                alpha_level=self.alpha_level
            )
            p_matrix = results_dict["p_matrix"]
            val_matrix = results_dict["val_matrix"]

            val_matrices.append(val_matrix)
            p_matrices.append(p_matrix)

        val_matrices = np.stack(val_matrices, axis=0)  # shape=(n_times, N, N, L)
        p_matrices = np.stack(p_matrices, axis=0)

        # 1) Average val_matrix
        avg_val_matrix = np.mean(val_matrices, axis=0)

        # 2) Combine p_matrices using Fisher’s method element-wise
        n_runs, N, _, L = p_matrices.shape
        combined_p_matrix = np.zeros((N, N, L), dtype=float)
        for i in range(N):
            for j in range(N):
                for k in range(L):
                    pvals_for_link = p_matrices[:, i, j, k]
                    combined_p_matrix[i, j, k] = self.fishers_method(pvals_for_link)

        # Print "significant links" using a dummy PCMCI object
        import tigramite.data_processing as pp
        from tigramite.independence_tests.parcorr import ParCorr
        from tigramite.pcmci import PCMCI

        dummy_df = pp.DataFrame(data=obs)
        dummy_pcmci = PCMCI(dataframe=dummy_df, cond_ind_test=ParCorr())

        significant_txt_file = os.path.join(run_dir, f"fisher_significant_links_{label}.txt")
        old_stdout = sys.stdout
        try:
            with open(significant_txt_file, "w") as f:
                sys.stdout = f
                dummy_pcmci.print_significant_links(
                    p_matrix=combined_p_matrix,
                    val_matrix=avg_val_matrix,
                    alpha_level=self.alpha_level,
                )
        finally:
            sys.stdout = old_stdout

        # Compute Markov violation score
        markov_score = get_markov_violation_score(
            p_matrix=combined_p_matrix,
            val_matrix=avg_val_matrix,
            alpha_level=self.alpha_level
        )
        logging.info(f"[{label}] Markov violation score: {markov_score:.4f}")

        # Save combined results
        combined_npz_file = os.path.join(run_dir, f"fisher_val_and_p_matrices_{label}.npz")
        np.savez_compressed(
            combined_npz_file,
            val_matrix=avg_val_matrix,
            p_matrix=combined_p_matrix,
            markovian_violation_score=markov_score
        )
        logging.info(f"[{label}] Combined PCMCI results saved to {combined_npz_file}")

        # Also store Markov violation score in a text file
        markov_txt_file = os.path.join(run_dir, f"markovian_violation_score_{label}.txt")
        with open(markov_txt_file, "w") as f:
            f.write(f"{markov_score:.6f}\n")
        logging.info(f"[{label}] Score saved to {markov_txt_file}")

    def plot_learning_curve(self, df, output_file, smoothing_window=10):
        """
        Plots the learning curve (episode vs. total reward) with smoothing.

        df must have columns: ["Episode", "TotalReward"].
        """
        df['SmoothedReward'] = df['TotalReward'].rolling(
            window=smoothing_window, min_periods=1).mean()

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x="Episode", y="SmoothedReward",
                     label="Smoothed Reward", color="blue")
        sns.lineplot(data=df, x="Episode", y="TotalReward",
                     label="Raw Reward", color="red", alpha=0.3)

        plt.title(f"Learning Curve (with smoothing) for {self.environment}")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

    def train(self):
        """
        Perform a single training run (PPO) for this environment.
        Saves the model, learning curve CSV, and plot in the standard locations.
        """
        self.last_seed = random.randint(0, 1000)
        self.baseline_seeds.append(self.last_seed)
        env_uid = str(uuid.uuid4())

        logging.info(f"Creating env {self.environment}, UID={env_uid}, seed={self.last_seed}")
        vec_env = make_vec_env(self.environment, n_envs=self.num_envs, seed=self.last_seed)

        # Train the RL model

        model = PPO("MlpPolicy", vec_env, verbose=0, learning_rate=self.learning_rate)
        model.learn(total_timesteps=self.time_steps, callback=self.reward_callback)

        # Save model
        model_path = os.path.join(self.models_path, f"{env_uid}.zip")
        model.save(model_path)
        logging.info(f"Model saved to: {model_path}")

        # Extract final episode rewards
        episode_rewards = self.reward_callback.get_rewards()
        results = []
        for i, reward_val in enumerate(episode_rewards, start=1):
            results.append({
                "Environment": self.environment,
                "UID": env_uid,
                "Seed": self.last_seed,
                "Episode": i,
                "TotalReward": reward_val
            })

        # Save to CSV
        df = pd.DataFrame(results)
        csv_file_name = "baseline_learning_curve.csv"
        csv_full_path = os.path.join(self.csv_path, csv_file_name)
        df.to_csv(csv_full_path, index=False)
        logging.info(f"Saved learning curve CSV => {csv_full_path}")

        # Plot smoothed vs raw
        plot_file = os.path.join(
            self.plots_path, f"{self.environment}_{env_uid}_learning_curve.png"
        )
        self.plot_learning_curve(df, plot_file, smoothing_window=10)
        logging.info(f"Saved learning curve plot => {plot_file}")

    def get_used_baseline_seeds(self):
        """Return the list of seeds used for baseline runs."""
        return self.baseline_seeds

    def run_experiments(self):
        """
        Orchestrates the pipeline:
          1) PCMCI with random policy (pre-training)
          2) Train RL
          3) PCMCI with trained policy (post-training)
        """
        # 1) PCMCI with random
        self.run_ci_tests_fisher(n_times=3, policy='random', label='random')

        # 2) Train RL
        self.train()

        # 3) PCMCI with trained policy
        self.run_ci_tests_fisher(n_times=3, policy='trained', label='trained')

        return self.last_seed


def run(config_path="config.json", env_name=None):
    """
    Reads config from config_path, picks the environment (env_name),
    runs random-policy CI tests, trains a PPO model, and then runs
    trained-policy CI tests, logging results. Returns the final seed used.

    Typically called from main.py.
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    # Get environment config from list
    if env_name is None:
        environment_config = config["environments"][0]
    else:
        environment_config = None
        for env_item in config["environments"]:
            if env_item["name"] == env_name:
                environment_config = env_item
                break
        if environment_config is None:
            raise ValueError(f"Could not find environment '{env_name}' in {config_path}")

    start_time = time.perf_counter()
    baseline = BaselineExperiments(environment_config["name"], environment_config)
    used_seed = baseline.run_experiments()
    end_time = time.perf_counter()

    logging.info(f"Baseline for {env_name} finished! Time taken: {end_time - start_time:.2f} seconds")
    return used_seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline RL Experiments")
    parser.add_argument("--config_path", type=str, default="config.json",
                        help="Path to the config JSON file.")
    parser.add_argument("--env", type=str, default=None,
                        help="Name of the environment to use from config.json.")
    args = parser.parse_args()

    run(config_path=args.config_path, env_name=args.env)