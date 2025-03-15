# File: noised_gaussian_action.py
"""
noised_gaussian_action.py
-------------------------
Runs experiments by adding Gaussian-like noise to each action dimension
for both continuous and discrete action spaces. In discrete spaces, we
convert variance into a "randomization probability" for overriding the
chosen action.

Overview of Steps
-----------------
1. Check whether the environment's action space is continuous (Box) or discrete.
2. For each dimension in the action space:
   a) If continuous, add Gaussian noise to that dimension of the action.
   b) If discrete, interpret 'variance' as controlling a random override probability.
      - If dimension_to_noisify is i, and the chosen action = i, randomly override it
        with probability p. If dimension_to_noisify is None, override any chosen action
        with probability p.
3. Train a PPO agent in this "noised" environment, record episode rewards.
4. Optionally run multiple PCMCI analyses with repeated rollouts, combining
   partial correlations & p-values via Fisher's method, and compute Markov scores.
5. Collect Markov violation scores and store them in CSV. Produce the same set of
   plots as in noised_gaussian.py but with "action" dimension labeling.

Usage
-----
Call the main function `run_noised_gaussian_action(config_path, env_name, baseline_seed)`
or run from the command line (see `main()` below).
"""

import argparse
import json
import os
import random
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

# From our own package:
from markovianess.ci.conditional_independence_test import (
    ConditionalIndependenceTest,
    get_markov_violation_score
)

import logging
logging.basicConfig(level=logging.INFO)


###############################################################################
# Reward Tracking Callback
###############################################################################
class RewardTrackingCallback(BaseCallback):
    """
    Callback that records total reward per episode in a single-env scenario.
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


###############################################################################
# NoisyGaussianActionWrapper
###############################################################################
class NoisyGaussianActionWrapper(gym.Wrapper):
    """
    A single wrapper that handles both continuous (Box) and discrete actions.

    - If the action space is continuous:
        We add Gaussian noise to one (or all) dimensions of the action vector.

    - If the action space is discrete:
        We interpret 'variance' as controlling a random override probability p:
          p = 1 - exp(-max(0, variance)).
        If dimension_to_noisify is i, we only override the action if the chosen
        action == i. If dimension_to_noisify is None, we can override any chosen
        action with probability p.

    dimension_to_noisify: int or None
      - For continuous: which dimension gets the noise? (None => all dims)
      - For discrete: which action index is "noised"? (None => all possible actions)
    """

    def __init__(self, env, dimension_to_noisify=None, mean=0.0, variance=0.01):
        super().__init__(env)
        self.dim_to_noisify = dimension_to_noisify
        self.mean = mean
        self.variance = variance
        self.std = np.sqrt(max(0.0, variance))

        self.action_space = env.action_space
        if isinstance(self.action_space, gym.spaces.Box):
            # Continuous environment
            self.is_discrete = False
            self.action_dim = self.action_space.shape[0]
            if (dimension_to_noisify is not None) and (
                dimension_to_noisify < 0 or dimension_to_noisify >= self.action_dim
            ):
                raise ValueError(
                    f"Invalid dimension_to_noisify={dimension_to_noisify} "
                    f"for action_dim={self.action_dim} in continuous space."
                )
        elif isinstance(self.action_space, gym.spaces.Discrete):
            # Discrete environment
            self.is_discrete = True
            # There's effectively a single 'dimension', but multiple possible actions
            # We'll interpret dimension_to_noisify as "which action index to randomize"
            # if dimension_to_noisify is not None, must be in [0, n-1].
            self.n_actions = self.action_space.n
            if (dimension_to_noisify is not None) and (
                dimension_to_noisify < 0 or dimension_to_noisify >= self.n_actions
            ):
                raise ValueError(
                    f"Invalid dimension_to_noisify={dimension_to_noisify} "
                    f"for n_actions={self.n_actions} in discrete space."
                )

            # Convert variance -> random override probability p
            # You can choose a different mapping if desired:
            self.override_prob = 1.0 - np.exp(-max(0.0, variance))
        else:
            raise NotImplementedError(
                "NoisyGaussianActionWrapper only supports Box or Discrete actions."
            )

    def step(self, action):
        """
        1) Possibly modify `action` by adding noise (continuous) or randomizing it (discrete).
        2) Step the environment with the modified action.
        """
        if not self.is_discrete:
            # Continuous case
            noisy_action = self._add_continuous_noise(action)
            # Step
            return self.env.step(noisy_action)
        else:
            # Discrete case
            # 'action' is an integer or an array of shape (1,) for a single discrete env
            if isinstance(action, np.ndarray) and action.shape == (1,):
                chosen_act = action[0]
            else:
                chosen_act = int(action)

            final_act = self._maybe_randomize_discrete(chosen_act)
            return self.env.step(final_act)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def _add_continuous_noise(self, action):
        """Add Gaussian noise to the chosen action vector in the continuous case."""
        noisy_action = np.array(action, copy=True, dtype=np.float32)

        if self.dim_to_noisify is None:
            # Add noise to all dims
            noise = np.random.normal(self.mean, self.std, size=noisy_action.shape)
            noisy_action += noise
        else:
            # Only to the chosen dimension
            noise = np.random.normal(self.mean, self.std)
            noisy_action[self.dim_to_noisify] += noise

        # Clip to respect the env's action space
        low, high = self.action_space.low, self.action_space.high
        noisy_action = np.clip(noisy_action, low, high)
        return noisy_action

    def _maybe_randomize_discrete(self, chosen_act):
        """With probability p, override the chosen action in the discrete case."""
        if self.dim_to_noisify is not None:
            # Only randomize if chosen_act == dim_to_noisify
            if chosen_act == self.dim_to_noisify:
                if np.random.rand() < self.override_prob:
                    # choose a random action among [0..(n_actions-1)], but
                    # optionally exclude the original action if you want
                    # This is a design choice; we'll exclude the same action for clarity
                    action_candidates = list(range(self.n_actions))
                    action_candidates.remove(chosen_act)
                    return np.random.choice(action_candidates)
                else:
                    return chosen_act
            else:
                return chosen_act
        else:
            # dimension_to_noisify = None => randomize any chosen action with probability p
            if np.random.rand() < self.override_prob:
                # random action from [0..n-1] (could be same or different)
                return np.random.randint(self.n_actions)
            else:
                return chosen_act


###############################################################################
# Utility functions for plotting
###############################################################################
def _smooth_reward_curve(episodes, rewards, window=10):
    """
    Rolling-mean smoothing of the reward data.
    """
    if len(rewards) < window:
        return episodes, rewards
    cumsum = np.cumsum(np.insert(rewards, 0, 0))
    smoothed = (cumsum[window:] - cumsum[:-window]) / float(window)
    return episodes[window - 1:], smoothed


def plot_noised_learning_curves_all_dims(
    df_rewards: pd.DataFrame,
    baseline_csv: str = None,
    output_dir: str = ".",
    smooth_window: int = 10
):
    """
    For each noise variance, produce a single figure with multiple lines
    (one line per 'ActionDim') if continuous, or 'ActionDim' meaning the
    discrete 'dimension index' if we interpret it that way.

    If baseline_csv is available, overlay the baseline in black.
    """
    os.makedirs(output_dir, exist_ok=True)
    if df_rewards.empty:
        logging.info("[Noised Action Gaussian] No data to plot.")
        return

    env_name = df_rewards["Environment"].iloc[0]

    # Attempt to load baseline
    baseline_df = None
    if baseline_csv and os.path.isfile(baseline_csv):
        bdf = pd.read_csv(baseline_csv)
        bdf = bdf[bdf["Environment"] == env_name]
        if not bdf.empty:
            baseline_df = bdf

    # Group by noise variance
    for var_val, df_var in df_rewards.groupby("NoiseVariance"):
        plt.figure(figsize=(8, 5))

        # Group by dimension => label lines with dimension index
        for dim_id, df_dim in df_var.groupby("ActionDim"):
            df_dim = df_dim.sort_values("Episode")
            episodes = df_dim["Episode"].values
            rewards = df_dim["Reward"].values
            x_smooth, y_smooth = _smooth_reward_curve(episodes, rewards, window=smooth_window)
            label_str = f"ActionDim={dim_id}"
            plt.plot(x_smooth, y_smooth, label=label_str, linewidth=2)

        # Overlay baseline if available
        if baseline_df is not None:
            baseline_sorted = baseline_df.sort_values("Episode")
            b_eps = baseline_sorted["Episode"].values
            b_rew = baseline_sorted["TotalReward"].values
            bx_smooth, by_smooth = _smooth_reward_curve(b_eps, b_rew, window=smooth_window)
            plt.plot(bx_smooth, by_smooth, color="black", linewidth=3, label="Baseline")

        plt.title(f"{env_name}, Action Noise Var={var_val} (All Action Dims + Baseline)")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True)

        out_fname = f"{env_name}_action_var_{var_val}_all_dims_overlaid.png"
        out_path = os.path.join(output_dir, out_fname)
        plt.savefig(out_path, dpi=150)
        plt.close()
        logging.info(f"[Noised Action Gaussian] Saved noised multi-action-dim plot => {out_path}")


def plot_rewards_vs_noise(df_markov, output_dir="."):
    """
    X-axis=NoiseVariance, Y-axis=MeanFinalReward. A line per "ActionDim".
    In discrete envs, 'ActionDim' might correspond to the integer action index
    we specifically target for randomization.
    """
    os.makedirs(output_dir, exist_ok=True)
    if df_markov.empty:
        return

    envs = df_markov["Environment"].unique()
    for env_name in envs:
        env_df = df_markov[df_markov["Environment"] == env_name]
        if env_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(6, 5))
        for dim_id, gdf in env_df.groupby("ActionDim"):
            gdf_sorted = gdf.sort_values("NoiseVariance")
            ax.plot(
                gdf_sorted["NoiseVariance"],
                gdf_sorted["MeanFinalReward"],
                marker="o",
                label=f"action_dim={dim_id}"
            )
        ax.set_title(f"Rewards vs Noise (Actions) - {env_name}")
        ax.set_xlabel("NoiseVariance")
        ax.set_ylabel("MeanFinalReward")
        ax.legend()
        ax.grid(True)
        outpath = os.path.join(output_dir, f"{env_name}_action_rewards_vs_noise.png")
        fig.savefig(outpath, dpi=150)
        plt.close(fig)


def plot_rewards_vs_markov(df_markov, output_dir="."):
    """
    Plot: X-axis=MarkovScore, Y-axis=MeanFinalReward. A line per "ActionDim".
    """
    os.makedirs(output_dir, exist_ok=True)
    if df_markov.empty:
        return

    envs = df_markov["Environment"].unique()
    for env_name in envs:
        env_df = df_markov[df_markov["Environment"] == env_name]
        if env_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(6, 5))
        for dim_id, gdf in env_df.groupby("ActionDim"):
            gdf_sorted = gdf.sort_values("MarkovScore")
            ax.plot(
                gdf_sorted["MarkovScore"],
                gdf_sorted["MeanFinalReward"],
                marker="o",
                label=f"action_dim={dim_id}"
            )
        ax.set_title(f"Rewards vs MarkovScore (Actions) - {env_name}")
        ax.set_xlabel("MarkovScore")
        ax.set_ylabel("MeanFinalReward")
        ax.legend()
        ax.grid(True)
        outpath = os.path.join(output_dir, f"{env_name}_action_rewards_vs_markov.png")
        fig.savefig(outpath, dpi=150)
        plt.close(fig)


def plot_noise_vs_markov_corr(df_markov, output_dir="."):
    """
    Plot: X-axis=NoiseVariance, Y-axis=MarkovScore, with lines for each 'ActionDim'.
    Also compute correlation across all points, if we want, just as an example.
    """
    os.makedirs(output_dir, exist_ok=True)
    if df_markov.empty:
        return

    envs = df_markov["Environment"].unique()
    for env_name in envs:
        env_df = df_markov[df_markov["Environment"] == env_name]
        if env_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(6,5))
        for dim_id, gdf in env_df.groupby("ActionDim"):
            gdf_sorted = gdf.sort_values("NoiseVariance")
            ax.plot(
                gdf_sorted["NoiseVariance"],
                gdf_sorted["MarkovScore"],
                marker="o",
                label=f"action_dim={dim_id}"
            )

        if len(env_df) >= 2:
            corr_all = env_df[["NoiseVariance","MarkovScore"]].corr().iloc[0,1]
        else:
            corr_all = float('nan')

        ax.set_title(f"{env_name} - Action Noise vs MarkovScore (corr={corr_all:.2f})")
        ax.set_xlabel("NoiseVariance")
        ax.set_ylabel("MarkovScore")
        ax.legend()
        ax.grid(True)
        outpath = os.path.join(output_dir, f"{env_name}_action_noise_vs_markov_corr.png")
        fig.savefig(outpath, dpi=150)
        plt.close(fig)


###############################################################################
# Main Class for Action-Noised Gaussian Experiments
###############################################################################
class NoisedGaussianActionExperiments:
    """
    Class for systematically injecting Gaussian-like noise in each *action* dimension
    (continuous or discrete), training a PPO model, and measuring Markov violation
    using PCMCI.

    Steps:
    ------
    1) Identify the action space. If Box, we have 'action_dim = shape[0]'. If Discrete,
       we have 'n_actions'. We'll loop over each dimension or action index.
    2) For each dimension, for each noise param (mean,var):
       - Create a "noised" environment that uses NoisyGaussianActionWrapper.
       - Train PPO => record reward per episode.
       - Gather multiple rollouts => run PCMCI => combine partial correlations & p-values => Markov Score.
    3) Save final metrics to CSV, produce plots, etc.
    """

    def __init__(self, config, root_path="."):
        """
        config : Dict loaded from config.json, must contain a "noise_strategies" dict
                 with "gaussian" info, e.g. { "gaussian": [ {"mean":0.0, "variance":0.01}, ... ] }
        root_path : Base directory path where results/<ENV> will be created.
        """
        self.config = config
        self.learning_rate = self.config.get("learning_rate", 3e-4)
        self.root_path = root_path
        self.reward_records = []
        self.markov_records = []

    def fishers_method(self, pvals, epsilon=1e-15):
        """
        Combine p-values via Fisher's method.
        """
        pvals = np.array(pvals, dtype=float)
        pvals = np.clip(pvals, epsilon, 1 - epsilon)
        statistic = -2.0 * np.sum(np.log(pvals))
        df = 2 * len(pvals)
        return 1.0 - chi2.cdf(statistic, df)

    def gather_and_run_pcmci(
        self, model, env_name, dim_id, mean, variance, steps=2000, seed=None
    ):
        """
        Creates an action-noised environment, collects steps with the given model,
        then runs PCMCI and returns (val_matrix, p_matrix).
        """
        test_env = self.make_noisy_env(
            env_name=env_name,
            dimension_to_noisify=dim_id,
            mean=mean,
            variance=variance,
            seed=seed
        )

        obs = test_env.reset()
        obs_list = []
        for _ in range(steps):
            action, _ = model.predict(obs, deterministic=True)
            result = test_env.step(action)
            if len(result) == 5:
                obs, reward, done, truncated, info = result
            else:
                obs, reward, done, info = result
                truncated = done

            obs_list.append(obs[0])
            if done[0] or truncated[0]:
                obs = test_env.reset()
        test_env.close()

        obs_array = np.array(obs_list)

        # Now run PCMCI
        cit = ConditionalIndependenceTest()
        results_dict = cit.run_pcmci(
            observations=obs_array,
            tau_max=5,       # or read from config if you prefer
            alpha_level=0.05
        )
        return results_dict["val_matrix"], results_dict["p_matrix"]

    def run_multiple_pcmci_fisher(
        self, model, env_name, dim_id, mean, variance,
        num_runs=5, steps=2000
    ):
        """
        Perform multiple rollouts for PCMCI analysis, combine partial-corr & p-values.
        Return markov_score.
        """
        val_list = []
        p_list = []
        for _ in range(num_runs):
            seed = random.randint(0, 9999)
            val_m, p_m = self.gather_and_run_pcmci(
                model, env_name, dim_id, mean, variance,
                steps=steps, seed=seed
            )
            val_list.append(val_m)
            p_list.append(p_m)

        val_arr = np.stack(val_list, axis=0)
        p_arr = np.stack(p_list, axis=0)

        avg_val_matrix = np.mean(val_arr, axis=0)

        n_runs, N, _, L = p_arr.shape
        combined_p_matrix = np.zeros((N, N, L), dtype=float)

        for i in range(N):
            for j in range(N):
                for k in range(L):
                    pvals_for_link = p_arr[:, i, j, k]
                    combined_p_matrix[i, j, k] = self.fishers_method(pvals_for_link)

        mk_score = get_markov_violation_score(
            p_matrix=combined_p_matrix,
            val_matrix=avg_val_matrix,
            alpha_level=0.05
        )
        return mk_score

    def make_noisy_env(
        self,
        env_name,
        dimension_to_noisify=None,
        mean=0.0,
        variance=0.01,
        seed=None
    ):
        """
        Creates a vectorized environment that adds "Gaussian-like" noise
        to the action dimension. For discrete action spaces, we interpret
        'variance' as controlling a random override probability.
        """
        def _env_fn():
            e = gym.make(env_name)
            if seed is not None:
                e.reset(seed=seed)
            return NoisyGaussianActionWrapper(
                e,
                dimension_to_noisify=dimension_to_noisify,
                mean=mean,
                variance=variance
            )

        return make_vec_env(_env_fn, n_envs=1, seed=seed)

    def run(self, env_name=None, baseline_seed=None):
        """
        Main pipeline:
          - For each environment in config (or a chosen env_name),
          - Identify the action space dimension (Box => shape[0], Discrete => n),
          - For each dimension, for each noise param (mean,var),
            train PPO => gather PCMCI => record Markov score.
        """
        envs = self.config["environments"]
        if env_name:
            envs = [e for e in envs if e["name"] == env_name]

        if not envs:
            logging.warning(f"[Noised Action Gaussian] No matching environment for env_name={env_name}. Exiting.")
            return

        noise_list = self.config["noise_strategies"].get("action_gaussian", [])
        if not noise_list:
            logging.warning("[Noised Action Gaussian] 'gaussian' noise config not found.")
            return

        for env_item in envs:
            name = env_item["name"]
            time_steps = env_item["time_steps"]

            env_path = os.path.join(self.root_path, "results", name)
            noised_path = os.path.join(env_path, "noised_action")
            os.makedirs(noised_path, exist_ok=True)

            # Make a dummy env to inspect the action space
            dummy_env = gym.make(name)
            action_space = dummy_env.action_space
            dummy_env.close()

            # Determine the number of "dimensions" to iterate over
            if isinstance(action_space, gym.spaces.Box):
                # Continuous actions
                n_dims = action_space.shape[0]
                discrete_mode = False
            elif isinstance(action_space, gym.spaces.Discrete):
                # Discrete actions
                n_dims = action_space.n
                discrete_mode = True
            else:
                logging.warning(
                    f"[Noised Action Gaussian] Environment {name} has an unsupported action space. Skipping."
                )
                continue

            logging.info(f"[Noised Action Gaussian] => Env={name}, action_dims={n_dims}, discrete={discrete_mode}")

            # For each dimension in the action space
            for dim_id in range(n_dims):
                # For each noise param
                for noise_params in noise_list:
                    mean = noise_params.get("mean", 0.0)
                    var = noise_params.get("variance", 0.01)

                    used_seed = baseline_seed
                    logging.info(
                        f"[Noised Action Gaussian] env={name}, action_dim={dim_id}, mean={mean}, var={var}"
                    )

                    # 1) Create environment with noise
                    venv = self.make_noisy_env(
                        env_name=name,
                        dimension_to_noisify=dim_id,
                        mean=mean,
                        variance=var,
                        seed=used_seed
                    )

                    # 2) Train from scratch
                    model = PPO("MlpPolicy", venv, verbose=0, learning_rate=self.learning_rate)
                    callback = RewardTrackingCallback()
                    model.learn(total_timesteps=time_steps, callback=callback)
                    ep_rewards = callback.get_rewards()
                    venv.close()

                    # Store entire reward curve
                    for i, rew in enumerate(ep_rewards):
                        self.reward_records.append({
                            "Environment": name,
                            "ActionDim": dim_id,
                            "NoiseVariance": var,
                            "Episode": i + 1,
                            "Reward": rew
                        })

                    # 3) Multiple PCMCI runs => Fisher
                    NUM_PCMCI_RUNS = 3
                    mk_score = self.run_multiple_pcmci_fisher(
                        model=model,
                        env_name=name,
                        dim_id=dim_id,
                        mean=mean,
                        variance=var,
                        num_runs=NUM_PCMCI_RUNS,
                        steps=2000
                    )
                    logging.info(f"[Noised Action Gaussian] Markov Score => {mk_score:.4f}")

                    # 4) Store final metrics
                    final_reward = (
                        np.mean(ep_rewards[-10:])
                        if len(ep_rewards) >= 10
                        else np.mean(ep_rewards)
                    )
                    self.markov_records.append({
                        "Environment": name,
                        "ActionDim": dim_id,
                        "NoiseVariance": var,
                        "MarkovScore": mk_score,
                        "MeanFinalReward": final_reward
                    })

            # After finishing this environment, write out CSV & produce plots
            env_rewards = [r for r in self.reward_records if r["Environment"] == name]
            env_markov = [m for m in self.markov_records if m["Environment"] == name]

            df_rewards = pd.DataFrame(env_rewards)
            df_markov = pd.DataFrame(env_markov)

            # Save CSV
            reward_csv = os.path.join(noised_path, "noised_action_rewards.csv")
            markov_csv = os.path.join(noised_path, "noised_action_markov.csv")
            df_rewards.to_csv(reward_csv, index=False)
            df_markov.to_csv(markov_csv, index=False)
            logging.info(f"[Noised Action Gaussian] Wrote noised action rewards => {reward_csv}")
            logging.info(f"[Noised Action Gaussian] Wrote noised action Markov => {markov_csv}")

            # Produce single-figure learning curves for each noise variance
            baseline_csv_path = os.path.join(env_path, "csv", "baseline_learning_curve.csv")
            plot_noised_learning_curves_all_dims(
                df_rewards,
                baseline_csv=baseline_csv_path,
                output_dir=noised_path,
                smooth_window=10
            )
            # Additional plots
            plot_rewards_vs_noise(df_markov, output_dir=noised_path)
            plot_rewards_vs_markov(df_markov, output_dir=noised_path)
            plot_noise_vs_markov_corr(df_markov, output_dir=noised_path)


###############################################################################
# Top-level function to run from an external script
###############################################################################
def run_noised_gaussian_action(config_path="config.json", env_name=None, baseline_seed=None):
    """
    Reads config, sets up a NoisedGaussianActionExperiments instance, and runs it
    for the specified environment. Typically called from main.py or directly.

    config_path : str
        Path to the config JSON file, which must have:
         {
           "noise_strategies": {
             "gaussian": [ {"mean":0.0, "variance":0.01}, ... ],
             ...
           },
           "environments": [
             {
               "name": "CartPole-v1",
               "time_steps": 50000,
               ...
             },
             ...
           ]
         }
    env_name : str
        If provided, filters to just this environment. Otherwise loops over all.
    baseline_seed : int or None
        If you want to reuse a certain seed (for random noise, environment init).
    """
    if not os.path.exists(config_path):
        logging.error(f"[Noised Action Gaussian] Config file '{config_path}' not found.")
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    root_path = os.path.abspath(".")
    runner = NoisedGaussianActionExperiments(config, root_path=root_path)

    start_t = time.perf_counter()
    runner.run(env_name=env_name, baseline_seed=baseline_seed)
    end_t = time.perf_counter()
    logging.info(f"[Noised Action Gaussian] Done! Total time: {((end_t - start_t)/60):.2f}m")


###############################################################################
# Command-line entry point
###############################################################################
def main():
    """
    Allows command-line usage, e.g.:
      python noised_gaussian_action.py --config_path config.json --env CartPole-v1
    """
    parser = argparse.ArgumentParser(
        description="Noised Actions (Gaussian) Experiments (handles discrete or continuous)."
    )
    parser.add_argument("--config_path", type=str, default="config.json",
                        help="Path to the config JSON file.")
    parser.add_argument("--env", type=str, default=None,
                        help="Environment name to use from config.json.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional baseline seed to reuse.")
    args = parser.parse_args()

    run_noised_gaussian_action(
        config_path=args.config_path,
        env_name=args.env,
        baseline_seed=args.seed
    )


if __name__ == "__main__":
    main()