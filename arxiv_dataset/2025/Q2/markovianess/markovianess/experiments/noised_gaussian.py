# File: markovianess/experiments/noised_gaussian.py
"""
noised_gaussian.py
------------------
Runs experiments by adding Gaussian noise to one observation dimension
at a time. Trains a PPO agent for each noise level, then performs PCMCI
to measure Markov violation.

Overview of Steps
-----------------
1. For each dimension in the observation space, create a gym wrapper that injects
   Gaussian noise into that dimension at every step.
2. Train a PPO agent on this "noised" environment, record episode rewards.
3. Optionally run multiple PCMCI analyses with different rollouts, combining results
   (partial correlations + p-values) via averaging + Fisher's method.
4. Collect Markov violation scores and store them in CSV. Produce any relevant plots.

Usage
-----
Call the main function `run_noised_gaussian(config_path, env_name, baseline_seed)`
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
# If you have a custom logger in utils, you can import it. Otherwise, we use standard logging:
import logging
logging.basicConfig(level=logging.INFO)


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


class NoisyGaussianObservationWrapper(gym.Wrapper):
    """
    Injects Gaussian noise into a single dimension of the observation.
    If dimension_to_noisify is None, it applies noise to *all* dimensions.
    """
    def __init__(self, env, dimension_to_noisify=None, mean=0.0, variance=0.01):
        super().__init__(env)
        self.dim_to_noisify = dimension_to_noisify
        self.mean = mean
        self.variance = variance
        self.std = np.sqrt(variance)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        # In Gymnasium v0.26+, step returns 5 items (obs, reward, terminated, truncated, info)
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, done, truncated, info = result
        else:
            obs, reward, done, info = result
            truncated = done  # older gym fallback

        # Inject noise
        if self.dim_to_noisify is None:
            # noise on all dims
            obs = obs + np.random.normal(self.mean, self.std, size=obs.shape)
        else:
            noise = np.random.normal(self.mean, self.std)
            obs = np.array(obs, dtype=np.float32)
            obs[self.dim_to_noisify] += noise

        return obs, reward, done, truncated, info


def make_noisy_env(
    env_name,
    dimension_to_noisify=None,
    mean=0.0,
    variance=0.01,
    seed=None
):
    """
    Creates a vectorized environment that adds Gaussian noise to one dimension
    of the observation. If dimension_to_noisify is None, apply noise to all dims.
    """
    def _env_fn():
        e = gym.make(env_name)
        if seed is not None:
            e.reset(seed=seed)
        return NoisyGaussianObservationWrapper(
            e,
            dimension_to_noisify=dimension_to_noisify,
            mean=mean,
            variance=variance
        )

    return make_vec_env(_env_fn, n_envs=1, seed=seed)


def _smooth_reward_curve(episodes, rewards, window=10):
    """
    Returns a rolling-mean smoothing of the reward data.
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
    For each noise variance, produce one figure with multiple lines:
    - one line per (dimension, obs_name)
    - optionally overlay baseline if found
    """
    os.makedirs(output_dir, exist_ok=True)
    if df_rewards.empty:
        logging.info("[Noised Gaussian] No data to plot.")
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

        # Group by dimension => label lines with dimension & name
        for (dim_id, obs_name), df_dim in df_var.groupby(["ObsDim", "ObsName"]):
            df_dim = df_dim.sort_values("Episode")
            episodes = df_dim["Episode"].values
            rewards = df_dim["Reward"].values
            x_smooth, y_smooth = _smooth_reward_curve(episodes, rewards, window=smooth_window)
            label_str = f"Dim={dim_id}({obs_name})"
            plt.plot(x_smooth, y_smooth, label=label_str, linewidth=2)

        # Overlay baseline if available
        if baseline_df is not None:
            baseline_sorted = baseline_df.sort_values("Episode")
            b_eps = baseline_sorted["Episode"].values
            b_rew = baseline_sorted["TotalReward"].values
            bx_smooth, by_smooth = _smooth_reward_curve(b_eps, b_rew, window=smooth_window)
            plt.plot(bx_smooth, by_smooth, color="black", linewidth=3, label="Baseline")

        plt.title(f"{env_name}, Noise Var={var_val} (All Dims + Baseline)")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True)

        out_fname = f"{env_name}_var_{var_val}_all_dims_overlaid.png"
        out_path = os.path.join(output_dir, out_fname)
        plt.savefig(out_path, dpi=150)
        plt.close()
        logging.info(f"[Noised Gaussian] Saved noised multi-dim plot => {out_path}")


def plot_rewards_vs_noise(df_markov, output_dir="."):
    """
    Creates a plot: X-axis=NoiseVariance, Y-axis=MeanFinalReward. A line per dimension.
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
        for dim_id, gdf in env_df.groupby("ObsDim"):
            gdf_sorted = gdf.sort_values("NoiseVariance")
            ax.plot(
                gdf_sorted["NoiseVariance"],
                gdf_sorted["MeanFinalReward"],
                marker="o",
                label=f"dim={dim_id}"
            )
        ax.set_title(f"Rewards vs Noise - {env_name}")
        ax.set_xlabel("NoiseVariance")
        ax.set_ylabel("MeanFinalReward")
        ax.legend()
        ax.grid(True)
        outpath = os.path.join(output_dir, f"{env_name}_rewards_vs_noise.png")
        fig.savefig(outpath, dpi=150)
        plt.close(fig)


def plot_rewards_vs_markov(df_markov, output_dir="."):
    """
    Plot: X-axis=MarkovScore, Y-axis=MeanFinalReward. A line per dimension.
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
        for dim_id, gdf in env_df.groupby("ObsDim"):
            gdf_sorted = gdf.sort_values("MarkovScore")
            ax.plot(
                gdf_sorted["MarkovScore"],
                gdf_sorted["MeanFinalReward"],
                marker="o",
                label=f"dim={dim_id}"
            )
        ax.set_title(f"Rewards vs MarkovScore - {env_name}")
        ax.set_xlabel("MarkovScore")
        ax.set_ylabel("MeanFinalReward")
        ax.legend()
        ax.grid(True)
        outpath = os.path.join(output_dir, f"{env_name}_rewards_vs_markov.png")
        fig.savefig(outpath, dpi=150)
        plt.close(fig)


def plot_noise_vs_markov_corr(df_markov, output_dir="."):
    """
    Plot: X-axis=NoiseVariance, Y-axis=MarkovScore, with lines for each dimension.
    Also compute correlation across all points if you wish.
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
        for dim_id, gdf in env_df.groupby("ObsDim"):
            gdf_sorted = gdf.sort_values("NoiseVariance")
            ax.plot(
                gdf_sorted["NoiseVariance"],
                gdf_sorted["MarkovScore"],
                marker="o",
                label=f"dim={dim_id}"
            )

        if len(env_df) >= 2:
            corr_all = env_df[["NoiseVariance","MarkovScore"]].corr().iloc[0,1]
        else:
            corr_all = float('nan')

        ax.set_title(f"{env_name} - Noise vs MarkovScore (corr={corr_all:.2f})")
        ax.set_xlabel("NoiseVariance")
        ax.set_ylabel("MarkovScore")
        ax.legend()
        ax.grid(True)
        outpath = os.path.join(output_dir, f"{env_name}_noise_vs_markov_corr.png")
        fig.savefig(outpath, dpi=150)
        plt.close(fig)


class NoisedGaussianExperiments:
    """
    Class for systematically injecting Gaussian noise in each observation dimension,
    training a PPO model, and measuring Markov violation using PCMCI.

    Steps:
    ------
    1. For each dimension 'dim_id':
        a) For each noise param (mean, variance), create a "noised" environment.
        b) Train PPO => record reward per episode.
        c) Gather multiple rollouts with the trained model => run PCMCI => combine
           partial correlations & p-values => get Markov Score.
    2. Save final metrics to CSV, produce plots, etc.

    `run()` method orchestrates the above.
    """
    def __init__(self, config, root_path="."):
        """
        config : Dict loaded from config.json, must contain a "noise_strategies" dict
                 with "gaussian" info, e.g. { "gaussian": [ { "mean":0.0, "variance":0.01 }, ... ] }
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

    def gather_and_run_pcmci(self, model, env_name, dim_id, mean, variance, steps=2000, seed=None):
        """
        Creates a noised environment, collects steps with the given model,
        then runs PCMCI and returns (val_matrix, p_matrix).
        """
        test_env = make_noisy_env(
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
            tau_max=5,
            alpha_level=0.05
        )
        return results_dict["val_matrix"], results_dict["p_matrix"]

    def run_multiple_pcmci_fisher(
        self, model, env_name, dim_id, mean, variance,
        num_runs=5, steps=2000
    ):
        """
        Perform multiple rollouts for PCMCI analysis, combine partial-corr & p-values.
        Return (avg_val_matrix, combined_p_matrix, markov_score).
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
        return avg_val_matrix, combined_p_matrix, mk_score

    def run(self, env_name=None, baseline_seed=None):
        """
        Main pipeline:
          - For each environment in config (or a chosen env_name),
          - For each dimension, for each noise param (mean,var),
            train PPO => gather PCMCI => record Markov score.
        """
        envs = self.config["environments"]
        if env_name:
            envs = [e for e in envs if e["name"] == env_name]

        if not envs:
            logging.warning(f"[Noised Gaussian] No matching environment for env_name={env_name}. Exiting.")
            return

        noise_list = self.config["noise_strategies"].get("gaussian", [])

        for env_item in envs:
            name = env_item["name"]
            time_steps = env_item["time_steps"]
            n_envs = env_item["n_envs"]
            obs_names = env_item.get("observations", [])
            obs_dim = len(obs_names)

            # Setup environment-specific results directories
            env_path = os.path.join(self.root_path, "results", name)
            noised_path = os.path.join(env_path, "noised")
            pcmci_path = os.path.join(noised_path, "pcmci")
            os.makedirs(noised_path, exist_ok=True)
            os.makedirs(pcmci_path, exist_ok=True)

            logging.info(f"[Noised Gaussian] Start environment: {name}, obs_dim={obs_dim}")

            # For each dimension
            for dim_id in range(obs_dim):
                obs_dim_name = obs_names[dim_id]
                # For each noise param
                for noise_params in noise_list:
                    mean = noise_params.get("mean", 0.0)
                    var = noise_params.get("variance", 0.01)

                    logging.info(f"[Noised Gaussian] env={name}, dim_id={dim_id}, mean={mean}, var={var}")

                    # 1) Create environment with noise
                    used_seed = baseline_seed
                    venv = make_noisy_env(
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
                            "ObsDim": dim_id,
                            "ObsName": obs_dim_name,
                            "NoiseVariance": var,
                            "Episode": i + 1,
                            "Reward": rew
                        })

                    # 3) Multiple PCMCI runs => Fisher
                    NUM_PCMCI_RUNS = 3
                    _, _, mk_score = self.run_multiple_pcmci_fisher(
                        model=model,
                        env_name=name,
                        dim_id=dim_id,
                        mean=mean,
                        variance=var,
                        num_runs=NUM_PCMCI_RUNS,
                        steps=2000
                    )
                    logging.info(f"[Noised Gaussian] Markov Score => {mk_score:.4f}")

                    # 4) Store final metrics
                    final_reward = np.mean(ep_rewards[-10:]) if len(ep_rewards) >= 10 else np.mean(ep_rewards)
                    self.markov_records.append({
                        "Environment": name,
                        "ObsDim": dim_id,
                        "ObsName": obs_dim_name,
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
            reward_csv = os.path.join(noised_path, "noised_rewards.csv")
            markov_csv = os.path.join(noised_path, "noised_markov.csv")
            df_rewards.to_csv(reward_csv, index=False)
            df_markov.to_csv(markov_csv, index=False)
            logging.info(f"[Noised Gaussian] Wrote noised rewards => {reward_csv}")
            logging.info(f"[Noised Gaussian] Wrote noised Markov => {markov_csv}")

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


def run_noised_gaussian(config_path="config.json", env_name=None, baseline_seed=None):
    """
    Reads config, sets up a NoisedGaussianExperiments instance, and runs it
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
               "time_steps": 30000,
               "n_envs": 1,
               "observations": ["CartPos","CartVel","PoleAngle","PoleAngVel"]
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
        logging.error(f"[Noised Gaussian] Config file '{config_path}' not found.")
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    root_path = os.path.abspath(".")
    runner = NoisedGaussianExperiments(config, root_path=root_path)

    start_t = time.perf_counter()
    runner.run(env_name=env_name, baseline_seed=baseline_seed)
    end_t = time.perf_counter()
    logging.info(f"[Noised Gaussian] Done! Total time: {((end_t - start_t)/60):.2f}m")


def main():
    """
    Allows command-line usage, e.g.:
      python noised_gaussian.py --config_path config.json --env CartPole-v1
    """
    parser = argparse.ArgumentParser(description="Noised Observations (Gaussian) Experiments")
    parser.add_argument("--config_path", type=str, default="config.json",
                        help="Path to the config JSON file.")
    parser.add_argument("--env", type=str, default=None,
                        help="Environment name to use from config.json.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional baseline seed to reuse.")
    args = parser.parse_args()

    run_noised_gaussian(
        config_path=args.config_path,
        env_name=args.env,
        baseline_seed=args.seed
    )


if __name__ == "__main__":
    main()