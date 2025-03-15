#!/usr/bin/env python3
"""
-------------------------------------------------------------------------------
File: dropped_observation_dimension.py

Description:
    A unified script to run dimension-drop experiments on one of three
    Gymnasium environments:
      - Pendulum-v1 (3D observation)
      - CartPole-v1 (4D observation)
      - Acrobot-v1 (6D observation)

    We optionally drop one dimension (or none) from the observation space,
    train a PPO agent multiple times (in parallel), compute average rewards
    across test episodes, and compare to a random-action baseline.

    The final result is a bar plot of average rewards ± std for each
    dimension-drop scenario (plus random), using skyblue bars.

Usage:
    python dropped_observation_dimension.py
        --env_name [Pendulum-v1 | CartPole-v1 | Acrobot-v1]
        --total_timesteps 30000
        --episodes_for_test 50
        --num_runs 5
        --save_plot results.png
        --n_jobs -1

Parameters:
    --env_name: Which environment to use. Must be one of {Pendulum-v1, CartPole-v1, Acrobot-v1}.
    --total_timesteps: Number of PPO training timesteps (default: 30000).
    --episodes_for_test: Number of test episodes (default: 50).
    --num_runs: Number of independent runs per dimension setting (default: 5).
    --save_plot: Output filename for the bar plot (default: results.png).
    --n_jobs: How many parallel workers to use (default: -1 uses all cores).

Dependencies:
    - Python 3
    - stable-baselines3
    - gymnasium
    - matplotlib
    - joblib
-------------------------------------------------------------------------------
"""

import argparse
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


class DropDimensionWrapper(gym.ObservationWrapper):
    """
    A custom wrapper that drops a specified dimension from an N-dimensional observation.
    It automatically updates the observation space accordingly.
    """

    def __init__(self, env, drop_idx):
        super().__init__(env)
        self.drop_idx = drop_idx

        # Original Box shape
        orig_low = self.observation_space.low
        orig_high = self.observation_space.high

        # Remove the specified dimension from the low/high arrays
        new_low = np.delete(orig_low, drop_idx)
        new_high = np.delete(orig_high, drop_idx)

        # Define the new observation space
        self.observation_space = spaces.Box(
            low=new_low,
            high=new_high,
            dtype=self.observation_space.dtype
        )

    def observation(self, obs):
        return np.delete(obs, self.drop_idx)


def get_env_config(env_name):
    """
    Return a dict containing:
      - 'dims_to_test': A list of dimension indices to drop, plus None for no drop.
      - 'dimension_labels': A list of short labels for each dimension
      - 'dimension_labels_random': Label for the random baseline
      - 'original_obs_dim': integer dimension of the original observation

    We also demonstrate how to label each dimension for plotting.
    """
    if env_name == "Pendulum-v1":
        # 3D observation: [cos(theta), sin(theta), theta_dot]
        return {
            "dims_to_test": [None, 0, 1, 2],
            "dimension_labels": [
                "No drop",
                "Drop 0\n(cosθ)",
                "Drop 1\n(sinθ)",
                "Drop 2\n(θ_dot)"
            ],
            "dimension_labels_random": "Random\n(No drop)",
            "original_obs_dim": 3
        }
    elif env_name == "CartPole-v1":
        # 4D observation: [cart_pos, cart_vel, pole_angle, pole_ang_vel]
        return {
            "dims_to_test": [None, 0, 1, 2, 3],
            "dimension_labels": [
                "No drop",
                "Drop 0\n(Cart Pos.)",
                "Drop 1\n(Cart Vel.)",
                "Drop 2\n(Pole Angle)",
                "Drop 3\n(Pole Ang. Vel.)"
            ],
            "dimension_labels_random": "Random\n(No drop)",
            "original_obs_dim": 4
        }
    elif env_name == "Acrobot-v1":
        # 6D observation:
        #   0: cos(theta1), 1: sin(theta1),
        #   2: cos(theta2), 3: sin(theta2),
        #   4: ang_vel_theta1, 5: ang_vel_theta2
        return {
            "dims_to_test": [None, 0, 1, 2, 3, 4, 5],
            "dimension_labels": [
                "No drop",
                "Drop 0\n(cosθ1)",
                "Drop 1\n(sinθ1)",
                "Drop 2\n(cosθ2)",
                "Drop 3\n(sinθ2)",
                "Drop 4\n(ang.vel.θ1)",
                "Drop 5\n(ang.vel.θ2)",
            ],
            "dimension_labels_random": "Random\n(No drop)",
            "original_obs_dim": 6
        }
    else:
        raise ValueError(f"Unsupported env_name: {env_name}")


def make_custom_env(env_name, drop_idx=None, seed=None):
    """
    Creates a Gymnasium environment for the given env_name,
    optionally drops one dimension (using DropDimensionWrapper),
    and optionally seeds the environment for reproducibility.
    """
    env = gym.make(env_name)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

    if drop_idx is not None:
        env = DropDimensionWrapper(env, drop_idx)
    return env


def train_model(env_name, drop_idx, total_timesteps, seed=None):
    """
    Train a PPO model on the specified environment, optionally dropping
    one dimension. Returns the trained model.
    """
    def _make_training_env():
        return make_custom_env(env_name, drop_idx=drop_idx, seed=seed)

    # If you want to parallelize environment sampling:
    #   train_vec_env = make_vec_env(_make_training_env, n_envs=4)
    train_vec_env = make_vec_env(_make_training_env, n_envs=1)
    model = PPO("MlpPolicy", train_vec_env, verbose=0, seed=seed)
    model.learn(total_timesteps)
    return model


def evaluate_model(env_name, model, drop_idx, n_episodes=50, seed=None):
    """
    Evaluate a trained model for n_episodes, returning the average total reward.
    """
    test_env = make_custom_env(env_name, drop_idx=drop_idx, seed=seed)

    total_rewards = []
    for _ in range(n_episodes):
        obs, info = test_env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = test_env.step(action)
            ep_reward += reward
            done = terminated or truncated
        total_rewards.append(ep_reward)

    return np.mean(total_rewards)


def evaluate_random_action(env_name, drop_idx=None, n_episodes=50, seed=None):
    """
    Evaluate random actions in the given environment (optionally dropping one dim),
    returning the average total reward over n_episodes.
    """
    env = make_custom_env(env_name, drop_idx=drop_idx, seed=seed)

    total_rewards = []
    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        total_rewards.append(ep_reward)

    return np.mean(total_rewards)


def run_single_ppo_experiment(env_name, drop_idx, total_timesteps, seed, episodes_for_test):
    """
    Train + evaluate PPO for a single run, returning average total reward.
    """
    model = train_model(env_name, drop_idx, total_timesteps, seed=seed)
    avg_return = evaluate_model(env_name, model, drop_idx, n_episodes=episodes_for_test, seed=seed)
    return avg_return


def run_single_random_experiment(env_name, drop_idx, seed, episodes_for_test):
    """
    Random actions for a single run, returning average total reward.
    """
    return evaluate_random_action(env_name, drop_idx=drop_idx, n_episodes=episodes_for_test, seed=seed)


def main():
    parser = argparse.ArgumentParser(description="Unified dimension-drop experiments on various Gym environments.")
    parser.add_argument(
        "--env_name",
        type=str,
        required=True,
        help="Which environment to run. Must be one of {Pendulum-v1, CartPole-v1, Acrobot-v1}."
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=30000,
        help="Number of timesteps for PPO training per dimension setting."
    )
    parser.add_argument(
        "--episodes_for_test",
        type=int,
        default=50,
        help="Number of test episodes after training each model."
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=5,
        help="Number of independent runs (with different seeds) per dimension + random."
    )
    parser.add_argument(
        "--save_plot",
        type=str,
        default="results.png",
        help="Filename for the resulting plot."
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of parallel processes to use with joblib."
    )
    args = parser.parse_args()

    # Get environment-specific configuration (dims to test, labels, etc.)
    env_config = get_env_config(args.env_name)

    dims_to_test = env_config["dims_to_test"]  # e.g., [None, 0, 1, 2] for Pendulum
    dim_labels = env_config["dimension_labels"]  # labels for each dimension
    random_label = env_config["dimension_labels_random"]

    # We want an extra bar for the random-action baseline, so let's track them together
    labels = list(dim_labels) + [random_label]

    means = []
    stds = []

    print(f"\nEnvironment: {args.env_name}")
    print(f"Dimension scenarios: {dims_to_test}")
    print(f"Running {args.num_runs} independent runs per scenario.\n")

    # 1) PPO runs for each dimension
    for dim_idx, drop_idx in enumerate(dims_to_test):
        label_str = dim_labels[dim_idx]
        print(f"=== Parallel runs for {label_str} (drop_idx={drop_idx}) ===")

        # Build argument list
        run_args_list = []
        for run_id in range(args.num_runs):
            seed = 1000 + 100 * dim_idx + run_id  # example systematic seeding
            run_args_list.append((args.env_name, drop_idx, args.total_timesteps, seed, args.episodes_for_test))

        # Run them in parallel
        results_across_runs = Parallel(n_jobs=args.n_jobs)(
            delayed(run_single_ppo_experiment)(*r_args) for r_args in run_args_list
        )

        mean_r = np.mean(results_across_runs)
        std_r = np.std(results_across_runs)
        means.append(mean_r)
        stds.append(std_r)

        print(f"=> Mean reward for {label_str}: {mean_r:.2f} ± {std_r:.2f}\n")

    # 2) Random baseline (no drop)
    print("=== Parallel runs for Random Action (No drop) ===")
    random_run_args_list = []
    for run_id in range(args.num_runs):
        seed = 2000 + run_id
        random_run_args_list.append((args.env_name, None, seed, args.episodes_for_test))

    random_results = Parallel(n_jobs=args.n_jobs)(
        delayed(run_single_random_experiment)(*rr_args) for rr_args in random_run_args_list
    )

    random_mean = np.mean(random_results)
    random_std = np.std(random_results)
    means.append(random_mean)
    stds.append(random_std)

    print(f"=> Mean reward for Random Action: {random_mean:.2f} ± {random_std:.2f}\n")

    # 3) Create bar plot (skyblue color)
    x_positions = np.arange(len(labels))
    plt.figure(figsize=(10, 5))

    plt.bar(x_positions, means, yerr=stds, capsize=5, color='skyblue', width=0.6)
    plt.xticks(x_positions, labels)
    plt.xlabel("Dimension Dropped / Agent Type")
    plt.ylabel("Avg. Episode Reward")
    plt.title(
        f"PPO + Dimension-Drop on {args.env_name}\n"
        f"{args.total_timesteps} steps, {args.episodes_for_test} test eps, {args.num_runs} runs"
    )

    # 4) Label each bar with mean ± std
    for i, (m, s) in enumerate(zip(means, stds)):
        offset = max(1.0, abs(s) * 0.5)
        plt.text(i, m + offset, f"{m:.2f}±{s:.2f}", ha='center', va='bottom', fontsize=9)

    # Some envs (e.g. Pendulum) can have negative rewards, so let's set
    # a margin around min and max
    min_val = min(means) - abs(min(stds)) - 10
    max_val = max(means) + abs(max(stds)) + 10
    plt.ylim([min_val, max_val])

    plt.tight_layout()
    plt.savefig(args.save_plot, dpi=200)
    print(f"\nPlot saved to '{args.save_plot}'")

    # plt.show()


if __name__ == "__main__":
    main()