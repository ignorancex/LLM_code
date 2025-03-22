#!/usr/bin/env python3
"""
-------------------------------------------------------------------------------
File: pendulum_drop_experiment.py

Description:
    This script trains a PPO agent on the Pendulum-v1 environment multiple times,
    each time dropping one dimension (or none) from the original 3-dimensional
    observation space:
      0) cos(theta)
      1) sin(theta)
      2) theta_dot (angular velocity)

    After each training, the script evaluates the resulting agent for a number
    of test episodes and computes the average total reward (which is typically
    negative unless perfectly upright).

    We then produce a bar plot showing those average returns, from a negative
    range up to near 0. Each bar is labeled with both its raw average and the
    difference from the "No drop" baseline in parentheses, in the same style
    as the CartPole script.

Usage:
    python pendulum_drop_experiment.py
        --total_timesteps 30000
        --episodes_for_test 50
        --save_plot pendulum_drop_experiment.png

Parameters:
    --total_timesteps: Number of PPO training timesteps for each condition
                       (default: 30000).
    --episodes_for_test: Number of test episodes to average over (default: 50).
    --save_plot: Filename where the resulting plot is saved
                 (default: pendulum_drop_experiment.png).

Dependencies:
    - Python 3
    - stable-baselines3
    - gymnasium
    - matplotlib

-------------------------------------------------------------------------------
"""

import argparse
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt


class DropDimensionWrapper(gym.ObservationWrapper):
    """
    A custom wrapper that drops a specified dimension from the
    3D observation of Pendulum-v1: [cos(theta), sin(theta), theta_dot].
    """

    def __init__(self, env, drop_idx):
        super().__init__(env)
        self.drop_idx = drop_idx

        # Original observation space is shape=(3,) for Pendulum.
        orig_low = self.observation_space.low
        orig_high = self.observation_space.high

        # Remove the specified dimension from the low/high arrays
        new_low = np.delete(orig_low, drop_idx)
        new_high = np.delete(orig_high, drop_idx)

        # Define the new (reduced) observation space
        self.observation_space = spaces.Box(
            low=new_low,
            high=new_high,
            dtype=self.observation_space.dtype
        )

    def observation(self, obs):
        # Remove the dimension at index = self.drop_idx
        return np.delete(obs, self.drop_idx)


def make_custom_env(render_mode, drop_idx=None):
    """
    Creates a Pendulum-v1 environment with a specified render_mode and
    optionally drops a dimension in the observation space.
    """
    env = gym.make("Pendulum-v1", render_mode=render_mode)
    if drop_idx is not None:
        env = DropDimensionWrapper(env, drop_idx)
    return env


def train_model(drop_idx, total_timesteps):
    """
    Train a PPO model on Pendulum-v1 with the specified dimension dropped.
    Returns the trained model.
    """
    def make_training_env():
        # Use "rgb_array" for training to avoid pop-up windows
        return make_custom_env(render_mode="rgb_array", drop_idx=drop_idx)

    train_vec_env = make_vec_env(make_training_env, n_envs=1)
    model = PPO("MlpPolicy", train_vec_env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    return model


def evaluate_model(model, drop_idx, n_episodes=50):
    """
    Evaluate a trained PPO model for n_episodes using the specified drop_idx.
    Returns the average total reward (negative unless perfectly upright).
    """
    test_env = make_custom_env(render_mode="rgb_array", drop_idx=drop_idx)

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

    avg_reward = np.mean(total_rewards)
    return avg_reward


def main():
    parser = argparse.ArgumentParser(
        description="Experiment: drop each dimension of Pendulum-v1 one by one and measure performance."
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
        "--save_plot",
        type=str,
        default="pendulum_drop_experiment.png",
        help="Where to save the resulting plot (e.g., 'results.png')."
    )
    args = parser.parse_args()

    # Dimensions: 0 => cos(theta), 1 => sin(theta), 2 => theta_dot
    # None => No dimension dropped
    dimensions_to_test = [None, 0, 1, 2]
    avg_returns = []

    print("Starting experiment over dimensions:", dimensions_to_test)
    for drop_idx in dimensions_to_test:
        label_str = "No drop" if drop_idx is None else f"Drop {drop_idx}"
        print(f"\n=== Training with {label_str} ===")

        # Train model
        model = train_model(drop_idx, args.total_timesteps)

        # Evaluate
        print(f"Evaluating model trained with {label_str} ...")
        avg_ret = evaluate_model(model, drop_idx, n_episodes=args.episodes_for_test)
        avg_returns.append(avg_ret)

        print(f"Average return over {args.episodes_for_test} episodes: {avg_ret:.2f}")

    # The first entry is "No drop" (our baseline)
    baseline = avg_returns[0]

    # Prepare data for plotting
    labels = [
        "No drop",
        "Drop 0\n(cosθ)",
        "Drop 1\n(sinθ)",
        "Drop 2\n(θ_dot)"
    ]

    # Create the bar plot
    plt.figure(figsize=(8.0, 5.0))
    bars = plt.bar(labels, avg_returns, color='lightcoral', width=0.6)

    plt.xlabel("Dimension Dropped")
    plt.ylabel("Avg. Episode Return (closer to 0 is better)")
    plt.title(
        f"Performance of PPO on Pendulum-v1 Dropping Each Dimension\n"
        f"(Trained {args.total_timesteps} steps, {args.episodes_for_test} test episodes)"
    )

    # Label each bar with both the raw average return and difference from baseline
    for i, val in enumerate(avg_returns):
        diff = val - baseline  # how many reward points more/less than baseline
        sign_diff = f"{diff:+.1f}"
        label_text = f"{val:.1f}\n({sign_diff})"

        # Place the text above the bar
        # Since returns are negative, we place the text a bit above val.
        offset = 2 if val > -10 else 10  # adjust if you have large negative
        plt.text(i, val + offset, label_text, ha='center', va='bottom', fontsize=9)

    # Optionally, you can fix the y-limits if your returns are around [-2000, 0].
    # This ensures consistent scale across different runs:
    min_val = min(avg_returns) - 50
    max_val = 10  # A bit above 0
    plt.ylim([min_val, max_val])

    plt.tight_layout()
    plt.savefig(args.save_plot, dpi=200)
    print(f"Plot saved to '{args.save_plot}'")

    # Uncomment for a pop-up window
    # plt.show()


if __name__ == "__main__":
    main()