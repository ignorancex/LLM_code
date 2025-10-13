#!/usr/bin/env python3
"""
-------------------------------------------------------------------------------
File: acrobot_drop_experiment.py

Description:
    This script trains a PPO agent on the Acrobot-v1 environment multiple times,
    each time dropping one dimension (or none) from the default 6D observation:
        0. cos(theta1)
        1. sin(theta1)
        2. cos(theta2)
        3. sin(theta2)
        4. angular velocity of theta1
        5. angular velocity of theta2

    After training each variant (with dimension i removed), the script evaluates
    the agent for a number of test episodes and computes the average episode length.
    We then create a bar plot showing how performance changes when each dimension
    is dropped, compared to the "No drop" case.

Usage:
    python acrobot_drop_experiment.py
        --total_timesteps 30000
        --episodes_for_test 50
        --save_plot acrobot_drop_experiment.png

Parameters:
    --total_timesteps: Number of timesteps for PPO training (default: 30000).
    --episodes_for_test: Number of test episodes to compute average length (default: 50).
    --save_plot: Filename for saving the resulting bar plot (default: acrobot_drop_experiment.png).

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
    A custom wrapper that drops a specified dimension from the 6D observation of Acrobot-v1.
    The default observation is:
        [cos(theta1), sin(theta1), cos(theta2), sin(theta2), ang_vel_theta1, ang_vel_theta2]
    """

    def __init__(self, env, drop_idx):
        super().__init__(env)
        self.drop_idx = drop_idx

        # For Acrobot-v1, observation_space.shape is typically (6,)
        # We remove the specified dimension from the observation space:
        orig_low = self.observation_space.low
        orig_high = self.observation_space.high

        new_low = np.delete(orig_low, drop_idx)
        new_high = np.delete(orig_high, drop_idx)

        # Define the new (reduced) observation space
        self.observation_space = spaces.Box(
            low=new_low,
            high=new_high,
            dtype=self.observation_space.dtype
        )

    def observation(self, obs):
        # Remove the dimension at index = drop_idx
        return np.delete(obs, self.drop_idx)


def make_custom_env(render_mode, drop_idx=None):
    """
    Creates an Acrobot-v1 environment with the specified render_mode and
    optionally wraps it to drop one dimension from the observation.
    """
    env = gym.make("Acrobot-v1", render_mode=render_mode)
    if drop_idx is not None:
        env = DropDimensionWrapper(env, drop_idx)
    return env


def train_model(drop_idx, total_timesteps):
    """
    Train a PPO model on Acrobot-v1 with one dimension dropped (drop_idx).
    Returns the trained model.
    """
    def make_training_env():
        # Use "rgb_array" (or None) for training to avoid real-time rendering overhead
        return make_custom_env(render_mode="rgb_array", drop_idx=drop_idx)

    # Create a vectorized environment for training (n_envs=1 for simplicity)
    train_vec_env = make_vec_env(make_training_env, n_envs=1)

    # Initialize and train the PPO model
    model = PPO("MlpPolicy", train_vec_env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    return model


def evaluate_model(model, drop_idx, n_episodes=100):
    """
    Evaluate a trained PPO model for n_episodes with the same dimension dropped.
    Returns the average episode length (how many steps each episode lasted).
    """
    test_env = make_custom_env(render_mode="rgb_array", drop_idx=drop_idx)

    all_ep_lengths = []
    for _ in range(n_episodes):
        obs, info = test_env.reset()
        done = False
        steps = 0

        while not done:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = test_env.step(action)
            steps += 1
            done = terminated or truncated

        all_ep_lengths.append(steps)

    return np.mean(all_ep_lengths)


def main():
    parser = argparse.ArgumentParser(description="Experiment: drop each dimension of Acrobot-v1 one by one.")
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
        default="acrobot_drop_experiment.png",
        help="Where to save the resulting plot (e.g., 'results.png')."
    )
    args = parser.parse_args()

    # The 6D observation of Acrobot:
    #   0: cos(theta1)
    #   1: sin(theta1)
    #   2: cos(theta2)
    #   3: sin(theta2)
    #   4: angular velocity of theta1
    #   5: angular velocity of theta2
    dimensions_to_test = [None, 0, 1, 2, 3, 4, 5]
    avg_episode_lengths = []

    print("Starting experiment over dimensions:", dimensions_to_test)
    for drop_idx in dimensions_to_test:
        label_str = "No drop" if drop_idx is None else f"Drop {drop_idx}"
        print(f"\n=== Training with {label_str} ===")

        # 1) Train
        model = train_model(drop_idx, total_timesteps=args.total_timesteps)

        # 2) Evaluate
        print(f"Evaluating model trained with {label_str}...")
        avg_length = evaluate_model(model, drop_idx, n_episodes=args.episodes_for_test)
        avg_episode_lengths.append(avg_length)

        print(f"Average episode length over {args.episodes_for_test} episodes: {avg_length:.2f}")

    # Prepare data for plotting
    labels = [
        "No drop",
        "Drop 0\n(cosθ1)",
        "Drop 1\n(sinθ1)",
        "Drop 2\n(cosθ2)",
        "Drop 3\n(sinθ2)",
        "Drop 4\n(ang.vel.θ1)",
        "Drop 5\n(ang.vel.θ2)"
    ]

    # Create the bar plot
    plt.figure(figsize=(9, 5))
    plt.bar(labels, avg_episode_lengths, color='lightcoral', width=0.6)
    plt.xlabel("Dimension Dropped")
    plt.ylabel("Avg. Episode Length (Max 500)")
    plt.title(
        f"PPO on Acrobot-v1: Dropping Each Observation Dimension\n"
        f"(Trained {args.total_timesteps} steps, {args.episodes_for_test} test episodes)"
    )

    # Label each bar with the actual average length
    for i, val in enumerate(avg_episode_lengths):
        plt.text(i, val + 5, f"{val:.1f}", ha='center', va='bottom', fontsize=9)

    # Increase lower margin so x-labels don’t overlap
    plt.subplots_adjust(bottom=0.15)

    plt.ylim([0, 520])  # 500 step limit + a little margin
    plt.tight_layout()

    # Save and optionally show the plot
    plt.savefig(args.save_plot, dpi=200)
    print(f"Plot saved to '{args.save_plot}'")

    # Uncomment if you want to see the plot window
    # plt.show()


if __name__ == "__main__":
    main()