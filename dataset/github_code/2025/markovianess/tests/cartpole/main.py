#!/usr/bin/env python3
"""
-------------------------------------------------------------------------------
File: cartpole_drop_experiment.py

Description:
    This script trains a PPO agent on the CartPole-v1 environment multiple times,
    each time dropping one dimension (or none) from the original 4-dimensional
    observation space. Specifically, we investigate how dropping each dimension:
      0) Cart Position
      1) Cart Velocity
      2) Pole Angle
      3) Pole Angular Velocity
    affects training performance.

    After each training, the script evaluates the resulting agent for a number of
    test episodes and computes the average episode length (i.e., the number of steps
    per episode, which also corresponds to the total reward in CartPole).

    Finally, it creates a bar plot illustrating the average episode length across
    each "drop" setting (including "No drop" for the original 4D environment),
    and saves this figure to a file.

Usage:
    python cartpole_drop_experiment.py
        --total_timesteps 30000
        --episodes_for_test 50
        --save_plot cartpole_drop_experiment.png

Parameters:
    --total_timesteps: Number of PPO training timesteps for each condition (default: 30000).
    --episodes_for_test: Number of test episodes to average over (default: 50).
    --save_plot: Filename to which the resulting plot is saved (default: cartpole_drop_experiment.png).

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
    A custom wrapper that drops a specified dimension from the observation.
    For CartPole-v1, the default observation has shape (4,).
    """

    def __init__(self, env, drop_idx):
        super().__init__(env)
        self.drop_idx = drop_idx

        # Original observation space is typically shape=(4,) for CartPole
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
        # Remove the dimension (index = self.drop_idx) from the current observation
        return np.delete(obs, self.drop_idx)


def make_custom_env(render_mode, drop_idx=None):
    """
    Creates a CartPole-v1 environment with a specified render_mode and
    optionally drops a dimension in the observation space.
    """
    env = gym.make("CartPole-v1", render_mode=render_mode)
    if drop_idx is not None:
        env = DropDimensionWrapper(env, drop_idx)
    return env


def train_model(drop_idx, total_timesteps):
    """
    Train a PPO model with the specified dimension dropped.
    Returns the trained model.
    """
    def make_training_env():
        # Use "rgb_array" (or None) for training to avoid real-time rendering overhead
        return make_custom_env(render_mode="rgb_array", drop_idx=drop_idx)

    # Create a vectorized environment for training (n_envs=1 for simplicity)
    train_vec_env = make_vec_env(make_training_env, n_envs=1)

    # Create and train the model
    model = PPO("MlpPolicy", train_vec_env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    return model


def evaluate_model(model, drop_idx, n_episodes=100):
    """
    Evaluate a trained PPO model for n_episodes using the specified drop_idx.
    Returns the average episode length (number of steps) over those episodes.
    """
    # Create a separate environment for testing (no pop-up windows).
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

    avg_length = np.mean(all_ep_lengths)
    return avg_length


def main():
    parser = argparse.ArgumentParser(
        description="Experiment: drop each dimension of CartPole-v1 one by one and measure performance."
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
        default="cartpole_drop_experiment.png",
        help="Where to save the resulting plot (e.g., 'results.png')."
    )
    args = parser.parse_args()

    dimensions_to_test = [None, 0, 1, 2, 3]
    avg_episode_lengths = []

    print("Starting experiment over dimensions:", dimensions_to_test)
    for drop_idx in dimensions_to_test:
        label_str = "No drop" if drop_idx is None else f"Drop {drop_idx}"
        print(f"\n=== Training with {label_str} ===")

        # 1) Train the model
        model = train_model(drop_idx, total_timesteps=args.total_timesteps)

        # 2) Evaluate
        print(f"Evaluating model trained with {label_str} ...")
        avg_length = evaluate_model(model, drop_idx, n_episodes=args.episodes_for_test)
        avg_episode_lengths.append(avg_length)

        print(f"Average episode length over {args.episodes_for_test} episodes: {avg_length:.2f}")

    # Prepare data for plotting
    labels = [
        "No drop",
        "Drop 0\n(Cart Pos.)",
        "Drop 1\n(Cart Vel.)",
        "Drop 2\n(Pole Angle)",
        "Drop 3\n(Pole Ang. Vel.)"
    ]

    # Create the bar plot
    plt.figure(figsize=(8.5, 5))
    plt.bar(labels, avg_episode_lengths, color='skyblue', width=0.6)
    plt.xlabel("Dimension Dropped")
    plt.ylabel("Avg. Episode Length (out of 500 max)")
    plt.title(
        f"Performance of PPO on CartPole-v1 Dropping Each Dimension\n"
        f"(Trained {args.total_timesteps} steps, {args.episodes_for_test} test episodes)"
    )

    # Label each bar with the actual average length
    for i, val in enumerate(avg_episode_lengths):
        plt.text(i, val + 5, f"{val:.1f}", ha='center', va='bottom', fontsize=10)

    # Increase lower margin so x-labels don’t overlap
    plt.subplots_adjust(bottom=0.15)

    plt.ylim([0, 520])  # A bit higher than 500 to fit labels
    plt.tight_layout()

    # Save the plot to file
    plt.savefig(args.save_plot, dpi=200)
    print(f"Plot saved to '{args.save_plot}'")

    # Uncomment the next line if you want a pop-up window after saving
    # plt.show()


if __name__ == "__main__":
    main()