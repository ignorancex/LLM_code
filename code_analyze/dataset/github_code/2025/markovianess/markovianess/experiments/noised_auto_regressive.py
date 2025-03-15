# File: markovianess/experiments/noised_auto_regressive.py
"""
noised_auto_regressive.py
-------------------------
Runs experiments by adding auto-regressive (AR) noise to one observation dimension
at a time. Steps:

1) Train a "no-noise" PPO policy on the given environment.
2) Gather transitions (S, A, S_next) using that policy -> train a neural network f(S,A)->S_next.
3) For each dimension:
    - Wrap environment in an AR noise wrapper that blends the real next state with
      the predicted next state, plus AR noise in that dimension.
    - Train a PPO policy on that AR-noised environment; record rewards.
    - Run PCMCI multiple times to measure Markov violation (averaged or combined).
4) Save results to CSV and produce optional plots.

Usage:
------
Call the function `run_noised_auto_regressive(config_path, env_name, baseline_seed)` from your
main script or command line.

The config JSON might look like:
{
  "environments": [
    {
      "name": "CartPole-v1",
      "time_steps": 30000,
      "n_envs": 1,
      "observations": ["CartPos", "CartVel", "PoleAngle", "PoleAngVel"],
      "samples_for_fit": 10000,
      "epochs_for_fit": 20
    }
  ],
  "noise_strategies": {
    "auto_regressive": {
      "ar1": [
        { "alphas": [0.9], "sigma": 0.5, "description": "AR(1), alpha=0.9" },
        { "alphas": [0.8], "sigma": 0.2, "description": "AR(1), alpha=0.8" }
      ],
      "ar2": [
        { "alphas": [0.5, 0.3], "sigma": 0.5, "description": "AR(2), alphas=0.5,0.3" }
      ]
      ...
    }
  }
}

"""

import argparse
import json
import os
import random
import time
from collections import deque

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torch.optim as optim
from scipy.stats import chi2
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from markovianess.ci.conditional_independence_test import (
    ConditionalIndependenceTest,
    get_markov_violation_score
)

import logging
logging.basicConfig(level=logging.INFO)


###############################################################################
# (A) Train No-Noise Policy, then gather transitions => train f(S,A)->S_next
###############################################################################
class RewardTrackingCallback(BaseCallback):
    """
    Tracks the total reward per episode in a single-environment scenario.
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


def train_noiseless_policy(env_name, total_timesteps=50000):
    """
    Creates a standard PPO model with no modifications, trains on env_name,
    and returns the trained model.
    """
    logging.info(f"[ARNoised] Training no-noise PPO on {env_name} for {total_timesteps} steps.")
    vec_env = make_vec_env(env_name, n_envs=1)
    model = PPO("MlpPolicy", vec_env, verbose=0)
    model.learn(total_timesteps=total_timesteps)
    vec_env.close()
    return model


def gather_transition_data(model, env_name, n_samples=10000):
    """
    Runs the environment with the trained 'model' to collect (S,A,S_next) transitions
    for fitting a transition model f(S,A)->S_next.

    Returns:
    --------
    states: shape=(n_samples, state_dim)
    actions: shape=(n_samples, action_dim)
    next_states: shape=(n_samples, state_dim)
    """
    env = gym.make(env_name)
    obs, _ = env.reset()

    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    states = []
    actions = []
    next_states = []

    for _ in range(n_samples):
        action, _ = model.predict(obs, deterministic=True)
        if discrete:
            a_vec = float(action)
            a_vec = np.array([a_vec], dtype=np.float32)
        else:
            a_vec = action.astype(np.float32, copy=False)

        new_obs, reward, done, truncated, info = env.step(action)

        states.append(obs)
        actions.append(a_vec)
        next_states.append(new_obs)

        if done or truncated:
            obs, _ = env.reset()
        else:
            obs = new_obs

    env.close()

    states = np.array(states, dtype=np.float32)
    actions = np.stack(actions, axis=0).astype(np.float32)
    next_states = np.array(next_states, dtype=np.float32)

    logging.info(f"[ARNoised] Transitions gathered: states={states.shape}, actions={actions.shape}, next_states={next_states.shape}")
    return states, actions, next_states


class TransitionModel(nn.Module):
    """
    A simple MLP that inputs (S,A) and predicts S_next.
    If state_dim=D, action_dim=A, input size=(D+A), output size=D.
    """
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, state_dim)
        )

    def forward(self, s, a):
        x = th.cat([s, a], dim=-1)
        return self.net(x)


def train_transition_model(states, actions, next_states,
                           epochs=20, batch_size=64, lr=1e-3):
    """
    Trains a neural network to approximate f(S,A)->S_next.
    Returns the trained model and a list of losses per epoch.
    """
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    N, state_dim = states.shape
    action_dim = actions.shape[1]

    X_s = th.from_numpy(states).to(device)
    X_a = th.from_numpy(actions).to(device)
    Y = th.from_numpy(next_states).to(device)

    model = TransitionModel(state_dim, action_dim, hidden_size=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    indices = np.arange(N)
    losses_per_epoch = []
    for ep in range(epochs):
        np.random.shuffle(indices)
        epoch_loss = 0.0
        num_batches = 0

        for start_idx in range(0, N, batch_size):
            end_idx = start_idx + batch_size
            batch_idx = indices[start_idx:end_idx]

            s_batch = X_s[batch_idx]
            a_batch = X_a[batch_idx]
            y_batch = Y[batch_idx]

            optimizer.zero_grad()
            y_pred = model(s_batch, a_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        losses_per_epoch.append(avg_loss)
        logging.info(f"[ARNoised] Epoch {ep+1}/{epochs}, MSE={avg_loss:.6f}")

    return model, losses_per_epoch


###############################################################################
# (B) AR(p) Wrapper that uses the learned transition f(S,A)->S_next
###############################################################################
class ARObservationWrapper(gym.Wrapper):
    """
    Gym Wrapper that:
    1) Gets the real next state from env.
    2) Predicts next state from f(S,A).
    3) Blends them (e.g. a fixed ratio) and adds an AR(p) noise vector.
    4) If dimension_to_noisify is not None, only that dimension is affected by AR noise.
    Otherwise, all dims get AR noise.

    :param transition_model: The learned f(S,A)->S_next (a PyTorch model).
    :param alphas: list of AR coefficients [alpha_1, alpha_2, ..., alpha_p].
    :param sigma: stdev for new white noise each step.
    :param beta: scale factor for adding AR noise to the final observation.
    """
    def __init__(self, env, transition_model,
                 dimension_to_noisify=None,
                 alphas=None,
                 sigma=0.5, beta=1.0,
                 device=th.device("cpu")):
        super().__init__(env)
        self.transition_model = transition_model.to(device)
        self.transition_model.eval()
        self.device = device

        self.action_space_dim = (1 if isinstance(env.action_space, gym.spaces.Discrete)
                                 else env.action_space.shape[0])
        self.dim_to_noisify = dimension_to_noisify
        self.sigma = sigma
        self.beta = beta

        if not alphas or len(alphas) == 0:
            # Default to AR(1) with alpha=0.9
            self.alphas = np.array([0.9], dtype=np.float32)
        else:
            self.alphas = np.array(alphas, dtype=np.float32)

        self.p = len(self.alphas)
        logging.info(f"[ARWrapper] AR order={self.p}, alphas={self.alphas}, sigma={sigma}, beta={beta}")

        obs_dim = self.observation_space.shape[0]
        from collections import deque
        self.eta_hist = deque([np.zeros(obs_dim, dtype=np.float32) for _ in range(self.p)],
                              maxlen=self.p)

        self.current_obs = None

    def reset(self, **kwargs):
        underlying_obs, info = self.env.reset(**kwargs)

        obs_dim = self.observation_space.shape[0]
        self.eta_hist = deque([np.zeros(obs_dim, dtype=np.float32) for _ in range(self.p)],
                              maxlen=self.p)
        self.current_obs = underlying_obs
        return self.current_obs, info

    def step(self, action):
        # 1) Step the underlying env => real next obs
        real_next_obs, reward, done, truncated, info = self.env.step(action)

        # 2) Predict next state from transition model
        s_t = th.tensor(self.current_obs, dtype=th.float32, device=self.device).unsqueeze(0)
        if self.action_space_dim == 1 and not isinstance(action, np.ndarray):
            a_t = th.tensor([float(action)], dtype=th.float32, device=self.device).unsqueeze(0)
        else:
            a_t = th.from_numpy(action.astype(np.float32)).unsqueeze(0).to(self.device)

        with th.no_grad():
            predicted_next = self.transition_model(s_t, a_t).cpu().numpy().squeeze(0)

        # 3) Blend real & predicted
        w = 0.2
        blended_next = w * real_next_obs + (1.0 - w) * predicted_next

        # 4) AR update => new_eta = sum_i alphas[i]*past_eta[i] + eps
        eps = np.random.normal(0.0, self.sigma, size=blended_next.shape)
        new_eta = np.zeros_like(self.eta_hist[0], dtype=np.float32)
        for i in range(self.p):
            new_eta += self.alphas[i] * self.eta_hist[self.p - 1 - i]
        new_eta += eps
        self.eta_hist.append(new_eta)

        # 5) final_next_obs = blended_next + beta * new_eta (optionally only in one dimension)
        final_next_obs = blended_next.copy()
        if self.dim_to_noisify is None:
            final_next_obs += self.beta * new_eta
        else:
            final_next_obs[self.dim_to_noisify] += self.beta * new_eta[self.dim_to_noisify]

        self.current_obs = final_next_obs
        return final_next_obs, reward, done, truncated, info


###############################################################################
# (C) Main Class for AR-Noised Experiments
###############################################################################
def _smooth_reward_curve(episodes, rewards, window=10):
    if len(rewards) < window:
        return episodes, rewards
    cumsum = np.cumsum(np.insert(rewards, 0, 0))
    smoothed = (cumsum[window:] - cumsum[:-window]) / float(window)
    return episodes[window - 1:], smoothed


class ARNoisedExperiments:
    """
    1) Train no-noise PPO => gather transitions => train f(S,A)->S_next.
    2) For each dimension => for each AR order => for each parameter set (alphas, sigma, etc.),
       create AR environment, train PPO, run PCMCI => measure Markov violation.
    """
    def __init__(self, config, root_path="."):
        self.config = config
        self.learning_rate = self.config.get("learning_rate", 3e-4)
        self.root_path = root_path
        self.reward_records = []
        self.markov_records = []

    def fishers_method(self, pvals, epsilon=1e-15):
        """
        Combine multiple p-values via Fisher's method (like in other scripts).
        """
        pvals = np.array(pvals, dtype=float)
        pvals = np.clip(pvals, epsilon, 1 - epsilon)
        stat = -2.0 * np.sum(np.log(pvals))
        df = 2 * len(pvals)
        return 1.0 - chi2.cdf(stat, df)

    def gather_and_run_pcmci(
            self, model, env_name, dim_id, alphas, sigma, beta,
            steps=2000, seed=None, trained_transition_model=None, device=th.device("cpu")
    ):
        """
        1) Build environment with AR(...) noise using the trained transition model.
        2) Collect `steps` transitions with the provided `model`.
        3) Run PCMCI => returns (val_matrix, p_matrix).
        """
        def _env_fn():
            e = gym.make(env_name)
            return ARObservationWrapper(
                env=e,
                transition_model=trained_transition_model,
                dimension_to_noisify=dim_id,
                alphas=alphas,
                sigma=sigma,
                beta=beta,
                device=device
            )

        test_env = make_vec_env(_env_fn, n_envs=1, seed=seed)
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

    def run_multiple_pcmci_fisher(self, model, env_name, dim_id, alphas, sigma, beta,
                                  trained_transition_model, device,
                                  num_runs=5, steps=2000):
        """
        Repeats gather_and_run_pcmci multiple times, combining partial correlations
        and p-values via average + Fisher.
        Returns the final Markov violation score.
        """
        val_list = []
        p_list = []

        for _ in range(num_runs):
            seed = random.randint(0, 9999)
            val_m, p_m = self.gather_and_run_pcmci(
                model, env_name, dim_id, alphas, sigma, beta,
                steps=steps, seed=seed,
                trained_transition_model=trained_transition_model,
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

    def run(self, env_name=None, baseline_seed=None):
        """
        Orchestrates for each environment:
         - Train no-noise policy + gather transitions => train f(S,A)->S_next
         - For each dimension, for each AR order, for each param set => train PPO & measure Markov score
        """
        envs = self.config["environments"]
        if env_name:
            envs = [e for e in envs if e["name"] == env_name]
        if not envs:
            logging.warning(f"[ARNoised] No matching environment for env={env_name}.")
            return

        ar_noise_config = self.config["noise_strategies"].get("auto_regressive", {})
        if not ar_noise_config:
            logging.warning("[ARNoised] 'auto_regressive' noise config not found.")
            return

        device = th.device("cuda" if th.cuda.is_available() else "cpu")

        for env_item in envs:
            name = env_item["name"]
            time_steps = env_item["time_steps"]
            obs_names = env_item.get("observations", [])
            obs_dim = len(obs_names)

            samples_for_fit = env_item.get("samples_for_fit", 10000)
            epochs_for_fit = env_item.get("epochs_for_fit", 20)

            env_path = os.path.join(self.root_path, "results", name)
            noised_path = os.path.join(env_path, "noised_ar")
            pcmci_path = os.path.join(noised_path, "pcmci")
            os.makedirs(noised_path, exist_ok=True)
            os.makedirs(pcmci_path, exist_ok=True)

            logging.info(f"[ARNoised] => Env={name}, obs_dim={obs_dim}")

            # Step A) Train a no-noise PPO
            used_seed = baseline_seed if baseline_seed is not None else random.randint(0, 9999)
            no_noise_model = train_noiseless_policy(name, total_timesteps=time_steps)

            # Step B) Gather transitions => train f(S,A)->S_next
            states, actions, next_states = gather_transition_data(no_noise_model, name, n_samples=samples_for_fit)
            transition_model, losses = train_transition_model(
                states, actions, next_states,
                epochs=epochs_for_fit, batch_size=64, lr=1e-3
            )
            transition_model = transition_model.to(device)

            # Plot the training loss for f(S,A)
            plt.figure()
            plt.plot(losses, marker='o')
            plt.title(f"{name} - f(S,A) Training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("MSE")
            plt.grid(True)
            plt.tight_layout()
            fsa_loss_png = os.path.join(env_path, f"{name}_transition_loss.png")
            plt.savefig(fsa_loss_png, dpi=150)
            plt.close()
            logging.info(f"[ARNoised] Transition MSE plot => {fsa_loss_png}")

            # Step C) For each dimension & AR order => create AR wrapper => train PPO => measure Markov
            for dim_id in range(obs_dim):
                obs_dim_name = obs_names[dim_id]

                for ar_order_key, ar_param_list in ar_noise_config.items():
                    # e.g. "ar1", "ar2", "ar3"...
                    for ar_params in ar_param_list:
                        alphas = np.array(ar_params.get("alphas", [0.9]), dtype=np.float32)
                        sigma = ar_params.get("sigma", 0.5)
                        beta = 1.0  # or from config
                        desc = ar_params.get("description", "")

                        logging.info(f"[ARNoised] {ar_order_key} => env={name}, dim={dim_id}, alphas={alphas}, sigma={sigma}, desc={desc}")

                        # Build environment
                        def _env_builder():
                            e = gym.make(name)
                            wrapper = ARObservationWrapper(
                                env=e,
                                transition_model=transition_model,
                                dimension_to_noisify=dim_id,
                                alphas=alphas,
                                sigma=sigma,
                                beta=beta,
                                device=device
                            )
                            return wrapper

                        venv = make_vec_env(_env_builder, n_envs=1, seed=used_seed)
                        model = PPO("MlpPolicy", venv, verbose=0, learning_rate=self.learning_rate)
                        callback = RewardTrackingCallback()
                        model.learn(total_timesteps=time_steps, callback=callback)
                        ep_rewards = callback.get_rewards()
                        venv.close()

                        # Store reward curve
                        alphas_str = str(alphas.tolist())
                        for i, rew in enumerate(ep_rewards):
                            self.reward_records.append({
                                "Environment": name,
                                "ObsDim": dim_id,
                                "ObsName": obs_dim_name,
                                "AR_order": ar_order_key,
                                "AR_alphas": alphas_str,
                                "AR_sigma": sigma,
                                "Beta": beta,
                                "Episode": i + 1,
                                "Reward": rew
                            })

                        # Markov analysis
                        NUM_PCMCI_RUNS = 3
                        mk_score = self.run_multiple_pcmci_fisher(
                            model=model,
                            env_name=name,
                            dim_id=dim_id,
                            alphas=alphas,
                            sigma=sigma,
                            beta=beta,
                            trained_transition_model=transition_model,
                            device=device,
                            num_runs=NUM_PCMCI_RUNS,
                            steps=2000,
                        )
                        logging.info(f"[ARNoised] Markov Score => {mk_score:.4f}")

                        final_reward = (np.mean(ep_rewards[-10:])
                                        if len(ep_rewards) >= 10 else np.mean(ep_rewards))
                        self.markov_records.append({
                            "Environment": name,
                            "ObsDim": dim_id,
                            "ObsName": obs_dim_name,
                            "AR_order": ar_order_key,
                            "AR_alphas": alphas_str,
                            "AR_sigma": sigma,
                            "Beta": beta,
                            "MarkovScore": mk_score,
                            "MeanFinalReward": final_reward
                        })

            # After finishing this environment, save CSV & produce any plots
            env_rewards = [r for r in self.reward_records if r["Environment"] == name]
            env_markov = [m for m in self.markov_records if m["Environment"] == name]

            df_rewards = pd.DataFrame(env_rewards)
            df_markov = pd.DataFrame(env_markov)

            reward_csv = os.path.join(noised_path, "ar_noised_rewards.csv")
            markov_csv = os.path.join(noised_path, "ar_noised_markov.csv")
            df_rewards.to_csv(reward_csv, index=False)
            df_markov.to_csv(markov_csv, index=False)
            logging.info(f"[ARNoised] Wrote AR noised rewards => {reward_csv}")
            logging.info(f"[ARNoised] Wrote AR noised Markov => {markov_csv}")

            #####################################################################
            # ADDED PLOT FUNCTIONS + CALLS (following other scripts' examples)
            #####################################################################

            # 1) Plot AR learning curves, grouped by AR_order & (optionally) AR_sigma
            baseline_csv_path = os.path.join(env_path, "csv", "baseline_learning_curve.csv")
            plot_ar_learning_curves_all_orders(
                df_rewards,
                baseline_csv=baseline_csv_path,
                output_dir=noised_path,
                smooth_window=10
            )

            # 2) Plot Rewards vs Markov Score
            plot_ar_rewards_vs_markov(df_markov, output_dir=noised_path)

            # 3) Plot AR sigma vs Markov correlation
            plot_ar_sigma_vs_markov_corr(df_markov, output_dir=noised_path)

###############################################################################
# (D) NEW PLOT FUNCTIONS FOR AR NOISE
###############################################################################

def plot_ar_learning_curves_all_orders(
    df_rewards: pd.DataFrame,
    baseline_csv: str = None,
    output_dir: str = ".",
    smooth_window: int = 10
):
    """
    Similar to the noised_gaussian approach:
      For each AR_order, produce a figure. On each figure, we can group by (AR_sigma, AR_alphas)
      or produce sub-lines. We'll do a single figure for each unique (AR_order, AR_sigma),
      overlaying lines for each dimension + alpha combo.

    This ensures all AR variations get a plot, plus optional baseline in black.
    """
    os.makedirs(output_dir, exist_ok=True)
    if df_rewards.empty:
        logging.info("[ARNoisedPlots] No data to plot for AR noise.")
        return

    env_name = df_rewards["Environment"].iloc[0]

    # Attempt baseline overlay
    baseline_df = None
    if baseline_csv and os.path.isfile(baseline_csv):
        bdf = pd.read_csv(baseline_csv)
        bdf = bdf[bdf["Environment"] == env_name]
        if not bdf.empty:
            baseline_df = bdf

    # Group by AR_order => produce separate plots
    for ar_order, df_order in df_rewards.groupby("AR_order"):
        # Also group by AR_sigma so we produce different figures for each sigma
        for sigma_val, df_sigma in df_order.groupby("AR_sigma"):
            # We will produce lines for each (dim_id, alpha_str).
            plt.figure(figsize=(8, 5))
            # Sort so lines are consistent
            df_sigma = df_sigma.sort_values(["ObsDim", "Episode"])
            # Group lines by dimension + AR_alphas
            for (dim_id, alpha_str), df_line in df_sigma.groupby(["ObsDim", "AR_alphas"]):
                episodes = df_line["Episode"].values
                rewards = df_line["Reward"].values
                x_smooth, y_smooth = _smooth_reward_curve(episodes, rewards, window=smooth_window)
                label_str = f"Dim={dim_id}, alphas={alpha_str}"
                plt.plot(x_smooth, y_smooth, label=label_str)

            # Overlay baseline if found
            if baseline_df is not None:
                base_sorted = baseline_df.sort_values("Episode")
                bx = base_sorted["Episode"].values
                by = base_sorted["TotalReward"].values
                bx_smooth, by_smooth = _smooth_reward_curve(bx, by, window=smooth_window)
                plt.plot(bx_smooth, by_smooth, color="black", linewidth=3, label="Baseline")

            plt.title(f"{env_name} - {ar_order}, sigma={sigma_val}")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.legend()
            plt.grid(True)

            out_fname = f"{env_name}_{ar_order}_sigma_{sigma_val}_learning_curves.png"
            out_path = os.path.join(output_dir, out_fname)
            plt.savefig(out_path, dpi=150)
            plt.close()
            logging.info(f"[ARNoisedPlots] Saved AR learning curves => {out_path}")


def plot_ar_rewards_vs_markov(df_markov: pd.DataFrame, output_dir="."):
    """
    Plot: X-axis=MarkovScore, Y-axis=MeanFinalReward, a line or scatter for each dimension or AR_order.
    We'll group by AR_order (like "ar1", "ar2") and then plot dimension with different markers.
    """
    os.makedirs(output_dir, exist_ok=True)
    if df_markov.empty:
        return

    envs = df_markov["Environment"].unique()
    for env_name in envs:
        env_df = df_markov[df_markov["Environment"] == env_name]
        if env_df.empty:
            continue

        # For each AR_order => produce a figure
        for ar_order, gdf_order in env_df.groupby("AR_order"):
            fig, ax = plt.subplots(figsize=(6, 5))

            # sort by MarkovScore so lines are consistent
            gdf_order = gdf_order.sort_values("MarkovScore")

            # We'll group by dimension to get separate lines
            for dim_id, gdf_dim in gdf_order.groupby("ObsDim"):
                ax.plot(
                    gdf_dim["MarkovScore"],
                    gdf_dim["MeanFinalReward"],
                    marker="o",
                    label=f"dim={dim_id}"
                )
            ax.set_title(f"{env_name} - Rewards vs Markov (Order={ar_order})")
            ax.set_xlabel("MarkovScore")
            ax.set_ylabel("MeanFinalReward")
            ax.legend()
            ax.grid(True)
            outpath = os.path.join(output_dir, f"{env_name}_{ar_order}_rewards_vs_markov.png")
            fig.savefig(outpath, dpi=150)
            plt.close(fig)
            logging.info(f"[ARNoisedPlots] Saved => {outpath}")


def plot_ar_sigma_vs_markov_corr(df_markov: pd.DataFrame, output_dir="."):
    """
    Plot: X-axis=AR_sigma, Y-axis=MarkovScore, line for each dimension or alpha.
    We'll group by AR_order as well.
    This is analogous to the "noise vs markov corr" from noised_gaussian.
    """
    os.makedirs(output_dir, exist_ok=True)
    if df_markov.empty:
        return

    envs = df_markov["Environment"].unique()
    for env_name in envs:
        env_df = df_markov[df_markov["Environment"] == env_name]
        if env_df.empty:
            continue

        # For each AR_order => produce a figure
        for ar_order, gdf_order in env_df.groupby("AR_order"):
            fig, ax = plt.subplots(figsize=(7, 5))

            # Sort by AR_sigma so lines look nice
            gdf_order = gdf_order.sort_values("AR_sigma")

            # group by dimension + AR_alphas to label them distinctly
            for (dim_id, alpha_str), gdf_subset in gdf_order.groupby(["ObsDim", "AR_alphas"]):
                ax.plot(
                    gdf_subset["AR_sigma"],
                    gdf_subset["MarkovScore"],
                    marker="o",
                    label=f"dim={dim_id}, alpha={alpha_str}"
                )

            # Optional correlation across entire subset
            if len(gdf_order) >= 2:
                corr_all = gdf_order[["AR_sigma", "MarkovScore"]].corr().iloc[0,1]
            else:
                corr_all = float('nan')

            ax.set_title(f"{env_name} - AR_sigma vs Markov (Order={ar_order}, corr={corr_all:.2f})")
            ax.set_xlabel("AR_sigma")
            ax.set_ylabel("MarkovScore")
            ax.legend()
            ax.grid(True)
            outpath = os.path.join(output_dir, f"{env_name}_{ar_order}_sigma_vs_markov_corr.png")
            fig.savefig(outpath, dpi=150)
            plt.close(fig)
            logging.info(f"[ARNoisedPlots] Saved => {outpath}")


###############################################################################
# (D) Entry point function
###############################################################################
def run_noised_auto_regressive(config_path="config.json", env_name=None, baseline_seed=None):
    if not os.path.exists(config_path):
        logging.error(f"[ARNoised] Config file '{config_path}' not found.")
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    root_path = os.path.abspath(".")
    runner = ARNoisedExperiments(config, root_path=root_path)

    start_t = time.perf_counter()
    runner.run(env_name=env_name, baseline_seed=baseline_seed)
    end_t = time.perf_counter()
    logging.info(f"[ARNoised] Done! Total time: {(end_t - start_t)/60:.2f} min")


def main():
    """
    CLI usage:
      python noised_auto_regressive.py --config_path config.json --env CartPole-v1
    """
    parser = argparse.ArgumentParser(description="Auto-Regressive (AR) Noised Observations + Learned f(S,A)")
    parser.add_argument("--config_path", type=str, default="config.json",
                        help="Path to the config JSON file.")
    parser.add_argument("--env", type=str, default=None,
                        help="Name of the environment to use from config.json.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional baseline seed.")
    args = parser.parse_args()

    run_noised_auto_regressive(
        config_path=args.config_path,
        env_name=args.env,
        baseline_seed=args.seed
    )


if __name__ == "__main__":
    main()