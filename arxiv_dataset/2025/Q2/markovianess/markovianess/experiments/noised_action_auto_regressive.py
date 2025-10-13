# File: noised_action_auto_regressive.py
"""
noised_action_auto_regressive.py
--------------------------------
Runs experiments by adding auto-regressive (AR) noise to each *action* dimension
(or using a probability-shift approach for discrete spaces).

Overview of Steps:
------------------
1) Train a "no-noise" PPO policy on the given environment.
2) Gather transitions (S,A,S_next) using that policy -> train a neural network
   f(S,A)->S_next. (Kept for consistency, though not strictly needed for action noise.)
3) For each dimension:
   - Wrap the environment in an ARActionWrapper that injects AR noise into the
     chosen dimension if continuous, or uses probability shift if discrete.
   - Train a new PPO policy on that AR-noised environment; record episode rewards.
   - Run PCMCI multiple times to measure Markov violation (averaged or combined
     via Fisher's method).
4) Save results to CSV and produce the same plots as noised_auto_regressive.py,
   but focusing on how AR noise in the *action* affects performance and Markov score.
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
    Creates a standard PPO model (no modifications) on env_name, trains it,
    and returns the trained model.
    """
    logging.info(f"[ARAction] Training no-noise PPO on {env_name} for {total_timesteps} steps.")
    vec_env = make_vec_env(env_name, n_envs=1)
    model = PPO("MlpPolicy", vec_env, verbose=0)
    model.learn(total_timesteps=total_timesteps)
    vec_env.close()
    return model


def gather_transition_data(model, env_name, n_samples=10000):
    """
    Runs the environment with the trained 'model' to collect (S,A,S_next) transitions.
    Even though it's not strictly needed for action-noise, we keep it for consistency.

    Returns:
      states: shape=(n_samples, state_dim)
      actions: shape=(n_samples, action_dim or 1 if discrete)
      next_states: shape=(n_samples, state_dim)
    """
    import gymnasium as gym
    env = gym.make(env_name)
    obs, _ = env.reset()

    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    states, actions, next_states = [], [], []

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

    logging.info(
        f"[ARAction] Collected transitions: states={states.shape}, actions={actions.shape}, "
        f"next_states={next_states.shape}"
    )
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
    Returns the trained model and a list of MSE losses per epoch.
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
        logging.info(f"[ARAction] TransitionModel Epoch {ep+1}/{epochs}, MSE={avg_loss:.6f}")

    return model, losses_per_epoch


###############################################################################
# (B) AR Action Wrapper
###############################################################################
class ARActionWrapper(gym.Wrapper):
    """
    Gym Wrapper that injects an auto-regressive noise vector or scalar into the *action*.

    - If action space is continuous (Box), we do the original approach:
        a_final = a_policy + beta * eta_t  (clipped to action bounds).
      If dimension_to_noisify is set, we only apply AR noise to that dimension.

    - If action space is discrete, we interpret the AR noise as a "probability shift":
        1) We keep only a single AR dimension (since the action is a single integer).
        2) We compute new_eta = sum_{i=1 to p} alpha_i * eta_{t-i} + eps_t, eps ~ N(0, sigma^2).
        3) prob = logistic( beta * new_eta[0] )  (a scalar in [0,1])
        4) with probability=prob, we keep the policy's chosen action;
           with probability=(1-prob), we pick a random "other" action from the discrete set.

    This is an approximate way to inject "noise" into discrete actions.

    Steps for discrete:
      - If n_actions=2, it effectively flips the action with probability (1-prob).
      - If n_actions>2, it picks uniformly among the other (n_actions-1) with probability (1-prob).

    By default, if dimension_to_noisify is None and the space is discrete, we treat everything
    as "1D AR noise," ignoring dimension_to_noisify.
    """

    def __init__(self, env, dimension_to_noisify=None,
                 alphas=None, sigma=0.5, beta=1.0):
        super().__init__(env)
        self.env = env
        self.alphas = np.array(alphas if alphas else [0.9], dtype=np.float32)
        self.sigma = sigma
        self.beta = beta
        self.p = len(self.alphas)

        self.is_continuous = isinstance(env.action_space, gym.spaces.Box)
        if self.is_continuous:
            # We do standard approach with AR noise on a vector
            self.action_dim = env.action_space.shape[0]
            self.dim_to_noisify = dimension_to_noisify
            if dimension_to_noisify is not None and (
                dimension_to_noisify < 0 or dimension_to_noisify >= self.action_dim
            ):
                raise ValueError(
                    f"Invalid dimension_to_noisify={dimension_to_noisify} "
                    f"for action_dim={self.action_dim}"
                )
            # Keep a history of AR states, shape=(action_dim,)
            self.eta_hist = deque(
                [np.zeros(self.action_dim, dtype=np.float32) for _ in range(self.p)],
                maxlen=self.p
            )
            logging.info(f"[ARAction] (Continuous) AR order={self.p}, alphas={self.alphas}, sigma={sigma}, beta={beta}")

        else:
            # Discrete => single AR dimension
            self.action_dim = env.action_space.n  # number of discrete actions
            self.dim_to_noisify = None  # not used for discrete
            # Keep a single scalar for AR noise
            self.eta_hist = deque(
                [np.zeros(1, dtype=np.float32) for _ in range(self.p)],
                maxlen=self.p
            )
            logging.info(f"[ARAction] (Discrete) AR order={self.p}, alphas={self.alphas}, sigma={sigma}, beta={beta}")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # reset AR hist
        if self.is_continuous:
            self.eta_hist = deque(
                [np.zeros(self.action_dim, dtype=np.float32) for _ in range(self.p)],
                maxlen=self.p
            )
        else:
            self.eta_hist = deque(
                [np.zeros(1, dtype=np.float32) for _ in range(self.p)],
                maxlen=self.p
            )
        return obs, info

    def step(self, action):
        if self.is_continuous:
            final_act = self._apply_continuous_ar_noise(action)
            return self.env.step(final_act)
        else:
            final_discrete_act = self._apply_discrete_prob_shift(action)
            return self.env.step(final_discrete_act)

    def _apply_continuous_ar_noise(self, policy_action):
        """
        For continuous action space:
          - new_eta = sum alpha_i * past_eta[i] + eps
          - if dimension_to_noisify is None => add to all dims
          - else only to that dimension
          - clip to env bounds
        """
        new_eta = np.zeros(self.action_dim, dtype=np.float32)
        for i in range(self.p):
            new_eta += self.alphas[i] * self.eta_hist[self.p - 1 - i]

        eps = np.random.normal(0.0, self.sigma, size=self.action_dim)
        new_eta += eps

        self.eta_hist.append(new_eta)

        final_act = np.array(policy_action, copy=True, dtype=np.float32)
        if self.dim_to_noisify is None:
            final_act += self.beta * new_eta
        else:
            final_act[self.dim_to_noisify] += self.beta * new_eta[self.dim_to_noisify]

        # clip
        low, high = self.env.action_space.low, self.env.action_space.high
        final_act = np.clip(final_act, low, high)
        return final_act

    def _apply_discrete_prob_shift(self, policy_action):
        """
        For discrete spaces:
          - We keep a single scalar AR noise.
          - new_eta = sum alpha_i * past_eta[i] + eps
          - Convert to probability: p = logistic(beta * new_eta[0])
          - with prob p => keep policy_action
            with prob (1-p) => choose a random different action among the available
        """
        # 1) build new_eta
        new_eta = 0.0
        for i in range(self.p):
            new_eta += self.alphas[i] * self.eta_hist[self.p - 1 - i][0]

        eps = np.random.normal(0.0, self.sigma)
        new_eta += eps
        self.eta_hist.append(np.array([new_eta], dtype=np.float32))

        # 2) logistic => probability of "keep"
        x = self.beta * new_eta
        p_keep = 1.0 / (1.0 + np.exp(-x))

        # 3) sample random uniform in [0,1]
        r = np.random.rand()
        if r <= p_keep:
            final_action = int(policy_action)  # keep original
        else:
            # pick among the other (n-1) actions
            all_acts = list(range(self.action_dim))
            current_act = int(policy_action)
            all_acts.remove(current_act)
            final_action = np.random.choice(all_acts)

        return final_action


###############################################################################
# (C) Main Class for AR Noised Action Experiments
###############################################################################
def _smooth_reward_curve(episodes, rewards, window=10):
    if len(rewards) < window:
        return episodes, rewards
    cumsum = np.cumsum(np.insert(rewards, 0, 0))
    smoothed = (cumsum[window:] - cumsum[:-window]) / float(window)
    return episodes[window - 1:], smoothed


class ARNoisedActionExperiments:
    """
    1) Train no-noise PPO => gather transitions => train f(S,A)->S_next
       (kept for structural consistency).
    2) For each dimension in the action space => for each AR param set
       => create ARActionWrapper => train PPO => measure Markov violation via PCMCI.

    Now handles BOTH continuous (Box) action spaces (additive AR noise)
    AND discrete action spaces (probability-shift approach).
    """

    def __init__(self, config, root_path="."):
        self.config = config
        self.root_path = root_path
        self.learning_rate = self.config.get("learning_rate", 3e-4)
        self.reward_records = []
        self.markov_records = []

    def fishers_method(self, pvals, epsilon=1e-15):
        """
        Combine multiple p-values via Fisher's method.
        """
        pvals = np.array(pvals, dtype=float)
        pvals = np.clip(pvals, epsilon, 1 - epsilon)
        stat = -2.0 * np.sum(np.log(pvals))
        df = 2 * len(pvals)
        return 1.0 - chi2.cdf(stat, df)

    def gather_and_run_pcmci(
            self, model, env_name, dim_id, alphas, sigma, beta,
            steps=2000, seed=None
    ):
        """
        Create an environment with AR noise on the chosen action dimension
        (continuous => additive, discrete => probability shift).
        Then gather steps, run PCMCI => returns (val_matrix, p_matrix).
        """
        def _env_fn():
            e = gym.make(env_name)
            return ARActionWrapper(
                env=e,
                dimension_to_noisify=dim_id,
                alphas=alphas,
                sigma=sigma,
                beta=beta
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

        # run PCMCI
        cit = ConditionalIndependenceTest()
        results_dict = cit.run_pcmci(
            observations=obs_array,
            tau_max=5,
            alpha_level=0.05
        )
        return results_dict["val_matrix"], results_dict["p_matrix"]

    def run_multiple_pcmci_fisher(self, model, env_name, dim_id, alphas, sigma, beta,
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
                model, env_name, dim_id,
                alphas=alphas,
                sigma=sigma,
                beta=beta,
                steps=steps,
                seed=seed
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
         1) Train no-noise PPO + gather transitions => train f(S,A)->S_next
         2) For each dimension in the action space => for each AR param => train PPO & measure Markov
            - If continuous => additive noise
            - If discrete => probability shift
        """
        envs = self.config["environments"]
        if env_name:
            envs = [e for e in envs if e["name"] == env_name]
        if not envs:
            logging.warning(f"[ARAction] No matching environment for env={env_name}.")
            return

        ar_noise_config = self.config["noise_strategies"].get("auto_regressive", {})
        if not ar_noise_config:
            logging.warning("[ARAction] 'auto_regressive' noise config not found.")
            return

        for env_item in envs:
            name = env_item["name"]
            time_steps = env_item["time_steps"]
            obs_names = env_item.get("observations", [])
            samples_for_fit = env_item.get("samples_for_fit", 10000)
            epochs_for_fit = env_item.get("epochs_for_fit", 20)

            env_path = os.path.join(self.root_path, "results", name)
            noised_path = os.path.join(env_path, "noised_ar_action")
            os.makedirs(noised_path, exist_ok=True)

            logging.info(f"[ARAction] => Env={name}")

            # Step A: Train no-noise PPO
            used_seed = baseline_seed if baseline_seed is not None else random.randint(0, 9999)
            no_noise_model = train_noiseless_policy(name, total_timesteps=time_steps)

            # Step B: Gather transitions => train f(S,A)->S_next
            states, actions, next_states = gather_transition_data(
                no_noise_model,
                name,
                n_samples=samples_for_fit
            )
            transition_model, losses = train_transition_model(
                states, actions, next_states,
                epochs=epochs_for_fit,
                batch_size=64,
                lr=1e-3
            )

            # Plot the training loss for f(S,A)
            plt.figure()
            plt.plot(losses, marker='o')
            plt.title(f"{name} - f(S,A) Training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("MSE")
            plt.grid(True)
            plt.tight_layout()
            loss_png = os.path.join(env_path, f"{name}_action_transition_loss.png")
            plt.savefig(loss_png, dpi=150)
            plt.close()
            logging.info(f"[ARAction] Transition MSE plot => {loss_png}")

            # Step C: Determine if action space is discrete or continuous
            dummy_env = gym.make(name)
            action_space = dummy_env.action_space
            dummy_env.close()

            if isinstance(action_space, gym.spaces.Discrete):
                action_dim = action_space.n
                logging.info(f"[ARAction] Environment {name} has discrete action space => Probability shift approach.")
            elif isinstance(action_space, gym.spaces.Box):
                action_dim = action_space.shape[0]
                logging.info(f"[ARAction] Environment {name} has continuous action space => Additive noise approach.")
            else:
                logging.warning(f"[ARAction] Unrecognized action space type. Skipping {name}.")
                continue

            # Step D: For each AR param => for each dimension in [0..(action_dim-1)]
            for ar_order_key, ar_param_list in ar_noise_config.items():
                for ar_params in ar_param_list:
                    alphas = np.array(ar_params.get("alphas", [0.9]), dtype=np.float32)
                    sigma = ar_params.get("sigma", 0.5)
                    beta = 1.0  # or from config
                    desc = ar_params.get("description", "")

                    # For each dimension if continuous,
                    # If discrete, dimension_to_noisify basically selects which discrete dimension?
                    # But for discrete we interpret dim_id as "which discrete dimension to apply prob shift."
                    # Typically you'd do 0 if 2 actions, but let's allow multiple dims if the user wants to approach multi-binary.
                    # We'll just do [0..(action_dim-1)] for consistency.
                    for dim_id in range(action_dim):
                        logging.info(
                            f"[ARAction] {ar_order_key} => env={name}, action_dim={dim_id}, "
                            f"alphas={alphas}, sigma={sigma}, desc={desc}"
                        )

                        from stable_baselines3.common.env_util import make_vec_env

                        def _env_builder():
                            e = gym.make(name)
                            return ARActionWrapper(
                                env=e,
                                dimension_to_noisify=dim_id,  # relevant for continuous; discrete uses it as an "index"
                                alphas=alphas,
                                sigma=sigma,
                                beta=beta
                            )

                        venv = make_vec_env(_env_builder, n_envs=1, seed=used_seed)
                        model = PPO("MlpPolicy", venv, verbose=0, learning_rate=self.learning_rate)
                        callback = RewardTrackingCallback()
                        model.learn(total_timesteps=time_steps, callback=callback)
                        ep_rewards = callback.get_rewards()
                        venv.close()

                        # Record learning curve
                        for i, rew_val in enumerate(ep_rewards):
                            self.reward_records.append({
                                "Environment": name,
                                "ActionDim": dim_id,
                                "AR_order": ar_order_key,
                                "AR_alphas": str(list(alphas)),
                                "AR_sigma": sigma,
                                "Beta": beta,
                                "Episode": i + 1,
                                "Reward": rew_val
                            })

                        # Markov analysis with multiple PCMCI runs
                        NUM_PCMCI_RUNS = 3
                        mk_score = self.run_multiple_pcmci_fisher(
                            model=model,
                            env_name=name,
                            dim_id=dim_id,
                            alphas=alphas,
                            sigma=sigma,
                            beta=beta,
                            num_runs=NUM_PCMCI_RUNS,
                            steps=2000
                        )
                        logging.info(f"[ARAction] Markov Score => {mk_score:.4f}")

                        final_reward = (
                            np.mean(ep_rewards[-10:])
                            if len(ep_rewards) >= 10
                            else np.mean(ep_rewards)
                        )
                        self.markov_records.append({
                            "Environment": name,
                            "ActionDim": dim_id,
                            "AR_order": ar_order_key,
                            "AR_alphas": str(list(alphas)),
                            "AR_sigma": sigma,
                            "Beta": beta,
                            "MarkovScore": mk_score,
                            "MeanFinalReward": final_reward
                        })

            # After finishing this environment, save CSV & produce the same plots
            env_rewards = [r for r in self.reward_records if r["Environment"] == name]
            env_markov = [m for m in self.markov_records if m["Environment"] == name]

            df_rewards = pd.DataFrame(env_rewards)
            df_markov = pd.DataFrame(env_markov)

            reward_csv = os.path.join(noised_path, "ar_noised_action_rewards.csv")
            markov_csv = os.path.join(noised_path, "ar_noised_action_markov.csv")
            df_rewards.to_csv(reward_csv, index=False)
            df_markov.to_csv(markov_csv, index=False)
            logging.info(f"[ARAction] Wrote AR noised action rewards => {reward_csv}")
            logging.info(f"[ARAction] Wrote AR noised action Markov => {markov_csv}")

            # Now produce the same plots used for the AR action approach:
            baseline_csv_path = os.path.join(env_path, "csv", "baseline_learning_curve.csv")
            plot_ar_action_learning_curves_all_orders(
                df_rewards,
                baseline_csv=baseline_csv_path,
                output_dir=noised_path,
                smooth_window=10
            )
            plot_ar_action_rewards_vs_markov(df_markov, output_dir=noised_path)
            plot_ar_action_sigma_vs_markov_corr(df_markov, output_dir=noised_path)


###############################################################################
# (D) Plot Functions (reused from your AR approach, but keyed to "ActionDim")
###############################################################################
def plot_ar_action_learning_curves_all_orders(
    df_rewards: pd.DataFrame,
    baseline_csv: str = None,
    output_dir: str = ".",
    smooth_window: int = 10
):
    """
    For each AR_order, produce a figure, grouping by AR_sigma.
    Then each line is (ActionDim, AR_alphas).
    We optionally overlay baseline if found.
    """
    os.makedirs(output_dir, exist_ok=True)
    if df_rewards.empty:
        logging.info("[ARActionPlots] No data to plot for AR action noise.")
        return

    env_name = df_rewards["Environment"].iloc[0]

    # Attempt to load baseline
    baseline_df = None
    if baseline_csv and os.path.isfile(baseline_csv):
        bdf = pd.read_csv(baseline_csv)
        bdf = bdf[bdf["Environment"] == env_name]
        if not bdf.empty:
            baseline_df = bdf

    # Group by AR_order => produce separate plots per AR_order & AR_sigma
    for ar_order, df_order in df_rewards.groupby("AR_order"):
        for sigma_val, df_sigma in df_order.groupby("AR_sigma"):
            plt.figure(figsize=(8, 5))
            df_sigma = df_sigma.sort_values(["ActionDim", "Episode"])

            # lines for (ActionDim, AR_alphas)
            for (dim_id, alpha_str), df_line in df_sigma.groupby(["ActionDim", "AR_alphas"]):
                episodes = df_line["Episode"].values
                rewards = df_line["Reward"].values
                x_smooth, y_smooth = _smooth_reward_curve(episodes, rewards, window=smooth_window)
                label_str = f"ActionDim={dim_id}, alphas={alpha_str}"
                plt.plot(x_smooth, y_smooth, label=label_str)

            # overlay baseline if available
            if baseline_df is not None:
                base_sorted = baseline_df.sort_values("Episode")
                bx = base_sorted["Episode"].values
                by = base_sorted["TotalReward"].values
                bx_smooth, by_smooth = _smooth_reward_curve(bx, by, window=smooth_window)
                plt.plot(bx_smooth, by_smooth, color="black", linewidth=3, label="Baseline")

            plt.title(f"{env_name} - AR_action {ar_order}, sigma={sigma_val}")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.legend()
            plt.grid(True)

            out_fname = f"{env_name}_ARaction_{ar_order}_sigma_{sigma_val}_learning_curves.png"
            out_path = os.path.join(output_dir, out_fname)
            plt.savefig(out_path, dpi=150)
            plt.close()
            logging.info(f"[ARActionPlots] Saved learning curves => {out_path}")


def plot_ar_action_rewards_vs_markov(df_markov: pd.DataFrame, output_dir="."):
    """
    Plot: X-axis=MarkovScore, Y-axis=MeanFinalReward,
    separate lines or points for each AR_order + ActionDim.
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
            gdf_order = gdf_order.sort_values("MarkovScore")

            # group by dimension => separate lines
            for dim_id, gdf_dim in gdf_order.groupby("ActionDim"):
                ax.plot(
                    gdf_dim["MarkovScore"],
                    gdf_dim["MeanFinalReward"],
                    marker="o",
                    label=f"ActionDim={dim_id}"
                )
            ax.set_title(f"{env_name} - AR_action {ar_order}: Rewards vs MarkovScore")
            ax.set_xlabel("MarkovScore")
            ax.set_ylabel("MeanFinalReward")
            ax.legend()
            ax.grid(True)
            outpath = os.path.join(output_dir, f"{env_name}_ARaction_{ar_order}_rewards_vs_markov.png")
            fig.savefig(outpath, dpi=150)
            plt.close(fig)
            logging.info(f"[ARActionPlots] Saved => {outpath}")


def plot_ar_action_sigma_vs_markov_corr(df_markov: pd.DataFrame, output_dir="."):
    """
    Plot: X-axis=AR_sigma, Y-axis=MarkovScore.
    Group by AR_order => produce separate figure.
    Lines for each (ActionDim, AR_alphas).
    Also compute correlation across entire subset if desired.
    """
    os.makedirs(output_dir, exist_ok=True)
    if df_markov.empty:
        return

    envs = df_markov["Environment"].unique()
    for env_name in envs:
        env_df = df_markov[df_markov["Environment"] == env_name]
        if env_df.empty:
            continue

        # For each AR_order => figure
        for ar_order, gdf_order in env_df.groupby("AR_order"):
            fig, ax = plt.subplots(figsize=(7, 5))
            gdf_order = gdf_order.sort_values("AR_sigma")

            for (dim_id, alpha_str), gdf_subset in gdf_order.groupby(["ActionDim", "AR_alphas"]):
                ax.plot(
                    gdf_subset["AR_sigma"],
                    gdf_subset["MarkovScore"],
                    marker="o",
                    label=f"ActionDim={dim_id}, alpha={alpha_str}"
                )

            if len(gdf_order) >= 2:
                corr_all = gdf_order[["AR_sigma", "MarkovScore"]].corr().iloc[0,1]
            else:
                corr_all = float('nan')

            ax.set_title(f"{env_name} - AR_action {ar_order}, sigma vs Markov (corr={corr_all:.2f})")
            ax.set_xlabel("AR_sigma")
            ax.set_ylabel("MarkovScore")
            ax.legend()
            ax.grid(True)
            outpath = os.path.join(output_dir, f"{env_name}_ARaction_{ar_order}_sigma_vs_markov_corr.png")
            fig.savefig(outpath, dpi=150)
            plt.close(fig)
            logging.info(f"[ARActionPlots] Saved => {outpath}")


###############################################################################
# (E) Entry point function
###############################################################################
def run_noised_action_auto_regressive(config_path="config.json", env_name=None, baseline_seed=None):
    """
    Main top-level function. Reads config, sets up ARNoisedActionExperiments,
    and runs the pipeline for the specified environment(s).

    config_path : str
        Path to the config JSON file, which includes a "noise_strategies" section
        with "auto_regressive" details.
    env_name : Optional[str]
        If provided, only runs for that environment. Otherwise runs for all in config.
    baseline_seed : Optional[int]
        If provided, reuses this seed for the "no-noise" baseline.
    """
    if not os.path.exists(config_path):
        logging.error(f"[ARAction] Config file '{config_path}' not found.")
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    root_path = os.path.abspath(".")
    runner = ARNoisedActionExperiments(config, root_path=root_path)

    start_t = time.perf_counter()
    runner.run(env_name=env_name, baseline_seed=baseline_seed)
    end_t = time.perf_counter()
    logging.info(f"[ARAction] Done! Total time: {(end_t - start_t)/60:.2f} min")


def main():
    """
    CLI usage:
      python noised_action_auto_regressive.py --config_path config.json --env Pendulum-v1
    """
    parser = argparse.ArgumentParser(description="Auto-Regressive Action Noise Experiments (Discrete or Continuous)")
    parser.add_argument("--config_path", type=str, default="config.json",
                        help="Path to the config JSON file.")
    parser.add_argument("--env", type=str, default=None,
                        help="Name of the environment to use from config.json.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional baseline seed.")
    args = parser.parse_args()

    run_noised_action_auto_regressive(
        config_path=args.config_path,
        env_name=args.env,
        baseline_seed=args.seed
    )


if __name__ == "__main__":
    main()