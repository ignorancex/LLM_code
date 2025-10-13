import os
import time
from typing import Dict, List

import jax.lax as lax
import jax.numpy as np
import jax.random as random
from jax import Array, jit, vmap
from jax.random import split as split_key
from tqdm import tqdm

import wandb
from cong_alg_common import (
    Q_function,
    all_states,
    get_envs_rewards,
    get_visitdistr_valfunc,
    load_policy,
    update_step,
)
from util.util import (
    add_experiment_replication_args,
    add_log_save_args,
    add_performative2_args,
    add_pga_train_args,
    get_filename,
)

dtype_int = np.int32


def compute_reward_singleenv_singleep_singleagent(h: Array, rs: Array):
    def _compute_reward_inner(i, R):
        R += rs[i]
        return R

    R = lax.fori_loop(h[0], h[0] + h[1], _compute_reward_inner, np.zeros(1))
    return R


compute_reward_singleenv_singleep = vmap(compute_reward_singleenv_singleep_singleagent, in_axes=(0, 1))
compute_reward_singleenv = vmap(compute_reward_singleenv_singleep, in_axes=(0, 0))
compute_reward = jit(vmap(compute_reward_singleenv, in_axes=(0, 0)))


def run(config) -> None:
    seed = config.seed
    n_experiment_replications = config.n_experiment_replications  # num envs / repetitions / replications (for mean, std)
    n_rounds = config.n_rounds  # T
    n_episodes = config.n_episodes  # K
    gamma = config.gamma
    lr = config.lr  # eta
    n_steps_visitdistr = 20
    n_steps_qval = 20
    ding = config.ding  # use the version of the algorithm without the visitation distr. (Ding et al., 2022 vs. Leonardos et al., 2021)
    method = "ding" if config.ding else "leo"

    continue_round = config.continue_round

    log_interval = config.log_interval
    save_interval = config.save_interval

    key = random.PRNGKey(seed)

    n_agents = 4
    n_states = all_states.shape[0]
    n_actions = 2

    track_reward = True

    log_data: List[Dict[str, float]] = []
    policies_rounds = []
    policy = load_policy(n_experiment_replications, n_agents, n_states, n_actions, method, config, continue_round)

    if config.logging == "wandb":
        run = wandb.init(project="performative-mpgs", name=get_filename(method, "congestion2", config, n_rounds=n_rounds, n_experiment_replications=n_experiment_replications))
    print("Starting..." if continue_round is None else f"Continuing from round {continue_round}...")

    c_r = config.omega_r / ((1 - gamma) * np.sqrt(n_states * n_actions)).item()
    c_p = config.omega_p / ((1 - gamma) * np.sqrt(n_states * n_actions)).item()

    for round in tqdm(
        range(continue_round if continue_round is not None else 0, n_rounds),
        desc="Train",
    ):
        b_dist = np.zeros((n_experiment_replications, n_states))
        val = np.zeros((n_experiment_replications, n_states, n_agents))

        if track_reward:
            key, subkey_rewards = split_key(key)
            subkeys_rewards = split_key(subkey_rewards, n_experiment_replications)
            rew_avgs = get_envs_rewards(policy, n_agents, 50, c_r, c_p, subkeys_rewards)

            for repl in range(n_experiment_replications):
                log_data.append({"round": round, f"multi/reward[{repl}]": rew_avgs[repl]})

        if not ding:
            for state_idx, state in enumerate(all_states):
                key, subkey = split_key(key)
                subkeys_visitation = split_key(subkey, n_experiment_replications)
                a_dist, _ = get_visitdistr_valfunc(state, state_idx, policy, gamma, n_states, n_agents, n_steps_visitdistr, n_episodes, c_r, c_p, subkeys_visitation)
                b_dist = b_dist.at[:, state_idx].set(np.mean(a_dist, axis=1))

        for state_idx, state in enumerate(all_states):
            key, subkey = split_key(key)
            subkeys_val = split_key(subkey, n_experiment_replications)
            _, val_add = get_visitdistr_valfunc(state, state_idx, policy, gamma, n_states, n_agents, n_steps_visitdistr, n_episodes, c_r, c_p, subkeys_val)
            val = val + val_add

        key, subkey_qval = split_key(key)
        subkeys_qval = split_key(subkey_qval, n_experiment_replications)
        qval = Q_function(policy, gamma, val, n_agents, n_states, n_actions, n_steps_qval, c_r, c_p, subkeys_qval, 1)

        if ding:
            grads = qval
        else:
            grads = b_dist[:, None, :, None] * qval[:, :, :, :]

        policy_new = update_step(policy, grads, lr)
        for repl in range(n_experiment_replications):
            policy_delta_norm = np.sum(np.linalg.norm(policy_new[repl] - policy[repl], axis=(-2, -1)), axis=-1).item()
            log_data.append({"round": round, f"multi/policy_delta_norm[{repl}]": policy_delta_norm})
        policy = policy_new

        policies_rounds.append(policy)

        if (round % save_interval == 0 and round != 0 and round != continue_round) or round == n_rounds - 1:
            policies_envs = np.array(policies_rounds).swapaxes(0, 1)
            filename = f"data/{get_filename(method, 'congestion2', config, n_rounds=(n_rounds if round == n_rounds - 1 else round), n_experiment_replications=n_experiment_replications)}.npy"
            np.save(filename, policies_envs)
            print(f"Saved '{filename}'.")
            del policies_envs

        if config.logging == "wandb" and (round % log_interval == 0 or round < 50 or round == n_rounds - 1):
            for item in log_data:
                run.log(item, step=item.pop("round"), commit=False)
            run.log({"round": round}, step=round, commit=True)
            log_data.clear()

    if config.logging == "wandb":
        run.finish()
    print("Finished.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    add_pga_train_args(parser)
    add_performative2_args(parser)
    add_experiment_replication_args(parser)
    add_log_save_args(parser)
    parser.add_argument("--continue_round", type=int, required=False, default=None, help="Round number to continue from")
    args = parser.parse_args()

    time_start = time.time()
    run(args)
    time_end = time.time()

    print(f"Total runtime: {time_end - time_start:.2f} s")
