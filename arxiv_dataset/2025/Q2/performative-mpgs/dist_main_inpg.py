import os
import time
from typing import Dict, List

import jax.lax as lax
import jax.numpy as np
import jax.random as random
from jax.random import split as split_key
from tqdm import tqdm

import wandb
from dist_alg_common import (
    Q_function_intervention,
    all_states,
    envs_get_qvals_intervention,
    get_envs_rewards,
    get_visitdistr_valfunc_intervention,
    load_policy,
)
from dist_env import reward_weights
from util.util import (
    add_experiment_replication_args,
    add_general_train_args,
    add_log_save_args,
    add_performative_args,
    get_filename,
)


def run(config) -> None:
    seed = config.seed
    n_experiment_replications = config.n_experiment_replications  # num envs / repetitions / replications (for mean, std)
    n_rounds = config.n_rounds  # T
    n_episodes = config.n_episodes  # K
    gamma = config.gamma
    lr = config.lr  # eta
    n_steps = 20
    method = "inpg"
    performative_temperature = config.beta
    performative_prob = config.alpha

    n_qvaliter_samples = 10  # TODO
    qvaliter_threshold = 1e-5  # TODO
    continue_round = config.continue_round

    log_barrier_reg = config.log_barrier_reg

    log_interval = config.log_interval
    save_interval = config.save_interval

    n_agents = 8  # N
    n_states = 2  # |S|
    n_actions = 4  # |A|

    key = random.PRNGKey(seed)

    policy = load_policy(n_experiment_replications, n_agents, n_states, n_actions, method, config, continue_round)

    key, subkey = split_key(key)
    reward_weights_many = np.tile(reward_weights, (n_experiment_replications, n_agents, 1))
    perturbation_probabilities = random.uniform(subkey, reward_weights_many.shape)
    reward_weights_permuted = random.permutation(subkey, reward_weights_many, axis=-1, independent=True)
    reward_weights_perturbed = lax.select(perturbation_probabilities > 0.7, reward_weights_permuted, reward_weights_many)

    track_reward = True

    log_data: List[Dict[str, float]] = []
    policies_rounds = []
    if config.logging == "wandb":
        run = wandb.init(project="performative-mpgs", name=get_filename(method, "distancing", config, n_rounds=n_rounds, n_experiment_replications=n_experiment_replications))
    print("Starting...")

    for round in tqdm(range(continue_round if continue_round is not None else 0, n_rounds), desc="Train"):
        val = np.zeros((n_experiment_replications, n_states, n_agents))
        visit_distr = np.zeros((n_experiment_replications, n_states))

        key, subkey = split_key(key)
        subkey_qvals = split_key(key, (n_experiment_replications, n_agents))
        qvals_intervention = envs_get_qvals_intervention(policy, n_actions, gamma, reward_weights_perturbed, n_qvaliter_samples, subkey_qvals, qvaliter_threshold)
        qvals_intervention = performative_temperature * qvals_intervention

        if track_reward:
            key, subkey_rewards = split_key(key)
            rew_avgs = get_envs_rewards(policy, n_experiment_replications, n_episodes, n_agents, n_actions, qvals_intervention, performative_prob, subkey_rewards)

            for repl in range(n_experiment_replications):
                log_data.append({"round": round, f"multi/reward[{repl}]": rew_avgs[repl]})

        for state_idx, state in enumerate(all_states):
            key, subkey = split_key(key)
            subkeys_val = split_key(subkey, n_experiment_replications)
            visit_distr_add, val_add = get_visitdistr_valfunc_intervention(
                state, state_idx, policy, qvals_intervention, gamma, n_states, n_agents, n_steps, n_episodes, performative_prob, subkeys_val
            )
            val = val + val_add
            visit_distr = visit_distr + visit_distr_add

        key, subkey = split_key(key)
        subkeys_qval = split_key(subkey, n_experiment_replications)
        qval = Q_function_intervention(policy, qvals_intervention, gamma, val, n_agents, n_episodes, n_states, n_actions, performative_prob, subkeys_qval)

        advantage = qval - val.swapaxes(1, 2)[:, :, :, None]

        if log_barrier_reg == 0:
            policy_multiplier = np.exp(lr / (1 - gamma) * advantage)
            if np.any(advantage == 0):
                policy_multiplier = policy_multiplier.at[advantage == 0].set(1)  # TODO: is this a good heuristic?
        else:
            visit_distr = visit_distr / np.sum(visit_distr, axis=1, keepdims=True)
            policy_multiplier = np.exp(
                lr * (1 / (1 - gamma) * advantage + log_barrier_reg / (visit_distr[:, None, :, None] * policy) - (log_barrier_reg * n_actions / visit_distr)[:, None, :, None])
            )
            # if np.any(advantage == 0):
            #     policy_multiplier = policy_multiplier.at[advantage == 0].set(
            #         np.exp(lr * (log_barrier_reg / (visit_distr[:, None, :, None] * policy) - (log_barrier_reg * n_actions / visit_distr)[:, None, :, None]))
            #     )
        policy_multiplier = policy_multiplier.at[np.isneginf(policy_multiplier)].set(-2)
        policy_multiplier = policy_multiplier.at[np.isposinf(policy_multiplier)].set(2)
        policy_multiplier = policy_multiplier.at[np.isnan(policy_multiplier)].set(1)

        policy_new = policy * policy_multiplier
        policy_new = policy_new / policy_new.sum(axis=-1, keepdims=True)

        for repl in range(n_experiment_replications):
            policy_delta_norm = np.sum(np.linalg.norm(policy_new[repl] - policy[repl], axis=(-2, -1)), axis=-1).item()
            log_data.append({"round": round, f"multi/policy_delta_norm[{repl}]": policy_delta_norm})
        policy = policy_new

        policies_rounds.append(policy)

        if (round % save_interval == 0 and round != 0 and round != continue_round) or round == n_rounds - 1:
            policies_envs = np.array(policies_rounds).swapaxes(0, 1)
            filename = f"data/{get_filename(method, 'distancing', config, n_rounds=(n_rounds if round == n_rounds - 1 else round), n_experiment_replications=n_experiment_replications)}.npy"
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
    add_general_train_args(parser)
    add_performative_args(parser)
    add_experiment_replication_args(parser, 10)
    parser.add_argument("--continue_round", type=int, required=False, default=None, help="Number of round to continue from")
    # lambda = 0.003 is a good value
    parser.add_argument("--log_barrier_reg", type=float, required=False, default=0, help="Log barrier regularization (lambda) value (default: 0)")
    add_log_save_args(parser)
    args = parser.parse_args()

    time_start = time.time()
    run(args)
    time_end = time.time()

    print(f"Total runtime: {time_end - time_start:.2f} s")
