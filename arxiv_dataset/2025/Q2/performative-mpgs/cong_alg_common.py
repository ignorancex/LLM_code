import os
from typing import Tuple

import jax.numpy as np
import jax.random as random
from jax import Array, jit, lax, vmap
from jax.nn import one_hot
from jax.random import split as split_key

from cong_env import all_states, env_reset
from cong_env import env_step_perf as env_step
from util.util import get_filename

dtype = np.float32
dtype_int = np.int32

reward_penalty_intervention = 20

all_actions = np.identity(2, dtype=dtype_int)


def sample_env_action(policy: Array, key: Array) -> Array:
    return random.choice(key, np.arange(policy.shape[-1]), p=policy)


sample_env_actions = vmap(sample_env_action, in_axes=(0, 0))


def sample_env_policy(policy: Array, key: Array) -> Array:
    # assert policy.shape == (n_agents, n_actions)
    actions = sample_env_action(policy, key)
    return one_hot(actions, policy.shape[-1], axis=-1, dtype=dtype_int)


sample_env_policies = vmap(sample_env_policy, in_axes=(0, 0))


def get_env_rewards(policy: Array, n_agents: int, n_samples: int, c_r: float, c_p: float, key: Array) -> Array:
    key, subkey = split_key(key)
    state = env_reset(n_agents, subkey)  # draw initial state s_0 ~ \rho
    total_rewards = np.zeros(n_agents)
    for _ in range(n_samples):
        key, subkey = split_key(key)
        subkeys_actions = split_key(subkey, n_agents)
        actions = sample_env_policies(policy[np.arange(n_agents), state, :], subkeys_actions)

        key, subkey_step = split_key(key)
        perf_influence = policy[np.arange(n_agents), state, actions.argmax(axis=-1)]
        state_next, rewards = env_step(subkey_step, state, actions, perf_influence, c_r, c_p)

        total_rewards += rewards
        state = state_next

    return np.average(total_rewards) / n_samples  # avg. per agent per step rewards


get_envs_rewards = jit(
    vmap(get_env_rewards, in_axes=(0, None, None, None, None, 0)),
    static_argnames=("n_agents", "n_samples", "c_r", "c_p"),
)


def projection_simplex_sort(v: Array, z: float = 1) -> Array:
    # Courtesy: EdwardRaff/projection_simplex.py
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0

    def find_rho(carry, x):
        i, cond_value = x
        return carry + cond_value, carry + cond_value

    rho, _ = lax.scan(find_rho, 0, (ind, cond.astype(np.int32)))
    rho -= 1
    theta = cssv[rho] / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    return w


def update_step_general(policy: Array, grads: Array, lr: float) -> Array:
    return projection_simplex_sort(policy[:] + lr * grads[:])


update_step_state = vmap(update_step_general, (0, 0, None))
update_step_agent_state = vmap(update_step_state, (0, 0, None))
update_step = jit(vmap(update_step_agent_state, (0, 0, None)), static_argnames="lr")


def index_of(arr: Array, needle: Array):
    mask = arr == needle
    idx = np.argmax(mask)
    return idx


def get_visitdistr_valfunc_env(
    state: Array,
    state_idx: int,
    policy: Array,
    gamma: float,
    n_states: int,
    n_agents: int,
    n_steps: int,
    n_episodes: int,
    c_r: float,
    c_p: float,
    key: Array,
) -> Tuple[Array, Array]:
    """Unnormalized visitation distribution. Since we take finite trajectories, the normalization constant is (1-gamma**T)/(1-gamma)."""

    def _get_episode_inner(step: int, args: Tuple[Array, Array, Array, Array, Array]):
        visit_distr, val, state, state_idx, key = args
        visit_distr = visit_distr.at[state_idx, step].add(1)

        key, subkey = split_key(key)
        subkeys_actions = split_key(subkey, n_agents)
        actions = sample_env_policies(policy[:, state_idx, :], subkeys_actions)

        key, subkey_step = split_key(key)
        perf_influence = policy[np.arange(n_agents), state, actions.argmax(axis=-1)]
        state, rewards = env_step(subkey_step, state, actions, perf_influence, c_r, c_p)
        state_idx = index_of(all_states, state)

        val = val.at[state_idx, :].add((gamma**step) * rewards)
        return visit_distr, val, state, state_idx, key

    visit_distr = np.zeros((n_states, n_steps))
    val = np.zeros((n_states, n_agents))

    visit_distr, val, _, _, _ = lax.fori_loop(
        0,
        n_episodes,
        lambda _, args: lax.fori_loop(0, n_steps, _get_episode_inner, (args[0], args[1], state, state_idx, args[4])),
        (visit_distr, val, state, state_idx, key),
    )

    dist = np.dot(visit_distr / n_episodes, gamma ** np.arange(n_steps))
    return dist, val / n_episodes


get_visitdistr_valfunc = jit(
    vmap(get_visitdistr_valfunc_env, (None, None, 0, None, None, None, None, None, None, None, 0), (0, 0)),
    static_argnames=("gamma", "n_states", "n_agents", "n_steps", "n_episodes", "c_r", "c_p"),
)


def Q_function_meta(
    agent: Array,
    state: Array,
    action: Array,
    policy: Array,
    gamma: float,
    val: Array,
    all_agents: Array,
    n_samples: int,
    c_r: float,
    c_p: float,
    key: Array,
    reward_multiplier: float = 1,
) -> Array:
    def _Q_function_inner(step, args):
        tot_reward, key = args
        state_idx = index_of(all_states, state)
        actions = sample_env_policies(policy[:, state_idx, :], key[step])
        actions = actions.at[agent].set(action)
        perf_influence = policy[all_agents, state, actions.argmax(axis=-1)]
        next_states, rewards = env_step(key[step, agent], state, actions, perf_influence, c_r, c_p)
        tot_reward += rewards[agent] * reward_multiplier + gamma * val[index_of(all_states, next_states), agent]

        return tot_reward, key

    tot_reward, _ = lax.fori_loop(0, n_samples, _Q_function_inner, (np.array(0), key))
    return tot_reward / n_samples


Q_function_actions = vmap(Q_function_meta, (None, None, 0, None, None, None, None, None, None, None, None, None))
Q_function_states_actions = vmap(Q_function_actions, (None, 0, None, None, None, None, None, None, None, None, 0, None))
Q_function_agents_states_actions = jit(vmap(Q_function_states_actions, (0, None, None, None, None, None, None, None, None, None, 0, None)))


def Q_function_env(
    policy: Array,
    gamma: float,
    val: Array,
    n_agents: int,
    n_states: int,
    n_actions: int,
    n_samples: int,
    c_r: float,
    c_p: float,
    key: Array,
    reward_multiplier: float = 1,
):
    keys = split_key(key, (n_agents, n_states, n_actions, n_agents))
    all_agents = np.arange(n_agents)
    qval = Q_function_agents_states_actions(all_agents, all_states, all_actions, policy, gamma, val, all_agents, n_samples, c_r, c_p, keys, reward_multiplier)
    return qval


Q_function = jit(
    vmap(Q_function_env, (0, None, 0, None, None, None, None, None, None, 0, None)),
    static_argnames=("gamma", "n_agents", "n_states", "n_actions", "n_samples", "c_r", "c_p", "reward_multiplier"),
)


def load_policy(n_experiment_replications: int, n_agents: int, n_states: int, n_actions: int, method: str, config, continue_round: int | None = None) -> Array:
    if continue_round == 0 or continue_round == None:
        return np.ones((n_experiment_replications, n_agents, n_states, n_actions)) / n_actions

    filename = get_filename(method, "congestion2", config, n_rounds=continue_round, n_experiment_replications=n_experiment_replications)
    if os.path.exists(f"data/{filename}.npy"):
        all_policies = np.load(f"data/{filename}.npy")
        if all_policies.shape[1] >= continue_round:
            print(f"Warning: saved file contains more rounds than requested ({all_policies.shape[1]} vs requested {continue_round})")
        policy = all_policies[:, continue_round - 1, :, :]
        del all_policies
        print(f"Loaded 'data/{filename}.npy'.")
    else:
        raise RuntimeError(f"Cannot continue from round {continue_round}, file 'data/{filename}.npy' do not exist.")

    return policy
