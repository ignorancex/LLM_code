from typing import Tuple

import jax.numpy as np
from jax import Array, jit, lax, vmap
from jax.random import split as split_key

from dist_alg_common import sample_env_policies
from dist_env import env_step


def get_visitdistr_valfunc_env(state: Array, policy: Array, gamma: float, n_states: int, n_agents: int, n_steps: int, n_episodes: int, key: Array) -> Tuple[Array, Array]:
    def _get_episode_inner(step: int, args: Tuple[Array, Array, Array, Array]):
        visit_distr, val, state, key = args
        visit_distr = visit_distr.at[state, step].add(1)

        key, subkey = split_key(key)
        subkeys_actions = split_key(subkey, n_agents)
        actions = sample_env_policies(policy[:, state, :], subkeys_actions)
        state, rewards = env_step(state, actions)

        val = val.at[state, :].add((gamma**step) * rewards)
        return visit_distr, val, state, key

    visit_distr = np.zeros((n_states, n_steps))
    val = np.zeros((n_states, n_agents))

    visit_distr, val, _, _ = lax.fori_loop(
        0,
        n_episodes,
        lambda _, args: lax.fori_loop(0, n_steps, _get_episode_inner, (args[0], args[1], state, args[3])),
        (visit_distr, val, state, key),
    )

    dist = np.dot(visit_distr / n_episodes, gamma ** np.arange(n_steps))
    return dist, val / n_episodes


get_visitdistr_valfunc = jit(
    vmap(get_visitdistr_valfunc_env, (None, 0, None, None, None, None, None, 0), (0, 0)),
    static_argnames=("gamma", "n_states", "n_agents", "n_steps", "n_episodes"),
)


def Q_function_meta(agent: Array, state: Array, action: Array, policy: Array, gamma: float, val: Array, n_samples: int, key: Array) -> Array:
    def _Q_function_inner(step, args):
        tot_reward, key = args
        actions = sample_env_policies(policy[:, state, :], key[step])
        actions = actions.at[agent].set(action)
        next_state, rewards = env_step(state, actions)
        tot_reward += rewards[agent] + gamma * val[next_state, agent]

        return tot_reward, key

    tot_reward, _ = lax.fori_loop(0, n_samples, _Q_function_inner, (np.array(0), key))
    return tot_reward / n_samples


Q_function_actions = vmap(Q_function_meta, (None, None, 0, None, None, None, None, None))
Q_function_states_actions = vmap(Q_function_actions, (None, 0, None, None, None, None, None, 0))
Q_function_agents_states_actions = jit(vmap(Q_function_states_actions, (0, None, None, None, None, None, None, 0)))


def Q_function_env(policy: Array, gamma: float, val: Array, n_agents: int, n_samples: int, n_states: int, n_actions: int, key: Array):
    keys = split_key(key, (n_agents, n_states, n_actions, n_agents))
    qval = Q_function_agents_states_actions(np.arange(n_agents), np.array([0, 1]), np.identity(4), policy, gamma, val, n_samples, keys)
    return qval


Q_function = jit(
    vmap(Q_function_env, (0, None, 0, None, None, None, None, 0)),
    static_argnames=("gamma", "n_agents", "n_samples", "n_states", "n_actions"),
)
