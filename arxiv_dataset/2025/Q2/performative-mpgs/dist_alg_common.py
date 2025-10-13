import os
from typing import Tuple

import jax.numpy as np
import jax.random as random
from jax import Array, jit, lax, vmap
from jax.nn import one_hot, softmax
from jax.random import split as split_key

from dist_env import env_reset, env_step, env_step_general
from util.util import get_filename

dtype = np.float32
dtype_int = np.int32

reward_penalty_intervention = 20


all_states = np.array([0, 1])
all_actions = np.identity(4, dtype=dtype_int)
sa_leading_into = np.empty((0,))


def sample_env_action(policy: Array, key: Array) -> Array:
    return random.choice(key, np.arange(policy.shape[-1]), p=policy)


sample_env_actions = vmap(sample_env_action, in_axes=(0, 0))


def sample_env_policy(policy: Array, key: Array) -> Array:
    # assert policy.shape == (n_agents, n_actions)
    actions = sample_env_action(policy, key)
    return one_hot(actions, policy.shape[-1], axis=-1, dtype=dtype_int)


sample_env_policies = vmap(sample_env_policy, in_axes=(0, 0))


def get_env_episode(dataset: Array, policy: Array, n_agents: int, n_samples: int, qvals: Array, performative_prob: float, key: Array) -> Array:
    key, subkey = split_key(key)
    state = env_reset(n_agents, subkey)  # draw initial state s_0 ~ \rho
    for step in range(n_samples):
        key, subkey = split_key(key)
        subkeys_actions = split_key(subkey, n_agents)
        actions = sample_env_policies(policy[np.arange(n_agents), state, :], subkeys_actions)

        key, subkey_qactions = split_key(key)
        actions = conditionally_intervene(qvals[np.arange(n_agents), state, :], actions, performative_prob, subkey_qactions)

        state_next, rewards = env_step(state, actions)
        new_element = np.concatenate((state.reshape(-1), actions.reshape(-1), rewards, state_next.reshape(-1)), axis=-1)

        dataset = dataset.at[step, :].set(new_element)
        state = state_next

    return dataset


get_env_episodes = vmap(get_env_episode, in_axes=(0, None, None, None, None, None, 0))
get_envs_episodes = vmap(get_env_episodes, in_axes=(0, 0, None, None, 0, None, 0))
get_envs_episodes_jit = jit(get_envs_episodes, static_argnames=("n_agents", "n_samples", "performative_prob"))


def get_envs_rewards_nojit(
    policy: Array, n_experiment_replications: int, n_episodes: int, n_agents: int, n_actions: int, qvals_intervention: Array, performative_prob: float, key: Array
):
    subkeys_get_episodes = split_key(key, (n_experiment_replications, n_episodes))
    dataset = np.zeros((n_experiment_replications, n_episodes, 3, 1 + n_agents * n_actions + n_agents + 1))
    dataset = get_envs_episodes_jit(dataset, policy, n_agents, 3, qvals_intervention, performative_prob, subkeys_get_episodes)

    rew_avg = np.sum(dataset[:, :, :, n_agents + n_actions * n_agents : n_agents + n_actions * n_agents + n_agents], axis=(1, 2, 3)) / n_agents / n_episodes
    return rew_avg


get_envs_rewards = jit(get_envs_rewards_nojit, static_argnames=("n_experiment_replications", "n_episodes", "n_agents", "n_actions", "performative_prob"))


def get_env_rewards_2(policy: Array, n_agents: int, n_samples: int, qvals: Array, performative_prob: float, key: Array) -> Array:
    key, subkey = split_key(key)
    state = env_reset(n_agents, subkey)  # draw initial state s_0 ~ \rho
    total_rewards = np.zeros(n_agents)
    for _ in range(n_samples):
        key, subkey = split_key(key)
        subkeys_actions = split_key(subkey, n_agents)
        actions = sample_env_policies(policy[np.arange(n_agents), state, :], subkeys_actions)

        key, subkey_qactions = split_key(key)
        actions = conditionally_intervene(qvals[np.arange(n_agents), state, :], actions, performative_prob, subkey_qactions)

        state_next, rewards = env_step(state, actions)

        total_rewards += rewards
        state = state_next

    return np.average(total_rewards) / n_samples  # avg. per agent per step rewards


get_envs_rewards_2 = jit(
    vmap(get_env_rewards_2, in_axes=(0, None, None, 0, None, 0)),
    static_argnames=("n_agents", "n_samples", "performative_prob"),
)


def predict_q_actions(qvals: Array, key: Array) -> Array:
    keys = split_key(key, qvals.shape[0])
    distr = softmax(qvals, axis=-1)
    actions = sample_env_actions(distr, keys)
    return one_hot(actions, qvals.shape[-1], dtype=dtype_int)


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


def conditionally_intervene(qvals_intervention_state: Array, actions: Array, performative_prob: float, key: Array):
    actions_intervention = predict_q_actions(qvals_intervention_state, key)
    cond = random.uniform(key, (actions_intervention.shape[0], 1)) > performative_prob
    actions_intervention = np.where(cond, np.zeros((actions_intervention.shape[-1]), dtype=dtype_int).at[-1].set(1), actions_intervention)
    actions = np.where(np.all(actions_intervention[:, :-1] == np.zeros(actions.shape[-1], dtype=dtype_int), axis=-1)[:, None], actions, actions_intervention[:, :-1])
    return actions


def index_of(arr: Array, needle: Array):
    mask = arr == needle
    idx = np.argmax(mask)
    return idx


def get_visitdistr_valfunc_intervention_env(
    state: Array,
    state_idx: int,
    policy: Array,
    qvals_intervention: Array,
    gamma: float,
    n_states: int,
    n_agents: int,
    n_steps: int,
    n_episodes: int,
    performative_prob: float,
    key: Array,
) -> Tuple[Array, Array]:
    """Unnormalized visitation distribution. Since we take finite trajectories, the normalization constant is (1-gamma**T)/(1-gamma)."""

    def _get_episode_inner(step: int, args: Tuple[Array, Array, Array, Array, Array]):
        visit_distr, val, state, state_idx, key = args
        visit_distr = visit_distr.at[state_idx, step].add(1)

        key, subkey = split_key(key)
        subkeys_actions = split_key(subkey, n_agents)
        actions = sample_env_policies(policy[:, state_idx, :], subkeys_actions)

        key, subkey_qactions = split_key(key)
        actions = conditionally_intervene(qvals_intervention[:, state_idx, :], actions, performative_prob, subkey_qactions)

        state, rewards = env_step(state, actions)
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


get_visitdistr_valfunc_intervention = jit(
    vmap(get_visitdistr_valfunc_intervention_env, (None, None, 0, 0, None, None, None, None, None, None, 0), (0, 0)),
    static_argnames=("gamma", "n_states", "n_agents", "n_steps", "n_episodes", "performative_prob"),
)


def get_visitdistr_intervention_individual_env(
    state: Array,
    policy: Array,
    qvals_intervention: Array,
    gamma: float,
    n_states: int,
    n_actions: int,
    n_agents: int,
    n_steps: int,
    n_episodes: int,
    performative_prob: float,
    key: Array,
) -> Array:
    def _get_episode_inner(step: int, args: Tuple[Array, Array, Array]):
        visit_distr, state, key = args

        key, subkey = split_key(key)
        subkeys_actions = split_key(subkey, n_agents)
        actions = sample_env_policies(policy[np.arange(n_agents), state, :], subkeys_actions)

        key, subkey_qactions = split_key(key)
        actions = conditionally_intervene(qvals_intervention[np.arange(n_agents), state, :], actions, performative_prob, subkey_qactions)

        visit_distr = visit_distr.at[np.arange(n_agents), state, np.argmax(actions, axis=-1), step].add(1)

        state_next, _ = env_step(state, actions)

        return visit_distr, state_next, key

    visit_distr = np.zeros((n_agents, n_states, n_actions, n_steps))

    visit_distr, _, _ = lax.fori_loop(
        0,
        n_episodes,
        lambda _, args: lax.fori_loop(0, n_steps, _get_episode_inner, (args[0], state, args[2])),
        (visit_distr, state, key),
    )

    dist = np.dot(visit_distr, (gamma ** np.arange(n_steps)) / n_episodes)
    dist_sum = np.sum(dist, axis=(-1, -2), keepdims=True)
    dist_sum = np.where(dist_sum == 0, 1, dist_sum)
    dist = dist / dist_sum
    return dist


get_visitdistr_intervention_individual = jit(
    vmap(get_visitdistr_intervention_individual_env, (0, 0, 0, None, None, None, None, None, None, None, 0)),
    static_argnames=("gamma", "n_states", "n_actions", "n_agents", "n_steps", "n_episodes", "performative_prob"),
)


def Q_function_intervention_meta(
    agent: Array,
    state: Array,
    action: Array,
    policy: Array,
    qvals_intervention: Array,
    gamma: float,
    val: Array,
    n_samples: int,
    performative_prob: float,
    key: Array,
) -> Array:
    def _Q_function_inner(step, args):
        tot_reward, key = args
        state_idx = index_of(all_states, state)
        actions = sample_env_policies(policy[:, state_idx, :], key[step])

        _, subkey_qactions = split_key(key[step, agent])
        actions = conditionally_intervene(qvals_intervention[:, state_idx, :], actions, performative_prob, subkey_qactions)

        actions = actions.at[agent].set(action)
        next_state, rewards = env_step(state, actions)
        tot_reward += rewards[agent] + gamma * val[index_of(all_states, next_state), agent]

        return tot_reward, key

    tot_reward, _ = lax.fori_loop(0, n_samples, _Q_function_inner, (np.array(0), key))
    return tot_reward / n_samples


Q_function_intervention_actions = vmap(Q_function_intervention_meta, (None, None, 0, None, None, None, None, None, None, None))
Q_function_intervention_states_actions = vmap(Q_function_intervention_actions, (None, 0, None, None, None, None, None, None, None, 0))
Q_function_intervention_agents_states_actions = jit(vmap(Q_function_intervention_states_actions, (0, None, None, None, None, None, None, None, None, 0)))


def Q_function_intervention_env(
    policy: Array,
    qvals_intervention: Array,
    gamma: float,
    val: Array,
    n_agents: int,
    n_samples: int,
    n_states: int,
    n_actions: int,
    performative_prob: float,
    key: Array,
):
    keys = split_key(key, (n_agents, n_states, n_actions, n_agents))
    qval = Q_function_intervention_agents_states_actions(np.arange(n_agents), all_states, all_actions, policy, qvals_intervention, gamma, val, n_samples, performative_prob, keys)
    return qval


Q_function_intervention = jit(
    vmap(Q_function_intervention_env, (0, 0, None, 0, None, None, None, None, None, 0)),
    static_argnames=("gamma", "n_agents", "n_samples", "n_states", "n_actions", "performative_prob"),
)


def env_get_qval_intervention(
    policy: Array, agent_index: Array, states: Array, n_actions: int, gamma: float, reward_weights_perturbed: Array, n_qvaliter_samples: int, key: Array, qvaliter_threshold: float
):
    qvaliter_sample_normalizer = 1 / n_qvaliter_samples

    def _q_value_iteration_inner_sample(_, args: Tuple[Array, Array, Array, Array, Array, Array, Array, Array]) -> Tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
        key, prob, state, state_idx, action, qvals, new_qval, new_qval_nointervention = args

        key, subkey = split_key(key)
        subkeys = split_key(subkey, policy.shape[0])
        actions = sample_env_actions(policy[:, state_idx, :], subkeys)
        actions = actions.at[agent_index].set(action)
        state_next, rewards = env_step_general(state, one_hot(actions, n_actions, dtype=dtype_int), reward_weights_perturbed)

        # qvals = qvals.at[state, action].add(prob * (rewards[agent_index] - reward_penalty_intervention + gamma * qvals[state_next].max()) * qvaliter_sample_normalizer)
        new_qval += prob * (rewards[agent_index] - reward_penalty_intervention + gamma * qvals[index_of(all_states, state_next)].max()) * qvaliter_sample_normalizer

        state_next_clean, rewards_clean = env_step(state, one_hot(actions, n_actions, dtype=dtype_int))
        # qvals = qvals.at[state, -1].add(prob * (rewards_clean[agent_index] + gamma * qvals[state_next_clean].max()) * qvaliter_sample_normalizer_nointervention)
        new_qval_nointervention += prob * (rewards_clean[agent_index] + gamma * qvals[index_of(all_states, state_next_clean)].max()) * qvaliter_sample_normalizer

        return key, prob, state, state_idx, action, qvals, new_qval, new_qval_nointervention

    def _q_value_iteration_inner(args: Tuple[Array, Array, Array, Array]) -> Tuple[Array, Array, Array, Array]:
        qvals, delta, key, i = args

        for state_idx, state in enumerate(states):
            # numerify state?
            for action in range(n_actions):
                old_qvals_state_action = qvals[state_idx, action]

                prob = policy[agent_index, state_idx, action]

                key, _, _, _, _, _, new_qval, new_qval_nointervention = lax.fori_loop(
                    0, n_qvaliter_samples, _q_value_iteration_inner_sample, (key, prob, state, state_idx, action, qvals, 0, 0)
                )

                qvals = qvals.at[state_idx, action].set(new_qval)
                qvals = qvals.at[state_idx, -1].set(new_qval_nointervention)

                delta = np.maximum(delta, np.abs(qvals[state_idx, action] - old_qvals_state_action))

        return qvals, delta, key, i + 1

    max_steps = 20
    qvals, _, _, _ = lax.while_loop(
        lambda qdki: np.logical_and(qdki[1] > qvaliter_threshold, qdki[3] < max_steps),
        _q_value_iteration_inner,
        (
            np.zeros((states.shape[0], n_actions + 1)),  # n_actions + 1 action for the no intervention
            np.array(1 + qvaliter_threshold),
            key,
            np.array(0),
        ),
    )

    return qvals


env_get_qvals_intervention = vmap(env_get_qval_intervention, in_axes=(None, 0, None, None, None, 0, None, 0, None))
envs_get_qvals_intervention_manual = jit(
    vmap(env_get_qvals_intervention, in_axes=(0, None, None, None, None, 0, None, 0, None)), static_argnames=("n_actions", "gamma", "n_qvaliter_samples", "qvaliter_threshold")
)


def envs_get_qvals_intervention(policy: Array, n_actions: int, gamma: float, reward_weights_perturbed: Array, n_qvaliter_samples: int, key: Array, qvaliter_threshold: float):
    return envs_get_qvals_intervention_manual(
        policy, np.arange(policy.shape[1]), all_states, n_actions, gamma, reward_weights_perturbed, n_qvaliter_samples, key, qvaliter_threshold
    )


def load_policy(n_experiment_replications: int, n_agents: int, n_states: int, n_actions: int, method: str, config, continue_round: int | None = None) -> Array:
    policy = np.ones((n_experiment_replications, n_agents, n_states, n_actions)) / n_actions

    if continue_round == 0 or continue_round == None:
        return policy
    if os.path.exists(f"data/{get_filename(method, 'distancing', config, continue_round, 0)}.npy"):
        for repl in range(n_experiment_replications):
            policy = policy.at[repl].set(np.load(f"data/{get_filename(method, 'distancing', config, continue_round, repl)}.npy"))
        print(f"Loaded 'data/{get_filename(method, 'distancing', config, continue_round)}.npy'.")
    elif os.path.exists(f"data/{get_filename(method, 'distancing', config)}.npy"):
        all_policies = np.load(f"data/{get_filename(method, 'distancing', config)}.npy")
        if all_policies.shape[1] >= continue_round:
            policy = all_policies[:, continue_round - 1, :, :]
            del all_policies
        print(f"Loaded 'data/{get_filename(method, 'distancing', config)}.npy'.")
    else:
        raise RuntimeError(
            f"Cannot continue from round {continue_round}, files 'data/{get_filename(method, 'distancing', config, continue_round, 0)}.npy' or 'data/{get_filename(method, 'distancing', config)}.npy' do not exist."
        )

    return policy
