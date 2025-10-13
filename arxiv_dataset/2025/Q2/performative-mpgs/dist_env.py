from typing import List, Tuple

import jax.numpy as np
import jax.random as random
import matplotlib.pyplot as plt
from jax import Array, jit, lax, vmap

dtype = np.float32
dtype_int = np.int32

reward_weights = np.array([4, 3, 2, 1], dtype=dtype)
reward_penalty = 100


def env_reset(n_agents: int, key: Array) -> Array:
    state = random.choice(key, np.array([0, 1], dtype=dtype_int), p=np.array([0.5, 0.5]))
    return state


def env_step_general(state: Array, actions: Array, reward_weights: Array) -> Tuple[Array, Array]:
    actions_indices = actions.argmax(axis=-1)
    action_counts = np.sum(actions, axis=0)

    rewards = action_counts[actions_indices] * reward_weights[actions_indices]
    threshold: int = lax.cond(state, lambda: 4, lambda: 2)

    any_over_threshold = np.any(action_counts > actions.shape[0] // threshold)

    state = any_over_threshold.astype(dtype_int)
    rewards = rewards - state * reward_penalty

    return state, rewards


def env_step(state: Array, actions: Array) -> Tuple[Array, Array]:
    return env_step_general(state, actions, reward_weights)


def env_show_plot(state: Array, actions: Array, n_agents: int, n_actions: int) -> None:
    assert actions.shape == (1, n_agents, n_actions)

    fig, (ax1, ax2) = plt.subplots(2, n_actions // 2, figsize=(4, 2), sharex=True)
    ax = [*ax1, *ax2]  # type: ignore
    fig.subplots_adjust(wspace=0, hspace=0)

    cell_width = 1 / n_agents
    for i in range(n_actions):
        indices = np.where(actions[0, :, i] == 1)[0]
        if state == 0:
            ax[i].set_facecolor("#eeffee")
        else:
            ax[i].set_facecolor("#ffeeee")

        for idx in indices:
            x = (idx + 0.5) * cell_width
            ax[i].text(x, 0.5, str(idx), ha="center", va="center", fontsize=12)
            ax[i].set_xlim(0, 1)
            ax[i].set_ylim(0, 1)
            ax[i].tick_params(axis="both", which="both", bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
            ax[i].axis("on")

    plt.show()


def get_env_occupancy_measure(dataset: Array, n_agents: int, n_states: int, n_actions: int, dim_state: int, dim_action: int, gamma: float) -> Array:
    def get_episode_occupancy(i, args):
        d, total_mass = args

        for t, transition in enumerate(dataset[i]):
            state = transition[:dim_state].astype(dtype_int)
            actions = transition[dim_state : dim_state + n_agents * dim_action].reshape(n_agents, dim_action)

            gamma_factor = gamma**t
            total_mass += gamma_factor

            d = d.at[np.arange(n_agents), state, actions.argmax(axis=-1)].add(gamma_factor)
        return d, total_mass

    d, total_mass = lax.fori_loop(0, dataset.shape[0], get_episode_occupancy, (np.zeros((n_agents, n_states, n_actions)), 0.0))

    d = d / (total_mass * (1.0 - gamma))
    return d


get_envs_occupancy_measure = vmap(get_env_occupancy_measure, in_axes=(0, None, None, None, None, None, None))
get_envs_occupancy_measure_jit = jit(get_envs_occupancy_measure, static_argnames=("n_agents", "n_states", "n_actions", "dim_state", "dim_action", "gamma"))
