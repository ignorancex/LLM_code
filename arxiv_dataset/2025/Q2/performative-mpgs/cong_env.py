from typing import List, Tuple

import jax.numpy as np
from jax import Array, random, vmap

dtype = np.float32
dtype_int = np.int32

n_states = 5


def env_reset(n_agents: int, key: Array) -> Array:
    return np.zeros((n_agents,), dtype=dtype_int)


transition_kernel = np.array(
    [
        [
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
        ],
        [
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ],
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
        ],
        [
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
        ],
        [
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
        ],
    ],
    dtype=dtype,
)

perturb_mask = np.array(
    [
        [
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0],
        ],
        [
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
        ],
        [
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
        ],
        [
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
        ],
        [
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
        ],
    ],
    dtype=dtype_int,
)

#   . 1 - 3 .
# 0     X     5 / 0
#   ' 2 - 4 '

step_rewards = np.array([0, 50, 15, 5, 1])

all_states = np.array(
    [
        [0, 0, 0, 0],
        #
        [1, 1, 1, 1],
        [1, 1, 1, 2],
        [1, 1, 2, 1],
        [1, 1, 2, 2],
        [1, 2, 1, 1],
        [1, 2, 1, 2],
        [1, 2, 2, 1],
        [1, 2, 2, 2],
        [2, 1, 1, 1],
        [2, 1, 1, 2],
        [2, 1, 2, 1],
        [2, 1, 2, 2],
        [2, 2, 1, 1],
        [2, 2, 1, 2],
        [2, 2, 2, 1],
        [2, 2, 2, 2],
        #
        [3, 3, 3, 3],
        [3, 3, 3, 4],
        [3, 3, 4, 3],
        [3, 3, 4, 4],
        [3, 4, 3, 3],
        [3, 4, 3, 4],
        [3, 4, 4, 3],
        [3, 4, 4, 4],
        [4, 3, 3, 3],
        [4, 3, 3, 4],
        [4, 3, 4, 3],
        [4, 3, 4, 4],
        [4, 4, 3, 3],
        [4, 4, 3, 4],
        [4, 4, 4, 3],
        [4, 4, 4, 4],
    ]
)

sa_leading_into: List[Tuple[Array, Array]] = [
    (np.array([3, 3, 4, 4]), np.array([0, 1, 0, 1])),
    (np.array([0]), np.array([0])),
    (np.array([0]), np.array([1])),
    (np.array([1, 2]), np.array([0, 1])),
    (np.array([1, 2]), np.array([1, 0])),
]


def sample_transition(key: Array, n_states: int, states: Array, action: Array, perf_perturb: Array):
    actions_idxs = action.argmax(axis=-1)
    new_transition_kernel = transition_kernel[states, actions_idxs] + perturb_mask[states, actions_idxs] * perf_perturb
    new_transition_kernel = new_transition_kernel / new_transition_kernel.sum(axis=-1, keepdims=True)  # renormalize
    return random.choice(key, np.arange(n_states), p=new_transition_kernel)


sample_transitions = vmap(sample_transition, (0, None, 0, 0, 0))


def env_step_perf(key: Array, states: Array, actions: Array, perf_influence: Array, c_r: float, c_p: float) -> Tuple[Array, Array]:
    # state: (n_agents,)
    # actions: (n_agents, 2)

    state_action_counts = np.zeros((n_states, 2), dtype=dtype_int)
    state_action_counts = state_action_counts.at[states].add(actions)

    state_rewards = step_rewards[state_action_counts]
    state_rewards = np.where(~np.isfinite(state_rewards), 0, state_rewards)

    rewards = np.einsum("ij,ij->i", state_rewards[states], actions)

    policy_diff = perf_influence - 1 / actions.shape[-1]
    rewards = rewards + c_r * policy_diff

    keys = random.split(key, states.shape[0])
    states = sample_transitions(keys, n_states, states, actions, c_p * policy_diff)

    return states, rewards
