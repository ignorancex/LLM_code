import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from agents.pbvi import PBVI
from agents.dpbvi import DPBVI


if __name__ == '__main__':
    # states: A, B
    states = np.array([0, 1])
    # actions = stay, move
    actions = np.array([0, 1])
    # terminal states
    terminal_states = np.zeros(2, dtype=bool)
    # transition function
    transition_function = np.zeros((2, 2, 2), dtype=np.float32)
    transition_function[0, 0, 0] = 0.9  # try stay and stay in A
    transition_function[0, 0, 1] = 0.1  # try stay but move to B
    transition_function[0, 1, 0] = 0.1  # try move but stay in A
    transition_function[0, 1, 1] = 0.9  # try move and move to B
    # B
    transition_function[1, 0, 1] = 0.9  # try stay and stay in B
    transition_function[1, 0, 0] = 0.1  # try stay but move to A
    transition_function[1, 1, 1] = 0.1  # try move but stay in B
    transition_function[1, 1, 0] = 0.9  # try move and move to A
    assert np.isclose(transition_function.sum(-1), 1.0).all(), 'Transition function is not normalized'
    # reward function
    reward_function = np.zeros((2, 2), dtype=np.float32)  # state, action
    # 0.0 for being in state A, 1.0 for being in state B
    reward_function[0, 0] = transition_function[0, 0, 0] * 0.0 + transition_function[0, 0, 1] * 1.0
    reward_function[0, 1] = transition_function[0, 1, 0] * 0.0 + transition_function[0, 1, 1] * 1.0
    reward_function[1, 0] = transition_function[1, 0, 0] * 0.0 + transition_function[1, 0, 1] * 1.0
    reward_function[1, 1] = transition_function[1, 1, 0] * 0.0 + transition_function[1, 1, 1] * 1.0
    # sensor model
    sensor_model = np.zeros((2, 2), dtype=np.float32)  # state, observation
    sensor_model[0, 0] = 0.6  # observe A when in A
    sensor_model[0, 1] = 0.4  # observe B when in A
    sensor_model[1, 0] = 0.4  # observe A when in B
    sensor_model[1, 1] = 0.6  # observe B when in B
    assert np.isclose(sensor_model.sum(-1), 1.0).all(), 'Sensor model is not normalized'
    gamma = 0.99 # discount factor - NOTE was 1.0 but Distributional RL doesnt support 1.0
    convergence_eps = 1e-3 # convergence threshold for PBVI and DPBVI
    max_itrs_ = 10000
    steps_before_belief_expansion_ = 160000 # NOTE: disabled
    beliefs_: np.ndarray[float] = np.linspace(0.0, 1.0, 20).reshape(-1, 1)
    beliefs_ = np.hstack([beliefs_, 1.0 - beliefs_])
    pbvi = PBVI(transition_function, sensor_model, reward_function, terminal_states, gamma)
    policy, beliefs = pbvi.plan(beliefs_, steps_before_belief_expansion_, max_itrs_, convergence_eps=convergence_eps)
    print(f'PBVI Policy: {policy} Values: {pbvi.values[:beliefs_.shape[0]]}')
    dpbvi = DPBVI(transition_function, sensor_model, reward_function, terminal_states, gamma, (0, 100))
    policy, beliefs = dpbvi.plan(beliefs_, max_itrs_, steps_before_belief_expansion=steps_before_belief_expansion_, convergence_eps=convergence_eps)
    print(f'DPBVI Policy: {policy} Values: {dpbvi.values[:beliefs_.shape[0]]}')
    print(f'All Values Close: {np.allclose(pbvi.values, dpbvi.values)}')
    denom = np.where(np.abs(pbvi.cached_values) > 1e-8, np.abs(pbvi.cached_values), 1e-8) # avoid division by zero
    max_abs_cached_val_rel_error = (np.abs(pbvi.cached_values - dpbvi.cached_values) / denom).max(axis=1) # take max across beliefs
    fig, ax = plt.subplots(figsize=(12, 8))
    ax = sns.lineplot(x=np.arange(max_abs_cached_val_rel_error.shape[0]), y=max_abs_cached_val_rel_error, ax=ax)
    ax.set_title('Maximum Value Relative Error per Iteration', fontsize=20)
    ax.set_xlabel('Iteration', fontsize=16)
    ax.set_ylabel('Max Relative Error', fontsize=16)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.yaxis.offsetText.set_fontsize(12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    plt.show()
