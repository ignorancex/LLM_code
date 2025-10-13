import time
import numpy as np
from tqdm import tqdm
import seaborn as sns
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from agents.dvi import DVI


if __name__ == '__main__':
    # states: 4x12 grid - 36 is start, 47 is goal
    #         37 - 46 are the cliff
    # actions: up, right, down, left
    # transition function
    deterministic = False
    record_video = False
    use_random_policy = False
    transition_function = np.zeros((48, 4, 48), dtype=np.float32)
    if deterministic:
        for s in range(48):
            transition_function[s, 0, s - 12 if s - 12 >= 0 else s] = 1.0  # handles top row (no up action)
            transition_function[s, 1, s + 1 if s % 12 != 11 else s] = 1.0  # handles right edge (no right action)
            transition_function[s, 2, s + 12 if s + 12 <= 47 else s] = 1.0  # handles bottom row (no down action)
            transition_function[s, 3, s - 1 if s % 12 != 0 else s] = 1.0  # handles left edge (no left action)
            # handle cliff
            if 37 <= s <= 46:  # cliff states - should never actually be reached though
                transition_function[s, :, :] = 0.0
                transition_function[s, :, 36] = 1.0  # return to start
            if 25 <= s <= 34:  # above cliff
                transition_function[s, 2, :] = 0.0
                transition_function[s, 2, 36] = 1.0  # down is off cliff - return to start
            # handle start
            if s == 36:
                transition_function[s, 1, :] = 0.0
                transition_function[s, 1, 36] = 1.0  # right is off cliff - return to start
            # handle goal
            if s == 47:
                transition_function[s, :, :] = 0.0
                transition_function[s, :, 47] = 1.0 # NOTE: Goal state cant transition anywhere
            if s != 47:
                assert np.isclose(transition_function[s].sum(), 4.0), 'Transition function is non-deterministic'
    else:
        # slippery transitions move in direction of action 1/3 of time, and move in perpendicular directions 1/3 of time each (i.e., for action up, 1/3 of time move up, 1/3 of time move left, 1/3 of time move right)
        for s in range(48):
            up_next_state = s - 12 if s - 12 >= 0 else s
            up_slip_left_next_state = s - 1 if s % 12 != 0 else s
            up_slip_right_next_state = s + 1 if s % 12 != 11 else s

            for next_state in [up_next_state, up_slip_left_next_state, up_slip_right_next_state]:
                transition_function[s, 0, next_state] += 1/3

            right_next_state = s + 1 if s % 12 != 11 else s
            right_slip_up_next_state = s - 12 if s - 12 >= 0 else s
            right_slip_down_next_state = s + 12 if s + 12 <= 47 else s

            for next_state in [right_next_state, right_slip_up_next_state, right_slip_down_next_state]:
                transition_function[s, 1, next_state] += 1/3

            down_next_state = s + 12 if s + 12 <= 47 else s
            down_slip_left_next_state = s - 1 if s % 12 != 0 else s
            down_slip_right_next_state = s + 1 if s % 12 != 11 else s

            for next_state in [down_next_state, down_slip_left_next_state, down_slip_right_next_state]:
                transition_function[s, 2, next_state] += 1/3

            left_next_state = s - 1 if s % 12 != 0 else s
            left_slip_up_next_state = s - 12 if s - 12 >= 0 else s
            left_slip_down_next_state = s + 12 if s + 12 <= 47 else s

            for next_state in [left_next_state, left_slip_up_next_state, left_slip_down_next_state]:
                transition_function[s, 3, next_state] += 1/3

            # handle cliff
            if 37 <= s <= 46:  # cliff states - should never actually be reached though
                transition_function[s, :, :] = 0.0
                transition_function[s, :, 36] = 1.0  # return to start
            if 25 <= s <= 34:  # above cliff
                transition_function[s, 2, :] = 0.0
                transition_function[s, 2, 36] = 1/3  # down is off cliff - return to start
                transition_function[s, 2, s+1] = 1/3
                transition_function[s, 2, s-1] = 1/3
            # handle start
            if s == 36:
                transition_function[s, 1, :] = 0.0
                transition_function[s, 1, 36] = 2/3  # right is off cliff - return to start AND down stays at start
                transition_function[s, 1, 24] = 1/3
            # handle goal
            if s == 47:
                transition_function[s, :, :] = 0.0
                transition_function[s, :, 47] = 1.0 # NOTE: Goal state cant transition anywhere
            if s != 47:
                assert np.isclose(transition_function[s].sum(), 4.0), 'Transition function is non-deterministic'
    # reward function
    reward_function = np.full((48, 4), -1, dtype=np.float32) # -1 for almost all transitions
    # handle cliff transitions
    reward_function[37:47, :] = -100 # cliff states - should never actually be reached though
    reward_function[36, 1] = -100 # start state - right off cliff
    reward_function[25:35, 2] = -100 # above cliff - down off cliff
    # handle goal
    reward_function[47, :] = 0 # goal state
    # returns function - initial is uniform distribution
    num_particles = 51
    min_value, max_value = -150, 0
    particles = np.linspace(min_value, max_value, num_particles)
    particle_step_size = (max_value - min_value) / (num_particles - 1)
    gamma = 0.99
    theta = 0.01
    delta = 1.0
    max_itrs = 400

    # DEBUG: test DVI
    terminal_states = np.isin(np.arange(48), [47])
    dvi = DVI(transition_function, reward_function, terminal_states, gamma, (min_value, max_value), num_particles=num_particles, cache_etas=True)
    policy = dvi.plan(max_itrs=max_itrs, convergence_thresh=theta)
    print(policy)
    # plot value function
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(dvi.eta[36, 0], ax=ax)
    ax.set_xticks([i for i, p in enumerate(dvi._particles) if i % 5 == 0 or i == dvi._n_particles - 1],
                  labels=[p.item() for i, p in enumerate(dvi._particles) if i % 5 == 0 or i == dvi._n_particles - 1])
    ax.set_title(f'Sample Distribution') # Distribution of Up Action for State {36}
    ax.set_ylim(0.0, 1.0)
    ax.axvline((np.abs(dvi._particles - dvi.eta[36, 0].dot(dvi._particles)) < np.abs(dvi._particles[1] - dvi._particles[0]) / 2).nonzero()[0][0],
               color='r', linestyle='--', label='Expectation') # HACK
    ax.legend()
    plt.show()
    # plot etas
    f, axes = plt.subplots(2, 2, figsize=(10, 6))
    def update(frame: int):
        # plt.clf()
        #f, axes = plt.subplots(2, 12, figsize=(40, 10))
        action_lbls = ['Up', 'Right', 'Down', 'Left']
        x_ticks = [i for i, p in enumerate(particles) if i % 5 == 0 or i == num_particles - 1]
        x_tick_lbls = [f'{int(p)}' for i, p in enumerate(particles) if i % 5 == 0 or i == num_particles - 1]
        for r in range(2):
            for c in range(2):
                ax = axes[r][c] #r-2
                ax.clear()
                #s = r * 12 + c
                s = 36
                a = r * 2 + c
                #optimal_action = dvi.eta_cache[frame][s].dot(particles).argmax(-1)
                sns.barplot(dvi.eta_cache[frame][s, a], ax=ax)
                ax.set_xticks(x_ticks, x_tick_lbls)
                ax.set_title(f'Action {action_lbls[a]}')
                ax.set_ylim(0.0, 1.0)
        f.suptitle(f'Start State Action Distributions at Iteration {frame}')
        plt.tight_layout()
    anim = FuncAnimation(f, update, frames=len(dvi.eta_cache), repeat=False, interval=100)
    anim.save('./dvi_start_state_action_dists.mp4')
    plt.show()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(dvi.eta.dot(particles).max(-1).reshape(4, 12), annot=True, fmt='.1f', cmap='Blues_r', linewidths=1.0, ax=ax)
    ax.set_title(f'Value function converged after {dvi.converge_itrs} iterations')
    plt.show()
    # extract policy
    policy = dvi.eta.dot(particles).argmax(-1)
    # plot policy
    x, y = np.meshgrid(np.arange(12), np.arange(4))
    policy_reshape = policy.reshape(4, 12)
    # actions: up, right, down, left
    x_dir, y_dir = np.zeros_like(policy_reshape), np.zeros_like(policy_reshape)
    up_mask = policy_reshape == 0
    right_mask = policy_reshape == 1
    down_mask = policy_reshape == 2
    left_mask = policy_reshape == 3
    y_dir[up_mask] = 1
    x_dir[right_mask] = 1
    y_dir[down_mask] = -1
    x_dir[left_mask] = -1
    quiver = plt.quiver(x, y, x_dir, y_dir)
    quiver.axes.set_title('Learned Policy')
    quiver.axes.invert_yaxis()
    plt.show()
    # plot state-value distributions
    # f, axes = plt.subplots(4, 12, figsize=(40, 10))
    # for r in range(4):
    #     for c in range(12):
    #         ax = axes[r][c]
    #         s = r * 12 + c
    #         sns.barplot(dvi.eta[s].sum(0) / 4, ax=ax)
    #         ax.set_xticklabels([f'{p:.0f}' for p in particles])
    #         ax.set_title(f'State {s}')
    #         ax.set_ylim(0.0, 1.0)
    # f.suptitle('Value Distributions by State')
    # plt.tight_layout()
    # plt.show()
    # plot greedy policy value distributions
    f, axes = plt.subplots(4, 12, figsize=(40, 10))
    x_ticks = [i for i, p in enumerate(particles) if i % 5 == 0 or i == num_particles - 1]
    x_tick_lbls = [f'{int(p)}' for i, p in enumerate(particles) if i % 5 == 0 or i == num_particles - 1]
    for r in range(4):
        for c in range(12):
            ax = axes[r][c]
            s = r * 12 + c
            optimal_action = dvi.eta[s].dot(particles).argmax(-1)
            sns.barplot(dvi.eta[s, optimal_action], ax=ax)
            ax.set_xticks(x_ticks, x_tick_lbls)
            ax.set_title(f'State {s}')
            ax.set_ylim(0.0, 1.0)
    f.suptitle('Greedy Policy Value Distributions by State')
    plt.tight_layout()
    plt.show()
    env = gym.make('CliffWalking-v0', render_mode='rgb_array' if record_video else 'human', is_slippery=not deterministic)
    if record_video:
        env = RecordVideo(env, video_folder='cliff_walking', name_prefix=f'{"deterministic" if deterministic else "nondeterministic"}_cliff_walking_{"random" if use_random_policy else "learned"}_dist_policy')
    for ep in range(1 if record_video else 10):
        state, info = env.reset()
        done = False
        while not done:
            if use_random_policy:
                action = env.action_space.sample()
            else:
                action = policy[state]
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
            time.sleep(0.5)
    env.close()
