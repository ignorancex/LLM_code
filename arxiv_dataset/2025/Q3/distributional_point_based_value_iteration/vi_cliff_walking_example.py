import time
import numpy as np
from tqdm import tqdm
import seaborn as sns
import gymnasium as gym
from gymnasium.wrappers import RecordVideo, TimeLimit
from gymnasium.utils.play import play
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.animation import FuncAnimation


if __name__ == '__main__':
    # states: 4x12 grid - 36 is start, 47 is goal
    #         37 - 46 are the cliff
    # actions: up, right, down, left
    # transition function
    play_game = False
    deterministic = True
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
            assert np.isclose(transition_function[s].sum(), 4.0), 'Transition function is non-deterministic'
    # reward function
    reward_function = np.full((48, 4), -1, dtype=np.float32) # -1 for almost all transitions
    # handle cliff transitions
    reward_function[37:47, :] = -100 # cliff states - should never actually be reached though
    reward_function[36, 1] = -100 # start state - right off cliff
    reward_function[25:35, 2] = -100 # above cliff - down off cliff
    # handle goal
    reward_function[47, :] = 0 # goal state
    # print reward function
    # print('reward function for up')
    # print(' '.join([f'{r}' for r in reward_function[:12, 0]]))
    # print(' '.join([f'{r}' for r in reward_function[12:24, 0]]))
    # print(' '.join([f'{r}' for r in reward_function[24:36, 0]]))
    # print(' '.join([f'{r}' for r in reward_function[36:, 0]]))
    # print('reward function for right')
    # print(' '.join([f'{r}' for r in reward_function[:12, 1]]))
    # print(' '.join([f'{r}' for r in reward_function[12:24, 1]]))
    # print(' '.join([f'{r}' for r in reward_function[24:36, 1]]))
    # print(' '.join([f'{r}' for r in reward_function[36:, 1]]))
    # print('reward function for down')
    # print(' '.join([f'{r}' for r in reward_function[:12, 2]]))
    # print(' '.join([f'{r}' for r in reward_function[12:24, 2]]))
    # print(' '.join([f'{r}' for r in reward_function[24:36, 2]]))
    # print(' '.join([f'{r}' for r in reward_function[36:, 2]]))
    # print('reward function for left')
    # print(' '.join([f'{r}' for r in reward_function[:12, 3]]))
    # print(' '.join([f'{r}' for r in reward_function[12:24, 3]]))
    # print(' '.join([f'{r}' for r in reward_function[24:36, 3]]))
    # print(' '.join([f'{r}' for r in reward_function[36:, 3]]))
    # value function
    value_fn_dict = {
        'itr': list(),
        'values': list()
    }
    value_function = np.zeros(48, dtype=np.float32)
    gamma = 0.99
    theta = 0.01
    max_itrs = 400
    itr = 0
    for _ in tqdm(range(max_itrs)):
        value_fn_dict['itr'].append(itr)
        value_fn_dict['values'].append(value_function.copy())
        delta = 0.0
        for s in range(48):
            v = value_function[s]
            value_function[s] = (transition_function[s] * (np.repeat(np.expand_dims(reward_function[s, :], -1), 48, -1) + gamma * np.repeat(np.expand_dims(value_function, 0), 4,0))).sum(-1).max()  # NOTE: no sum bc transitions are deterministic
            delta = max(delta, np.abs(v - value_function[s]))
        itr += 1
        if delta < theta:
            break
    """
    Bellman Operator
    TV(s) = max_a sum_s' T(s, a, s')[R(s, a, s') + gamma * V(s')]
    """
    value_fn_dict['itr'].append(itr)
    value_fn_dict['values'].append(value_function.copy())
    # print(f'Value function converged after {itr} iterations')
    # print(value_function)
    if not play_game:
        # Plot value function over iterations
        fig = plt.figure(figsize=(10, 6))
        #fig, ax = plt.subplots(figsize=(10, 6))
        #sns.heatmap(value_fn_dict['values'][0].reshape(4, 12), annot=True, fmt='.1f', cmap='Blues_r', linewidths=1.0, ax=ax)
        #ax.set_title(f'Value function after {value_fn_dict["itr"][0]} iterations')
        def init():
            plt.clf()
            ax = sns.heatmap(value_fn_dict['values'][0].reshape(4, 12), annot=True, fmt='.1f', cmap='Blues_r', linewidths=1.0, vmin=-150, vmax=0)
            ax.set_title(f'Value function after {value_fn_dict["itr"][0]} iterations')
        def update(frame: int):
            #ax = fig.axes[0]
            #ax.clear()
            plt.clf()
            ax = sns.heatmap(value_fn_dict['values'][frame].reshape(4, 12), annot=True, fmt='.1f', cmap='Blues_r', linewidths=1.0, vmin=-150, vmax=0)
            ax.set_title(f'Value function after {value_fn_dict["itr"][frame]} iterations')
            return ax
        anim = FuncAnimation(fig, update, frames=len(value_fn_dict['itr']), repeat=False, interval=1000)
        anim.save('./value_function_convergence.mp4', fps=1)
        #plt.show()
        # Plot final value function
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(value_function.reshape(4, 12), annot=True, fmt='.1f', cmap='Blues_r', linewidths=1.0, ax=ax)
        ax.set_title(f'Value function converged after {itr} iterations')
        plt.show()
    # optimal policy
    policy = np.zeros(48, dtype=np.int64)
    for s in range(48):
        policy[s] = (transition_function[s] * (np.repeat(np.expand_dims(reward_function[s, :], -1), 48, -1) + gamma * np.repeat(np.expand_dims(value_function, 0), 4, 0))).sum(-1).argmax()  # NOTE: no sum bc transitions are deterministic
    if not play_game:
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
        # quiver.axes.xaxis.set_minor_locator(AutoMinorLocator(2))
        # quiver.axes.yaxis.set_minor_locator(AutoMinorLocator(2))
        quiver.axes.set_xticks(np.arange(12))
        quiver.axes.set_yticks(np.arange(4))
        quiver.axes.invert_yaxis()
        plt.grid(which='minor')
        plt.show()
    env = gym.make('CliffWalking-v0', render_mode='rgb_array' if record_video or play_game else 'human', is_slippery=not deterministic)
    if play_game:
        keys_to_actions = {
            'w': 0,  # up
            'd': 1,  # right
            's': 2,  # down
            'a': 3,  # left
        }
        play(env, keys_to_action=keys_to_actions, wait_on_player=True)
    else:
        if record_video:
            env = TimeLimit(env, max_episode_steps=500)
            env = RecordVideo(env, video_folder='cliff_walking', name_prefix=f'{"deterministic" if deterministic else "nondeterministic"}_cliff_walking_{"random" if use_random_policy else "learned"}_policy')
        for ep in range(1 if use_random_policy else 3):
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
