import glob
import os

import imageio
import matplotlib.pyplot as plt
from rich import print

from .configs import SIMULATOR


def default_pre_step(simulator, state, action):
    simulator.render(state)


def default_post_step(simulator, next_state, action, reward, terminal):
    print(f'action:     {simulator.ACTIONS[action]}\n'
          f'reward:     {reward}\n'
          f'terminal:   {terminal}\n')


def default_pre_episode(*args):
    plt.ion()
    plt.show()

    if not os.path.exists('runs/visualiser'):
        os.makedirs('runs/visualiser')
    os.system('rm runs/visualiser/*.png')


def default_post_episode(state):
    SIMULATOR.render(state)
    images = sorted(glob.glob('runs/visualiser/*.png'))
    if images:
        images = [imageio.imread(file) for file in images]
        imageio.mimsave('runs/visualiser/movie.gif', images, fps=5)
        os.system('rm runs/visualiser/*.png')


def simulate_episode(get_simulator,
                     get_action,
                     pre_step=default_pre_step,
                     post_step=default_post_step,
                     pre_episode=default_pre_episode,
                     post_episode=default_post_episode):
    simulator, state = get_simulator()

    pre_episode(simulator, state)

    episode_reward = 0
    success = False
    for i in range(simulator.max_steps):
        action = get_action(state)

        pre_step(simulator, state, action)

        next_state, reward, terminal = simulator.step(action)
        episode_reward += reward * (simulator.discount ** i)

        post_step(simulator, next_state, action, reward, terminal)
        state = next_state

        if terminal:
            success = simulator.is_solved(state)
            break

    post_episode(state)

    return episode_reward if success else SIMULATOR.failed
