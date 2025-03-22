import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch as t

from .base import BaseSimulator

device = 'cuda' if t.cuda.is_available() else 'cpu'


class Actions:
    EAST = 0
    SOUTH = 1
    WEST = 2
    NORTH = 3


class Navigation(BaseSimulator):
    N = 50

    n_channels = 3
    max_steps = 200
    discount = 1
    failed = -200

    n_actions = 4

    ACTIONS = ['RIGHT', 'DOWN', 'LEFT', 'UP']

    levels = t.load('data/navigation/levels/test-5K.data')
    n_levels = len(levels)

    def __init__(self):
        super().__init__()

    @staticmethod
    def load_levels_from(file):
        Navigation.levels = t.load(file)
        Navigation.n_levels = len(Navigation.levels)

    def reset(self, idx=None, level=None):
        idx = random.randint(0, self.n_levels - 1) if idx is None else idx
        self.state = self.levels[idx]
        self.state = self.state[0]
        return self.state

    @staticmethod
    def render(state, as_image=True, searched_paths={}):
        grid, goal, robot = state

        string = np.array([' '] * Navigation.N ** 2)

        string[np.array(grid) == 1] = '#'
        string[goal] = '.'
        string[robot] = '@'

        if searched_paths.get('during_search') or searched_paths.get('during_rollouts'):
            visited_nodes = set()
            searched_paths['during_rollout'].discard('root')
            for path in searched_paths['during_rollout']:
                itr = state
                for action in path.lstrip('root->').split('->'):
                    itr, _, _ = Navigation.simulate(itr, int(action))
                    visited_nodes.add(itr[-1])
            for node in visited_nodes:
                if node != robot:
                    string[node] = 'R'

            visited_nodes = set()
            searched_paths['during_search'].discard('root')
            for path in searched_paths['during_search']:
                itr = state
                for action in path.lstrip('root->').split('->'):
                    itr, _, _ = Navigation.simulate(itr, int(action))
                    visited_nodes.add(itr[-1])
            for node in visited_nodes:
                if node != robot:
                    string[node] = 'S'

        string = string.reshape(Navigation.N, Navigation.N)
        if as_image:
            image = np.zeros((Navigation.N, Navigation.N, 3))
            image[string == '#', 0] = 0.5

            image[string == '.', 1] = 0.85

            image[string == '@', 0] = 1
            image[string == '@', 1] = 1
            image[string == '@', 2] = 0

            image[string == 'S', 0] = 0.5
            image[string == 'S', 1] = 0.5
            image[string == 'S', 2] = 0.5

            image[string == 'R', 0] = 0.25
            image[string == 'R', 1] = 0.25
            image[string == 'R', 2] = 0.25

            plt.clf()
            plt.imshow(image)
            plt.xticks([])
            plt.yticks([])
            plt.draw()
            plt.pause(0.0001)
            plt.savefig(f'runs/visualiser/{time.time()}.png')

        else:
            for i in range(Navigation.N):
                for j in range(Navigation.N):
                    print(string[i, j], end=' ')
                print()

    @staticmethod
    def simulate(state, action):
        grid, goal, robot = state

        reward = 0
        collision = False
        while action >= 0:
            x, y = robot // Navigation.N, robot % Navigation.N

            if (action % 4) == Actions.EAST:
                y = min(y + 1, Navigation.N - 1)
            elif (action % 4) == Actions.WEST:
                y = max(y - 1, 0)
            elif (action % 4) == Actions.NORTH:
                x = max(x - 1, 0)
            elif (action % 4) == Actions.SOUTH:
                x = min(x + 1, Navigation.N - 1)

            robot_next = x * Navigation.N + y

            reward += -1

            # if collision
            if grid[robot_next] == 1:
                robot_next = robot
                collision = True

            terminal = (robot_next == goal)

            action -= 4
            robot = robot_next

        reward += -1 * collision
        return (grid, goal, robot), reward, terminal

    @staticmethod
    def hash(state):
        grid, goal, robot = state
        return robot

    @staticmethod
    def is_solved(state):
        grid, goal, robot = state
        return goal == robot

    @staticmethod
    def to_tensor(state):
        grid, goal, agent = state
        t_grid = t.from_numpy(grid).float().to(device)

        t_agent = t.zeros_like(t_grid)
        t_agent[agent] = 1

        t_goal = t.zeros_like(t_grid)
        t_goal[goal] = 1

        return t.stack((t_grid, t_goal, t_agent)).reshape(1, 3, Navigation.N, Navigation.N)

    @staticmethod
    def legal_actions(state):
        return np.ones(Navigation.n_actions)

    @classmethod
    def get_sample_tensor(cls):
        return t.zeros(1, 3, Navigation.N, Navigation.N).to(device)
