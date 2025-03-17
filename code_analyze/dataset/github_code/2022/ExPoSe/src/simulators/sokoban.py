import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch as t

from .base import BaseSimulator

device = 'cuda' if t.cuda.is_available() else 'cpu'


class Sokoban(BaseSimulator):
    n_channels = 4
    max_steps = 120
    discount = 1

    failed = -1000

    n_actions = 4

    ACTIONS_EAST = 0
    ACTIONS_SOUTH = 1
    ACTIONS_WEST = 2
    ACTIONS_NORTH = 3

    ACTIONS = ['RIGHT', 'DOWN', 'LEFT', 'UP']

    TRANSITION = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 49,
         51, 52, 53, 54, 55, 56, 57, 58, 59, 59, 61, 62, 63, 64, 65, 66, 67, 68, 69, 69, 71, 72, 73, 74, 75, 76, 77, 78, 79, 79, 81, 82, 83, 84, 85, 86, 87, 88, 89, 89, 91, 92, 93, 94, 95, 96, 97, 98,
         99, 99],
        [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
         58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 90, 91, 92, 93, 94, 95,
         96, 97, 98, 99],
        [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 40, 41, 42, 43, 44, 45, 46, 47, 48,
         50, 50, 51, 52, 53, 54, 55, 56, 57, 58, 60, 60, 61, 62, 63, 64, 65, 66, 67, 68, 70, 70, 71, 72, 73, 74, 75, 76, 77, 78, 80, 80, 81, 82, 83, 84, 85, 86, 87, 88, 90, 90, 91, 92, 93, 94, 95, 96,
         97, 98],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
         43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89]
    ]

    REWARD_MOVE = -1

    levels = np.asarray(np.load('data/sokoban/levels/test/hard.data', allow_pickle=True, encoding='latin1'), dtype=object)
    n_levels = len(levels)

    d_state = (4, 10, 10)

    def __init__(self):
        super().__init__()

    @staticmethod
    def load_levels_from(file):
        Sokoban.levels = np.asarray(np.load(file, allow_pickle=True, encoding='latin1'), dtype=object)
        Sokoban.n_levels = len(Sokoban.levels)

    def reset(self, idx=None, level=None):
        idx = idx if idx is not None else random.randint(0, self.n_levels - 1)
        self.state = self.levels[idx]
        if level:
            level = str(level).replace('\n', '')
            level = np.array(list(level))

            walls = (level == '#')
            agent = np.argwhere(level == '@').reshape(-1)
            boxes = np.argwhere(level == '$').reshape(-1)
            goals = np.argwhere(level == '.').reshape(-1)

            self.state = (walls, agent[0], boxes, goals)

        return self.state

    @staticmethod
    def render(state, as_image=True, searched_paths=()):
        walls, agent, boxes, goals = state

        string = np.array([' '] * 100)

        string[walls == 1] = '#'
        string[goals] = '.'
        string[boxes] = '$'
        string[agent] = '@'

        string = string.reshape(10, 10)

        if as_image:
            image = np.zeros((10, 10, 3))
            image[string == '#', 0] = 0.5

            image[string == '.', 1] = 0.85

            image[string == '$', 2] = 0.85

            image[string == '@', 0] = 1
            image[string == '@', 1] = 1

            plt.clf()
            plt.imshow(image)
            plt.xticks([])
            plt.yticks([])
            plt.draw()
            plt.pause(0.0001)
            plt.savefig(f'runs/visualiser/{time.time()}.png')

        else:
            for i in range(10):
                for j in range(10):
                    print(string[i, j], end=' ')
                print()

    @staticmethod
    def simulate(state, action):
        walls, agent, boxes, goals = state

        reward = -1
        agent_next = Sokoban.TRANSITION[action][agent]

        # if agent's next location is a wall, then revert
        if walls[agent_next] == 1:
            agent_next = agent
            reward += -1

        # check if agent pushes a box
        boxes_next = list(boxes)
        if agent_next in boxes:
            box_next = Sokoban.TRANSITION[action][agent_next]

            # if next position for moved box is a box or wall, then revert
            if walls[box_next] == 1 or box_next in boxes:
                agent_next = agent
                reward += -1
            # else change the box position
            else:
                if agent_next in goals:
                    reward += -10
                if box_next in goals:
                    reward += 10
                idx = boxes_next.index(agent_next)
                boxes_next[idx] = box_next

                # if a box is moved to terminal failure position, then return fail with negative reward
                for a in range(Sokoban.n_actions):
                    if walls[Sokoban.TRANSITION[a][box_next]] and walls[Sokoban.TRANSITION[(a + 1) % 4][box_next]] and box_next not in goals:
                        return (walls, agent_next, boxes_next, goals), -1000, True

        if set(boxes_next) == set(goals):
            reward = 100

        return (walls, agent_next, boxes_next, goals), reward, (set(boxes_next) == set(goals))

    @staticmethod
    def hash(state):
        walls, robot, boxes, goals = state
        return int(robot * 1e8 + boxes[0] * 1e6 + boxes[1] * 1e4 + boxes[2] * 1e2 + boxes[3])

    @staticmethod
    def is_solved(state):
        walls, robot, boxes, goals = state
        return set(boxes) == set(goals)

    @staticmethod
    def to_tensor(state):
        walls, agent, boxes, goals = state

        t_walls = t.from_numpy(walls).float().to(device)

        t_agent = t.zeros_like(t_walls)
        t_agent[agent] = 1

        t_boxes = t.zeros_like(t_walls)
        t_boxes[boxes] = 1

        t_goals = t.zeros_like(t_walls)
        t_goals[goals] = 1

        return t.stack((t_walls, t_agent, t_boxes, t_goals)).reshape(1, 4, 10, 10)

    @staticmethod
    def legal_actions(state):
        return np.ones(Sokoban.n_actions)

    @staticmethod
    def get_sample_tensor():
        return t.zeros(1, 4, 10, 10).to(device)
