import multiprocessing as mp
import os
import pickle

import numpy as np


def convert(f):
    data = []

    file_level = open('../sokoban-solver/levels/{:03d}.txt'.format(f), 'r')
    file_expert = open('../sokoban-solver/experts/{:03d}.txt'.format(f), 'r')
    print('{:03d}.txt'.format(f))
    while True:
        # don't store level id
        idx_level = file_level.readline()
        if not idx_level:
            break
        idx_level = int(idx_level.rstrip('\n').lstrip('; '))

        level = ''
        for i in range(10):
            level += file_level.readline().rstrip('\n')

        level = np.array(list(level))

        walls = (level == '#')
        agent = np.argwhere(level == '@').reshape(-1)
        boxes = np.argwhere(level == '$').reshape(-1)
        goals = np.argwhere(level == '.').reshape(-1)

        file_level.readline()

        idx_expert, trajectory = file_expert.readline().rstrip('\n').split(':')
        idx_expert = int(idx_expert)

        if 'Failed' in trajectory:
            continue

        # sanity check
        if idx_level != idx_expert:
            print('IDs do not match!!', idx_level, idx_expert)

        trajectory = np.array(list(trajectory.replace(' ', '')))

        data.append(((walls, agent[0], boxes, goals), trajectory))

    file_level.close()
    file_expert.close()

    file = open('{}/{:03d}.data'.format(folder, f), 'wb')
    pickle.dump(data, file)
    file.close()


folder = 'data/sokoban/trajectories/'
folder = os.path.dirname(folder)
if not os.path.exists(folder):
    os.makedirs(folder)

mp.Pool(50).map(func=convert, iterable=range(900))
