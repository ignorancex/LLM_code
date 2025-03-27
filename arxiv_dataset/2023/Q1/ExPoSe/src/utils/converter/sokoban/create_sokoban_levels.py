import glob
import multiprocessing as mp
import os
import pickle

import numpy as np
from tqdm import tqdm


def convert(f):
    data = []

    file_level = open('../boxoban-levels/unfiltered/train/{:03d}.txt'.format(f), 'r')
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

        data.append((walls, agent[0], boxes, goals))

    file_level.close()

    file = open('{}/{:03d}.data'.format(folder, f), 'wb')
    pickle.dump(data, file)
    file.close()


folder = 'data/sokoban/temp/'
folder = os.path.dirname(folder)
if not os.path.exists(folder):
    os.makedirs(folder)

mp.Pool(50).map(func=convert, iterable=range(900))

# combine files into single file
levels = []
for fname in tqdm(glob.glob('data/sokoban/temp/*')):
    with open(fname, 'rb') as f:
        levels.extend(pickle.load(f))
    os.remove(fname)

with open('data/sokoban/temp/combined.data', 'wb') as f:
    pickle.dump(levels, f)
    f.close()
