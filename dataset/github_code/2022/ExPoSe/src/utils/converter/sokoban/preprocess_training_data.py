import multiprocessing as mp
import os
import pickle

import numpy as np
from tqdm import tqdm

from src.simulators import Sokoban

actions_mapping = {'r': 0, 'd': 1, 'l': 2, 'u': 3}


def preprocess(f):
    print('data/sokoban/trajectories/{:03d}.data'.format(f))
    file = open('data/sokoban/trajectories/{:03d}.data'.format(f), 'rb')
    dataset = pickle.load(file)
    file.close()

    states, actions, q_values = [], [], []
    for state, trajectory in tqdm(dataset, position=f):
        # Calculate state Q-values from the trajectory
        t_states = []
        t_actions = []

        rewards = []
        for i, action in enumerate(trajectory):
            action = actions_mapping[action]
            t_states.append(Sokoban.to_tensor(state).squeeze().bool().numpy())
            t_actions.append(action)

            state, r, _ = Sokoban.simulate(state, action)
            rewards.append(r)
            i += 1

        rewards.reverse()
        rewards = np.cumsum(rewards).tolist()
        rewards.reverse()

        states.extend(t_states)
        actions.extend(t_actions)
        q_values.extend(rewards)

    data = {
        'states': states,
        'actions': np.asarray(actions),
        'q_values': np.asarray(q_values)
    }

    file = open('data/sokoban/preprocessed/{:03d}.data'.format(f), 'wb')
    pickle.dump(data, file)
    file.close()


folder = 'data/sokoban/preprocessed/'
folder = os.path.dirname(folder)
if not os.path.exists(folder):
    os.makedirs(folder)

# Run in parallel
mp.Pool(50).map(func=preprocess, iterable=range(900))
