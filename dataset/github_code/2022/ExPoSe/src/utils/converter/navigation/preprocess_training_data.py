import pickle

import numpy as np
from tqdm import tqdm

from src.simulators.navigation import Navigation

states, actions, q_values = [], [], []

file = open('../qmdp-net/grid10_train.data', 'rb')
dataset = pickle.load(file, encoding='latin1')
file.close()

for state, trajectory in tqdm(dataset):
    # Calculate state Q-values from the trajectory
    t_states = []
    t_actions = []
    t_q_values = []
    l_trajectory = len(trajectory[1:])
    i = 0
    for action in trajectory[1:]:
        t_states.append(Navigation.to_tensor(state).squeeze().bool().numpy())
        t_actions.append(action)
        t_q_values.append(-(l_trajectory - i))

        state, _, _ = Navigation.simulate(state, action)
        i += 1

    states.extend(t_states)
    actions.extend(t_actions)
    q_values.extend(t_q_values)

data = {
    'states': states,
    'actions': np.asarray(actions),
    'values': np.asarray(q_values)
}

f = open('data/navigation/train/000.data', 'wb')
pickle.dump(data, f)
f.close()
