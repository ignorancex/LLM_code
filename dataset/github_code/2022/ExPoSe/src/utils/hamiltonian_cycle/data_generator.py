import argparse
import os
from time import sleep

import numpy as np
import torch as t
import torch.multiprocessing as mp
from tqdm import tqdm

from ...simulators import HamiltonianCycle

t.set_num_threads(1)

os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'


def define_a_graph():
    n_nodes = HamiltonianCycle.n_nodes
    cycle = np.random.RandomState().permutation(n_nodes)

    # choose random sparsely connected edges
    edges = np.triu(np.random.RandomState().binomial(n=1, p=HamiltonianCycle.sparsity, size=(n_nodes, n_nodes)))

    # add self edges
    edges = np.logical_or(edges, np.eye(n_nodes)).astype(np.uint8)

    # match the lower half with upper half
    edges = np.logical_or(edges, edges.T).astype(np.uint8)

    # add the edges from the cycle
    edges[cycle[0], cycle[-1]] = edges[cycle[-1], cycle[0]] = 1
    for i, j in zip(cycle[:-1], cycle[1:]):
        edges[i, j] = edges[j, i] = 1

    start_state = cycle[0]
    state = (edges, start_state, [start_state])

    return cycle, state


def episode():
    cycle, state = define_a_graph()

    simulator = HamiltonianCycle()
    simulator.reset(state)

    t_states = []
    t_actions = []
    t_values = []

    cycle = cycle.tolist()
    cycle.append(cycle[0])

    rewards = []
    for action in cycle[1:]:
        t_edges, t_features = simulator.to_tensor(state)
        t_states.append((t_edges.cpu().bool().numpy(), t_features.cpu().bool().numpy()))
        t_actions.append(action)
        state, reward, _ = simulator.simulate(state, action)
        rewards.append(reward)

    rewards.reverse()
    rewards = np.cumsum(rewards).tolist()
    rewards.reverse()

    t_values.extend(rewards)

    return t_states, t_actions, t_values


def collector(idx, params):
    episodes, itr, lock, n_levels = params['episodes'], params['itr'], params['lock'], params['n_levels']

    t.cuda.set_device(-1)

    while True:
        lock.acquire()
        local_itr = itr.value
        itr.value += 1
        lock.release()

        if local_itr >= n_levels:
            return

        with t.no_grad():
            states, actions, values = episode()
            episodes[local_itr] = (states, actions, values)


if __name__ == '__main__':
    mp.set_start_method('spawn', True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--save', default='data/hamiltonian_cycle/0.data')
    parser.add_argument('--n-workers', type=int, default=1)
    parser.add_argument('--n-levels', type=int, default=1)
    args = parser.parse_args()

    print(args)

    episodes = mp.Manager().list([None] * args.n_levels)
    lock = mp.RLock()
    itr = mp.Value('i', 0)

    params = {
        'episodes': episodes,
        'itr': itr,
        'lock': lock,
        'n_levels': args.n_levels
    }

    processes = [mp.Process(target=collector, args=(idx, params)) for idx in range(args.n_workers)]
    [p.start() for p in processes]

    # Start the progress bar
    pbar = tqdm(total=args.n_levels, desc='Progress')
    prev_itr = 0
    local_itr = 0
    while local_itr < args.n_levels:
        sleep(0.5)
        local_itr = itr.value

        pbar.set_description('Progress')
        pbar.update(local_itr - prev_itr)

        prev_itr = local_itr

    [p.join() for p in processes]
    pbar.update(args.n_levels - prev_itr)

    list_states, list_actions, list_values = [], [], []
    for states, actions, values in list(episodes):
        list_states.extend(states)
        list_actions.extend(actions)
        list_values.extend(values)

    data = {'states': list_states,
            'actions': list_actions,
            'values': list_values}

    t.save(data, args.save)
