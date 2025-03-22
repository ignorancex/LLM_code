import time
from copy import deepcopy

import matplotlib.pyplot as plt
import networkx
import numpy as np
import torch as t
from rich import print

from .base import BaseSimulator

device = 'cuda' if t.cuda.is_available() else 'cpu'


class HamiltonianCycle(BaseSimulator):
    n_nodes = 50
    sparsity = 0.10

    max_steps = n_nodes
    discount = 1

    failed = -1000

    n_actions = n_nodes

    ACTIONS = np.arange(n_actions)

    levels = np.asarray(t.load('data/hamiltonian_cycle/tsp-n_{}-s_{:.2f}.level'.format(n_nodes, sparsity)), dtype=object)
    n_levels = len(levels)

    def __init__(self):
        super().__init__()

    @staticmethod
    def load_levels_from(file):
        HamiltonianCycle.levels = np.asarray(t.load(file), dtype=object)
        HamiltonianCycle.n_levels = len(HamiltonianCycle.levels)

    def reset(self, idx=None, state=None):
        if idx is not None:
            state = self.levels[idx]

        if state is None:
            cycle = np.random.RandomState().permutation(HamiltonianCycle.n_nodes)

            # choose random sparsely connected edges
            edges = np.triu(np.random.RandomState().binomial(n=1, p=HamiltonianCycle.sparsity, size=(HamiltonianCycle.n_nodes, HamiltonianCycle.n_nodes)))

            # add self edges
            edges = np.logical_or(edges, np.eye(HamiltonianCycle.n_nodes)).astype(np.uint8)

            # match the lower half with upper half
            edges = np.logical_or(edges, edges.T).astype(np.uint8)

            # add the edges from the cycle
            edges[cycle[0], cycle[-1]] = edges[cycle[-1], cycle[0]] = 1
            for i, j in zip(cycle[:-1], cycle[1:]):
                edges[i, j] = edges[j, i] = 1

            start_state = cycle[0]
            state = (edges, start_state, [start_state])

        self.state = state
        return self.state

    @staticmethod
    def render(state, as_image=True, searched_paths={}):
        edges, start_state, visited = state

        if as_image:
            graph = networkx.convert_matrix.from_numpy_matrix(edges)

            visited_edges = [(visited[i - 1], visited[i]) for i in range(1, len(visited))] + [(visited[i], visited[i - 1]) for i in range(1, len(visited))]
            searched_edges = set()

            if searched_paths.get('during_search') or searched_paths.get('during_rollouts'):
                searched_paths['during_rollout'].discard('root')
                searched_paths['during_search'].discard('root')
                for path in searched_paths['during_search'].union(searched_paths['during_rollout']):
                    searched_nodes = [visited[-1]] + list(path.lstrip('root->').split('->'))
                    searched_edges = searched_edges.union({(int(searched_nodes[i - 1]), int(searched_nodes[i])) for i in range(1, len(searched_nodes))}) \
                        .union({(int(searched_nodes[i]), int(searched_nodes[i - 1])) for i in range(1, len(searched_nodes))})

            plt.clf()
            networkx.draw_shell(graph,
                                node_color=['red' if n == visited[-1] else ('yellow' if n == visited[0] else ('green' if n in visited else 'blue')) for n in graph.nodes],
                                edge_color=['green' if e in visited_edges else ('red' if e in searched_edges else 'black') for e in graph.edges],
                                width=[1 if e in visited_edges else (0.5 if e in searched_edges else 0.1) for e in graph.edges],
                                with_labels=True)
            plt.xticks([])
            plt.yticks([])
            plt.title(f'Nodes: {HamiltonianCycle.n_nodes}, Sparsity: {HamiltonianCycle.sparsity}')
            plt.draw()
            plt.pause(0.0001)
            plt.savefig(f'runs/visualiser/{time.time()}.png')

            pass
        else:
            for s in visited[:-1]:
                print(f'{s}->', end='')
            print(visited[-1])

    @staticmethod
    def simulate(state, action):
        edges, start_state, visited = deepcopy(state)

        reward = -1
        terminal = (len(visited) == len(edges)) and action == visited[0] and edges[visited[-1], action] == 1
        if terminal:
            visited.append(action)
        else:
            # check if the edge exists or if the node has been visited before
            if edges[visited[-1], action] == 0 or action in visited:
                reward = -100
                terminal = True
            else:
                visited.append(action)

        return (edges, start_state, visited), reward, terminal

    @classmethod
    def hash(cls, state):
        edges, start_state, visited = state
        return tuple(visited)

    @classmethod
    def is_solved(cls, state):
        edges, start_state, visited = state
        return (len(visited) == len(edges) + 1) and visited[-1] == visited[0]

    @staticmethod
    def to_tensor(state):
        edges, start_state, visited = deepcopy(state)
        visited.insert(0, visited[0])

        t_edges = t.from_numpy(edges).float().to(device)

        t_features = t.FloatTensor([[0, 1, 0, 1, 0, 1, 0, 1]]).repeat(len(edges), 1).to(device)

        t_features[start_state, 0] = 1
        t_features[start_state, 1] = 0

        t_features[visited[-2], 2] = 1
        t_features[visited[-2], 3] = 0

        t_features[visited[-1], 4] = 1
        t_features[visited[-1], 5] = 0

        for s in visited:
            t_features[s, 6] = 1
            t_features[s, 7] = 0

        return t_edges, t_features

    @staticmethod
    def legal_actions(state):
        edges, _, visited = state
        current_node = visited[-1]
        return edges[current_node]

    @staticmethod
    def get_sample_tensor():
        return t.ones(1, HamiltonianCycle.n_nodes, HamiltonianCycle.n_nodes).float().to(device), t.ones(1, HamiltonianCycle.n_nodes, 8).float().to(device)
