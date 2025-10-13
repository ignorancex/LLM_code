import numpy as np


def randomized_di_cut(cost: np.ndarray):
    """
    Randomly assign nodes to either side of a directed cut
    :param cost:
    :return:
    """
    assert cost.ndim == 2
    assert cost.shape[0] == cost.shape[1]
    n = cost.shape[0]

    random_assignment = np.random.randint(2, size=n)
    adjacency = np.outer(1-random_assignment, random_assignment)
    adjacency[cost <= 0] = 0
    adjacency[np.diag_indices_from(adjacency)] = 1
    return (cost*adjacency).sum(), adjacency


def greedy_di_cut(cost: np.ndarray):
    """
    Greedy variant of the de-randomized version of the randomized di-cut algorithm.
    Algorithm 1 from the paper.
    :param cost:
    :return:
    """
    assert cost.ndim == 2
    assert cost.shape[0] == cost.shape[1]
    n = cost.shape[0]

    # set all negative costs to 0
    pos_cost = cost.copy()
    pos_cost[cost < 0] = 0
    # signed gains for assigning nodes to either side of the di-cut
    g = (pos_cost.sum(axis=1) - pos_cost.sum(axis=0)) / 4
    # node assignments. Initially all nodes are unassigned (-1).
    assignment = -np.ones(n, dtype=int)
    # greedily assign nodes to either 0 or 1
    for _ in range(n):
        # get unassigned node with best grain
        abs_g = np.abs(g)
        abs_g[assignment != -1] = -1
        i = np.argmax(abs_g)
        assert assignment[i] == -1

        # assign node based on sign of gain
        if g[i] >= 0:
            assignment[i] = 0
            g[assignment == -1] -= (pos_cost[i, assignment == -1] + pos_cost[assignment == -1, i]) / 4
        else:
            assignment[i] = 1
            g[assignment == -1] += (pos_cost[i, assignment == -1] + pos_cost[assignment == -1, i]) / 4

    # return value of di-cut and di-cut as adjacency matrix
    adjacency = np.outer(1 - assignment, assignment)
    adjacency[cost <= 0] = 0
    adjacency[np.diag_indices_from(adjacency)] = 1
    return (cost*adjacency).sum(), adjacency
