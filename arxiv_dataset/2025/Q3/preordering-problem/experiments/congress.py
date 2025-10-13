import networkx as nx
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton, MouseEvent

import sys
sys.path.append('../')

from preordering_problem.ilp_solver import Preorder
from preordering_problem.decompose_preorder import decompose_preorder


"""
Preordering Twitter accounts of members of the 117th US Congress.
The data can be downloaded from here: https://snap.stanford.edu/data/congress-twitter.html 
The party membership file is contained in the `data` directory.
"""


data_root = "../data"


colors = {
    "REP": (232, 27, 35),
    "DEM": (0, 174, 243),
    "IND": (170, 170, 170)
}

colors = {p: np.array(c) / 255 for p, c in colors.items()}


def load_congress():
    """
    :return: usernames, party membership and weight matrix for US congress graph.
    """
    with open(f"{data_root}/congress_network/congress_network_data.json", "r") as f:
        d = json.load(f)[0]

    user_names = d["usernameList"]
    n = len(user_names)
    # load weights
    weights = np.zeros((n, n))
    for i, (j, w) in enumerate(zip(d["outList"], d["outWeight"])):
        weights[i, j] = w
    # load parties
    parties = []
    with open(f"{data_root}/congress_network/parties.txt", "r") as f:
        for i, line in enumerate(f.readlines()):
            name, party = line.strip().split(", ")
            assert name == user_names[i]
            parties.append(party)
    return user_names, parties, weights


def plot_congress(g, clustering, parties, user_names):
    """
    Visualize preorder of congress members.
    Click on a node to print the usernames of the members in that cluster

    :param g:
    :param clustering:
    :param parties:
    :param user_names:
    :return:
    """
    pos = nx.nx_pydot.graphviz_layout(g, prog="dot")
    pos = {i: [-pos[i][1], pos[i][0]] for i in pos}

    fig, ax = plt.subplots(figsize=(7, 15))
    node_color = [np.array([colors[parties[x]] for x in clustering[i]]).mean(axis=0) for i in g.nodes]
    nx.draw(g, ax=ax, pos=pos, node_size=[5 * len(clustering[i]) for i in g.nodes],
            node_color=node_color, width=0.2, arrowsize=6)
    pos_arr = np.array([pos[i] for i in range(len(pos))])
    fig.tight_layout()
    handles = []

    def on_click(event: MouseEvent):
        if event.button == MouseButton.RIGHT:
            for h in handles:
                h.remove()
            handles.clear()
        if event.button == MouseButton.LEFT and event.inaxes == ax:
            for h in handles:
                h.remove()
            handles.clear()
            dist = np.linalg.norm(pos_arr - [event.xdata, event.ydata], axis=1)
            i = np.argmin(dist)
            names = [user_names[c] for c in clustering[i]]
            print(names)
            x, y = pos_arr[i]
            handles.append(plt.scatter([x], [y], edgecolors=(1, 1, 0, 1), color=(1, 1, 0, 0.5), s=100, linewidths=2))
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()


def main():
    # load data
    user_names, parties, weights = load_congress()
    # create cost matrix
    offset = 0.01
    costs = weights - offset
    costs[np.diag_indices_from(costs)] = 0

    # solve preorder problem
    preorder = Preorder(costs, binary=True, suppress_log=True)
    preorder_obj = preorder.solve()
    adjacency = preorder.get_variable_values()
    print("Preorder objective:", preorder_obj)
    print("Preorder bound:", costs[costs > 0].sum())

    # plot results
    g, clustering = decompose_preorder(adjacency)
    plot_congress(g, clustering, parties, user_names)

    # solve clustering problem
    clustering = Preorder(costs, binary=True, suppress_log=True)
    clustering.add_symmetric_constraints()
    clustering_obj = clustering.solve()
    print("Clustering objective:", clustering_obj)


if __name__ == "__main__":
    main()
