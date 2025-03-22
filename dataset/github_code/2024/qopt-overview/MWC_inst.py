"""Generates MaxCut ("MWC") instances by generating random graphs.

USAGE:
    python -m MWC_inst ./instances

Generates a set of MaxCut instances, saving original instance data in
``./instances`` (or another specified directory): ``orig`` subdirectory
for original instances and ``QUBO``, respectively, for QUBOs.

Instance parameters are specified in :py:func:`main`, which in turn calls the
core generation function, :py:func:`generate_MWC_instances`. This module also
contains a few cross-checks, auxiliary and
visualization code.
"""

import argparse
from pathlib import Path
from math import ceil
from time import time
import re
import os
import random
import json
import numpy as np
import networkx as nx
import gurobipy as gp
from gurobipy import GRB
from glob import glob
from qubo_utils import save_QUBO, load_QUBO, solve_QUBO
import matplotlib.pyplot as plt


###############################################################################
# Instance generation
def generate_instance(p, N, C):
    """Generates a (single) graph for a MaxCut instance.

    Uses Erdos-Renyi model for random graphs.

    Args:
        p: Erdos-Renyi parameter (probability of an edge between any two given vertices),
        N: number of vertices,
        C: maximum (absolute) value of the edge weight.

    Returns:
        Weighted graph G.
    """
    G = nx.erdos_renyi_graph(p=p, n=N)

    for (u, v) in G.edges():
        G.edges[u, v]['weight'] = C*random.random()

    return G


def draw_instance(G, ax=plt.gca(), node_color='blue',
                  node_size=200, with_labels=True):
    """Draws weighted graph G."""
    pos = nx.nx_agraph.graphviz_layout(G)
    edge_labels = {(u, v): f'{d["weight"]:.2f}'
                   for u, v, d in G.edges(data=True)}

    style_net = dict(node_color=node_color, node_size=node_size,
                     font_color='white',
                     with_labels=with_labels, width=1)
    nx.draw(G, pos, ax=ax, **style_net)
    if with_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                    ax=ax)


def draw_inst_grid(selected_files, Ncols = 3, directory="./instances-new/QUBO",
                   cross_check=True):
    """Draws a grid of instances from ``selected_files``.

    Args:
        selected_files(list): list of files (\*.qubo.json)
        Ncols(int): number of columns in the grid.
        directory(str): directory with QUBO files (\*.qubo.json)
    """
    k = 0
    Ncols = min(Ncols, len(selected_files))
    Nrows = ceil(len(selected_files) / Ncols)
    fig, ax = plt.subplots(Nrows, Ncols, figsize=(20,20 * Nrows / Ncols))

    for qubofile in selected_files:
        Q, P, C, qubojs = load_QUBO(qubofile)
        # solve classically -- MIP
        G, origjs = load_orig_MWC_instance(qubojs["description"]["original_instance_file"])

        if cross_check:
            t0 = time()
            model, z, x = create_MWC_LBOP(G, quiet=True)
            model.setParam("OutputFlag", 0)
            model.update()
            model.optimize()
            miptime = time()-t0
            assert model.status == 2, f"status = {model.status}"
            mipObj = model.objVal

            # solve as QUBO
            t0 = time()
            qubo_model, qubo_x = solve_QUBO(Q, P, C,quiet=True)
            qubotime = time() - t0
            assert qubo_model.status==2

            assert abs(mipObj + qubo_model.objVal)<1e-5, f"MIP: {mipObj}, QUBO: {qubo_model.objVal}"

            # show the results
            Gp_qubo = nx.Graph()
            Gp_qubo.add_nodes_from([j for j in G.nodes if qubo_x[j-1].X==1.0])

        N = len(G.nodes)
        if Nrows == 1:
            axk = ax[k]
        else:
            axk = ax[k//Ncols][k-Ncols*(k//Ncols)]

        draw_instance(G, ax = axk, node_color = 'blue', with_labels=False,
                      node_size=50)
        # ax_title = f"{os.path.basename(os.path.normpath(qubofile))[3:-10]} \n N:{N} | E:{len(G.edges)}/{N*(N-1)/2:.0f})"

        ax_title = f"{origjs['description']['instance_id']}: {N} nodes, {len(G.edges)} edges."

        if cross_check:
            draw_instance(Gp_qubo, ax=axk, node_color='red',
                        with_labels=False, node_size=50)
            ax_title += f"\nTmip:{miptime:.2f} s. | Tqubo:{qubotime:.2f}"

        axk.set_title(ax_title, fontsize=20)
        k+=1


def draw_IDs_grid(sel_IDs, directory = "./instances-new/QUBO", Ncols=3,
                  cross_check=False):
    """Draws a grid of instances by (the number part of) IDs.

    Args:
        sel_IDs(list[int]): list of instance IDs (numbers)
        directory(str): directory with QUBO files (\*.qubo.json)
        Ncols(int): number of columns in the grid

    Returns:
        selected file names.
    """
    selected_files = []

    for ID in sel_IDs:
        file_list = [fname for fname in glob(directory + "/MWC*.json") if re.search(r'.+MWC[0]*' + re.escape(str(ID)) + r'_.*\.qubo.json', fname)]
        if len(file_list)==0:
            raise ValueError(f"Instance with ID '{ID}' not found in {directory}.")
        elif len(file_list)>1:
            raise ValueError(f"Multiple instances with ID '{ID}' in {directory}:\n{file_list}.")

        selected_files.append(file_list[0])

    draw_inst_grid(selected_files, Ncols=Ncols, cross_check=cross_check)
    return selected_files

def create_MWC_QUBO(G: nx.Graph):
    """Given the graph G, creates a QUBO formulation.

    Args:
        G (nx.Graph): the underlying weighted graph,

    Returns:
        Q, P: QUBO coefficients as np.arrays

    Notes:
        - We assume QUBO in the form:
            min (1/2) x' Q x + P x
    """
    N = len(G.nodes)
    Q = np.zeros((N, N))
    P = np.zeros(N)

    for (i, j) in G.edges:
        Q[i, j] += 2*G.edges[i, j]["weight"]
        Q[j, i] += 2*G.edges[i, j]["weight"]
        P[i] -= G.edges[i, j]["weight"]
        P[j] -= G.edges[i, j]["weight"]

    return Q, P


def create_MWC_LBOP(G, quiet=False, timeout=None):
    """Creates a linear binary problem from the underlying graph G.

    Args:
        G (nx.Graph): the underlying weighted graph,
        quiet (bool): suppress model output (flag),
        timeout (int): timeout for the solver (seconds)

    Returns:
        a tuple of m (gurobipy model), z (z-vars), x (x-vars)
    """
    model = gp.Model("LBOP")

    # Defining variables
    z = {(i, j): model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{j}")
         for (i, j) in G.edges}

    x = {i: model.addVar(vtype=GRB.BINARY, name=f"x_{i}")
         for i in G.nodes}

    # Setting the objective
    model.setObjective(gp.quicksum(G.edges[i, j]["weight"]*z[(i, j)]
                                   for (i, j) in G.edges), GRB.MAXIMIZE)

    # Defining constraints
    for (i, j) in G.edges:
        model.addConstr(z[(i, j)] <= x[i] + x[j])
        model.addConstr(z[(i, j)] <= 2 - (x[i] + x[j]))

    if quiet:
        model.setParam("OutputFlag", 0)

    if timeout is not None:
        model.setParam("TimeLimit", timeout)

    return model, z, x


###############################################################################
# Original instance file I/O
def save_orig_MWC_instance(G, inst_id, problem_name, comment="",
                           directory="./instances/orig/", filename=None):
    """Saves a MaxCut instance encoded by graph G to a JSON (with metadata).

    Args:
        nodes: list of nodes,
        edges: list of tuples (i, j, w), where (i, j) is the edge, w is the weight
        inst_id (string): instance ID,
        problem_name (string): (relatively) human-readable name,
        comment: optional comment for the JSON file,
        directory: a dir to save files.
    """
    instance = {"nodes": [i for i in G.nodes],
               "edges": [(i, j, G.edges[i, j]["weight"]) for (i, j) in G.edges]}

    instance["description"] = {
        "instance_id": inst_id,
        "instance_type": "MWC",
        "original_instance_name": problem_name,
        "contents": "orig_MWC_G",
        "comment": "Nodes and weighted edges. " + comment
    }

    if filename is None:
        filename = f"{inst_id}_{problem_name}.orig.json"

    filename = directory + filename

    inst_json = json.dumps(instance, indent=4)
    with open(filename, "w") as outfile:
        outfile.write(inst_json)

    return filename


def extract_G_from_json(js) -> nx.Graph:
    """Helper: extracts the graph from the problem JSON.

    Assumes original JSON format (typically, in ``instances/orig/`` folder).
    """
    G = nx.Graph()
    G.add_nodes_from(js['nodes'])
    G.add_weighted_edges_from(js['edges'])
    return G


def load_instance_by_ID(ID, directory="./instances/orig/"):
    """Loads an instance by ID and returns graph G."""
    directory += "/"
    file_list = [fname for fname in glob(directory + ID + "_*.json")]

    if len(file_list)==0:
        raise ValueError(f"Instance with ID '{ID}' not found in {directory}.")
    elif len(file_list)>1:
        raise ValueError(f"Multiple instances with ID '{ID}' in {directory}:\n{file_list}.")

    with open(file_list[0], 'r') as ofile:
        js = json.load(ofile)

    if js["description"]["instance_type"] != "MWC":
        raise ValueError(f"Instance '{ID}' is not marked as MWC in 'instance_type' field. (see {file_list[0]})")

    return extract_G_from_json(js)

def load_orig_MWC_instance(filename):
    """Loads the instance from ``filename``."""
    with open(filename, 'r') as openfile:
        myjson = json.load(openfile)

    return extract_G_from_json(myjson), myjson


###############################################################################
# Instance generator
def generate_MWC_instances(K=3,
                           Ns=[5, 10, 15],
                           Ps=[0.1, 0.25, 0.5, 0.75, 0.9],
                           Cmin=1,
                           Cmax=10,
                           seed=2023,
                           instdir="./instances"):
    """Core function: generates a set of MWC instances with given parameters.

    Creates K * len(Ns) * len(Ps) instances, using the given random seed
    and the costs bounds, and saves them to directory ``instdir``.

    Args:
        K(int): number of instances per set of parameters,
        Ns(list[int]): instance sizes,
        Ps(list[float]): p parameter for ER random graph model,
        Cmin, Cmax(int): min/max edge costs,
        seed(int): random seed
        instdir(str): directory to save instances to.
    """
    assert (Cmin>=0) and (Cmax >0), \
        f"Can't sample costs from ({Cmin, Cmax}): nonnegative numbers expected."

    inst_id = 0

    random.seed(seed)

    for _ in range(K):
        for N in Ns:
            for p in Ps:
                inst_id += 1

                C = random.randint(Cmin, Cmax)
                G = generate_instance(p, N, C)
                Q, P = create_MWC_QUBO(G)

                orig_name = f"N{len(G.nodes)}E{len(G.edges)}_ERG_p{p}"
                # save the original instance
                orig_file = save_orig_MWC_instance(G, f"MWC{inst_id}",
                                                   orig_name,
                                                   directory = instdir+"/orig/")
                # save the QUBO
                save_QUBO(Q, P, 0.0, "MWC", orig_name, orig_file,
                                      f"MWC{inst_id}",
                                      comment = "Optimal objective is *minus* the QUBO objective.",
                          directory=instdir+"/QUBO/")
                print(".", end="", flush=True)


    print(f"\n✅ Done. Generated {inst_id} instances.")


def get_objective_from_sol(G: nx.Graph, sol: str) -> float:
    """Calculates a MaxCut objective given the graph and solution bistring.

    Note:
        Assumes **direct** order of bits in the solution.
        E.g., ``[n0, n1, n2, ...]``.

    """
    obj = 0.0
    for (i,j) in G.edges:
        if sol[i] != sol[j]:
            obj += G.edges[i, j]['weight']

    return obj

# The code to be run from the command line
def main():
    """The main function specifying instance generation parameters (see the source code)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("instdir", help="(output) instances directory",
                        default="./instances")
    args = parser.parse_args()
    if not Path(args.instdir).is_dir():
        raise ValueError(f"{args.instdir}: does not exist, or is not a directory.")

    generate_MWC_instances(Ns = [j for j in range(5, 256 + 1, 10)],
                           Ps = [0.25, 0.5, 0.75],
                           K=5,
                           instdir=args.instdir)

if __name__ == '__main__':
    main()

###############################################################################
# Cross-checks
def assert_MWC_MIP(orig_json, qubo_model,
                   quiet=True, tol=1e-3, timeout=None) -> None:
    """Asserts if the objectives for QUBO and original MWC (MIP) coincide."""
    G = extract_G_from_json(orig_json)
    m1, _, _ = create_MWC_LBOP(G, quiet=True)
    if quiet:
        m1.setParam("OutputFlag", 0)
    if timeout is not None:
        m1.setParam("TimeLimit", timeout)

    m1.update()
    m1.optimize()

    if timeout is None:
        assert m1.status == GRB.OPTIMAL
    else:
        assert (m1.status == GRB.OPTIMAL) or (m1.status == GRB.TIME_LIMIT)

    # note the "+": that's due to different sizes / QUBO formulation with 'min'
    assert abs(m1.objVal + qubo_model.objVal)<=tol
