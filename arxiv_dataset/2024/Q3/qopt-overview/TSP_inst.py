"""Generates TSP instances by sampling nodes from TSPLIB files.

USAGE:
    python -m TSP_inst ./instances

Generates a set of TSP instances, saving original instance data in
``./instances`` (or another specified directory): ``orig`` subdirectory
for original instances and ``QUBO``, respectively, for QUBOs.

Instance parameters are specified in :py:func:`main`, which in turn calls the
core generation function, :py:func:`generate_TSP_instances`. This module also
contains a few cross-checks (e.g., alternative MIP models), auxiliary and
visualization code.
"""

import tsplib95 as tsp
import argparse
from pathlib import Path
from haversine import haversine
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import math
import json
import os
import re
import random
import gurobipy as gp
from gurobipy import GRB
from glob import glob
from math import ceil
from time import time

from qubo_utils import save_QUBO, load_QUBO, solve_QUBO


###############################################################################
## Recalculating the distances
def pt_to_ddeg(val):
    """Converts single lat/lon value (``val``): Deg.Min to decimal degrees."""
    minutes, deg = math.modf(val)
    minutes = round(minutes*100)
    deg = round(deg)
    return deg + minutes/60.0


def get_correct_dist(NC, i, j):
    """Calculates the distance between i and j from node_coords ``NC``."""
    points_src = points_src = [NC[i] for i in [i, j]]
    points = [list(map(pt_to_ddeg, pi)) for pi in points_src]
    return round(haversine(points[0], points[1]))


# Then we can just calculate all the distances ourselves.
# Assuming our problem is loaded to `problem` by `tsplib95`
def calc_distances(problem):
    """Returns the distance matrix for ``problem`` (from ``tsp.load``)."""
    G = problem.get_graph()
    return np.array([[get_correct_dist(problem.node_coords, i, j)
                      for i in G.nodes] for j in G.nodes])


###############################################################################
# Formulating the problems
def k(i, t, N):
    """Helper: returns the (linear) vector-index (k) given the matrix-indices (i,t).

    k: (i,t) -> N*t + i

    Notes:
        Both ``i`` and ``t`` numbering is assumed zero-based.
        ``N`` is the total number of nodes addressed by ``i``.
    """
    return N*t+i


def get_QF_coeffs(D):
    """Builds TSP quadratic cost function coefficients from a graph.

    Args:
        D: distance matrix, (N x N)

    Returns:

        - ``Q`` -- an (N-1)^2 x (N-1)^2 matrix of quadratic coefficients,
        - ``P`` -- a vector of (N-1)^2 numbers containing linear coefficients.

    Note:
        The resulting form is assumed to be (1/2) x' Q x + P' x.
    """
    N = len(D)

    Q = np.zeros(((N-1)**2, (N-1)**2))
    P = np.zeros((N-1)**2)

    for i in range(1, N):
        # linear terms
        P[k(i-1, 0, N-1)] = D[0, i]  # first hop (node 0 --> somewhere)
        P[k(i-1, N-2, N-1)] = D[i, 0]  # last hop (somewhere --> node 0)

        # quadratic terms
        for j in range(1, N):
            if i == j: continue
            for t in range(1, N-1):
                k1 = k(i-1, t-1, N-1)
                k2 = k(j-1, t, N-1)
                Q[k1, k2] = D[i, j]
                Q[k2, k1] = D[i, j]

    return Q, P


def encode_constraints(Q, P, A, b, M):
    """Utility to create a QUBO: incorporates constraints in the form "Ax=b".

    Args:
        Q: quadratic coefficients matrix,
        P: linear coefficients vector,
        A: constraints matrix,
        b: constraints RHS vector,
        M: big M parameter (multiplicative const) for penalty terms.

    Returns:
        Q-prime, P-prime, and Const

    Notes:
        We assume the input problem in the form:

            | min_x   (1/2) x' Q x + P' x  (1)
            | s.t.        Ax = b

        The function transforms it to:

            min_x   (1/2) x' Qp x + Pp' x + Const,  (2)

        where Qp and Pp incorporate penalties (with multiple M)
        into the objective, so that solutions to (1) and (2) would be the same.
    """
    Qp = Q + 2*M*np.dot(A.transpose(), A)
    Pp = P - 2*M*np.dot(np.transpose(A), b)
    Const = M*np.dot(b, b)
    return Qp, Pp, Const


def get_constraints(D):
    """Returns the constraints for TSP in the matrix form.

    Args:
        D: the distance matrix,

    Returns:
        Matrix A and vector b, so that the constraints are ``Ax = b``,
        assuming x-variables.
    """
    N = len(D)
    n_constraints = 2*(N-1)

    b = [1.0 for _ in range(n_constraints)]
    A = np.zeros((n_constraints, (N-1)**2))

    u = 0  # constraints counter

    for i in range(1, N):
        for t in range(1, N):
            A[u, k(i-1, t-1, N-1)] = 1.0
        u += 1

    for t in range(1, N):
        for i in range(1, N):
            A[u, k(i-1, t-1, N-1)] = 1.0
        u += 1

    return A, b


def save_orig_instance(D, inst_id, problem, comment="", directory="./instances/orig/"):
    """Saves a TSP instance encoded by a distance matrix to a JSON (with metadata).

    Args:
        D: distance matrix,
        inst_id (string): instance ID,
        problem: original problem (as loaded by tsplib),
        comment: optional comment for the JSON file,
        directory: a dir to save files.
    """
    instance = {"D": D.tolist()}
    filename = f"{inst_id}_{len(D)}_{problem.name}.orig.json"
    instance["description"] = {
        "instance_id": inst_id,
        "instance_type": "TSP",
        "original_instance_name": problem.name,
        "original_instance_file": filename,
        "contents": "Distance matrix D.",
        "comment": f"Distances for GEO edge type are recalculated from coordinates.{comment}"
    }

    filename = directory + filename
    inst_json = json.dumps(instance, indent=4)
    with open(filename, "w") as outfile:
        outfile.write(inst_json)

    return filename


def load_instance_by_ID(ID, directory="./instances/orig/"):
    """Loads an instance by ID and returns matrix D."""
    directory += "/"
    file_list = [fname for fname in glob(directory + ID + "_*.json")]

    if len(file_list)==0:
        raise ValueError(f"Instance with ID '{ID}' not found in {directory}.")
    elif len(file_list)>1:
        raise ValueError(f"Multiple instances with ID '{ID}' in {directory}:\n{file_list}.")

    with open(file_list[0], 'r') as ofile:
        js = json.load(ofile)

    if js["description"]["instance_type"] != "TSP":
        raise ValueError(f"Instance '{ID}' is not marked as TSP in 'instance_type' field. (see {file_list[0]})")

    return np.array(js["D"]), js


def sample_nodes(Dfull, Ns=[5], K=5):
    """Creates distance matrices for sub-graphs of given size.

    Try to create K unique instances by generating N-node subgraphs from
    a weighted graph given by distance matrix `Dfull`.

    Args:
        Dfull(np.array): full distance matrix,
        N(list): numbers of nodes to sample (default: one value, 5),
        K(int): number  of samples to make per size (default: 5).

    Example:
        >>> seed(2023)
        >>> t = np.array([[1,2,3],[4,5,6],[7,8,9]])
        >>> sample_nodes(t, Ns=2, K=3)
        [array([[5, 6],
                    [8, 9]]),
         array([[1, 3],
                    [7, 9]]),
         array([[1, 2],
                    [4, 5]])]
    """
    if type(Ns) != list:
        Ns = [Ns]

    Ds = []

    for N in Ns:
        max_samples = math.comb(len(Dfull), N)
        if K > max_samples or K < 0:
            raise ValueError(f"Can't sample {K} unique subsets from {len(Dfull)} nodes! (Max is {max_samples}, min 1.)")

        samples = set()

        n_sampled = 0
        while n_sampled < K:
            nodes = np.sort(random.sample(range(len(Dfull)), N))
            if tuple(nodes) in samples:
                continue

            # else: we have found a new sample
            n_sampled += 1
            samples.add(tuple(nodes))
            Ds.append(Dfull[np.ix_(nodes, nodes)])

    return Ds


def generate_TSP_instances(downloads="./download/",
                           instdir="./instances",
                           Ns=[5, 7, 10],
                           K=1,
                           N_src_inst=15,
                           Ninst_max=1000,
                           N_total=100,
                           seed=2023):
    """The core function that generates TSP instances.

    Creates at least K * len(Ns) and at most ``N_total`` instances
    by sampling from existing TSP instances (looked for
    in ``downloads`` directory) and saves them to directory ``instdir``.

    Args:
        downloads(str): a directory with original instances to sample from
        instdir(str): directory for the generated instances,
        Ns(list[int]): number of nodes to sample
        K(int): number of instances to create per size
        N_src_inst(int): max # of source TSP instances to process.
        N_N(int): max # of source TSP instances to process.
        seed(int): random seed, for reproducibility
    """
    directory = os.fsencode(downloads)
    nfiles = 0

    edge_types = {}
    inst_num = 0
    filenum = 0

    random.seed(seed)
    files = [fle for fle in os.listdir(directory) if os.fsdecode(fle).endswith(".tsp")]
    random.shuffle(files)
    for file in files:
        filenum += 1
        if (inst_num >= N_total) or (filenum > N_src_inst):
            # we have generated enough instances
            break

        filename = os.fsdecode(file)
        nfiles += 1
        file = os.path.join(downloads, filename)
        p = tsp.load(file)
        print(f"Processing {file} ", end="", flush=True)
        EWT = p.edge_weight_type
        if EWT not in edge_types:
            edge_types[EWT] = 1
        else:
            edge_types[EWT] += 1

        # omit instances that are too large
        print(f"({p.dimension} nodes)", end="", flush=True)

        if (p.dimension > Ninst_max) or (p.dimension < np.max(Ns)):
            print("(skipped)")
            filenum -= 1
            continue

        if EWT == "GEO":
            D_full = np.array(calc_distances(p))  # calculate using our own approach from coordinates
        else:
            G = p.get_graph()
            D_full = nx.to_numpy_array(G)  # fall back to TSP lib otherwise

        Ds = sample_nodes(D_full, Ns=Ns, K=K)

        for D in Ds:
            # processing and saving each instance here
            inst_num += 1
            orig_filename = save_orig_instance(D, f"TSP{inst_num}", p,
                                                directory = instdir+"/orig/")
            Q, P = get_QF_coeffs(D)
            A, b = get_constraints(D)
            M = np.max(D)*(len(D)**2)

            Qp, Pp, Const = encode_constraints(Q, P, A, b, M)

            save_QUBO(Qp, Pp, Const, "TSP", f"{len(D)}_{p.name}",
                        orig_filename, f"TSP{inst_num}",
                        directory=instdir +"/QUBO/", comment="Constraints as penalties.")
        print("✅", flush=True)

    print(f"Processing finished.\nOriginal instances: {nfiles}")
    print(f"Edge types encountered: {edge_types}")
    print(f"Instances created with sampling: {inst_num}")
    print(f"({K} instances with each of {Ns} nodes out of each original TSP.")


def create_MTZ_TSP_model(D, quiet=False):
    """Creates an MTZ model from distance matrix ``D``.

    Note:
        All numberings start from one.

    Returns:
        A model (``gurobipy.Model``), ``x``, ``u`` (variables in the form of ``dict``-s)
    """
    N = lend(D)
    model = gp.Model("TSP_MTZ")

    x = {(i,j): model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
         for i in range(1, N+1)
         for j in range(1, N+1)
         if i<j}

    u = {i: model.addVar(vtype=GRB.INTEGER, name=f"u_{i}",
                         lb = 1, ub = N)
         for i in range(1, N+1)}

    # Setting the objective
    model.setObjective(gp.quicksum(D[i, j] * x[(i,j)]
                                   for i in range(1, N+1)
                                   for j in range(1, N+1)
                                   if i<j), GRB.MINIMIZE)

    for i in range(1, N+1):
        model.addConstr(gp.quicksum(x[(i, j)] for j in range(N)
                                    if i < j) == 1)

    for j in range(1, N+1):
        model.addConstr(gp.quicksum(x[(i, j)] for i in range(N)
                                    if i < j) == 1)

    for i in range(2, N+1):
        for j in range(i+1, N+1):
            model.addConstr(u[i] - u[j] + (N-1)*x[(i,j)] <= N-2)

    # removing the symmetry
    model.addConstr(u[1] == 1)

    if quiet:
        model.setParam("OutputFlag", 0)

    return model, x, u

###############################################################################
# Cross-checks
def create_simple_TSP_model(D, quiet=False):
    """Creates a MIP model (QAP form) from distance matrix `D` for cross-check.

    Args:
        D: distance matrix,

    Returns:
        model (``gurobipy.Model``) and y (a ``dict`` of variables).
    """
    N = len(D)  # number of nodes
    model = gp.Model("TSP_SQ")

    # Defining variables
    y = dict()

    y = {(i,t): model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{t}")
         for i in range(N)
         for t in range(N)}

    # Setting the objective
    model.setObjective(gp.quicksum(D[i, j]*y[(i, t)]*y[(j, t+1)]
                                   for t in range(N-1)
                                   for i in range(N)
                                   for j in range(N)
                                   if i != j) +
                       gp.quicksum(D[j, i]*y[(i, 0)] * y[(j, N-1)]
                                   for i in range(N)
                                   for j in range(N)), GRB.MINIMIZE)
    # Defining constraints
    for i in range(N):
        model.addConstr(gp.quicksum(y[(i, t)] for t in range(N)) == 1)

    for t in range(N):
        model.addConstr(gp.quicksum(y[(i, t)] for i in range(N)) == 1)

    # removing the obvious symmetry
    model.addConstr(y[(0, 0)] == 1)

    if quiet:
        model.setParam("OutputFlag", 0)

    return model, y


def unpack_tour_y(y, N):
    """Returns the tour given gurobi vars (matrix indexing) and the number of nodes.

    Args:
        y: a ``dict`` of variables returned by Gurobi, keyed by ``(i, t)``,
        N: number of nodes in the graph
    """
    tour = [0 for _ in range(N+1)]
    for i in range(N):
        for t in range(N):
            if y[(i, t)].X == 1:
                tour[t] = i

    tour[N] = tour[0]
    return tour

def unpack_tour_QUBO_bitstring(bs, N):
    """Unpacks the tour given the bitstring (assuming QUBO).

    Note:
        Assumes **direct** order of bits in the bistring.
        E.g., ``[n0, n1, n2, ... ]``.

    """
    tour = [0 for _ in range(N+1)]
    for t in range(1, N):
        have_node = False
        for i in range(N-1):
            if int(bs[k(i, t-1, N-1)]) == 1:
                if have_node:
                    return None  # multiple nodes per single time step t
                tour[t] = i+1
                have_node = True
        if not have_node:
            return None  # no node found per time step t

    tour[N] = tour[0]
    return tour

def unpack_tour_x(x, N):
    """Returns the tour given gurobi vars (linear indexing) and the number of nodes.

       Args:
        x: a ``dict`` of variables returned by Gurobi, keyed by a single number ``k(i, t, N-1)``,
        N: number of nodes in the graph
    """
    bs = "".join([str(int(xi.X)) for xi in x])
    return unpack_tour_QUBO_bitstring(bs, N)


# Let us define a few functions to show a nice summary
def is_feasible(tour, D):
    """Checks if ``tour`` is feasible (given the distance matrix ``D``).

    Essentially, checks if the number of nodes is correct, all nodes are
    visited once, and the last point coincides with the first one.
    """
    if tour is None:
        return False  # empty tour is assumed infeasible
                      # (e.g., as returned from unpack_tour_QUBO_bitstring)

    """Returns if tour ``tour`` is feasible."""
    if len(tour) != len(D)+1:
        raise ValueError(f"Number of nodes in a tour ({len(tour)}) must correspond to the number of nodes in the graph ({len(D)})")

    nodes = set()
    no_repeats = 0
    for i in tour:
        if (i in nodes):
            if no_repeats == 0:
                no_repeats += 1
            else:
                return False
        else:
            nodes.add(i)

    return (tour[0] == tour[-1])


def show_feasible(tour, D):
    """Wraps :py:func:`is_feasible` into a string, for readability."""
    if is_feasible(tour, D):
        return "feasible"
    else:
        return "infeasible"


def obj_val(D, tour):
    """Calculates the objective given the costs matrix and a tour."""
    if is_feasible(tour, D):
        return np.sum([D[tour[t]][tour[t+1]] for t in range(len(tour)-1)])
    else:
        return None


######################################################################
# Drawing / visualization code

def draw_instance(G, ax=plt.gca(), node_color='blue',
                  node_size=200, with_labels=True):
    """Draws a single TSP instance (weighted graph ``G``)."""
    pos = nx.spring_layout(G)
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

        with open(qubojs["description"]["original_instance_file"], 'r') as ofile:
            origjs = json.load(ofile)

        if origjs["description"]["instance_type"] != "TSP":
            raise ValueError(f"{qubojs['description']['original_instance_file']}: Original instance for '{qubofile}' is not marked as TSP in 'instance_type' field.")

        D = np.array(origjs["D"])

        G = nx.Graph()
        G.add_nodes_from([j for j in range(1, len(D)+1)])
        G.add_edges_from([(i,j) for i in range(1, len(D)+1) for j in range(i+1, len(D)+1)])
        for (i,j) in G.edges:
            G.edges[i,j]['weight'] = D[i-1,j-1]

        if cross_check:
        # solve classically -- MIP
            t0 = time()
            model, y = create_simple_TSP_model(D, quiet=True)
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

            assert abs(mipObj - qubo_model.objVal)<1e-5, f"MIP: {mipObj}, QUBO: {qubo_model.objVal}"

            # # show the results
            # Gp_qubo = nx.Graph()
            # Gp_qubo.add_nodes_from([j for j in G.nodes if qubo_x[j-1].X==1.0])

        N = len(G.nodes)
        if Nrows == 1:
            axk = ax[k]
        else:
            axk = ax[k//Ncols][k-Ncols*(k//Ncols)]

        draw_instance(G, ax = axk, node_color = 'blue', with_labels=False,
                      node_size=150)
        ax_title = f"{os.path.basename(os.path.normpath(qubofile))[3:-10]} \n N:{N})"

        # if cross_check:
        #     draw_instance(Gp_qubo, ax=axk, node_color='red',
        #                 with_labels=False, node_size=50)
        #     ax_title += f"\nTmip:{miptime:.2f} s. | Tqubo:{qubotime:.2f}"

        axk.set_title(ax_title)
        k+=1


def draw_IDs_grid(sel_IDs, directory = "./instances-new/QUBO", Ncols=3,
                  cross_check=False):
    """Draws a grid of instances by (the number part of) IDs.

    Args:
        sel_IDs(list[int]): list of instance IDs (numbers)
        directory(str): directory with QUBO files (\*.qubo.json)
        Ncols(int): number of columns in the grid

    Example:
        ``draw_IDs_grid([1,3,5])`` will draw a grid with instances
        ``TSP1``, ``TSP3``, and ``TSP5``.

    Returns:
        A list of selected file names.

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

def draw_IDs_grid(sel_IDs, directory = "./instances-new/QUBO", Ncols=3,
                  cross_check=True):
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
        file_list = [fname for fname in glob(directory + "/TSP*.json") if re.search(r'.+TSP[0]*' + re.escape(str(ID)) + r'_.*\.qubo.json', fname)]
        if len(file_list)==0:
            raise ValueError(f"Instance with ID '{ID}' not found in {directory}.")
        elif len(file_list)>1:
            raise ValueError(f"Multiple instances with ID '{ID}' in {directory}:\n{file_list}.")

        selected_files.append(file_list[0])

    draw_inst_grid(selected_files, Ncols=Ncols, cross_check=cross_check)
    return selected_files

######################################################################
# Testing code

def solve_TSP_MIP(instance_file, quiet=True, timeout=None):
    """Creates and solves a QAP MIP model for TSP."""
    with open(instance_file, 'r') as openfile:
        origjson = json.load(openfile)

        D = np.array(origjson["D"])
        model, y = create_simple_TSP_model(D)
        if quiet:
            model.setParam("OutputFlag", 0)

        if timeout is not None:
            model.setParam("TimeLimit", timeout)

        model.update()
        model.optimize()

        return model, y, D


def solve_TSP_MTZ(instance_file, quiet=True, timeout=None):
    """Creates and solves a MTZ MIP model for TSP."""
    with open(instance_file, 'r') as openfile:
        origjson = json.load(openfile)

        D = np.array(origjson["D"])
        model, y = create_simple_TSP_model(D)
        if quiet:
            model.setParam("OutputFlag", 0)

        if timeout is not None:
            model.setParam("TimeLimit", timeout)

        model.update()
        model.optimize()

        return model, y, D


def assert_TSP_MIP(instance_file, qubo_model, qubo_x,
                   quiet=True, tol=1e-3, timeout=None) -> None:
    """Asserts if the solutions to QUBO and 'natural' MIP coincide."""
    model, y, D = solve_TSP_MIP(instance_file, quiet, timeout)
    if timeout is None:
        assert model.status == GRB.OPTIMAL
    else:
        assert (model.status == GRB.OPTIMAL) or (model.status
                                                 == GRB.TIME_LIMIT)

    # check that the results (objectives) coincide
    x_tour = unpack_tour_x(qubo_x, len(D))
    y_tour = unpack_tour_y(y, len(D))

    dObj = abs(obj_val(D, y_tour) - obj_val(D, x_tour))
    print(f"x-tour: {x_tour} ({obj_val(D, x_tour):.3f})\ny-tour: {y_tour} ({obj_val(D, y_tour):.3f})")
    assert dObj <= tol


def main():
    """The main function specifying instance generation parameters (see the source code)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("instdir", help="(output) instances directory",
                        default="./instances")
    args = parser.parse_args()
    if not Path(args.instdir).is_dir():
        raise ValueError(f"{args.instdir}: does not exist, or is not a directory.")

    generate_TSP_instances(Ns = [j for j in range(5, 18, 1)],
                           K = 1,
                           N_src_inst=15,
                           N_total=500,
                           instdir=args.instdir)

if __name__ == '__main__':
    main()
