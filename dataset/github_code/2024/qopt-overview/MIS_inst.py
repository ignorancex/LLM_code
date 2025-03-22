"""Generates Unit Disc Max Independent Set (UDMIS) instances.

Creates each instance from a generated random collection of points on a 2D
plane.

USAGE:
    | python -m MIS_inst --dataset small ./instances
    | python -m MIS_inst --dataset large ./instances

Generates a set of UDMIS instances, saving original instance data in
``./instances`` (or another specified directory): ``orig`` subdirectory for
original instances and ``QUBO``, respectively, for QUBOs. Can create either
``small`` dataset (9 instances with suffix ``TG`` in the name, for manual
inspection), or a ``large`` one (the main dataset with ``EXT`` suffix in the
name).

The instance generation parameters are specified in
:py:func:`generate_main_dataset` and :py:func:`generate_TG_inst`, respectively,
for main and small subsets of instances. Both functions rely on the core
generation code in :py:func:`generate_UDMIS_instances`. This module also
contains a few cross-checks, auxiliary and visualization code.

"""

import argparse
import random
import json
import re
import os
from glob import glob
from math import sqrt, ceil

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
from math import ceil
from pathlib import Path
from time import time

from qubo_utils import save_QUBO, load_QUBO, solve_QUBO

def create_UDMIS_name(N, wwidth, wheight, R):
    """Helper: creates an instance name string (UD-MIS)."""
    return f"N{N}_W{max(wwidth, wheight) / min(wwidth, wheight)}_R{R / max(wwidth, wheight):.1f}"


def create_grid_points(N, wwidth, wheight, seed=2023):
    """Generates a random (broken) grid of points.

    Generates a full grid of (``wwidth`` x ``wheight``) points,
    and then randomly removes them one by one,
    until the necessary number of points ``N`` is reached.

    Returns:
        list[tuple[int, int]]: a list of points, each specified by an integer coordinates tuple
    """
    if seed is not None:
        random.seed(seed)

    assert N <= wwidth * wheight

    points = [(i,j) for i in range(wwidth) for j in range(wheight)]
    while len(points) > N:
        points.remove(random.sample(points,1)[0])

    return points


def create_random_points(N, wwidth, wheight, seed=2023):
    """Generates a set of random (int-coordinate) points.

    Point coordinates generated uniformly random from [0,1) x [0,1),
    scaled to [0, wwidth) x [0, wheight) and then rounded. If we happen to get
    duplicate coordinates, we just re-generate the point until we have
    ``N`` unique points.

    Args:
        N(int): number of points
        wwidth, wheight(int): width and height of the square
            to generate points in
        seed(int): random seed for `random`

    Returns:
        A list of tuples, each representing integer coordinates (x,y),
        where 0 <= x <= wwidth and 0 <= y <= wheight.
    """
    assert (wwidth >= 0) and (wheight >= 0)

    if seed is not None:
        random.seed(seed)

    points = set()
    while len(points) < N:
        cand = (round(random.random() * wwidth),
                round(random.random() * wheight))
        if cand in points:
            continue  # try to re-generate
        else:
            points.add(cand)

    return list(points)


def points_to_graph(points: tuple, R: float) -> nx.Graph:
    """Generate an ``nx.Graph`` out of a set of points.

    A node is created for each point, and an edge is created whenever the
    Euclidean distance between two corresponding points is no more than ``R``.
    """
    G = nx.Graph()
    N = len(points)
    G.add_nodes_from([j for j in range(1, N+1)])
    G.add_edges_from([(i, j)
                      for i in range(1, N+1)
                      for j in range(i+1, N+1)
                      if np.linalg.norm(
                              np.asarray(points[i-1]) - np.asarray(points[j-1])) <= R])

    return G

def save_orig_UDMIS(N, wwidth, wheight, R, instance_id, points, G,
                    directory="./instances/orig"):
    """Saves the original UD-WIS instance in JSON format.

    Args:
        N(int): number of vertices (points)
        wwidth(float), wheight(float): coordinate "window"
        R (float): radius for points generation,
        instance_id(str): a unique instance ID
        points(list): coordinates of points,
        G (nx.Graph): resulting graph,
        directory(str): dir to save the instance to

    Returns:
        filename (with the path) for the created file.

    File format:
        - js["nodes"]: list of nodes,
        - js["edges"]: list of edges, tuples (i,j) with i < j.
        - js["description"]: info concerning instance generation.
           UDMIS-specific subfields are:
           - points: dict of point coordinates: {i: (x,y)}
           - wwidth, wheight: max values for coordinates (window parameters)
           - R: radius value used

    """
    instname = create_UDMIS_name(N, wwidth, wheight, R)

    instance = {"nodes": [j for j in G.nodes()],
                "edges": [(i,j) for (i,j) in G.edges()]}

    instance["description"] = {
        "instance_id": instance_id,
        "instance_type": "UDMIS",
        "original_instance_name": instname,
        "contents": "orig_UDMIS",
        "wwidth": wwidth,
        "wheight": wheight,
        "R": R,
        "points": {i: points[i-1] for i in range(1, len(G.nodes())+1)},
        "comment": "Underlying graph and the point coordinates."
    }

    filename = directory + "/" + instance_id + "_" + instname + ".orig.json"

    inst_json = json.dumps(instance, indent=4)
    with open(filename, "w") as outfile:
        outfile.write(inst_json)

    return filename


def extract_G_from_json(js) -> nx.Graph:
    """Helper: Extracts the graph from JSON ``js``."""
    G = nx.Graph()
    G.add_nodes_from([j for j in js["nodes"]])
    G.add_edges_from([(i,j) for (i,j) in js["edges"]])
    return G

def load_orig_MIS(filename) -> tuple[nx.Graph, dict]:
    """Loads an original MIS instance from JSON file."""

    with open(filename, 'r') as openfile:
        myjson = json.load(openfile)

    return extract_G_from_json(myjson), myjson

def create_MIS_QUBO(G: nx.Graph) -> tuple[np.array, np.array]:
    """Takes a graph and returns QUBO coefficients for a MIS.

    Returns:
        Two ``np.array``-s,

        - ``Q`` (quadratic coefficients matrix), and
        - ``P`` (linear coefficients vector).
    """
    N = len(G.nodes())
    Q = np.zeros((N, N))
    P = -np.ones(N)

    M = N+1

    for (i,j) in G.edges():
        Q[i-1, j-1] += M

    Q = Q + Q.T

    return Q, P

def generate_UDMIS_instances(Ns=[5, 10, 15],
                             K=1,
                             pts_windows = [(100, 100)]*3,
                             Rs = [25]*3,
                             id_suffix = "",
                             id_offset = 0,
                             digits = None,
                             seed = 2023,
                             each_seed = None,
                             points_generator=create_random_points,
                             instdir="./instances"):
    """Main instance generation code for UDMIS instances.

    Creates len(Ns) * K instances, each stemming from a set of points from a 2D
    window scaled to wwidth x wheight (integer coordinates) specified in
    ``pts_windows``, and "blockade" radius being one of the values specified by
    ``Rs``. Points generation is delegated to a separate function given by
    ``points_generator`` (:py:func:`create_random_points` by default).

    Lists Ns, Rs, and pts_windows are assumed to have the same length,
    so that (Ns[i], Rs[i], pts_windows[i]) would specify a single set
    of parameters.

    Args:
        Ns(list[int]): instance sizes,
        id_offset(int): number to offset the ID numbering by (default: 0)
        K(int): number of instances per set of parameters,
        pts_windows(list[tuple]): a list of tuples of the form
            (wwidth, wheight) to generate points in.
        Rs (list): a list of "blockade" radiuses to generate the graph.
        id_suffix(str): suffix to append to the instance ID (after the number),
        digits(int): max number of digits in the ID
        seed (int): random seed for ``random``,
        each_seed (int): random seed to be used for *each* instance (if not None)
        instdir(str): directory to save the generated instances to

    """
    if len(Ns) != len(pts_windows):
        raise ValueError(f"len(Ns) != len(pts_windows), {len(Ns)} != {len(pts_windows)}")

    if len(Ns) != len(Rs):
        raise ValueError(f"len(Ns) != len(Rs), {len(Ns)} != {len(Rs)}")
    counter = 1

    if digits == None:
        digits = len(str(len(Ns)*K))

    if seed is not None:
        random.seed(seed)

    for k in range(len(Ns)):
        N = Ns[k]
        R = Rs[k]
        wwidth, wheight = pts_windows[k]

        for _ in range(K):
            points = points_generator(N, wwidth, wheight, each_seed)
            instance_id = "UDMIS" + \
                ("{:0" + str(digits) + "d}").format(id_offset+counter) + \
                id_suffix

            G = points_to_graph(points, R)

            orig_file = save_orig_UDMIS(N, wwidth, wheight, R,
                                        instance_id, points, G,
                                        directory=instdir+"/orig/")

            Q, P = create_MIS_QUBO(G)

            save_QUBO(Q, P, 0.0, "UDMIS", create_UDMIS_name(N, wwidth,
                                                            wheight, R),
                      orig_file,
                      instance_id,
                      comment="Optimum is *minus* QUBO objective",
                      directory=instdir + "/QUBO/")
            counter += 1
            print(".", end="", flush=True)

    print(f"\n✅ Done. {counter-1} instances generated.")


######################################################################
# Auxiliary code
#
def is_IS(G: nx.Graph, solution:list[int]) -> bool:
    """Tests if the solution is an independent set.

    Args:
        G(nx.Graph): original graph,
        solution(list[int]): candidate solution

    Notes:
        ``solution`` must be in the direct order of nodes
                    (i.e., ``[n0, n1, n2, ...]``)
    """
    for (i,j) in G.edges:
        if (solution[i-1]==1) and (solution[j-1]==1):
            return False

    return True


def load_instance_by_ID(ID, directory="./instances/orig/"):
    """Loads an instance by ID and returns graph G."""
    directory += "/"
    file_list = [fname for fname in glob(directory + ID + "_*.json")]

    if len(file_list)==0:
        raise ValueError(f"Instance with ID '{ID}' not found in {directory}.")
    elif len(file_list)>1:
        raise ValueError(f"Multiple instances with ID '{ID}' in {directory}:\n{file_list}.")

    return load_orig_MIS(file_list[0])

######################################################################
# Visualization and further helpers
#
def visualize_graph(G: nx.Graph, R=None, pts=None, ax=plt.gca(), z='blue',
                    with_labels=True, node_size=200):
    """Draws the graph (node labels as coordinates), with node color ``z``. """
    if pts is None:
        pos = None
    else:
        pos = dict()
        for j in pts:
            pos[int(j)] = pts[j]

    if R is not None:
        maxval = np.max([pt for j, pt in pos.items()], axis=0)

        xdata = []
        ydata = []
        for j in range(ceil(maxval[0] / R)+1):
            xdata.append(j*R)

        for j in range(ceil(maxval[1] / R)+1):
            ydata.append(j*R)

        ax.set_xticks(xdata)
        ax.set_yticks(ydata)
        # ax.grid()

    style_net = dict(node_color=z, edge_color='k', node_size=node_size,
                     font_color='white', with_labels=with_labels, width=3,
                     connectionstyle='arc3,rad=0.1')
    nx.draw_networkx(G, pos=pos, ax=ax, **style_net)


def draw_inst_grid(selected_files, Ncols = 3, directory="./instances-new/QUBO"):
    """Draws a grid of instances from ``selected_files``.

    Args:
        selected_files(list): list of files (\*.qubo.json)
        Ncols(int): number of columns in the grid.
        directory(str): directory with QUBO files (\*.qubo.json)
    """
    k = 0
    Ncols = min(Ncols, len(selected_files))
    Nrows = ceil(len(selected_files) / Ncols)
    fig, ax = plt.subplots(Nrows, min(Ncols, len(selected_files)), figsize=(20,20*Nrows/Ncols))

    for qubofile in selected_files:
        Q, P, C, qubojs = load_QUBO(qubofile)
        # solve classically -- MIP
        G, origjs = load_orig_MIS(qubojs["description"]["original_instance_file"])

        t0 = time()
        model, x = create_orig_MIS_IP(G)
        model.setParam("OutputFlag", 0)
        model.update()
        model.optimize()
        miptime = time()-t0
        assert model.status == 2
        mipObj = model.objVal

        # solve as QUBO
        t0 = time()
        qubo_model, qubo_x = solve_QUBO(Q, P, C,quiet=True)
        qubotime = time() - t0
        assert qubo_model.status ==2

        assert abs(mipObj + qubo_model.objVal)<1e-5, f"MIP:{mipObj}, QUBO:{qubo_model.objVal}"

        # show the results
        Gp_qubo = nx.Graph()
        Gp_qubo.add_nodes_from([j for j in G.nodes if qubo_x[j-1].X==1.0])

        R = origjs["description"]["R"]
        N = len(G.nodes)
        if Nrows == 1:
            if Ncols == 1:
                axk = ax  # ah, edge cases...
            else:
                axk = ax[k]
        else:
            axk = ax[k//Ncols][k-Ncols*(k//Ncols)]

        visualize_graph(G, R=R, pts=origjs["description"]["points"], ax = axk, z = 'blue',
                           with_labels=False, node_size=50)
        visualize_graph(Gp_qubo, pts = origjs["description"]["points"], ax=axk, z='red',
                           with_labels=False, node_size=50)

        # axk.set_title(f"{os.path.basename(os.path.normpath(qubofile))[5:-10]} \n N:{N} | E:{len(G.edges)}/{N*(N-1)/2:.0f} | (R={R:.1f}) \nTmip:{miptime:.2f} s. | Tqubo:{qubotime:.2f}")
        axk.set_title(f"{origjs['description']['instance_id']}: {N} nodes, {len(G.edges)} edges.", fontsize=20)
        k+=1


def draw_IDs_grid(sel_IDs, directory = "./instances-new/QUBO", Ncols=3):
    """Draws a grid of instances by (the unique part of) IDs.

    Example:
        ``draw_IDs_grid(['1EXT','3EXT','5TG'])`` will draw a grid with
        ``UMDIS001EXT``, ``UDMIS003EXT``, and ``UDMIS5TG``.

    Args:
        sel_IDs(list[int]): list of instance IDs (numbers)
        directory(str): directory with QUBO files (\*.qubo.json)
        Ncols(int): number of columns in the grid

    Returns:
        selected file names.
    """
    selected_files = []

    for ID in sel_IDs:
        file_list = [fname for fname in glob(directory + "/UDMIS*.json") if re.search(r'.+UDMIS[0]*' + re.escape(str(ID)) + r'_.*\.qubo.json', fname)]
        if len(file_list)==0:
            raise ValueError(f"Instance with ID '{ID}' not found in {directory}.")
        elif len(file_list)>1:
            raise ValueError(f"Multiple instances with ID '{ID}' in {directory}:\n{file_list}.")

        selected_files.append(file_list[0])

    draw_inst_grid(selected_files, Ncols=Ncols)
    return selected_files

######################################################################
# Testing / Cross-checks
def create_orig_MIS_IP(G: nx.Graph) -> tuple[gp.Model,
                                             dict[int, gp.Var]]:
    """Generates a MIP for the MIS instance given by G.

    Returns:
        A ``gurobipy.Model`` and a ``dict`` of variables (``x``).
    """
    model = gp.Model()
    x = model.addVars(G.nodes,vtype=GRB.BINARY, name="x")
    model.addConstrs(((x[i] + x[j]) <= 1 for (i, j) in G.edges), "edge_constr")
    model.setObjective(gp.quicksum(x[i] for i in G.nodes), sense=GRB.MAXIMIZE)
    return model, x


def assert_UDMIS_MIP(orig_filename, qubo_model, qubo_x,
                     quiet=True, timeout=None):
    """Asserts whether QUBO and original OP solutions coincide.

    Note:
        ``qubo_x`` argument is unused but kept for compatibility
        with the testing framework.
    """
    G, _ = load_orig_MIS(orig_filename)

    model, _ = create_orig_MIS_IP(G)

    if quiet:
        model.setParam("OutputFlag", 0)

    if timeout is not None:
        model.setParam("TimeLimit", timeout)

    model.update()
    model.optimize()

    if timeout is None:
        assert model.status == GRB.OPTIMAL
    else:
        assert model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT)

    np.testing.assert_allclose(-qubo_model.ObjVal, model.ObjVal)

def check_QuEra_geometry(orig_files, verbose=True):
    """Checks the geometric constraints for QuEra.

    Args:
        orig_files(list[str]): list of filenames (original instances)
        verbose(bool): technical output flag.

    Returns:
        true iff the constraints are satisfied; prints progress
        on the screen if ``verbose``
    """
    blockade_radius = 8.044e-06  # magical constant from the tutorial

    for ofile in orig_files:
        with open(ofile, 'r') as infile:
            instance_data = json.load(infile)

        assert instance_data['description']['instance_type'] == "UDMIS", \
            f"In {ofile}: unknown instance type {instance_data['description']['instance_type']}"

        x_min = np.min([x for x,y in instance_data['description']['points'].values()])
        y_min = np.min([y for x,y in instance_data['description']['points'].values()])
        R = instance_data['description']['R']

        factor = blockade_radius / R

        data = np.zeros((len(instance_data['nodes']),2))
        for i,(x,y) in instance_data['description']['points'].items():
            data[int(i)-1,0] = (x-x_min)*factor
            data[int(i)-1,1] = (y-y_min)*factor

        min_dist = np.min([np.linalg.norm(data[i] - data[j]) for i in range(len(data))
                            for j in range(len(data))
                            if i!=j])
        print(f"Min distance: {min_dist} (check: {min_dist > 4e-06})")

        max_coord = np.max(data)
        print(f"Max coord is: {max_coord} (check: {max_coord < 7.5e-05})")

        if (min_dist <= 4e-06) or (max_coord >= 7.5e-05):
            print(f"Test failed for: {ofile} ({instance_data['description']['instance_id']})")
            return False

    return True

######################################################################
# Command line usage

def generate_TG_inst(instdir):
    """Generates grid instances (``TG`` suffix).

    See the source code for specific parameter values. Relies on
    :py:func:`generate_UDMIS_instances` for actual instance generation, with
    :py:func:`create_grid_points` for points generation.
    """
    windows = [(5,5)]*3 + [(8,8)]*3 + [(13,13)]*3
    Ns = [8, 15, 25, 15, 30, 60, 30, 80, 150]
    Rs = [1.5 for j in range(len(windows))]

    K = 1  # number of instances per set of parameters

    generate_UDMIS_instances(Ns=Ns, pts_windows=windows,
                             points_generator=create_grid_points, K=K, Rs=Rs,
                             id_suffix="TG", seed=2023, instdir=instdir)

def generate_main_dataset(instdir):
    """Generates the main UD-MIS dataset (``EXT`` suffix).

    See the source code for specific parameter values. Relies on
    :py:func:`generate_UDMIS_instances` for actual instance generation, with
    :py:func:`create_grid_points` for points generation.
    """
    Ns = [15, 25, 30, 36, 40, 49, 60, 70, 80, 90, 100, 120]
    windows = [(ceil(sqrt(N))+1, ceil(sqrt(N))+1) for N in Ns]

    K=3

    for i, R in enumerate([1.25, 1.42, 1.85]):
        Rs = [R for _ in range(len(Ns))]
        generate_UDMIS_instances(Ns=Ns, pts_windows=windows,
                                 points_generator=create_grid_points, K=K,
                                 Rs=Rs, id_suffix="EXT", id_offset= len(Ns)*K*i,
                                 digits=3, seed=2023, instdir=instdir)

def main():
    """The main function specifying instance generation parameters (see the source code)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("instdir", help="(output) instances directory",
                        default="./instances")
    parser.add_argument('-t', "--dataset", help="dataset type: ``small`` or ``large`` ",
                        default="large")
    args = parser.parse_args()
    if not Path(args.instdir).is_dir():
        raise ValueError(f"{args.instdir}: does not exist, or is not a directory.")

    if args.dataset == "small":
        generate_TG_inst(args.instdir)  # <- used to generate TG instances
    elif args.dataset == "large":
        generate_main_dataset(instdir)
    else:
        print(f"{args.dataset}: wrong value ('small' or 'large' expected).")

if __name__ == '__main__':
    main()
