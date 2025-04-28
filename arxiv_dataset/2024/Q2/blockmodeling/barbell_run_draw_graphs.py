#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 09:20:08 2023

@author: adaskin
https://networkx.org/documentation/stable/reference/generated/networkx.linalg.graphmatrix.adjacency_matrix.html
"""
import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def savefig_tofile(plt, fname=None):
    plt.savefig(
        fname + ".eps",
        dpi="figure",
        bbox_inches="tight",
        pad_inches=0.1,
        facecolor="auto",
        edgecolor="auto",
        backend=None,
    )

    plt.savefig(
        fname + ".pdf",
        dpi="figure",
        bbox_inches="tight",
        pad_inches=0.1,
        facecolor="auto",
        edgecolor="auto",
        backend=None,
    )

    plt.savefig(
        fname + ".png",
        dpi="figure",
        bbox_inches="tight",
        pad_inches=0.1,
        facecolor="auto",
        edgecolor="auto",
        backend=None,
    )

    # plt.close()


###################################################################
#################################################################


def draw_matrix(A, filename=None, xlabels=None, ylabels=None):
    [m, n] = A.shape
    fig, ax = plt.subplots()
    im = ax.imshow(
        A, origin="upper", alpha=0.8, cmap="Blues", extent=[0, m, n, 0], aspect=1
    )

    # ax.pcolor(A, cmap=plt.cm.Blues)

    if xlabels is not None:
        # want a more natural, table-like display
        # ax.invert_yaxis()
        ax.xaxis.tick_top()

        # ax.set_xticklabels(column_labels, minor=False)
        # ax.set_yticklabels(row_labels, minor=False)

        ax.set_xticks(
            np.arange(len(xlabels)) + 0.5,
            labels=xlabels,
            fontsize=7,
            rotation=90,
            minor=False,
        )
        ax.set_yticks(
            np.arange(len(ylabels)) + 0.5, labels=ylabels, fontsize=7, minor=False
        )
    # plt.xlabel('Column')
    # plt.ylabel('Row')
    # plt.title('qubits-%d'%(q), fontsize='9')
    # Define grid with axis='y'
    # plt.grid()
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)

    # cax = ax.axis((0.75, 0.1, 0.03, 0.8))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation="vertical")
    cbar.set_ticks([0, 1])
    cbar.ax.set_yticklabels(["0", "1"])  # horizontal colorbar
    # plt.suptitle('Color Maps of the Qubit States')
    # fig.tight_layout()

    if filename != None:
        savefig_tofile(plt, filename)
    plt.show()
    plt.close(fig)


###################################################################
#################################################################


def find_draw_communities(G, filename=None):
    comp = nx.community.girvan_newman(G)
    communityList = tuple(sorted(c) for c in next(comp))

    # communityList = list(nx.community.k_clique_communities(G,3))

    pos = nx.spring_layout(
        G,
        seed=20160,
        k=1,
        pos=None,
        fixed=None,
        iterations=50,
        threshold=0.0001,
        weight="weight",
        scale=1,
        center=None,
        dim=2,
    )  # Seed for reproducible layout
    # pos = nx.shell_layout(G, nlist=communityList)  # Seed for reproducible layout

    colors = ["orange", "cornflowerblue", "red", "green", "black"]

    icolor = 0
    for c in communityList:
        # nx.draw_networkx_nodes(G,pos,nodelist=c,node_size=10,node_color=colors[icolor])
        # nx.draw_networkx_edges(G,pos, nodelist=c, edge_color=colors[-icolor])
        nx.draw(
            G,
            pos,
            nodelist=c,
            node_size=350,
            node_color=colors[icolor],
            edge_color=colors[-icolor],
            with_labels=True,
        )
        icolor += 1
    # nx.draw_networkx_nodes(G,pos,nodelist=communityList[1],node_size=10,node_color='blue')
    # nx.draw_networkx_edges(G,pos)
    # nx.draw(G, pos, node_color=list(partition.values())); plt.show()
    if filename != None:
        # fig = plt.figure()
        savefig_tofile(plt, filename)

    plt.show()


###################################################################
#################################################################


def barbell_graph_example(m1=15, m2=2, seed=12345):
    """Creates barbell graph with m1, m2 parameters"""
    # n = 100  # 10 nodes
    # m = 100  # 20 edges
    # seed = 20160  # seed random number generators for reproducibility

    # Use seed for reproducibility
    # G = nx.gnm_random_graph(n, m, seed=seed)
    # G = nx.erdos_renyi_graph(n, p=0.1)
    G = nx.barbell_graph(15, 2)
    # some properties
    print("node degree clustering")
    # for v in nx.nodes(G):
    #     print(f"{v} {nx.degree(G, v)} {nx.clustering(G, v)}")

    # print()
    # print("the adjacency list")
    # for line in nx.generate_adjlist(G):
    #     print(line)

    Adj = nx.to_numpy_array(G, dtype=int)

    # import tensorflow as tf
    # Adj_shuffled = tf.random.shuffle(Adj, seed=1234)

    rng = np.random.default_rng(seed=seed)
    indices = np.arange(Adj.shape[0])

    shuffled_indices = np.arange(Adj.shape[0])
    rng.shuffle(shuffled_indices)  # shuffle indices
    Adj_shuffled = Adj.copy()
    Adj_shuffled[indices, :] = Adj[shuffled_indices, :]  # swap rows
    Adj_shuffled[:, indices] = Adj_shuffled[:, shuffled_indices]  # swap columns

    # DRAW MATRICES
    find_draw_communities(G, "barbellgraph")
    draw_matrix(Adj, "barbell-matrix", xlabels=indices + 1, ylabels=indices + 1)
    draw_matrix(
        Adj_shuffled,
        "barbell-shuffled",
        xlabels=shuffled_indices + 1,
        ylabels=shuffled_indices + 1,
    )

    return G, Adj, Adj_shuffled


###################################################################
#################################################################


def permutation_matrix_fromX(indx, n):
    """
    generates a permutation matrix of the form
    for indx = (b1,...bn)_2 in binary form
    P = \otimes X^bj
    If by multiplicaiton with different block diagonal matrix different
    permutations are possible

    Parameters
    ----------
    indx : int
        its binary form used to either put X or I to generate permutation.
    n: int
        number of qubits
    Returns
    -------
    a permutation matrix

    """
    X = np.array([[0, 1], [1, 0]], dtype=int)
    I = np.eye(2, 2, dtype="int")
    ibin = bin(indx)[2:].zfill(n)
    P = 1
    if indx == 0:
        P = np.eye(2**n, dtype="int")
        return P
    else:
        for i in ibin:
            if i == "1":
                Gi = X
            else:
                Gi = I
            P = np.kron(P, Gi)
    return P


###################################################################
#################################################################


def multi_controlled_x(controlqubits, targetqubit, nqubits):
    """
    generates a permutation matrix which is the matrix representation of multi control gate
    Parameters
    ----------
    control_qubits : List
        the order of the qubits 0, 1, 2, ... in the binary form of the indices
    targetl_qubit: int
        a qubit
    nqubits: int

    Returns
    -------
    a permutation matrix

    """
    if targetqubit in controlqubits:
        return np.eye(2**nqubits, dtype=int)

    N = 2**nqubits
    P = np.zeros((N, N), dtype=int)

    for i in range(0, N):
        binState = bin(i)[2:].zfill(nqubits)
        controlZero = False
        for cq in controlqubits:
            if binState[cq] == "0":
                controlZero = True
                break
        if controlZero == True:
            P[i][i] = 1  # no change when control is zero
        else:
            if binState[targetqubit] == "1":
                changeState = list(binState)
                changeState[targetqubit] = "0"
                changeIndx = int("".join(changeState), 2)
                P[i][changeIndx] = 1

            else:  # targetl_qubit == '0'
                changeState = list(binState)
                changeState[targetqubit] = "1"
                changeIndx = int("".join(changeState), 2)
                P[i][changeIndx] = 1

    return P


###################################################################
#################################################################


def rand_multi_controlled_x(nqubits):
    """
    Parameters
    ----------
    nqubits : int
        DESCRIPTION.

    Returns
    -------
    P : a matrix
       a random multi controlled gate matrix.

    """
    rng = np.random.default_rng()
    controlLen = rng.integers(1, nqubits)
    controlqubits = random.sample(range(0, nqubits), controlLen)
    targetqubit = rng.integers(low=0, high=nqubits)
    # print("c: ", control_qubits)
    P = multi_controlled_x(controlqubits, targetqubit, nqubits)
    return P


###################################################################
#################################################################


def get_probs_for_qubit_group(state_matrix, qubits=[0, 5], m=32, n=32):
    """


    Parameters
    ----------
    qubits : List, []
        the probability of the qubits. The default is [0,4].
    state_matrix: an mxn numpy matrix
    m : TYPE, optional
        DESCRIPTION. The default is 16.
    n : TYPE, optional
        DESCRIPTION. The default is 16.

    Returns
    -------
    A : TYPE
        DESCRIPTION.

    """

    # a = np.reshape(state_matrix, (m*n,1))#vectorize matrix
    nq = int(np.log2(m * n))
    L = len(qubits)

    qstates = np.zeros(2**L)

    "scan row by row all matrix elements"
    ijelement = 0
    # sumOfElemetns = 0
    for i in range(0, m):
        for j in range(0, n):
            "ith state in the vector"
            # for q in range(0,nq): #reading orders n-1, n-2, ... 1,0
            bits = bin(ijelement)[2:].zfill(nq)
            indx = 0
            for q in range(L):
                if bits[qubits[q]] == "1":
                    indx += 2 ** (L - q - 1)
            # print(L, bits, q,(L-q-1), indx, qubits[q],nq)
            qstates[indx] += state_matrix[i][j]

            # state_matrix[i][j] = indx
            # sumOfElemetns += state_matrix[i][j]
            # print(qstates)
            # print('ind', indx, bits,ijelement)
            ijelement += 1  # next element
    return qstates


###################################################################
#################################################################


def fitness(state_matrix, qubits, targetState):
    """

    state_matrix: numpy matrix
    qubits: a list
        whose state is the solution..

    Returns
        nr: fitness value
    -------
    measure qubits and returns the norm of the targetState-foundState
    a value in [0, 1]. 1 indicates better

    """

    m, n = state_matrix.shape
    istate = get_probs_for_qubit_group(state_matrix, qubits, m, n)
    istate = istate / np.linalg.norm(istate)
    nr = np.dot(istate, targetState)

    return nr, istate


###################################################################
#################################################################


def find_perm_sequence_random_mcx(
    problem_matrix,
    niter=100,
    qubit_group=(0, 5),
    target_state=[212.0, 1.0, 1.0, 212.0],
    targetchoice=2,
    randomcontrol=False,
):
    """4 FACES OF RUBIKS CUBE
    Finds the solution sequence for blockmodeling:
        -Generates random P with random a and b,and swaps state a and b
        --if P improves the solution, it kepts that,
        --otherwise, repeats the process
        -repeat above for niter times

    Parameters
    ----------
    problem_matrix : TYPE
        DESCRIPTION.
    niter : int, optional
        umber of max iteration. The default is 100.
    qubit_group : tuple, optional
        ()). The default is (0,5).
    target_state : numpy vector, optional
        solution state for target qubits. The default is [212.,   1.,   1., 212.].

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    M, N = problem_matrix.shape
    MN = M * N
    n = int(np.log2(N))
    nqubits = n * 2

    if 2**nqubits != N * M:
        raise Exception("Shape of the problem matrix must be 2^n")
        return problem_matrix
    rng = np.random.default_rng()

    I = np.eye(2**n, dtype="int")

    # vectorize matrix
    vec_pmat = np.reshape(problem_matrix, (MN, 1))
    best_vec = vec_pmat.copy()
    target_state_normalized = target_state / np.linalg.norm(target_state)

    best_fitness, best_state = fitness(best_vec, qubit_group, target_state_normalized)
    print("target state:", target_state)
    print("initial fitness: ", best_fitness)

    iperm = 0
    perms = {}
    runs = [[best_fitness, best_fitness]]

    i = 0
    targetqubit = 0
    skipgeneratingCX = False
    if targetchoice == 2 and randomcontrol == False:
        targetqubit = 0
        if randomcontrol == False:
            controlqubits = list(range(n))
            controlqubits.remove(targetqubit)
        else:
            controlqubits = random.sample(list(range(n)), rng.integers(low=0, high=n))
            if targetqubit in controlqubits:
                controlqubits.remove(targetqubit)
        CX = multi_controlled_x(controlqubits, targetqubit, n)
        skipgeneratingCX = True

    while i < niter and best_fitness < 1.0:
        # generate random index
        indx = rng.integers(low=0, high=N)
        PX = permutation_matrix_fromX(indx, n)
   
        if skipgeneratingCX == False:
            # random target qubit
            if targetchoice == 0:
                targetqubit = rng.integers(low=0, high=n)
            elif targetchoice == 1:  # orderred target
                targetqubit = (targetqubit + 1) % n
            elif targetchoice == 2:  # fixed target
                targetqubit = 0

            if randomcontrol == False:
                controlqubits = list(range(n))
                controlqubits.remove(targetqubit)
            else:
                controlqubits = random.sample(list(range(n)), 
                                              rng.integers(low=0, high=n))
                if targetqubit in controlqubits:
                    controlqubits.remove(targetqubit)
            CX = multi_controlled_x(controlqubits, targetqubit, n)

        Pab = PX @ CX @ PX
        P = np.kron(Pab, Pab)
        ivec = P @ best_vec

        # C = np.reshape(ivec, (m,n))
        # draw_matrix(C)
        ifitness, istate = fitness(ivec, qubit_group, 
                                   target_state_normalized)
        best_state = istate
        print("inf{} ifitness: {}, best: {}".format(indx,ifitness, best_fitness))
        runs.append([ifitness, best_fitness])

        if ifitness > best_fitness:
            best_fitness = ifitness
            best_vec = ivec
            best_state = istate
            # perms[iperm] = (CX)
            iperm += 1
            print("fitness changed-fitness: {}, iperm: {}"
                  .format(best_fitness, iperm))

        i += 1

    return perms, best_fitness, best_state, best_vec, runs


def find_perm_sequence_classical_pab(
    problem_matrix, niter=100, qubit_group=(0, 5), target_state=[212.0, 1.0, 1.0, 212.0]
):
    """Finds the solution sequence for blockmodeling:
        -Generates random P with random a and b,and swaps state a and b
        --if P improves the solution, it kepts that,
        --otherwise, repeats the process
        -repeat above for niter times

    Parameters
    ----------
    problem_matrix : TYPE
        DESCRIPTION.
    niter : int, optional
        umber of max iteration. The default is 100.
    qubit_group : tuple, optional
        ()). The default is (0,5).
    target_state : numpy vector, optional
        solution state for target qubits. The default is [212.,   1.,   1., 212.].

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    M, N = problem_matrix.shape
    MN = M * N
    n = int(np.log2(N))
    nqubits = n * 2

    if 2**nqubits != N * M:
        raise Exception("Shape of the problem matrix must be 2^n")
        return problem_matrix
    rng = np.random.default_rng()

    I = np.eye(2**n, dtype="int")
    indices = np.arange(I.shape[0])

    # vectorize matrix
    vec_pmat = np.reshape(problem_matrix, (MN, 1))
    best_vec = vec_pmat.copy()
    target_state_normalized = target_state / np.linalg.norm(target_state)

    best_fitness, best_state = fitness(best_vec, qubit_group, target_state_normalized)
    print("target state:", target_state)
    print("initial fitness: ", best_fitness)

    iperm = 0
    perms = {}
    runs = [[best_fitness, best_fitness]]

    i = 0
    while i < niter and best_fitness < 1.0:
        # first X gates
        Pab = I.copy()
        a = rng.integers(low=0, high=I.shape[0])
        b = rng.integers(low=0, high=I.shape[0])
        Pab[[a, b], :] = I[[b, a], :]
        P = np.kron(Pab, Pab)
        # you can simply swap best_vec[a] and bes_vec[b]
        # for completeness we use matvec
        ivec = P @ best_vec

        # C = np.reshape(ivec, (m,n))
        # draw_matrix(C)
        ifitness, istate = fitness(ivec, qubit_group, target_state_normalized)
        best_state = istate
        print("ifitness: {}, best: {}".format(ifitness, best_fitness))
        runs.append([ifitness, best_fitness])

        if ifitness > best_fitness:
            best_fitness = ifitness
            best_vec = ivec
            best_state = istate
            perms[iperm] = P
            iperm += 1
            print("fitness changed-fitness: {}, iperm: {}".format(best_fitness, iperm))

        i += 1

    return perms, best_fitness, best_state, best_vec, runs


# if __name__=="__main__":

G, Adj, Adj_shuffled = barbell_graph_example()
target_state = [212.0, 1.0, 1.0, 212.0]


(perms, 
 best_fitness, 
 best_state, 
 best_vec, 
 runs) = find_perm_sequence_random_mcx(Adj_shuffled,
                                        niter=200, 
                                        qubit_group=(0, 5),
                                        targetchoice=2,
                                        randomcontrol=False
                                        )

fnameruns = "barbell_runs_random_mcx"
fnamematrix = "barbell_solmat_random_mcx"
(m, n) = Adj.shape
sol_matrix = np.reshape(best_vec, (m, n))

indices = np.arange(sol_matrix.shape[0])
draw_matrix(sol_matrix, fnamematrix)
            #xlabels=indices + 1, ylabels=indices + 1)

r = np.array(runs)
fig = plt.figure()
plt.plot(
    r[:, 0], color="blue", label="iteration fitness", linestyle="dotted", linewidth=1.5
)
plt.plot(
    r[:, 1], color="orange", label="best fitness", linestyle="dashed", linewidth=1.5
)
plt.legend()
ax = plt.gca()
# plt.axes([0.3, 0.23, 0.3, 0.5])
# plt.plot(r[:100, 0], color="blue",label="iteration fitness",
#          linestyle="dotted",linewidth=1.5)
# plt.plot(r[:100, 1], color="orange", label="best fitness",
#          linestyle="dashed", linewidth=1.5)
ax.text(int(len(runs) / 2), 0.85, r"fitness:%lf" % best_fitness, fontsize=9)
plt.show()

savefig_tofile(fig, fname=fnameruns)
