import pandas as pd
import numpy as np
from snakemake.script import Snakemake
import networkx as nx
from collections import defaultdict
import script_utils as su
from random_ccs.sampling import uniform_cc
from scipy.sparse.linalg import eigsh
from edge_flow_cell_complexes.generator import TriangulationComplexGenerator
from math import sqrt

def fix_smk() -> Snakemake:
    """
    Helper function to make linters think `snakemake` exists
    and to add type annotation. Doesn't change any code behavior.
    """
    return snakemake

snakemake = fix_smk()

n = int(snakemake.wildcards['n'])
p = float(snakemake.wildcards['p'])
N = int(snakemake.wildcards['N'])
samples = int(snakemake.wildcards['samples'])
runs = int(snakemake.wildcards['runs'])
model = snakemake.wildcards['model']

seeds = (720507023, 297554076, 10272976, 463024305, 90119225,
47776976, 691102297, 65747652, 759901654, 802456260,
342912204, 781417077, 637995996, 994109826, 815140230,
127723192, 107272427, 579731718, 318046254, 551533860,
993318543, 910535274, 979510462, 525271280, 682533905,
996125417, 923466556, 183489074, 676280881, 8401446,
790444476, 838337533, 628017784, 173164059, 437824856,
38284666, 931546790, 320072774, 416369951, 280544731,
453078248, 374164014, 954841428, 668062934, 5509502,
59513687, 867052939, 998941624, 549007697, 40111251,
128513815, 567603572, 331025506, 631135083, 140414700,
728342563, 198331443, 277111076, 206440743, 923337719,
545420357, 236045285, 469163432, 184359316, 728813996,
345501852, 680369203, 80435551, 649261253, 228462867,
900124757, 536102448, 987611379, 696310430, 578658172,
712805529, 98491716, 245414568, 884927859, 299820734,
198723355, 261062911, 474595575, 118980587, 58102329,
249640697, 610818636, 128661837, 77967708, 704864575,
58932385, 375904570, 722497187, 272692532, 708499431,
761852873, 142645345, 220226039, 825426679, 670693001)

def count_homology(A):
    if A.shape[0] <= 2:
        return 0
    ev, evec = eigsh(A, A.shape[0]-2, which='SM')
    return np.count_nonzero(np.abs(ev) < 1e-10)

def is_orientable(B2):
    if B2.shape[1] == 0:
        return True
    if max(np.sum(np.abs(B2), axis=1)) > 2:
        # If three cells share an edge, the CC is not orientable
        return False
    orientations = np.zeros(B2.shape[1])

    def orient(i, o):
        if orientations[i] == 0:
            orientations[i] = o
            indicator = np.zeros(B2.shape[1])
            indicator[i] = o
            edge_orientations = B2 @ indicator
            for edge, e_orient in enumerate(edge_orientations):
                if e_orient != 0:
                    indicator = np.zeros(B2.shape[0])
                    indicator[edge] = e_orient
                    for cell, c_orient in enumerate(B2.T @ indicator):
                        orient(cell, c_orient)
        elif orientations[i] == -o:
            # previously discovered with opposite orientation
            return False
        
    for i in range(B2.shape[1]):
        if orientations[i] == 0:
            orient(i, 1)

    return True


data = []
for i in range(runs):
    rnd = np.random.default_rng(seeds[i])
    if model == 'cellular_er':
        G, cells, _, _ = uniform_cc(n, p, N, samples, seed=rnd)
    elif model == 'triangulation':
        max_cell_len = int(2 * sqrt(n))
        cell_lengths = list(rnd.choice(list(range(3, max_cell_len + 1)), N, replace=True))
        gen = TriangulationComplexGenerator(seeds[i], n, cell_lengths, delete_edges=0, delete_nodes=0)
        cc = gen.generate()
        G = nx.Graph() 
        G.add_nodes_from(list(range(n)))
        G.add_edges_from(cc.get_cells(1))
        cells = cc.get_cells(2)

    B1 = nx.incidence_matrix(G, oriented=True)
    B2 = su.cell_boundary_matrix(list(G.edges), cells)

    L1 = B1.T @ B1 + B2 @ B2.T
    L2 = B2.T @ B2 #B3 is empty

    hom1 = count_homology(L1)
    hom2 = count_homology(L2)

    orientable = is_orientable(B2)

    data.append({'model': model, 'n': n, 'p': p, 'edges': len(G.edges), 'N': N, 'N_act': len(cells), 'hom1': hom1, 'hom2': hom2, 'orientable': orientable, 'run': i})

df_est = pd.DataFrame(data)
df_est.to_csv(snakemake.output[0])