"""
Evaluates the probability approximation for ER-graphs.
"""

import script_utils as su
from snakemake.script import Snakemake
import networkx as nx
import numpy as np
import pandas as pd
import time
from collections import defaultdict
from random_ccs.sampling import uniform_cc, estimate_len_count_fast as estimate_len_count, NP_EDGE
from random_ccs.spanning_trees import random_spanning_tree

def fix_smk() -> Snakemake:
    """
    Helper function to make linters think `snakemake` exists
    and to add type annotation. Doesn't change any code behavior.
    """
    return snakemake

SEEDS = (757821932, 251989800, 395713524, 661430121, 629645171, 933919130, 893559118, 434866178, 286768719, 525508293)

snakemake = fix_smk()

n = int(snakemake.wildcards['n'])
p = float(snakemake.wildcards['p'])
N = n * float(snakemake.wildcards['N_factor'])
samples = int(snakemake.wildcards['samples'])
method = snakemake.wildcards['method']

seed = np.random.default_rng()

data = []

for run in range(10):
    undersample = 0
    overcorrelate = 0
    results = 0
    len_res = defaultdict(lambda: 0)
    before = time.time()
    if method == 'er-est':
        _, cells, undersample, overcorrelate = uniform_cc(n, p, N, samples, seed=SEEDS[run], fast_sampling=False)
        results = len(cells)
        for c in cells:
            len_res[f'count-{len(c)}'] += 1
    elif method == 'er-approx-est':
        _, cells, undersample, overcorrelate = uniform_cc(n, p, N, samples, seed=SEEDS[run], fast_sampling=True)
        results = len(cells)
        for c in cells:
            len_res[f'count-{len(c)}'] += 1
    elif method == 'nx-rst': # very slow
        for _ in range(samples):
            G = nx.gnp_random_graph(n, p)
            while not nx.is_connected(G):
                G = nx.gnp_random_graph(n, p)
            _ = nx.random_spanning_tree(G)
    elif method == 'py-rst':
        for _ in range(samples):
            G = nx.gnp_random_graph(n, p)
            while not nx.is_connected(G):
                G = nx.gnp_random_graph(n, p)
            random_spanning_tree(G, seed)
    elif method == 'len-count':
        est_samples = samples // 10 if samples > 100 else samples
        G = nx.gnp_random_graph(n, p)
        edges = np.array([(u,v) for (u,v) in G.edges], dtype=NP_EDGE)
        estimate_len_count(G, edges, p, est_samples, seed)
    else:
        raise RuntimeError('Unknown method: ' + method)
    runtime = time.time() - before
    data.append({
        'method': method, 'n': n, 'p': p, 'N': N, 'samples': samples, 'runtime': runtime,
        'cell_count': results, 'undersample': undersample, 'overcorrelate': overcorrelate, **len_res
    })
    
df_result = pd.DataFrame(data)

df_result.to_csv(snakemake.output[0])
