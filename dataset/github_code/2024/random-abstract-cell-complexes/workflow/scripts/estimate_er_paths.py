"""
Evaluates the probability approximation for ER-graphs.
"""

from math import ceil
from collections import defaultdict
import script_utils as su
from snakemake.script import Snakemake
import networkx as nx
import numpy as np
import pandas as pd
from random_ccs.spanning_trees import normalize_cell

def fix_smk() -> Snakemake:
    """
    Helper function to make linters think `snakemake` exists
    and to add type annotation. Doesn't change any code behavior.
    """
    return snakemake

snakemake = fix_smk()

def pot_stoch(X, exp):
    """
    converts X to a row-stochastic matrix and calculates returns X^2^exp
    """
    row_sums = X @ np.ones(X.shape[0])
    X = np.diag([1/x if x != 0 else 1 for x in row_sums]) @ X
    for _ in range(exp):
        X = X @ X
    return X

def lapl_rw(G: nx.Graph, path: tuple) -> float:
    """
    Calculates the probability of cycle being induced by a uniform spanning tree on G.

    Through laplacian random walks:
    sum over all edges: Probability of the path without the edge
    """
    total_prob = 0 
    A = nx.adjacency_matrix(G).todense()
    walk = path
    X = nx.adjacency_matrix(G).todense()
    walk_prob = 1
    target_vec = np.zeros_like(X[:,walk[-1]])
    target_vec[walk[-1]] = 1
    X[walk[-1]] = target_vec
    X[walk[0]] = np.zeros_like(X[walk[0]])
    for i in range(1, len(walk)):
        f = pot_stoch(X, len(G.nodes)) @ target_vec
        walk_prob *= f[walk[i]] / (A @ f)[walk[i-1]]
        X[walk[i]] = np.zeros_like(X[walk[i]])
    total_prob += walk_prob
    return total_prob

def calc_probabilities(G: nx.Graph, seed, samples=10,)-> tuple[dict[tuple,int],int]:
    res = {}
    total = 0
    nx.set_edge_attributes(G, 1, 'weight')
    rnd = seed if not isinstance(seed, int) else np.random.default_rng(seed)
    for _ in range(samples):
        T = nx.random_spanning_tree(G, weight=None, seed=rnd)
        total += 1
        if total % 5 == 0:
            print(total)
        for e in G.edges:
            if e not in T.edges:
                u,v = e
                path = tuple(nx.astar_path(T, u, v))
                if path not in res:
                    res[path] = lapl_rw(G, path)
    return res

def last_step(d, n, l):
    return 1 / (1 + (d-2)*(n-l)/(n-3)/l)

def estimate_probabilities(G: nx.Graph, paths, q) -> dict[tuple, float]:
    res = {}
    n = len(G.nodes)

    for path in paths:
        deg_prod = 1
        l = len(path)
        for u in path[1:-2]:
            deg_prod *= G.degree[u] - 1
        first = 1/G.degree[path[0]]
        mod = (n-1)/n * ((n-2)/n)**(l-3)
        last = last_step(G.degree[path[-2]], n, l) # 1/(1+(G.degree[path[-2]] * (n-l)/(n-3)/l))
        res[path] = first * last * mod / deg_prod
    return res

SEED = 381920751

kind = snakemake.wildcards['kind']
n = int(snakemake.wildcards['size'])
p = float(snakemake.wildcards.get('p', 0))
q = float(snakemake.wildcards.get('q', 0))

if kind == 'full':
    p = 1

rnd = np.random.default_rng(SEED)
if kind == 'er':
    G = nx.gnp_random_graph(n, p, seed=rnd)
    while not nx.is_connected(G):
        G = nx.gnp_random_graph(n, p, seed=rnd)
elif kind == 'sbm':
    n1, n2 = n//2, n-n//2
    G = nx.stochastic_block_model([n1, n2], [[p,q], [q,p]], seed=rnd)
    while not nx.is_connected(G):
        G = nx.stochastic_block_model([n1, n2], [[p,q], [q,p]], seed=rnd)
elif kind == 'full':
    G = nx.complete_graph(n)
elif kind == 'fullbip':
    n1, n2 = n//2, n-n//2
    G = nx.complete_bipartite_graph(n1, n2) 
elif kind == 'config':
    seq = [ceil(2*n / (2+i/2)) for i in range(n)]
    if np.sum(seq) % 2 == 1:
        seq[-1] += 1
    G = nx.Graph(nx.configuration_model(seq, seed=rnd))
    while not nx.is_connected(G):
        G = nx.Graph(nx.configuration_model(seq, seed=rnd))
    G = nx.from_edgelist([(u,v) for (u,v) in set(G.edges) if u != v])
elif kind == 'barabasialbert':
    m = int(n * p)
    G = nx.barabasi_albert_graph(n, m, rnd)
else:
    raise RuntimeError(f'unknown model: {kind}')

if kind != 'er':
    p = len(G.edges) * 2 / (n*n - n)

probs = calc_probabilities(G, SEED)
estimates = estimate_probabilities(G, probs.keys(), p)
if kind == 'sbm' or kind == 'fullbip':
    same_cluster = {k : not (min(k) < n1 and max(k)) >= n1 for k in probs.keys()}
else:
    same_cluster = defaultdict(lambda: True)

comparison = np.array([(estimates[k],probs[k], same_cluster[k]) for k in probs.keys()])
df_result = pd.DataFrame(comparison, columns=['estimated', 'correct', 'same_cluster'])
df_result['len'] = [len(c) for c in probs.keys()]

df_result.to_csv(snakemake.output[0])
