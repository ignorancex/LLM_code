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

def lapl_rw(G: nx.Graph, cycle: tuple) -> float:
    """
    Calculates the probability of cycle being induced by a uniform spanning tree on G.

    Through laplacian random walks:
    sum over all edges: Probability of the path without the edge
    """
    total_prob = 0 
    A = nx.adjacency_matrix(G).todense()
    for start in range(len(cycle)):
        walk = cycle[start:] + cycle[:start]
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
                cycle = normalize_cell(tuple(nx.astar_path(T, u, v)))
                if cycle not in res:
                    res[cycle] = lapl_rw(G, cycle)
    return res

def last_step(d, n, l):
    return 1 / (1 + (d-2)*(n-l)/(n-3)/l)

def estimate_probabilities(G: nx.Graph, cycles, q) -> dict[tuple, float]:
    res = {}
    n = len(G.nodes)

    for cycle in cycles:
        deg_sum = 0
        deg_prod = 1
        l = len(cycle)
        for u in cycle:
            deg_prod *= G.degree[u] - 1
        for u,v,v_prime in zip(cycle, cycle[1:] + cycle[:1], cycle[2:] + cycle[:2]):
            deg_sum += (G.degree[u]-1) * (G.degree[v]-1) / G.degree[u] * (G.degree[v_prime]-1) * last_step(G.degree[v_prime], n, l)
        mod = (n-1)/n * ((n-2)/n)**(l-3)
        res[cycle] = deg_sum / deg_prod * mod
    return res

def approx_last_step(d, n, l):
    return (n-3) * l / ((d-2)*(n-l))

def approx_estimate_probabilities(G: nx.Graph, cycles, q) -> dict[tuple, float]:
    res = {}
    n = len(G.nodes)

    for cycle in cycles:
        deg_sum = 0
        deg_sum_alt = 0
        deg_prod = 1
        l = len(cycle)
        for u in cycle:
            deg_prod *= G.degree[u] - 1
        for u,v in zip(cycle, cycle[1:] + cycle[:1]):
            deg_sum += (G.degree[v]-1) * (G.degree[u]-1) * (n-3) / (G.degree[u] - 2) if G.degree[u] > 2 else 1
            deg_sum_alt += (G.degree[u]-1) * (G.degree[v]-1)
        deg_sum = deg_sum * l / (n-l) if n > l else deg_sum_alt
        mod = (n-1)/n * ((n-2)/n)**(l-3) * ((n-1)*q-1) / ((n-1)*q)
        res[cycle] = min(deg_sum,deg_sum_alt) / deg_prod * mod
    return res

def approx_estimate_probabilities2(G: nx.Graph, cycles, q) -> dict[tuple, float]:
    res = {}
    n = len(G.nodes)

    for cycle in cycles:
        deg_sum = 0
        deg_prod = 1
        l = len(cycle)
        for u in cycle:
            deg_prod *= G.degree[u] - 1
        for u,v in zip(cycle, cycle[1:] + cycle[:1]):
            deg_sum += (G.degree[u]-1) * (G.degree[v]-1)
        deg_sum = deg_sum / (1 + ((n-1)*q-2)*(n-l)/(n-3)/l)
        mod = (n-1)/n * ((n-2)/n)**(l-3) * ((n-1)*q-1) / ((n-1)*q)
        res[cycle] = deg_sum / deg_prod * mod
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
approx_estimates = approx_estimate_probabilities(G, probs.keys(), p)
approx_estimates2 = approx_estimate_probabilities2(G, probs.keys(), p)
if kind == 'sbm' or kind == 'fullbip':
    same_cluster = {k : not (min(k) < n1 and max(k)) >= n1 for k in probs.keys()}
else:
    same_cluster = defaultdict(lambda: True)

comparison = np.array([(estimates[k],probs[k],approx_estimates[k], approx_estimates2[k], same_cluster[k]) for k in probs.keys()])
df_result = pd.DataFrame(comparison, columns=['estimated', 'correct', 'approx_estimated', 'approx_estimated2', 'same_cluster'])
df_result['len'] = [len(c) for c in probs.keys()]

df_result.to_csv(snakemake.output[0])
