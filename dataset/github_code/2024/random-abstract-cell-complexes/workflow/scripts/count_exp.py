import pandas as pd
import numpy as np
from snakemake.script import Snakemake
import networkx as nx
from collections import defaultdict
import script_utils as su
from random_ccs.sampling import estimate_len_count_fast as estimate_len_count, NP_EDGE
from math import ceil

def fix_smk() -> Snakemake:
    """
    Helper function to make linters think `snakemake` exists
    and to add type annotation. Doesn't change any code behavior.
    """
    return snakemake

snakemake = fix_smk()

SEEDS = (754037574, 753916888, 878405526, 284910723, 490136116, 643595213, 359647163, 500837505, 411655577, 608270388, 387230776, 285543835, 629764367, 468466287, 981230837, 45531133, 16904131, 286347391, 567786555, 754310168, 257139682, 561448341, 343390849, 695695587, 431254817, 487509702, 469032745, 851658883, 486564388, 238093517, 664152317, 905027104, 771257888, 43201794, 639632436, 309853303, 705418535, 367973531, 240575967, 343427292, 172826508, 283056662, 377739114, 943539321, 965268193, 622578117, 673302424, 743645986, 67291836, 860230239, 773947807, 246583930, 1347798, 799989397, 523406397, 339606692, 898781469, 629118124, 15063413, 263629595, 348249844, 645627820, 105175836, 852879224, 212305831, 289942325, 588572490, 313684670, 861053471, 48173487, 30391100, 144769383, 998927226, 317625100, 262007253, 235800540, 766070342, 169406420, 456407560, 181331735, 261708019, 673706223, 970637432, 67859857, 603203796, 428181538, 561822346, 229580388, 815508389, 416158032, 919782092, 583784921, 445944949, 327458591, 412292816, 918189720, 910809961, 29503305, 423838683, 633065120)

kind = snakemake.wildcards['kind']
n = int(snakemake.wildcards['n'])
p = float(snakemake.wildcards.get('p', 0))
q = float(snakemake.wildcards.get('q', 0))
samples = int(snakemake.wildcards['samples'])
run = int(snakemake.wildcards['run'])

if kind == 'full':
    p = 1

data_est = []
i = run
rnd = np.random.default_rng(SEEDS[i])
if kind == 'er':
    G = nx.gnp_random_graph(n, p, seed=rnd)
    while not nx.is_connected(G):
        G = nx.gnp_random_graph(n, p, seed=rnd)
elif kind == 'sbm':
    G = nx.stochastic_block_model([n//2,n-n//2], [[p,q], [q,p]], seed=rnd)
    while not nx.is_connected(G):
        G = nx.stochastic_block_model([n//2,n-n//2], [[p,q], [q,p]], seed=rnd)
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
elif kind == 'karate':
    G = nx.karate_club_graph()
    n = len(G.nodes)
elif kind == 'november17':
    df_adj = pd.read_csv('data/november17/RHODESBOMBING.csv', index_col=0)
    G = nx.from_numpy_array(df_adj.to_numpy())
    n = len(G.nodes)
elif kind == 'barabasialbert':
    m = int(n * p)
    G = nx.barabasi_albert_graph(n, m, rnd)
else:
    raise RuntimeError(f'unknown model: {kind}')
edges = np.array([(u,v) for (u,v) in G.edges], dtype=NP_EDGE)
counts = defaultdict(lambda: 0)

a_priori_est_count = {}
for l in range(3, n+1):
    permutations = 1
    for j in range(l):
        permutations *= n-j
    a_priori_est_count[l] = permutations * p**l / 2 / l

if kind == 'full':
    counts = a_priori_est_count
elif kind == 'fullbip':
    for l in range(2, n2+1):
        permutations = 1
        for j in range(l):
            permutations *= n1-j
            permutations *= n2-j
        counts[2*l] = permutations / 2 / l
else:
    for cycle in nx.simple_cycles(G):
        counts[len(cycle)] += 1

estimation_p = len(G.edges) * 2 / (n*n - n)
est_count = {}
for l in range(3, n+1):
    permutations = 1
    for j in range(l):
        permutations *= n-j
    est_count[l] = permutations * estimation_p**l / 2 / l

sample_est, zeros, sample_occured = estimate_len_count(G, edges, estimation_p, samples, rnd)
sample_est = np.exp2(sample_est)
sample_est[zeros] = 0

sample_est_a_priori, zeros_a_priori, sample_occured_a_priori = estimate_len_count(G, edges, p, samples, rnd)
sample_est_a_priori = np.exp2(sample_est_a_priori)
sample_est_a_priori[zeros_a_priori] = 0

for l in est_count.keys():
    data_est.append({'l': l, 'count': counts[l], 'est': est_count[l], 'a_priori': a_priori_est_count[l], 'sample_est': sample_est[l], 'sample_occured': sample_occured[l], 'sample_est_a_priori': sample_est_a_priori[l], 'sample_occured_a_priori': sample_occured_a_priori[l], 'combined': sample_est[l] if sample_est[l] > 0 else est_count[l], 'run': i, 'kind': kind, 'p': p, 'est_p': estimation_p})

df_est = pd.DataFrame(data_est)
df_est.to_csv(snakemake.output[0])