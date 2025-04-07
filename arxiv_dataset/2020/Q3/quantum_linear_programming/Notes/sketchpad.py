# Connect to DWave
# from dwave.cloud import Client

# client = Client(endpoint=url, token=token)
# solver_names = client.get_solvers()
# print(solver_names)
# solver = client.get_solver("DW_2000Q_5")

# Map graph minor to hardware graph
# Find hardware graph
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite

# Sampler properties
url = "https://cloud.dwavesys.com/sapi"
token = open('./apikey.txt').read()
solver = "DW_2000Q_5"
sampler = DWaveSampler(endpoint=url, token=token, solver=solver)
embed = EmbeddingComposite(sampler)
qubo = {
    (0, 0): -3,
    (0, 1): 2,
    (1, 0): 2,
    (1, 1): -4,
    (2, 2): -3,
    (2, 3): 2,
    (3, 2): 2,
    (3, 3): -4,
}

# Sampler parameters
annealing_time = 2000  # integer microseconds [1, 2000]
answer_mode = "histogram" # "raw" or "histogram
auto_scale = True
num_reads = 1000  # raw will dump out all results
num_spin_reversal_transforms = 10  # ask Travis what this is

result = embed.sample_qubo(
    qubo,
    answer_mode=answer_mode,
    auto_scale=auto_scale,
    annealing_time=annealing_time,
    num_reads=num_reads,
    num_spin_reversal_transforms=num_spin_reversal_transforms,
)
print(result)
for d in result.data():
    print(d[0], "Energy: ", d[1], "Occurrences: ", d[2])

# sampler = DWaveSampler(endpoint=url, token=token)
# adjacency = sampler.adjacency

# Minor embedding
# from minorminer import find_embedding

# qubo = {(0, 0): 1.0, (0, 1): 1.0, (1, 0): 1.0, (1, 1): 1.0}
# emb = find_embedding(qubo, adjacency)
# print(emb)
