import torch
from heuristics import *
from data_loader import *
from learnable_mmf_model import *
from baseline_mmf_model import Baseline_MMF
from metaheuristics import evolutionary_algorithm, get_cost, directed_evolution

start_time = time.time()

# Data loading
karate_laplacian = karate_def('D:\codebase\Learnable_MMF\data')
N = karate_laplacian.size(0)

# Baseline (original) MMF
original_start_time = time.time()
original_mmf = Baseline_MMF(N=N, L=N - 8, dim=8)
A_rec, U, D, mother_coefficients, father_coefficients, mother_wavelets, father_wavelets = original_mmf(karate_laplacian)
original_end_time = time.time()
original_runtime = original_end_time - original_start_time
original_mmf_cost = torch.norm(karate_laplacian - A_rec, p='fro').item()

# Learnable MMF (random indices)
random_indices_start_time = time.time()
wavelet_indices, rest_indices = heuristics_random(karate_laplacian.to_sparse(), L=26, K=8, drop=1, dim=8)
A_rec, U, D, mother_coefficients, father_coefficients, mother_wavelets, father_wavelets = learnable_mmf_train(karate_laplacian, L=26, K=8, drop=1, dim=8, wavelet_indices=wavelet_indices, rest_indices=rest_indices, epochs=0, learning_rate=1e-4, early_stop=True)
random_indices_end_time = time.time()
random_indices_runtime = random_indices_end_time - random_indices_start_time
random_indices_cost = torch.norm(karate_laplacian - A_rec, p='fro').item()

# Learnable MMF (heuristics to select indices)
heuristics_start_time = time.time()
wavelet_indices, rest_indices = heuristics_k_neighbors_single_wavelet(karate_laplacian.to_sparse(), L=26, K=8, drop=1, dim=8)
A_rec, U, D, mother_coefficients, father_coefficients, mother_wavelets, father_wavelets = learnable_mmf_train(karate_laplacian, L=26, K=8, drop=1, dim=8, wavelet_indices=wavelet_indices, rest_indices=rest_indices, epochs=0, learning_rate=1e-3, early_stop=True)
heuristics_end_time = time.time()
heuristics_runtime = heuristics_end_time - heuristics_start_time
heuristics_cost = torch.norm(karate_laplacian - A_rec, p='fro').item()

# Learnable MMF (EA to select indices)
ea_start_time = time.time()
wavelet_indices, rest_indices, ea_cost, ea_min_cost_per_gen, ea_mean_cost_per_gen, ea_all_time_min_cost_per_gen = evolutionary_algorithm(get_cost, karate_laplacian, L=26, K=8, population_size=100, generations=100, mutation_rate=0.2)
ea_end_time = time.time()
ea_runtime = ea_end_time - ea_start_time

# Learnable MMF (DE to select indices)
de_start_time = time.time()
wavelet_indices, rest_indices, de_cost, de_min_cost_per_gen, de_mean_cost_per_gen, de_all_time_min_cost_per_gen = directed_evolution(get_cost, karate_laplacian, L=26, K=8, population_size=100, generations=100, sample_kept_rate=0.5)
de_end_time = time.time()
de_runtime = de_end_time - de_start_time

end_time = time.time()

# Calculate total runtime
total_runtime = end_time - start_time

# Display runtime
print("Original MMF runtime:", original_runtime, "seconds")
print("Random indices MMF runtime:", random_indices_runtime, "seconds")
print("Heuristics MMF runtime:", heuristics_runtime, "seconds")
print("EA MMF runtime:", ea_runtime, "seconds")
print("DE MMF runtime:", de_runtime, "seconds")
print("Total runtime:", total_runtime, "seconds")

# plt.plot(range(len(min_cost_per_gen)), min_cost_per_gen, label='Min evaluation')
generation = 100
plt.plot(range(generation), [original_mmf_cost] * generation, label='Original MMF')
plt.plot(range(generation), [random_indices_cost] * generation, label='Random indices')
plt.plot(range(generation), [heuristics_cost] * generation, label='K neighbours heuristics')
# plt.plot(range(generation), ea_mean_cost_per_gen, label='Mean EA evaluation')
plt.plot(range(generation), ea_all_time_min_cost_per_gen, label='Best EA evaluation')
# plt.plot(range(generation), de_mean_cost_per_gen, label='Mean DE evaluation')
plt.plot(range(generation), de_all_time_min_cost_per_gen, label='Best DE evaluation')

# Adding labels and title
plt.xlabel('Generation')
plt.ylabel('Cost')
plt.title('Metaheuristics versus baselines')
plt.legend()  # Adding legend

# Display plot
plt.show()

# Keep the plot window open until the user exits
while True:
    plt.pause(0.05)  # Pause for 0.05 seconds