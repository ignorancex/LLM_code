import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from mpi4py import MPI
import pickle
import sys
sys.path.append('..')
from library.py import compute_partial_kernel
import matplotlib.pyplot as plt
from matplotlib import style

print('Starting...')
style.use('bmh')

# MPI Setup: Initialize MPI communication and retrieve the rank and size of the current process
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Data Preprocessing (Executed by rank 0 only)
if rank == 0:
    df = pd.read_csv('framingham.csv')
    df = df.drop(['education'], axis=1)
    df.dropna(inplace=True)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    sampler = RandomOverSampler(sampling_strategy=0.6)
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    X_train, _, y_train, _ = train_test_split(X_resampled, y_resampled, train_size=0.8, stratify=y_resampled, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
else:
    X_scaled = None
    y_train = None

# Broadcast the preprocessed data to all processes
X_scaled = comm.bcast(X_scaled if rank == 0 else None, root=0)
y_train = comm.bcast(y_train if rank == 0 else None, root=0)

# Task Distribution: Divide the pairwise computations across the available processes
n_samples_X = X_scaled.shape[0]
print(f'shape is {X_scaled.shape}')
total_interactions = n_samples_X * (n_samples_X + 1) // 2
interactions_per_process = total_interactions // size
remainder = total_interactions % size
start_interaction = rank * interactions_per_process + min(rank, remainder)
end_interaction = start_interaction + interactions_per_process + (1 if rank < remainder else 0)

tasks = []
current_interaction = 0
outer_break = False

# Assign specific tasks (pairwise kernel computations) to this process
for i in range(n_samples_X):
    for j in range(i + 1):
        if current_interaction < start_interaction:
            current_interaction += 1
            continue
        if current_interaction >= end_interaction:
            outer_break = True
            break
        tasks.append((i, j))
        current_interaction += 1
    if outer_break:
        break

# Noise Levels: Iterate through different noise levels to compute kernel matrices
noise_levels = [0, 1, 2]
models = {}
for noise_level in noise_levels:
    K_partial = compute_partial_kernel(X_scaled, tasks, noise_level)
    all_K_partials = comm.gather(K_partial, root=0)

    if rank == 0:
        K = np.zeros((n_samples_X, n_samples_X))
        for part in all_K_partials:
            for i in part:
                for j in part[i]:
                    K[i, j] = part[i][j]
                    K[j, i] = part[i][j]

        with open(f'heart_kernel_matrix_noise_level_{noise_level}.pkl', 'wb') as f:
            pickle.dump(K, f)

# Centralized Computation: Compute the kernel matrix for the centralized quantum scenario
K_partial = compute_partial_kernel(X_scaled, tasks, noise_level, 1)
all_K_partials = comm.gather(K_partial, root=0)
if rank == 0:
    K = np.zeros((n_samples_X, n_samples_X))
    for part in all_K_partials:
        for i in part:
            for j in part[i]:
                K[i, j] = part[i][j]
                K[j, i] = part[i][j]

    with open(f'heart_centralized_kernel_matrix.pkl', 'wb') as f:
        pickle.dump(K, f)
