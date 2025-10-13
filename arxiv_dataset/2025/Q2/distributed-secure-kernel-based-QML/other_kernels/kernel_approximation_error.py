import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

def exact_rbf_kernel(X, sigma):
    sq_dists = euclidean_distances(X, X, squared=True)
    return np.exp(-sq_dists/(2*sigma**2))

def exact_laplacian_kernel(X, alpha):
    L1_dists = manhattan_distances(X, X)
    return np.exp(-L1_dists/alpha)

def quantum_feature_map_rbf(X, D, sigma):
    np.random.seed(42)
    N, d = X.shape
    W = np.random.normal(0, 1/sigma, size=(D, d))
    phi_cos = np.cos(X.dot(W.T))
    phi_sin = np.sin(X.dot(W.T))
    return np.sqrt(1.0/D) * np.hstack([phi_cos, phi_sin])

def quantum_feature_map_laplacian(X, D, alpha):
    np.random.seed(42)
    N, d = X.shape
    W = (1/alpha) * np.random.standard_cauchy(size=(D, d))
    phases = np.random.uniform(0, 2*np.pi, size=(D,))
    projection = X.dot(W.T) + phases
    phi_cos = np.cos(projection)
    phi_sin = np.sin(projection)
    return np.sqrt(1.0/D) * np.hstack([phi_cos, phi_sin])

N = 100
d_low = 100000
d_high = 100000
Ds = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
Dtwos = [32,128, 512, 2048, 8192, 32768]
params = [0.05,0.1, 0.5, 1.0,1.5, 2.0]
params_two= [1.0,1.5, 2.0, 2.5,3,3.5,4,4.5,5]
n_runs = 5

results_rbf = {p: [] for p in params}
results_lap = {p: [] for p in params_two}

for D in Ds:
    errors_rbf = {p: [] for p in params}
    errors_lap = {p: [] for p in params_two}
    for _ in range(n_runs):
        print(f'starting run number {_} for D = {D}')
        d = np.random.randint(d_low, d_high+1)
        X = np.random.randn(N, d)
        for sigma in params:
            K_exact = exact_rbf_kernel(X, sigma)
            Z = quantum_feature_map_rbf(X, D, sigma)
            err = np.linalg.norm(K_exact - Z.dot(Z.T), 'fro')/ np.linalg.norm(K_exact, 'fro')
            errors_rbf[sigma].append(err)
        for alpha in params_two:
            K_exact = exact_laplacian_kernel(X, alpha)
            Z = quantum_feature_map_laplacian(X, D, alpha)
            err = np.linalg.norm(K_exact - Z.dot(Z.T), 'fro')/ np.linalg.norm(K_exact, 'fro')
            errors_lap[alpha].append(err)
    for sigma in params:
        results_rbf[sigma].append((np.mean(errors_rbf[sigma]), np.std(errors_rbf[sigma])))
    for alpha in params_two:
        results_lap[alpha].append((np.mean(errors_lap[alpha]), np.std(errors_lap[alpha])))

fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
ax.set_facecolor('white')
for sigma in params:
    means = [val[0] for val in results_rbf[sigma]]
    stds = [val[1] for val in results_rbf[sigma]]
    ax.errorbar(Ds, means, yerr=stds, marker='o', label=f"$\\sigma = {sigma}$")
ax.set_xscale('log')
ticks = Dtwos
labels = [f"{d}\n({int(np.log2(d))} {'qubit' if int(np.log2(d))==1 else 'qubits'})" for d in Dtwos]
ax.set_xticks(ticks)
ax.tick_params(axis='x', labelsize=9)
ax.set_xticklabels(labels)
ax.set_xlabel('Number of Random Fourier Features (D)')
ax.set_ylabel('Relative Frobenius Norm Error')
ax.legend(frameon=True, facecolor='white', edgecolor='black')

ax.yaxis.grid(True, linestyle=':', linewidth=1)
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.2)

plt.show()

fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
ax.set_facecolor('white')
for alpha in params_two:
    means = [val[0] for val in results_lap[alpha]]
    stds = [val[1] for val in results_lap[alpha]]
    ax.errorbar(Ds, means, yerr=stds, marker='o', label=f"$\\alpha = {alpha}$")
ax.set_xscale('log')
ticks = Dtwos
labels = [f"{d}\n({int(np.log2(d))} {'qubit' if int(np.log2(d))==1 else 'qubits'})" for d in Dtwos]
ax.set_xticks(ticks)
ax.tick_params(axis='x', labelsize=9)
ax.set_xticklabels(labels)
ax.set_xlabel('Number of Random Fourier Features (D)')
ax.set_ylabel('Relative Frobenius Norm Error')
ax.legend(frameon=True, facecolor='white', edgecolor='black')
ax.yaxis.grid(True, linestyle=':', linewidth=1)
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.2)
plt.show()

