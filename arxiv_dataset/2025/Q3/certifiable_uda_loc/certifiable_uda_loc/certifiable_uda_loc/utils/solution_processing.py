import numpy as np


def compute_eigenvalue_ratio(moment_matrix: np.ndarray):
    unsorted_eigvals, _ = np.linalg.eig(moment_matrix)
    eigvals = np.sort(unsorted_eigvals)[::-1]
    ratio = np.abs(eigvals[0] / eigvals[1])
    return ratio


# This might take a big refactor later when we have bigger problem structures...
# What happens when we have multiple factors like this? This will be a mess lmao
def extract_lifted_solution(moment_matrix: np.ndarray):
    eigvals, eigvecs = np.linalg.eig(moment_matrix)
    max_idx = np.argmax(eigvals)
    max_eigval = eigvals[max_idx]
    max_eigvec = eigvecs[:, max_idx]
    if max_eigvec[0] < 0:  # homogenization variable
        max_eigvec *= -1
    return np.sqrt(max_eigval) * max_eigvec
