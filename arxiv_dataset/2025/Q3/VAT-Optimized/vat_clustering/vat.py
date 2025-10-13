import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

def vat(R):
    """
    Implements the VAT algorithm for Visual Assessment of Cluster Tendency.
    
    Parameters:
    R (numpy.ndarray): NxN dissimilarity matrix.
    
    Returns:
    (numpy.ndarray, list): VAT-reordered matrix, Reordering indices.
    """
    N = R.shape[0]
    J = list(range(N))
    I = [np.argmax(np.sum(R, axis=1))]
    J.remove(I[0])
    RV = np.zeros_like(R)

    for _ in range(1, N):
        min_dists = [R[j, I].min() for j in J]
        j_star = J[np.argmin(min_dists)]
        I.append(j_star)
        J.remove(j_star)

    for i in range(N):
        for j in range(N):
            RV[i, j] = R[I[i], I[j]]

    return RV, I

def plot_vat(R, title="VAT Image"):
    """
    Plots the VAT-reordered dissimilarity matrix.
    
    Parameters:
    R (numpy.ndarray): VAT-reordered dissimilarity matrix.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(R, cmap="gray", aspect="auto")
    plt.title(title)
    plt.colorbar(label="Dissimilarity")
    plt.show()
