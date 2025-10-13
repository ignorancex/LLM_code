import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np

from root.correlations.correlations import Correlations
from root.data.data import Data

mnist = Data.load("all_subsets")
artificial_correlated = Data.load("decreasingly_correlated_sets")

y1 = Data.load("k_max_subsets_at_n2")
y2 = Data.load("k_max_decor_n2")

# x1 = create_x(y1, mnist)
# x2 = create_x(y2, artificial_correlated)

print(mnist.shape)
print(artificial_correlated.shape)

for set in mnist:
    cov = Correlations.calc_empirical_covariance(set)
    np.savetxt("covariance_matrix.txt", cov)

    # Load the covariance matrix and verify its integrity

    ev = Correlations.calc_eigenvalue_spectra(set)
    print("top5 ", ev[:5])

    # Plot the eigenvalue spectrum
    plt.figure(figsize=(8, 5))
    plt.plot(ev, "o-", markersize=4)
    plt.title("Eigenvalue Spectrum of the Covariance Matrix")
    plt.xlabel("Eigenvalue Index")
    plt.ylabel("Eigenvalue")
    plt.grid(True)
    plt.show()
