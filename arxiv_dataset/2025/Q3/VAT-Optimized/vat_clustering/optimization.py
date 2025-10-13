import numpy as np
import matplotlib.pyplot as plt
from numba import jit

@jit(nopython=True)
def vat_optimized(R):
    """
    Optimized VAT Algorithm using Numba (JIT Compilation).
    """
    N = R.shape[0]
    J = list(range(N))
    I = [np.argmax(np.sum(R, axis=1))]
    J.remove(I[0])
    RV = np.zeros_like(R)
    
    # Convert I to a NumPy array for compatibility
    I = np.array(I)

    for _ in range(1, N):
        min_dists = np.array([np.min(R[j, I]) for j in J])  # Use NumPy operations
        j_star = J[np.argmin(min_dists)]
        I = np.append(I, j_star)  # Update I as a NumPy array
        J.remove(j_star)

    for i in range(N):
        for j in range(N):
            RV[i, j] = R[I[i], I[j]]

    return RV, I