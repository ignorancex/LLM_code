from scipy.spatial import distance
import numpy as np
from PASSta import WED

def ICED(X, Y):
    """
    Testing for Equal Distributions of two planetary systems samples using WED metric.

    Parameters
    ----------
    ~X : list of 'N' planetary systems X_i (see WED format) in following format: [X_1, X_2,...,X_N]
    ~Y : list of 'M' planetary systems Y_i (see WED format) in following format: [Y_1, Y_2,...,Y_N]

    Returns
    -------
     A two-sample Energy Distance

    Examples
    --------
    from PASSta import ICED

    ICED([[(1,2,3),(1.1,2.1,3.1)], [(1,2,3),(1.2,2.2,3.2)]],
        [[(1,2,3),(1.1,2.1,3.1)], [(1.5,2,3),(1.5,2.2,3.2)]]) #---> 0.623
    """

    N, M = len(X), len(Y)

    # Check if X or Y are empty
    if N == 0 or M == 0:
        raise ValueError("ICED assumes both X and Y are not empty")

    A = sum(sum(np.array([[WED(i, j) for j in Y] for i in X]))) / (N * M)

    B = sum(sum(np.array([[WED(i, j) for j in X] for i in X]))) / (N * N)

    C = sum(sum(np.array([[WED(i, j) for j in Y] for i in Y]))) / (M * M)

    e = 2 * A - B - C

    return e**0.5
