# A weighted Energy Distance approach
from scipy.spatial import distance
import numpy as np

def WED(X, Y):
    """
    Calculates the weighted Energy Distance between two sets of planetary systems (or any other user defined set).

    Parameters
    ----------
    X : list of 'n' planets (in d-dimensional phase space) in following format:
                                [(x_1,x_2,....,x_n, w_x)_1, (x_1,x_2,....,x_d, w_x)_2,....,(x_1,x_2,....,x_d, w_x)_n]
    Y : list of 'm' planets (in d-dimensional phase space) in following format:
                                [(y_1,y_2,....,y_n, w_y)_1, (y_1,y_2,....,y_d, w_y)_2,....,(y_1,y_2,....,y_d, w_y)_n]

    Returns
    -------
    Weighted Energy Distance

    Examples
    --------
    from PASSta import WED
    
    WED([(1,2,3),(1.1,2.1,3.1)], [(1,2,3),(1.2,2.2,3.2)]) #---> 0.274
    WED([(1,2,3)], [(1,2,3),(1.2,2.2,3.2)]) #---> 0.388

    """

    n, m = len(X), len(Y)

    # Check if X or Y are empty
    if n == 0 or m == 0:
        raise ValueError("WED assumes both X and Y are not empty")

    # Get phase space dimensional and check that all dimensions of X_i and Y_j are the same
    xdim = len(X[0])
    ydim = len(Y[0])
    if xdim != ydim:
        raise ValueError("Inconsistent planet phase space dimensions")

    for x in X:
        if xdim != len(x):
            raise ValueError("All X elements must be of same size")

    for y in Y:
        if ydim != len(y):
            raise ValueError("All Y elements must be of same size")


    # Get X,Y weight vectors and their sums
    W_x = np.array([xi[xdim-1] for xi in X])
    W_y = np.array([yi[ydim-1] for yi in Y])

    W_X, W_Y = sum(W_x), sum(W_y)

    Xd = [x[:xdim-1] for x in X]
    Yd = [y[:ydim-1] for y in Y]

    A_DistMat = distance.cdist(Xd, Yd, 'euclidean')
    A = sum(sum((np.outer(W_x, W_y) * A_DistMat))) / (W_X * W_Y)

    B_DistMat = distance.cdist(Xd, Xd, 'euclidean')
    B = sum(sum((np.outer(W_x, W_x) * B_DistMat))) / (W_X * W_X)

    C_DistMat = distance.cdist(Yd, Yd, 'euclidean')
    C = sum(sum((np.outer(W_y, W_y) * C_DistMat))) / (W_Y * W_Y)

    e = 2 * A - B - C

    return e**0.5
