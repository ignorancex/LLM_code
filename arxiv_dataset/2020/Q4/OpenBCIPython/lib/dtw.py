from numpy import array, zeros, argmin, inf, equal, ndim
from scipy.spatial.distance import cdist
import pandas as pd
import numpy as np

def dtw(x, y, dist):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.

    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure

    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    r, c = len(x), len(y)
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:] # view
    for i in range(r):
        for j in range(c):
            D1[i, j] = dist(x[i], y[j])
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            D1[i, j] += min(D0[i, j], D0[i, j+1], D0[i+1, j])
    if len(x)==1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1] / sum(D1.shape), C, D1, path

def fastdtw(x, y, dist):
    """
    Computes Dynamic Time Warping (DTW) of two sequences in a faster way.
    Instead of iterating through each element and calculating each distance,
    this uses the cdist function from scipy (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)

    :param array x: N1*M array
    :param array y: N2*M array
    :param string or func dist: distance parameter for cdist. When string is given, cdist uses optimized functions for the distance metrics.
    If a string is passed, the distance function can be 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'.
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    if ndim(x)==1:
        x = x.reshape(-1,1)
    if ndim(y)==1:
        y = y.reshape(-1,1)
    r, c = len(x), len(y)
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:]
    D0[1:,1:] = cdist(x,y,dist)
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            D1[i, j] += min(D0[i, j], D0[i, j+1], D0[i+1, j])
    if len(x)==1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1] / sum(D1.shape), C, D1, path

def _traceback(D):
    i, j = array(D.shape) - 2
    p, q = [i], [j]
    while ((i > 0) or (j > 0)):
        tb = argmin((D[i, j], D[i, j+1], D[i+1, j]))
        if (tb == 0):
            i -= 1
            j -= 1
        elif (tb == 1):
            i -= 1
        else: # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return array(p), array(q)


def nomalize_signal(input_signal):
    mean = np.mean(input_signal, axis=0)
    input_signal -= mean
    return input_signal / np.std(input_signal, axis=0)

if __name__ == '__main__':
    # if 0: # 1-D numeric
    from sklearn.metrics.pairwise import manhattan_distances


    kinect_angle_data = pd.read_csv("/home/runge/openbci/git/OpenBCI_Python/build/dataset/train/result/reconstructed_bycept_kinect__angles_.csv").dropna()
    nomalized_signal = nomalize_signal(kinect_angle_data)
    # mapping = interp1d([-1,1],[0,180])
    result=[]
    possion=[]
    x = np.array(nomalized_signal.ix[:, 1][4400:5000]).tolist()
    size = 5000-4400
    counter=3000
    for i in range(0,1000):
        y = np.array(nomalized_signal.ix[:, 1][counter:counter+size]).tolist()
        possion.append(counter)
        counter+=5
        dist_fun = manhattan_distances
        dist, cost, acc, path = dtw(x, y, dist_fun)
        print dist
        result.append(dist)

    print result
    print possion
    # else: # 2-D numeric
    #     from sklearn.metrics.pairwise import euclidean_distances
    #     x = [[0, 0], [0, 1], [1, 1], [1, 2], [2, 2], [4, 3], [2, 3], [1, 1], [2, 2], [0, 1]]
    #     y = [[1, 0], [1, 1], [1, 1], [2, 1], [4, 3], [4, 3], [2, 3], [3, 1], [1, 2], [1, 0]]
    #     dist_fun = euclidean_distances



    # vizualize
    # from matplotlib import pyplot as plt
    # plt.imshow(cost.T, origin='lower', cmap=plt.cm.Reds, interpolation='nearest')
    # plt.plot(path[0], path[1], '-o') # relation
    # plt.xticks(range(len(x)), x)
    # plt.yticks(range(len(y)), y)
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.axis('tight')
    # plt.title('Minimum distance: {}'.format(dist))
    # plt.show()
