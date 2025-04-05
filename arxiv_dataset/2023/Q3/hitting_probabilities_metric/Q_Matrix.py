import numpy as np
from numpy import matlib
def get_Q_matrix(M):
    # taken from https://github.com/zboyd2/hitting_probabilities_metric/blob/master/HittingTimes_L3.m


    N = M.shape[0]
 


    e1 = np.zeros((1, N))
    e1[0][0] = 1
    A1inv = np.eye(N) - M

    A1inv[0, :] = e1
    A1inv = np.linalg.inv(A1inv)

    Q = np.zeros((N, N))
    Q[:, 0] = A1inv[:, 0] / np.diagonal(A1inv)

    M = np.matmul(M, A1inv)

    detCj = ((1 + np.diagonal(M)) * (1 - M[0, 0])) + (M[:, 0] * M[0, :].transpose())
    CjInv = np.zeros((2, 2, N))
    CjInv[0, 0, :] = (1 - M[0, 0]) / detCj
    CjInv[0, 1, :] = M[:, 0] / detCj # WAS WRONG INDEX HERE
    CjInv[1, 0, :] = -M[0, :].transpose() / detCj
    CjInv[1, 1, :] = (1 + np.diagonal(M)) / detCj
 
    M1 = np.zeros((N, 2, N))
    M1[:, 0, :] = A1inv
    M1[:, 1, :] = matlib.repmat(-A1inv[:, 0], 1, N).reshape(N, N)
 
    M2 = np.zeros((2, N, N))
    M2[0, :, :] = M.transpose()
    M2[1, :, :] = matlib.repmat(M[0, :], N, 1).transpose()
 


    for j in range(1, N):
        # assert np.all(
        #    np.concatenate([A1inv[:, j].reshape((N, 1)), - A1inv[:, 0].reshape((N, 1))], axis=1) == M1[:, :, j])














        Ac = A1inv[:, j] - np.matmul(np.matmul(M1[:, :, j], CjInv[:, :, j]), M2[:, j, j])
        Ad = np.diagonal(A1inv) - sum(
            M1[:, :, j].transpose() * np.matmul(CjInv[:, :, j], M2[:, :, j]))
        Q[:, j] = Ac / Ad
 
    return Q  # - np.diagonal(np.diagonal(Q))
