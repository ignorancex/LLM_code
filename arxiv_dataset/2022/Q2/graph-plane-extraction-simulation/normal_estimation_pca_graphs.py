"""
Our proposed approach
"""
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize
from scipy.sparse.linalg import expm
from networkx.convert_matrix import from_scipy_sparse_matrix
from networkx.linalg import normalized_laplacian_matrix
import numpy as np
from sklearn.neighbors import NearestNeighbors
import math

# %% cvx code
import os
# os.chdir("U:\\Research Projects\\Norm Calculation for Arpan")
import cvxpy as cp

np.random.seed(1)


# read data
def calculate_normals_pca_graphs(X, num_neighbors, alpha):
    m = X.shape[0]
    p = X.shape[1]

    # row vector projector
    Tr = []
    for i in range(m):
        Tr_temp = np.zeros((p, p * m))
        Tr_temp[0: p, (i * p): (i * p + p)] = np.identity(p)
        Tr.append(Tr_temp)

    # column vector projector
    Tc = []
    for j in range(p):
        Tc_temp = np.zeros((m, p * m))
        Tc_temp[0: m, np.arange(j, p * m + j, p)] = np.identity(m)
        Tc.append(Tc_temp)

    # matrix of Tr[i].T @ Tr[i]
    TrTr = []
    for i in range(m):
        TrTr_temp = np.zeros((p * m, p * m))
        TrTr_temp[(i * p): (i * p + p), (i * p): (i * p + p)] = np.identity(p)
        TrTr.append(TrTr_temp)

    # L1 and L2 matrices
    num_neighbors_samples = num_neighbors
    num_neighbors_features = 3
    # creates sparse matrix of CSR format
    adjacency_matrix_samples = kneighbors_graph(X,
                                                num_neighbors_samples,
                                                mode='distance',
                                                include_self=True)
    np.square(adjacency_matrix_samples.data, out=adjacency_matrix_samples.data)
    adjacency_matrix_samples = -adjacency_matrix_samples
    np.exp(adjacency_matrix_samples.data, out=adjacency_matrix_samples.data)
    # convert CSR matrix to networkx graph
    graph_samples = from_scipy_sparse_matrix(adjacency_matrix_samples)
    # get normalized laplacian of graph
    laplacian_matrix_samples = normalized_laplacian_matrix(graph_samples).toarray()
    # creates sparse matrix of CSR format
    adjacency_matrix_features = kneighbors_graph(X.transpose(),
                                                 num_neighbors_features,
                                                 mode='distance',
                                                 include_self=True)
    np.square(adjacency_matrix_features.data, out=adjacency_matrix_features.data)
    adjacency_matrix_features = -adjacency_matrix_features
    np.exp(adjacency_matrix_features.data, out=adjacency_matrix_features.data)
    # convert CSR matrix to networkx graph
    graph_features = from_scipy_sparse_matrix(adjacency_matrix_features)
    # get normalized laplacian of graph
    laplacian_matrix_features = normalized_laplacian_matrix(graph_features).toarray()

    L1 = laplacian_matrix_features
    L2 = laplacian_matrix_samples

    # %%
    # calculate the neighbor-based matrices

    neigh = NearestNeighbors(n_neighbors=num_neighbors_samples)
    neigh.fit(X)
    neigh_id = neigh.kneighbors(X, num_neighbors_samples, return_distance=False)
    neigh_mat = np.zeros((m, num_neighbors_samples, p))
    for j in range(m):
        for k in range(num_neighbors_samples):
            neigh_mat[j, k, :] = X[neigh_id[j, k], :] - X[j, :]

    # calculate intermediate matrices for loss and gradient
    loss_mat_1 = 0
    loss_mat_2 = 0
    loss_mat_3 = 0
    for j in range(m):
        loss_mat_1 += Tr[j].T @ neigh_mat[j, :, :].T @ neigh_mat[j, :, :] @ Tr[j]
        loss_mat_2 += Tr[j].T @ L1 @ Tr[j]
    for j in range(p):
        loss_mat_3 += Tc[j].T @ L2 @ Tc[j]

    # %% start projected gradient descent
    n_true = np.zeros((m, p))
    n_true[0:100, 2] = 1
    n_true[100:200, 1] = 1
    n_true[200:300, 0] = 1
    # fig = plt.Figure()
    # ax = plt.subplot(projection='3d')
    # ax.quiver(X[:, 0], X[:, 1], X[:, 2], n_true[:, 0], n_true[:, 1], n_true[:, 2], length=0.1)
    # plt.show()

    # %%
    loss_pre = 100
    alpha = alpha
    N = np.random.rand(m, p)
    N_norm = np.linalg.norm(N, axis=1)
    N = N / N_norm[:, np.newaxis]
    n = N.reshape((m*p, 1))
    n_noweight = n
    n_weight_dotprod = n
    n_weight_dist = n
    n_weight_dotprod_dist = n
    niter = 10000
    gamma = 1
    X_change = X
    lr = 0.01

    # # compute spread of each point with respect to other neighboring points
    # planar_spread_pts = np.zeros((m,))
    #
    # for j in range(m):
    #     points = []
    #     for k in range(num_neighbors_samples):
    #         points.append(X[neigh_id[j, k], :])
    #     points = np.array(points)
    #     cov = np.cov(points.T)
    #     eigenvalues, eigenvectors = np.linalg.eig(cov)
    #     planar_spread_pts[j] = eigenvalues[2]/np.sum(eigenvalues)

    loss_noweight_singlerun = []
    loss_weight_dotprod_singlerun = []
    loss_weight_dist_singlerun = []
    loss_weight_dotprod_dist_singlerun = []
    deln_noweight_singlerun = []
    deln_weight_dotprod_singlerun = []
    deln_weight_dist_singlerun = []
    deln_weight_dotprod_dist_singlerun = []


    for i in range(niter):
        for j in range(m):
            for k in range(num_neighbors_samples):
                neigh_mat[j, k, :] = X_change[neigh_id[j, k], :] - X_change[j, :]
        loss_mat_1_noweight = 0
        loss_mat_1_dotprod = 0
        loss_mat_1_dist = 0
        loss_mat_1_dotprod_dist = 0

        for j in range(m):
            W_dotprod = np.zeros((num_neighbors_samples, num_neighbors_samples))
            W_dist = np.zeros((num_neighbors_samples, num_neighbors_samples))
            W_dotprod_dist = np.zeros((num_neighbors_samples, num_neighbors_samples))
            for k in range(num_neighbors_samples):
                neigh = neigh_id[j, k]
                dist = np.linalg.norm(neigh_mat[j, k, :])
                dot_prod = np.dot(n[(j * p): ((j + 1) * p), 0], n[(neigh * p): ((neigh + 1) * p), 0])
                if dist < 1e-5:
                    dist = 0.01
                W_dotprod[k, k] = dot_prod
                W_dist[k, k] = 1/dist
                W_dotprod_dist[k, k] = dot_prod/dist  # currently performing best
                # W[k, k] = 1/planar_spread_pts[neigh]
            loss_mat_1_noweight += Tr[j].T @ neigh_mat[j, :, :].T @ neigh_mat[j, :, :] @ Tr[j]
            loss_mat_1_dotprod += Tr[j].T @ neigh_mat[j, :, :].T @ W_dotprod @ neigh_mat[j, :, :] @ Tr[j]
            loss_mat_1_dist += Tr[j].T @ neigh_mat[j, :, :].T @ W_dist @ neigh_mat[j, :, :] @ Tr[j]
            loss_mat_1_dotprod_dist += Tr[j].T @ neigh_mat[j, :, :].T @ W_dotprod_dist @ neigh_mat[j, :, :] @ Tr[j]

        n_noweight = n_noweight - alpha * 2 * (loss_mat_1_noweight.T + gamma * loss_mat_3) @ n_noweight
        n_weight_dotprod = (n_weight_dotprod -
                           alpha * 2 * (loss_mat_1_dotprod.T + gamma * loss_mat_3) @ n_weight_dotprod)
        n_weight_dist = (n_weight_dist -
                            alpha * 2 * (loss_mat_1_dist.T + gamma * loss_mat_3) @ n_weight_dist)
        n_weight_dotprod_dist = (n_weight_dotprod_dist -
                                 alpha * 2 * (loss_mat_1_dotprod_dist.T + gamma * loss_mat_3) @ n_weight_dotprod_dist)
        for j in range(m):
            n_noweight[(j * p): ((j + 1) * p)] = normalize(n_noweight[(j * p): ((j + 1) * p)], axis=0)
            n_weight_dotprod[(j * p): ((j + 1) * p)] = normalize(n_weight_dotprod[(j * p): ((j + 1) * p)], axis=0)
            n_weight_dist[(j * p): ((j + 1) * p)] = normalize(n_weight_dist[(j * p): ((j + 1) * p)], axis=0)
            n_weight_dotprod_dist[(j * p): ((j + 1) * p)] = normalize(n_weight_dotprod_dist[(j * p): ((j + 1) * p)], axis=0)

        # loss = n.T @ loss_mat_2 @ n + n.T @ loss_mat_3 @ n
        loss_noweight = n_noweight.T @ loss_mat_1_noweight @ n_noweight + gamma * n_noweight.T @ loss_mat_3 @ n_noweight
        loss_weight_dotprod = n_weight_dotprod.T @ loss_mat_1_dotprod @ n_weight_dotprod + gamma * n_weight_dotprod.T @ loss_mat_3 @ n_weight_dotprod
        loss_weight_dist = n_weight_dist.T @ loss_mat_1_dist @ n_weight_dist + gamma * n_weight_dist.T @ loss_mat_3 @ n_weight_dist
        loss_weight_dotprod_dist = n_weight_dotprod_dist.T @ loss_mat_1_dotprod_dist @ n_weight_dotprod_dist + gamma * n_weight_dotprod_dist.T @ loss_mat_3 @ n_weight_dotprod_dist

        # for j in range(m):
        #     X_change[j, :] = X_change[j, :] + lr * ((loss - loss_pre)/loss_pre) * n[j, :]

        if np.abs(loss_weight_dotprod_dist - loss_pre) < 0.001:
            break
        else:
            loss_pre = loss_weight_dotprod_dist
            if i % 100 == 0:
                print([i, loss_weight_dotprod_dist])
            loss_noweight_singlerun.append(loss_noweight[0][0])
            loss_weight_dist_singlerun.append(loss_weight_dist[0][0])
            loss_weight_dotprod_singlerun.append(loss_weight_dotprod[0][0])
            loss_weight_dotprod_dist_singlerun.append(loss_weight_dotprod_dist[0][0])

            n_mat = n_noweight.reshape(m, p)
            del_n = np.linalg.norm(n_mat - n_true, axis=1)
            deln_noweight_singlerun.append(np.mean(del_n))
            n_mat = n_weight_dotprod.reshape(m, p)
            del_n = np.linalg.norm(n_mat - n_true, axis=1)
            deln_weight_dotprod_singlerun.append(np.mean(del_n))
            n_mat = n_weight_dist.reshape(m, p)
            del_n = np.linalg.norm(n_mat - n_true, axis=1)
            deln_weight_dist_singlerun.append(np.mean(del_n))
            n_mat = n_weight_dotprod_dist.reshape(m, p)
            del_n = np.linalg.norm(n_mat - n_true, axis=1)
            deln_weight_dotprod_dist_singlerun.append(np.mean(del_n))
            # print("Mean diff = ", np.mean(del_n))
    loss =  [loss_noweight_singlerun,
             loss_weight_dotprod_singlerun,
             loss_weight_dist_singlerun,
             loss_weight_dotprod_dist_singlerun]
    deln = [deln_noweight_singlerun,
            deln_weight_dotprod_singlerun,
            deln_weight_dist_singlerun,
            deln_weight_dotprod_dist_singlerun]

    # n_mat = n_weight_dotprod_dist.reshape(m, p)
    # fig1 = plt.figure(1)
    # ax1 = fig1.add_subplot(111, projection='3d')
    # ax1.quiver(X[:, 0], X[:, 1], X[:, 2], n_mat[:, 0], n_mat[:, 1], n_mat[:, 2], length=0.1)
    # fig2 = plt.figure(2)
    # ax2 = fig2.add_subplot(111)
    # line1, = ax2.plot(loss_noweight_singlerun, label='no weight loss')
    # line2, = ax2.plot(loss_weight_dotprod_singlerun, label='weight dot product loss')
    # line3, = ax2.plot(loss_weight_dist_singlerun, label='weight dist loss')
    # line4, = ax2.plot(loss_weight_dotprod_dist_singlerun, label='weight dot prod dist loss')
    # ax2.set_xlabel('Number of iterations')
    # ax2.set_ylabel('Loss value [units]')
    # ax2.legend(handles=[line1, line2, line3, line4])
    # fig3 = plt.figure(3)
    # ax3 = fig3.add_subplot(111)
    # line1, = ax3.plot(deln_noweight_singlerun, label='no weight diff')
    # line2, = ax3.plot(deln_weight_dotprod_singlerun, label='weight dot product diff')
    # line3, = ax3.plot(deln_weight_dotprod_singlerun, label='weight dist diff')
    # line4, = ax3.plot(deln_weight_dotprod_dist_singlerun, label='weight dot prod dist diff')
    # ax3.set_xlabel('Number of iterations')
    # ax3.set_ylabel('Mean difference between true and estimated normals [m]')
    # ax3.legend(handles=[line1, line2, line3, line4])
    # plt.show()
    # points_with_normals = np.hstack((X, n_mat))
    # np.savetxt('extracted_normals.txt', points_with_normals, fmt='%10.4f')
    return loss, deln


if __name__ == "__main__":
    pts_with_labels = np.loadtxt('points.txt')
    pts = pts_with_labels[:, 0:3]
    calculate_normals_pca_graphs(X=pts, num_neighbors=30, alpha=0.01)
