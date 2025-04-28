# Functions to compute mean with different

import os
import numpy as np
import torch
import random

from loguru import logger
import matplotlib.pyplot as plt

from optimization import optimize_weights_and_privacy_noise
from utils import find_max_l2_norm


def dme_intermittent_naive(
    transmit_probs: np.ndarray = None, client_data: np.ndarray = None
):
    """
    Distributed mean estimation without collaboration or any privacy consideration amongst clients
    :param transmit_probs: Array of transmission probabilities from each of the clients.
    :param client_data: matrix containing the vector for each of the clients
    -- each row contains data for a specific client
    Returns the mean estimated at the parameter server
    """

    # Renaming variables
    num_clients = len(transmit_probs)  # Number of clients
    dim = client_data.shape[1]  # Dimension of each vector

    assert (
        num_clients == client_data.shape[0]
    ), "prob_ngbrs and client_data should have dimensions consistent with the number of clients!"

    transmits = [transmit_probs[i] >= random.uniform(0, 1) for i in range(num_clients)]
    print(f"Transmission opportunities: {transmits}")

    mean_est = np.zeros(dim)
    for i in range(num_clients):
        if transmits[i]:
            mean_est += client_data[i]
    mean_est /= num_clients

    return mean_est


def dme_pricer(
    transmit_probs: np.ndarray = None,
    prob_ngbrs: np.ndarray = None,
    client_data: np.ndarray = None,
    weights: np.ndarray = None,
    priv_noise_var: np.ndarray = None,
):
    """
    Distributed mean estimation with collaboration with privacy constraints amongst clients
    :param transmit_probs: Array of transmission probabilities from each of the clients.
    :param prob_ngbrs: Matrix of probabilities for intermittent connectivity amongst clients
    :param client_data: matrix containing the vector for each of the clients
    -- each row contains data for a specific client
    :param eps_mat: Matrix of epsilons in pairwise differential privacy guarantee
    :param delta_mat: Matrix of deltas in pairwise differential privacy guarantee
    :param weights: Peer-to-peer collaboration weights
    :param priv_noise_var: Peer-to-peer privacy noise variance
    Returns the mean estimated at the parameter server
    """

    # Renaming variables
    p = transmit_probs  # Transmission probability to the PS
    P = prob_ngbrs  # (Intermittent) topology between clients
    X = client_data
    num_clients = len(transmit_probs)  # Number of clients
    dim = client_data.shape[1]  # Dimension of each vector
    A = weights
    sigma = priv_noise_var

    assert (
        num_clients
        == prob_ngbrs.shape[0]
        == prob_ngbrs.shape[1]
        == client_data.shape[0]
        == A.shape[0]
        == A.shape[1]
        == sigma.shape[0]
        == sigma.shape[1]
    ), "Dimensional inconsistency in parameters!"

    # Generate a random realization of connectivity amongst clients according to blockage model
    transmit_clients_colab = np.zeros([num_clients, num_clients])
    for i in range(num_clients):
        transmit_clients_colab[i][i] = 1
        for j in range(i + 1, num_clients):
            transmit_clients_colab[i][j] = P[i][j] >= random.uniform(0, 1)
            transmit_clients_colab[j][i] = transmit_clients_colab[i][j]

    # Assert that transmit_clients_colab is symmetric
    assert np.array_equal(
        transmit_clients_colab, np.transpose(transmit_clients_colab)
    ), "transmit_clients_colab should be symmetric!"

    # Compute vector to transmit at each client
    Y = np.zeros_like(X)
    for i in range(num_clients):
        for j in range(num_clients):
            if transmit_clients_colab[j][i]:
                Y[i] += A[j][i] * X[j] + sigma[j][i] * np.random.randn(dim)             # Gaussian mechanism

    # Compute the mean of locally averaged variables
    # Get a random realization of the connectivity of clients to the PS
    transmits = [p[i] >= random.uniform(0, 1) for i in range(num_clients)]

    mean_est = np.zeros(dim)
    for i in range(num_clients):
        if transmits[i]:
            mean_est += Y[i]
    mean_est /= num_clients

    return mean_est


# def intermittent_sole_good_client_naive(
#         data: np.ndarray = None, 
#         n_machines: int = 1,
#         tx_probs_PS: np.ndarray = None,
#         ):
#     """Clients have intermittent connectivity to the PS.
#     :param data: Local data at each node
#     :param n_machines: Number of nodes
#     :param tx_probs_PS: Connection probablities of nodes to the PS
#     Return: Estimated mean over a random realization on connectivity without any collaboration.
#     """

#     # p_bad = 0.1
#     # p_good = 0.9

#     # transmit_probs = p_bad * np.ones(n_machines)
#     # transmit_probs[int(n_machines / 2)] = p_good

#     logger.info(f"Transmit probabilities: {tx_probs_PS}")
#     return dme_intermittent_naive(transmit_probs=transmit_probs, client_data=data)


# def intermittent_sole_good_client_pricer_full_colab(updated_local_centroids):
#     """Clients have intermittent connectivity to the PS and to each other.
#     Connectivity is good and there are no privacy concerns.
#     Return: Estimated mean with PriCER
#     """

#     n_machines = len(updated_local_centroids)
#     k = updated_local_centroids[0].shape[0]
#     dim = updated_local_centroids[0].shape[1]

#     # Connectivity parameters: Fully connected topology
#     pc_good = 0.9  # Connectivity to other clients
#     ps_bad = 0.1  # Bad connectivity clients to PS
#     ps_good = 0.9  # Good connectivity clients to PS
#     p = ps_bad * np.ones(n_machines)
#     p[int(n_machines / 2)] = ps_good
#     P = pc_good * torch.ones(n_machines, n_machines)
#     P.fill_diagonal_(1)

#     # Compute the radius
#     radius = find_max_l2_norm(updated_local_centroids)
#     logger.info(f"Radius: {radius}")

#     # Privacy parameters: No privacy
#     delta = 1e-3
#     D = delta * torch.ones([n_machines, n_machines])
#     eps = 1e3
#     E = eps * torch.ones([n_machines, n_machines])

#     # Bias control paramters
#     reg_type = "L1"
#     reg_strength = 0.0

#     # Optimize collaboration weights and privacy noise variance
#     (
#         weights_opt,
#         priv_noise_var_opt,
#         optimization_loss,
#     ) = optimize_weights_and_privacy_noise(
#         p=p,
#         P=P,
#         E=E,
#         D=D,
#         radius=1.0,
#         dimension=dim,
#         reg_type=reg_type,
#         reg_strength=reg_strength,
#     )

#     # Save the optimization loss as a plot
#     with torch.no_grad():
#         xaxis = list(range(len(optimization_loss)))
#         plt.plot(xaxis, optimization_loss)
#         plt.title("Optimization loss")
#         plt.xlabel("Iterations")
#         plt.ylabel("Objective function value")
#         plt.savefig("optimization_loss.png")

#     logger.info(f"Initial optimization loss: {optimization_loss[0]}")
#     logger.info(f"Optimization loss[100]: {optimization_loss[100]}")
#     logger.info(f"Optimization loss[200]: {optimization_loss[200]}")
#     logger.info(f"Optimization loss[300]: {optimization_loss[300]}")
#     logger.info(f"Optimization loss[400]: {optimization_loss[400]}")
#     logger.info(f"Optimization loss[500]: {optimization_loss[500]}")
#     logger.info(f"Optimization loss[600]: {optimization_loss[600]}")
#     logger.info(f"Optimization loss[700]: {optimization_loss[700]}")
#     logger.info(f"Optimization loss[800]: {optimization_loss[800]}")
#     logger.info(f"Optimization loss[900]: {optimization_loss[900]}")
#     logger.info(f"Optimization loss[1000]: {optimization_loss[1000]}")
#     logger.info(f"Optimization loss[1100]: {optimization_loss[1100]}")
#     logger.info(f"Optimization loss[1200]: {optimization_loss[1200]}")
#     logger.info(f"Optimization loss[1300]: {optimization_loss[1300]}")  
#     logger.info(f"Final optimization loss: {optimization_loss[-1]}")

#     logger.info(f"Optimized collaboration weights:\n{weights_opt}")
#     logger.info(f"Optimized privacy noise variance:\n{priv_noise_var_opt}")

#     # Refactor the matrices
#     global_centroid = np.zeros([k, dim])

#     for c_idx in range(k):
#         T = np.zeros([n_machines, dim])

#         for mc_idx in range(n_machines):
#             T[mc_idx, :] = updated_local_centroids[mc_idx][c_idx, :]

#         global_centroid[c_idx, :] = dme_pricer(
#             transmit_probs=p,
#             prob_ngbrs=P,
#             client_data=T,
#             eps_mat=E,
#             delta_mat=D,
#             weights=weights_opt,
#             priv_noise_var=priv_noise_var_opt,
#         )

#     # Do the mean estimation and return the estimated mean
#     return global_centroid
