# Functions to generate different network topologies

import numpy as np
import torch


def client_locations_mmWave_clusters_intermittent(num_clients: int = 10):
    """mmWave network topology definition
    :param num_clients: Number of edge learning clients
    :return: probability of successful transmission to PS, and inter-client connectivity matrix
    """

    # PS is placed at origin
    PS_loc = np.array([0, 0])

    # Distance of good clients from PS
    circle_good_rad = 159  # meters. For prob success ~ 0.9

    # Clients angles
    client_vec_deg = np.zeros(num_clients)
    client_vec_deg[3] = 2 * np.pi / 3
    client_vec_deg[6] = 4 * np.pi / 3

    x = np.zeros(num_clients)
    y = np.zeros(num_clients)

    # Determining the Cartesian coordinates of the clients with good connectivity
    x[0] = circle_good_rad * np.cos(client_vec_deg[0])
    y[0] = circle_good_rad * np.sin(client_vec_deg[0])
    x[3] = circle_good_rad * np.cos(client_vec_deg[3])
    y[3] = circle_good_rad * np.sin(client_vec_deg[3])
    x[6] = circle_good_rad * np.cos(client_vec_deg[6])
    y[6] = circle_good_rad * np.sin(client_vec_deg[6])

    d = 159  # Distance of bad clients in each cluster to the good client of the cluster

    # Cluster 1
    ang1 = 1.582
    x[1] = x[0] + d * np.cos(ang1)
    y[1] = y[0] + d * np.sin(ang1)
    print("Distance = {}".format(np.sqrt(x[1] ** 2 + y[1] ** 2)))

    ang2 = -1.582
    x[2] = x[0] + d * np.cos(ang2)
    y[2] = y[0] + d * np.sin(ang2)

    # Cluster 2
    ang4 = 2 * np.pi / 3 + 1.582
    x[4] = x[3] + d * np.cos(ang4)
    y[4] = y[3] + d * np.sin(ang4)

    ang5 = 2 * np.pi / 3 - 1.582
    x[5] = x[3] + d * np.cos(ang5)
    y[5] = y[3] + d * np.sin(ang5)

    # Cluster 3
    ang7 = 4 * np.pi / 3 - 1.582
    x[7] = x[6] + d * np.cos(ang7)
    y[7] = y[6] + d * np.sin(ang7)

    ang8 = 4 * np.pi / 3 - 1.54
    x[8] = x[6] + d * np.cos(ang8)
    y[8] = y[6] + d * np.sin(ang8)

    ang9 = 4 * np.pi / 3 + 1.582
    x[9] = x[6] + d * np.cos(ang9)
    y[9] = y[6] + d * np.sin(ang9)

    # Client locations
    loc_clients = []
    for idx in range(num_clients):
        loc_clients.append([x[idx], y[idx]])
    loc_clients = np.array(loc_clients)

    # Compute distances of clients from PS
    sub_clients_PS = loc_clients - PS_loc
    dist_clients_PS2 = np.zeros(len(sub_clients_PS))
    for idx in range(len(sub_clients_PS)):
        dist_clients_PS2[idx] = (
            sub_clients_PS[idx][0] ** 2 + sub_clients_PS[idx][1] ** 2
        )
    dist_clients_PS = np.sqrt(dist_clients_PS2)

    # Compute pairwise distances between clients
    dist_clients_clients = np.zeros([num_clients, num_clients])
    for i in range(len(loc_clients)):
        for j in range(len(loc_clients)):
            dist_vector = loc_clients[i] - loc_clients[j]
            dist_clients_clients[i][j] = np.linalg.norm(dist_vector, 2)

    # Compute probability of successful transmission to PS
    prob_success_PS = np.zeros(num_clients)
    for i in range(len(dist_clients_PS)):
        p = min(1, np.exp(-dist_clients_PS[i] / 30 + 5.2))
        prob_success_PS[i] = np.round(p * 100) / 100

    # Determine connectivity amongst clients
    P = np.zeros([num_clients, num_clients])
    connectivity_mat_clients = np.zeros([num_clients, num_clients])
    for i in range(num_clients):
        for j in range(num_clients):
            p = min(1, np.exp(-dist_clients_clients[i][j] / 30 + 5.2))
            if p > 0.5:
                P[i][j] = p
                connectivity_mat_clients[i][j] = 1

    with open("prob_success_PS.npy", "wb") as f:
        np.save(f, prob_success_PS)
    with open("connectivity_mat_clients.npy", "wb") as f:
        np.save(f, connectivity_mat_clients)

    return torch.Tensor(prob_success_PS), torch.Tensor(P), torch.Tensor(connectivity_mat_clients)


