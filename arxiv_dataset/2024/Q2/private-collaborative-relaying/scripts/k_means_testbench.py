import numpy as np
import torch
import torchvision.datasets as datasets

import matplotlib.pyplot as plt
from datetime import datetime
import json
import os

from topology import client_locations_mmWave_clusters_intermittent
from k_means_clustering import *


def simulate_kmeans_perfect_conn(
    dataset: str = "mnist",
    n_machines: int = 1,
    k: int = 1,
    max_inner_iters: int = 100,
    max_outer_iters = 10,
    num_realizations: int = 1,
    partition: str = "uniform",
    metric: str = "euclidean"
):
    """Simulate kmeans for different realizations and return average relative inertia
    :param n_machines: Number of machines
    :param k: Number of clusters
    :param max_inner_iters: Number of inner iterations for (local) k-means clustering
    :param max_outer_iters: Number of outer iterations (communication rounds with PS) for k-means clustering 
    :param num_realizations: Number of realizations to average inertia over
    :param metric: Distance metric used for clustering
    Return average relative inertia
    """

    rel_inertia_list = []

    if dataset == "mnist":
        print(f"Loading MNIST dataset for K-means clustering!")
        X = load_mnist_data()

    elif dataset == "cifar10":
        print(f"Loading CIFAR-10 dataset for K-means clustering!")
        X = load_cifar10_data()

    elif dataset == "fashion-mnist-mobilenetv3-embs":
        print(f"Loading Fashion-MNIST embeddings extracted using MobileNet-v3!")
        train_embs, train_labels, test_embs, test_labels = load_fashion_mnist_embs()
        print(f"Shape of train embeddings: {train_embs.shape}")
        print(f"Shape of test embeddings: {test_embs.shape}")
        X = train_embs
    
    else:
        print(f"Dataset not configured!")

    print(f"k-means clustering with perfect connectivity.")
    for r in range(num_realizations):

        # Set manual seed
        manual_seed = (r + 1) * 1000
        np.random.seed(manual_seed)

        print(f"Doing distributed k-means clustering for realization: {r}")
        inertia_distr, global_centroids_dist = distributed_kmeans(
            X=X,
            n_machines=n_machines,
            k=k,
            max_inner_iters=max_inner_iters,
            max_outer_iters=max_outer_iters,
            partition=partition,
            consensus_type="perfect",
            manual_seed=manual_seed,
            metric=metric
        )

        print(f"Doing centralized k-means clustering for realization: {r}")
        inertia_central, global_centroids_central = distributed_kmeans(
            X=X, n_machines=1, k=k, max_inner_iters=max_inner_iters, 
            max_outer_iters=max_outer_iters, manual_seed=manual_seed, metric=metric
        )

        relative_inertia = inertia_distr / inertia_central

        rel_inertia_list.append(relative_inertia)
        print(f"Realization {r}, Relative inertia: {relative_inertia}")

    avg_rel_inertia = sum(rel_inertia_list) / len(rel_inertia_list)

    print(f"Average relative inertia over {num_realizations} realizations is {avg_rel_inertia}")

    if dataset == "fashion-mnist-mobilenetv3-embs":

        # Compute relative mismatch on the test set
        rel_mismatch_test = relative_clustering_mismatch(data=test_embs, 
                                                        centroids_dist=global_centroids_dist, 
                                                        centroids_cent=global_centroids_central,
                                                        metric=metric)
        print(f"Relative mismatch between dist. and cent. on test clustering is {rel_mismatch_test}")

        rel_mismatch_train = relative_clustering_mismatch(data=train_embs, 
                                                          centroids_dist=global_centroids_dist, 
                                                          centroids_cent=global_centroids_central,
                                                          metric=metric)
        print(f"Relative mismatch between dist. and cent. clustering on training dataset is {rel_mismatch_train}")

        # Compute classification accuracy on the test set
        # print(f"Computing classificaiton accuracy for test dataset!")
        # centroid_labels = assign_labels_to_centroids(train_data=X, train_labels=train_labels, centroids=global_centroids_dist)

        # test_acc = classify_test_data(test_data=test_embs, test_labels=test_labels, 
        #                               centroids=global_centroids_dist, centroid_labels=centroid_labels)
        
        # print(f"Test accuracy with k-means clustering is: {test_acc}")

    # Log results
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging_info = {
            "timestamp": current_time,
            "connectivity_type": "perfect connectivity to PS",
            "average_relative_inertia": avg_rel_inertia,
            "dist_kmeans_global_centroids_last_realization": global_centroids_dist.tolist(),
            "central_kmeans_global_centroids_last_realization": global_centroids_central.tolist(),
        }
    if dataset == "fashion-mnist-mobilenetv3-embs":
        logging_info.update({"relative_mismatch_test": rel_mismatch_test})
        logging_info.update({"relative_mismatch_train": rel_mismatch_train})

    logfile_path = f"misc/kmeans_clustering/{dataset}_partition_{partition}_perfect_conn.json"
    if os.path.exists(logfile_path):
        with open(logfile_path, "a") as json_file:
            json.dump(logging_info, json_file, indent=4)
            json_file.write("\n")
    else:
        with open(logfile_path, "w") as json_file:
            json.dump(logging_info, json_file, indent=4)
            json_file.write("\n")
            
            
def simulate_kmeans_intmt_sole_good_node_no_colab(
    dataset: str = "mnist",
    n_machines: int = 1,
    k: int = 1,
    max_inner_iters: int = 100,
    max_outer_iters = 10,
    num_realizations: int = 1,
    partition: str = "uniform",
    p_good: float = 0.9,
    p_bad: float = 0.1,
):
    """Simulate kmeans for different realizations and return average relative inertia
    :param n_machines: Number of machines
    :param k: Number of clusters
    :param max_inner_iters: Number of inner iterations for (local) k-means clustering
    :param max_outer_iters: Number of outer iterations (communication rounds with PS) for k-means clustering 
    :param num_realizations: Number of realizations to average inertia over
    :param p_good: Connectivity of good conn. client to the PS
    :param p_bad: Connectivity of bad conn. client to the PS
    Return average relative inertia
    """

    rel_inertia_list = []

    if dataset == "mnist":
        print(f"Loading MNIST dataset for K-means clustering!")
        X = load_mnist_data()

    elif dataset == "cifar10":
        print(f"Loading CIFAR-10 dataset for K-means clustering!")
        X = load_cifar10_data()
        
    elif dataset == "fmnist-mobilenetv3":
        print(f"Loading Fashion-MNIST embeddings extracted using MobileNet-v3!")
        train_embs, _, test_embs, _ = load_fashion_mnist_embs()
        print(f"Shape of train embeddings: {train_embs.shape}")
        print(f"Shape of test embeddings: {test_embs.shape}")
        X = train_embs
    
    else:
        print(f"Dataset not configured!")

    print(f"k-means clustering with intermittent connectivity: Sole good connectivity node to PS. No collaboration.")
    for r in range(num_realizations):
        print(f"Doing distributed k-means clustering for realization: {r}")
        np.random.seed((r + 1) * 1000)

        # Get connectivity probabilities to the PS
        tx_probs_PS = p_bad * np.ones(n_machines)
        tx_probs_PS[int(n_machines / 2)] = p_good

        inertia_distr, global_centroids_dist = distributed_kmeans(
            X=X,
            n_machines=n_machines,
            k=k,
            max_inner_iters=max_inner_iters,
            max_outer_iters=max_outer_iters,
            partition=partition,
            consensus_type="intermittent_sole_good_client_naive",
            tx_probs_PS = tx_probs_PS,
        )

        print(f"Doing centralized k-means clustering for realization: {r}")
        inertia_central, global_centroids_central = distributed_kmeans(
            X=X, n_machines=1, k=k, max_inner_iters=max_inner_iters, max_outer_iters=max_outer_iters
        )

        relative_inertia = inertia_distr / inertia_central

        rel_inertia_list.append(relative_inertia)
        print(f"Realization {r}, Relative inertia: {relative_inertia}")

    avg_rel_inertia = sum(rel_inertia_list) / len(rel_inertia_list)

    # Log results
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging_info = {
        "timestamp": current_time,
        "connectivity_type": "perfect connectivity to PS",
        "rel_inertia_list": rel_inertia_list,
        "average_relative_inertia": avg_rel_inertia,
        "dist_kmeans_global_centroids_last_realization": global_centroids_dist.tolist(),
        "central_kmeans_global_centroids_last_realization": global_centroids_central.tolist()
    }

    logfile_path = f"misc/kmeans_clustering/{dataset}_partition_{partition}_intmt_sole_good_node_no_colab.json"
    if os.path.exists(logfile_path):
        with open(logfile_path, "a") as json_file:
            json.dump(logging_info, json_file, indent=4)
            json_file.write("\n")
    else:
        with open(logfile_path, "w") as json_file:
            json.dump(logging_info, json_file, indent=4)
            json_file.write("\n")

    print(f"Average relative inertia over {num_realizations} realizations is {avg_rel_inertia}")
    
    
def simulate_kmeans_intmt_sole_good_node_pricer(
    dataset: str = "mnist",
    n_machines: int = 1,
    k: int = 1,
    max_inner_iters: int = 100,
    max_outer_iters = 10,
    num_realizations: int = 1,
    partition: str = "uniform",
    p_good: float = 0.9,
    p_bad: float = 0.1,
    p_node: float = 0.9,
    eps_trust: float = 1e3,
    eps_notrust: float = 1e-2,
    delta: float = 1e-3,
    num_ngbrs: int = 0,
    reg_type: str = "L1",
    reg_strength: float = 0.0,
    gd_num_iters: int = 1500,
    gd_learning_rate: float = 0.005,
):
    """Simulate kmeans for different realizations and return average relative inertia
    :param n_machines: Number of machines
    :param k: Number of clusters
    :param max_inner_iters: Number of inner iterations for (local) k-means clustering
    :param max_outer_iters: Number of outer iterations (communication rounds with PS) for k-means clustering 
    :param num_realizations: Number of realizations to average inertia over
    :param p_good: Connectivity of good conn. client to the PS
    :param p_bad: Connectivity of bad conn. client to the PS
    :param p_node: Connectivity probability between any two nodes
    :param eps_trust: Privacy parameter epsilon for trustworthy neighbors
    :param eps_notrust: Privacy parameter epsilon for non-trustworthy neighbors
    :param delta: Privacy parameter (common for all pairs of neighbors)
    :param num_ngbrs: Number of trustworthy neighbors with which a node can collaborate (by default, no collaboration) 
    :param reg_type: Type of bias regularization
    :param reg_strength: Regularization parameter strength
    :param gd_num_iters: Number of iterations of gradient descent for joint weight and nois evariance optimization
    :param gd_learning_rate: Learning rate of gradient descent
    Return average relative inertia
    """

    # List of relative inertia values for storing results of different realizations
    rel_inertia_list = []

    if dataset == "mnist":
        print(f"Loading MNIST dataset for K-means clustering!")
        X = load_mnist_data()

    elif dataset == "cifar10":
        print(f"Loading CIFAR-10 dataset for K-means clustering!")
        X = load_cifar10_data()
        
    elif dataset == "fmnist-mobilenetv3":
        print(f"Loading Fashion-MNIST embeddings extracted using MobileNet-v3!")
        train_embs, _, test_embs, _ = load_fashion_mnist_embs()
        print(f"Shape of train embeddings: {train_embs.shape}")
        print(f"Shape of test embeddings: {test_embs.shape}")
        X = train_embs
    
    else:
        print(f"Dataset not configured!")
    
    # Get dimension
    dim = X.shape[1]

    # Metrics to save
    losses_seed_variation = []
    optimized_mse_seed_variation = []
    bias_at_nodes_seed_variation = []
    optimized_tiv_seed_variation = []
    optimized_piv_seed_variation = []

    # Get connectivity probabilities to the PS (sole good client)
    tx_probs_PS = p_bad * torch.ones(n_machines)
    tx_probs_PS[int(n_machines / 2)] = p_good

    # Get connectivity probabilities between nodes
    tx_probs_colab = p_node * torch.ones(n_machines, n_machines)
    tx_probs_colab.fill_diagonal_(1)

    # Set the radius to 1 (assuming all data is normalized)
    radius = 1

    # Set privacy parameters depending on trustworthy neighbors
    E = eps_notrust * torch.ones([n_machines, n_machines])
    for i in range(n_machines):
        E[i][i] = eps_trust
        for j in range(num_ngbrs):
            E[i][(i + j + 1) % n_machines] = eps_trust
            E[i][(i + j - 1) % n_machines] = eps_notrust

    D = delta * torch.ones([n_machines, n_machines])

    print(f"Running PriCER with trustworthy neighbors: {2 * num_ngbrs} and dataset partition: {partition}")
    for r in range(num_realizations):
        print(f"Doing k-means clustering for realization: {r}")
        np.random.seed((r + 1) * 1000)
        torch.manual_seed((r + 1) * 1000)

        # Optimize collaboration weights and privacy noise variance (for every realization)
        print(f"Optimizing collaboration weights and privacy noise variance given network connectivity and privacy constraints.")
        (
            weights_opt,
            noise_opt,
            optimization_loss,
        ) = optimize_weights_and_privacy_noise(
            p=tx_probs_PS,
            P=tx_probs_colab,
            E=E,
            D=D,
            radius=radius,
            dimension=dim,
            reg_type=reg_type,
            reg_strength=reg_strength,
            num_iters=gd_num_iters,
            learning_rate=gd_learning_rate
        )

        # Append metrics to lists
        losses_seed_variation.append(optimization_loss)

        optimized_mse = evaluate_mse(
            p=tx_probs_PS, A=weights_opt, P=tx_probs_colab, radius=radius, sigma=noise_opt, dimension=dim
        )
        optimized_mse_seed_variation.append(optimized_mse)

        optimized_tiv = evaluate_tiv(p=tx_probs_PS, A=weights_opt, P=tx_probs_colab, radius=radius)
        optimized_tiv_seed_variation.append(optimized_tiv)

        optimized_piv = evaluate_piv(p=tx_probs_PS, P=tx_probs_colab, sigma=noise_opt, dimension=dim)
        optimized_piv_seed_variation.append(optimized_piv)

        bias_at_nodes = evaluate_bias_at_clients(p=tx_probs_PS, A=weights_opt, P=tx_probs_colab)
        bias_at_nodes_seed_variation.append(bias_at_nodes)

        # Plot the optimization loss for the first realization to see if hyper-parameter choice if ok
        if r == 0:
            plt.figure(figsize=(14, 7))
            with torch.no_grad():
                plt.plot(optimization_loss)

            plt.xlabel("Iterations")
            plt.ylabel("Objective function value")
            plt.yscale("log")
            plt.title("Collaboration weight and privacy noise optimization")
            plt.savefig(
                f"misc/kmeans_{dataset}_{partition}_ngbrs_{num_ngbrs}_bias_reg={reg_strength:.0f}_lr={gd_learning_rate}_pc={p_node}.png"
            )
        
        print(f"k-means clustering with intermittent connectivity | Sole good connectivity node to PS | PriCER")
        inertia_distr, global_centroids_distributed = distributed_kmeans(
            X=X,
            n_machines=n_machines,
            k=k,
            max_inner_iters=max_inner_iters,
            max_outer_iters=max_outer_iters,
            partition=partition,
            consensus_type="intermittent_sole_good_client_pricer",
            tx_probs_PS = tx_probs_PS,
            tx_probs_colab = tx_probs_colab,
            weights = weights_opt,
            noise = noise_opt
        )

        print(f"Doing centralized k-means clustering for realization: {r}")
        inertia_central, global_centroids_central = distributed_kmeans(
            X=X, 
            n_machines=1, 
            k=k, 
            max_inner_iters=max_inner_iters, 
            max_outer_iters=max_outer_iters, 
            consensus_type="perfect"
        )

        relative_inertia = inertia_distr / inertia_central

        rel_inertia_list.append(relative_inertia)
        print(f"Realization {r}, Relative inertia: {relative_inertia}")

    # Average quantities over different realizations
    avg_rel_inertia = sum(rel_inertia_list) / len(rel_inertia_list)

    losses_avg = torch.zeros((len(losses_seed_variation[0]),))
    for i in range(len(losses_seed_variation)):
        losses_avg += torch.Tensor(losses_seed_variation[i])
    losses_avg /= len(losses_seed_variation)

    bias_at_nodes_avg = torch.zeros((len(bias_at_nodes_seed_variation[0]),))
    for i in range(len(bias_at_nodes_seed_variation)):
        bias_at_nodes_avg += torch.Tensor(bias_at_nodes_seed_variation[i])
    bias_at_nodes_avg /= len(bias_at_nodes_seed_variation)

    optimized_mse_avg = torch.sum(torch.Tensor(optimized_mse_seed_variation)) / len(optimized_mse_seed_variation)
    optimized_tiv_avg = torch.sum(torch.Tensor(optimized_tiv_seed_variation)) / len(optimized_tiv_seed_variation)
    optimized_piv_avg = torch.sum(torch.Tensor(optimized_piv_seed_variation)) / len(optimized_piv_seed_variation)

    # Log results
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging_info = {
        "timestamp": current_time,
        "connectivity_type": "Intermittent connectivity to PS and amongst nodes: PriCER",
        "radius": radius,
        "dimension": dim,
        "connectivity_to_PS": tx_probs_PS.numpy().tolist(),
        "connectivity_between_clients": p_node,
        "privacy_delta": delta,
        "num_trustworthy_ngrs": num_ngbrs,
        "eps_trustworthy": eps_trust,
        "eps_others": eps_notrust,
        "bias_reg_type": reg_type,
        "bias_reg_strength": reg_strength,
        "gradient_descent_num_iters": gd_num_iters,
        "gradient_descent_learning_rate": gd_learning_rate,
        "rel_inertia_list": rel_inertia_list,
        "average_relative_inertia": avg_rel_inertia,
        "optimized_mse_avg": optimized_mse_avg.tolist(),
        "optimized_tiv_avg": optimized_tiv_avg.tolist(),
        "optimized_piv_avg": optimized_piv_avg.tolist(),
        "dist_kmeans_global_centroids_last_realization": global_centroids_distributed.tolist(),
        "central_kmeans_global_centroids_last_realization": global_centroids_central.tolist(),
        "optimized_weights_last_realization": weights_opt.tolist(),
        "optimized_privacy_noise_last_realization": noise_opt.tolist(),
        "optimization_loss_avg": losses_avg.tolist(),
        "bias_at_nodes_avg": bias_at_nodes_avg.tolist(),
    }

    logfile_path = f"misc/kmeans_clustering/{dataset}_partition_{partition}_intmt_sole_good_node_pricer_num_ngbrs_{num_ngbrs}_pc_{p_node}.json"
    if os.path.exists(logfile_path):
        with open(logfile_path, "a") as json_file:
            json.dump(logging_info, json_file, indent=4)
            json_file.write("\n")
    else:
        with open(logfile_path, "w") as json_file:
            json.dump(logging_info, json_file, indent=4)
            json_file.write("\n")

    print(f"Average relative inertia over {num_realizations} realizations is {avg_rel_inertia}")
 
 
def simulate_kmeans_cluster_no_colab(
    dataset: str = "mnist",
    n_machines: int = 1,
    k: int = 1,
    max_inner_iters: int = 100,
    max_outer_iters = 10,
    num_realizations: int = 1,
    partition: str = "uniform",
):
    """Simulate kmeans over a cluster topology but without collaboration
    :param n_machines: Number of machines
    :param k: Number of clusters
    :param max_inner_iters: Number of inner iterations for (local) k-means clustering
    :param max_outer_iters: Number of outer iterations (communication rounds with PS) for k-means clustering 
    :param num_realizations: Number of realizations to average inertia over
    Return average relative inertia
    """
    
    rel_inertia_list = []

    if dataset == "mnist":
        print(f"Loading MNIST dataset for K-means clustering!")
        X = load_mnist_data()

    elif dataset == "cifar10":
        print(f"Loading CIFAR-10 dataset for K-means clustering!")
        X = load_cifar10_data()
        
    elif dataset == "fmnist-mobilenetv3":
        print(f"Loading Fashion-MNIST embeddings extracted using MobileNet-v3!")
        train_embs, _, test_embs, _ = load_fashion_mnist_embs()
        print(f"Shape of train embeddings: {train_embs.shape}")
        print(f"Shape of test embeddings: {test_embs.shape}")
        X = train_embs
    
    else:
        print(f"Dataset not configured!")

    print(f"k-means clustering with intermittent connectivity: Nodes are clustered but no collaboration.")
    for r in range(num_realizations):
        print(f"Doing distributed k-means clustering for realization: {r}")
        np.random.seed((r + 1) * 1000)

        # Get connectivity probabilities to the PS
        tx_probs_PS, _, _ = client_locations_mmWave_clusters_intermittent()

        inertia_distr, global_centroids_dist = distributed_kmeans(
            X=X,
            n_machines=n_machines,
            k=k,
            max_inner_iters=max_inner_iters,
            max_outer_iters=max_outer_iters,
            partition=partition,
            consensus_type="cluster_no_colab",
            tx_probs_PS = tx_probs_PS,
        )

        print(f"Doing centralized k-means clustering for realization: {r}")
        inertia_central, global_centroids_central = distributed_kmeans(
            X=X, n_machines=1, k=k, max_inner_iters=max_inner_iters, max_outer_iters=max_outer_iters
        )

        relative_inertia = inertia_distr / inertia_central

        rel_inertia_list.append(relative_inertia)
        print(f"Realization {r}, Relative inertia: {relative_inertia}")

    avg_rel_inertia = sum(rel_inertia_list) / len(rel_inertia_list)

    # Log results
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging_info = {
        "timestamp": current_time,
        "connectivity_type": "intermittent connectivity to PS",
        "rel_inertia_list": rel_inertia_list,
        "average_relative_inertia": avg_rel_inertia,
        "dist_kmeans_global_centroids_last_realization": global_centroids_dist.tolist(),
        "central_kmeans_global_centroids_last_realization": global_centroids_central.tolist()
    }

    logfile_path = f"misc/kmeans_clustering/{dataset}_partition_{partition}_cluster_no_colab.json"
    if os.path.exists(logfile_path):
        with open(logfile_path, "a") as json_file:
            json.dump(logging_info, json_file, indent=4)
            json_file.write("\n")
    else:
        with open(logfile_path, "w") as json_file:
            json.dump(logging_info, json_file, indent=4)
            json_file.write("\n")

    print(f"Average relative inertia over {num_realizations} realizations is {avg_rel_inertia}")
    

def simulate_kmeans_cluster_colab(
    dataset: str = "mnist",
    n_machines: int = 1,
    k: int = 1,
    max_inner_iters: int = 100,
    max_outer_iters = 10,
    num_realizations: int = 1,
    partition: str = "uniform",
    eps_trust: float = 1e3,
    eps_notrust: float = 1e-2,
    delta: float = 1e-3,
    reg_type: str = "L1",
    reg_strength: float = 0.0,
    gd_num_iters: int = 1500,
    gd_learning_rate: float = 0.005,
):
    """Simulate kmeans over a clustered topology with collaboration via pricer
    :param n_machines: Number of machines
    :param k: Number of clusters
    :param max_inner_iters: Number of inner iterations for (local) k-means clustering
    :param max_outer_iters: Number of outer iterations (communication rounds with PS) for k-means clustering 
    :param num_realizations: Number of realizations to average inertia over
    :param eps_trust: Privacy parameter epsilon for trustworthy neighbors
    :param eps_notrust: Privacy parameter epsilon for non-trustworthy neighbors
    :param delta: Privacy parameter (common for all pairs of neighbors)
    :param reg_type: Type of bias regularization
    :param reg_strength: Regularization parameter strength
    :param gd_num_iters: Number of iterations of gradient descent for joint weight and nois evariance optimization
    :param gd_learning_rate: Learning rate of gradient descent
    Return average relative inertia
    """
    
    # List of relative inertia values for storing results of different realizations
    rel_inertia_list = []

    if dataset == "mnist":
        print(f"Loading MNIST dataset for K-means clustering!")
        X = load_mnist_data()

    elif dataset == "cifar10":
        print(f"Loading CIFAR-10 dataset for K-means clustering!")
        X = load_cifar10_data()
        
    elif dataset == "fmnist-mobilenetv3":
        print(f"Loading Fashion-MNIST embeddings extracted using MobileNet-v3!")
        train_embs, _, test_embs, _ = load_fashion_mnist_embs()
        print(f"Shape of train embeddings: {train_embs.shape}")
        print(f"Shape of test embeddings: {test_embs.shape}")
        X = train_embs
    
    else:
        print(f"Dataset not configured!")
    
    # Get dimension
    dim = X.shape[1]

    # Metrics to save
    losses_seed_variation = []
    optimized_mse_seed_variation = []
    bias_at_nodes_seed_variation = []
    optimized_tiv_seed_variation = []
    optimized_piv_seed_variation = []

    # Get connectivity probabilities to the PS (sole good client)
    tx_probs_PS, tx_probs_colab, conn_matrix = client_locations_mmWave_clusters_intermittent()

    # Set the radius to 1 (assuming all data is normalized)
    radius = 1

    # Set privacy parameters depending on trustworthy neighbors
    E = eps_notrust * torch.ones([n_machines, n_machines])
    for i in range(n_machines):
        for j in range(n_machines):
            if conn_matrix[i][j] == 1:
                E[i][j] = eps_trust

    D = delta * torch.ones([n_machines, n_machines])

    print(f"Running PriCER on cluster topology and dataset partition: {partition}")
    for r in range(num_realizations):
        print(f"Doing k-means clustering for realization: {r}")
        np.random.seed((r + 1) * 1000)
        torch.manual_seed((r + 1) * 1000)

        # Optimize collaboration weights and privacy noise variance (for every realization)
        print(f"Optimizing collaboration weights and privacy noise variance given network connectivity and privacy constraints.")
        (
            weights_opt,
            noise_opt,
            optimization_loss,
        ) = optimize_weights_and_privacy_noise(
            p=tx_probs_PS,
            P=tx_probs_colab,
            E=E,
            D=D,
            radius=radius,
            dimension=dim,
            reg_type=reg_type,
            reg_strength=reg_strength,
            num_iters=gd_num_iters,
            learning_rate=gd_learning_rate
        )

        # Append metrics to lists
        losses_seed_variation.append(optimization_loss)

        optimized_mse = evaluate_mse(
            p=tx_probs_PS, A=weights_opt, P=tx_probs_colab, radius=radius, sigma=noise_opt, dimension=dim
        )
        optimized_mse_seed_variation.append(optimized_mse)

        optimized_tiv = evaluate_tiv(p=tx_probs_PS, A=weights_opt, P=tx_probs_colab, radius=radius)
        optimized_tiv_seed_variation.append(optimized_tiv)

        optimized_piv = evaluate_piv(p=tx_probs_PS, P=tx_probs_colab, sigma=noise_opt, dimension=dim)
        optimized_piv_seed_variation.append(optimized_piv)

        bias_at_nodes = evaluate_bias_at_clients(p=tx_probs_PS, A=weights_opt, P=tx_probs_colab)
        bias_at_nodes_seed_variation.append(bias_at_nodes)

        # Plot the optimization loss for the first realization to see if hyper-parameter choice if ok
        if r == 0:
            plt.figure(figsize=(14, 7))
            with torch.no_grad():
                plt.plot(optimization_loss)

            plt.xlabel("Iterations")
            plt.ylabel("Objective function value")
            plt.yscale("log")
            plt.title("Collaboration weight and privacy noise optimization")
            plt.savefig(
                f"misc/kmeans_{dataset}_{partition}_cluster_bias_reg={reg_strength:.0f}_lr={gd_learning_rate}.png"
            )
        
        print(f"k-means clustering with intermittent connectivity | Sole good connectivity node to PS | PriCER")
        inertia_distr, global_centroids_distributed = distributed_kmeans(
            X=X,
            n_machines=n_machines,
            k=k,
            max_inner_iters=max_inner_iters,
            max_outer_iters=max_outer_iters,
            partition=partition,
            consensus_type="cluster_pricer",
            tx_probs_PS = tx_probs_PS,
            tx_probs_colab = tx_probs_colab,
            weights = weights_opt,
            noise = noise_opt
        )

        print(f"Doing centralized k-means clustering for realization: {r}")
        inertia_central, global_centroids_central = distributed_kmeans(
            X=X, 
            n_machines=1, 
            k=k, 
            max_inner_iters=max_inner_iters, 
            max_outer_iters=max_outer_iters, 
            consensus_type="perfect"
        )

        relative_inertia = inertia_distr / inertia_central

        rel_inertia_list.append(relative_inertia)
        print(f"Realization {r}, Relative inertia: {relative_inertia}")

    # Average quantities over different realizations
    avg_rel_inertia = sum(rel_inertia_list) / len(rel_inertia_list)

    losses_avg = torch.zeros((len(losses_seed_variation[0]),))
    for i in range(len(losses_seed_variation)):
        losses_avg += torch.Tensor(losses_seed_variation[i])
    losses_avg /= len(losses_seed_variation)

    bias_at_nodes_avg = torch.zeros((len(bias_at_nodes_seed_variation[0]),))
    for i in range(len(bias_at_nodes_seed_variation)):
        bias_at_nodes_avg += torch.Tensor(bias_at_nodes_seed_variation[i])
    bias_at_nodes_avg /= len(bias_at_nodes_seed_variation)

    optimized_mse_avg = torch.sum(torch.Tensor(optimized_mse_seed_variation)) / len(optimized_mse_seed_variation)
    optimized_tiv_avg = torch.sum(torch.Tensor(optimized_tiv_seed_variation)) / len(optimized_tiv_seed_variation)
    optimized_piv_avg = torch.sum(torch.Tensor(optimized_piv_seed_variation)) / len(optimized_piv_seed_variation)

    # Log results
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging_info = {
        "timestamp": current_time,
        "connectivity_type": "Intermittent connectivity to PS and amongst nodes: PriCER",
        "radius": radius,
        "dimension": dim,
        "connectivity_to_PS": tx_probs_PS.numpy().tolist(),
        "connectivity_between_clients": tx_probs_colab.numpy().tolist(),
        "privacy_delta": delta,
        "eps_trustworthy": eps_trust,
        "eps_others": eps_notrust,
        "bias_reg_type": reg_type,
        "bias_reg_strength": reg_strength,
        "gradient_descent_num_iters": gd_num_iters,
        "gradient_descent_learning_rate": gd_learning_rate,
        "rel_inertia_list": rel_inertia_list,
        "average_relative_inertia": avg_rel_inertia,
        "optimized_mse_avg": optimized_mse_avg.tolist(),
        "optimized_tiv_avg": optimized_tiv_avg.tolist(),
        "optimized_piv_avg": optimized_piv_avg.tolist(),
        "dist_kmeans_global_centroids_last_realization": global_centroids_distributed.tolist(),
        "central_kmeans_global_centroids_last_realization": global_centroids_central.tolist(),
        "optimized_weights_last_realization": weights_opt.tolist(),
        "optimized_privacy_noise_last_realization": noise_opt.tolist(),
        "optimization_loss_avg": losses_avg.tolist(),
        "bias_at_nodes_avg": bias_at_nodes_avg.tolist(),
    }

    logfile_path = f"misc/kmeans_clustering/{dataset}_partition_{partition}_intmt_cluster_pricer.json"
    if os.path.exists(logfile_path):
        with open(logfile_path, "a") as json_file:
            json.dump(logging_info, json_file, indent=4)
            json_file.write("\n")
    else:
        with open(logfile_path, "w") as json_file:
            json.dump(logging_info, json_file, indent=4)
            json_file.write("\n")

    print(f"Average relative inertia over {num_realizations} realizations is {avg_rel_inertia}")
    

if __name__ == "__main__":
    
    # Simulate k-means clustering with intermittent connectivity over clustered topology and no collaboration
    # simulate_kmeans_cluster_no_colab(
    #     dataset="fmnist-mobilenetv3", 
    #     n_machines=10, 
    #     k=10, 
    #     max_inner_iters=10,
    #     max_outer_iters=5, 
    #     num_realizations=3,
    #     partition="uniform"
    # )
    
    # Simulate k-means clustering with intermittent connectivity over clustered topology and PriCER
    simulate_kmeans_cluster_colab(
        dataset="fmnist-mobilenetv3", 
        n_machines=10, 
        k=10, 
        max_inner_iters=10,
        max_outer_iters=5, 
        num_realizations=3,
        partition="uniform",
        eps_trust=1e3,
        eps_notrust=1e-2,
        delta=1e-3,
        reg_type="L1",
        reg_strength=0,
        gd_num_iters=1500,
        gd_learning_rate=0.01,
    )

    # Fashion-MNIST dataset: Compute classification accuracy with k-means clustering with perfect connectivity
    # simulate_kmeans_perfect_conn(dataset="fashion-mnist-mobilenetv3-embs", n_machines=10, k=3, max_inner_iters=20, 
    #                              max_outer_iters=5, num_realizations=1, partition="uniform", metric="gaussian-kernel")
    # simulate_kmeans_perfect_conn(dataset="fashion-mnist-mobilenetv3-embs", n_machines=10, k=3, max_inner_iters=20, 
    #                              max_outer_iters=5, num_realizations=1, partition="uniform")

    # # Simulate k-means clustering with intermittent connectivity (sole good client) and PriCER
    # simulate_kmeans_intmt_sole_good_node_pricer(
    #     dataset="fmnist-mobilenetv3", 
    #     n_machines=10, 
    #     k=10, 
    #     max_inner_iters=10,
    #     max_outer_iters=5, 
    #     num_realizations=3,
    #     partition="uniform", 
    #     p_good=0.9, 
    #     p_bad=0.1,
    #     p_node=0.9,
    #     eps_trust=1e3,
    #     eps_notrust=1e-2,
    #     delta=1e-3,
    #     num_ngbrs=1,
    #     reg_type="L1",
    #     reg_strength=0.1,
    #     gd_num_iters=1500,
    #     gd_learning_rate=0.5,
    # )

    # Simulate k-means clustering with intermittent connectivity (sole good client) and no collaboration
    # simulate_kmeans_intmt_sole_good_node_no_colab(
    #     dataset="cifar10", 
    #     n_machines=10, 
    #     k=10, 
    #     max_inner_iters=20,
    #     max_outer_iters=5, 
    #     num_realizations=5,
    #     partition="uniform", 
    #     p_good=0.9, 
    #     p_bad=0.1
    # )

    # Simulate kmeans clustering with perfect connectivity
    # simulate_kmeans_perfect_conn(dataset="mnist", n_machines=10, k=3, max_inner_iters=20, max_outer_iters=5, num_realizations=1, partition="uniform")


    # parser = argparse.ArgumentParser()
    # parser.add_argument("--n_machines", type=int, default=10, help="Number of machines")
    # parser.add_argument("--k", type=int, default=10, help="Value of k")
    # parser.add_argument(
    #     "--max_iterations", type=int, default=10, help="Maximum number of iterations"
    # )
    # parser.add_argument(
    #     "--num_realizations", type=int, default=1, help="Number of realizations"
    # )
    # parser.add_argument(
    #     "--dataset", type=str, default="mnist", help="Dataset being clustered"
    # )
    # parser.add_argument(
    #     "--partition",
    #     type=str,
    #     default="uniform",
    #     help="Type of dataset split across clients",
    # )
    # parser.add_argument(
    #     "--consensus_type",
    #     type=str,
    #     default="perfect",
    #     help="Topology parameters and algorithm used for computing global centroid",
    # )

    # args = parser.parse_args()

    # n_machines = args.n_machines
    # k = args.k
    # max_iterations = args.max_iterations
    # num_realizations = args.num_realizations
    # dataset = args.dataset
    # partition = args.partition
    # consensus_type = args.consensus_type

    # logger.info(
    #     f"K-means clustering: Dataset: {dataset}, Partition: {partition}, Consensus_type: {consensus_type}"
    # )
    # logger.info(
    #     f"n_machines = {n_machines}, k = {k}, max_iterations = {max_iterations}"
    # )

    # inertia_arr = simulate_kmeans(
    #     dataset=dataset,
    #     n_machines=n_machines,
    #     k=k,
    #     max_iterations=max_iterations,
    #     num_realizations=num_realizations,
    #     partition=partition,
    #     consensus_type=consensus_type,
    # )

    # logger.info(
    #     f"K-means clustering: Dataset: {dataset}, Partition: {partition}, Consensus_type: {consensus_type}"
    # )
    # logger.info(
    #     f"n_machines = {n_machines}, k = {k}, max_iterations = {max_iterations}"
    # )
    # logger.info(
    #     f"Inertia: Mean = {np.mean(inertia_arr):.3f}, Std. dev = {np.std(inertia_arr):.3f}"
    # )
    
    