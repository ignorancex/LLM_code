import numpy as np
import torch
import torchvision.datasets as datasets

from loguru import logger

from mean_estimation import (
    dme_intermittent_naive,
    dme_pricer
)

from optimization import optimize_weights_and_privacy_noise
from objectives import evaluate_mse, evaluate_tiv, evaluate_piv
from utils import evaluate_bias_at_clients, load_mnist_data, load_cifar10_data, load_fashion_mnist_embs


def split_dataset(X, n_machines, partition: str = "uniform", manual_seed=42):
    """Split the dataset into n_machines subsets
    :param X: Complete dataset
    :param n_machines: No. of machine across which X is to be distributed
    :param partition: Dataset split type
    :param manual_seed: Random seed to do a different dataset splitting
    Return: List of (array like) datasets
    """

    # Manually initialize seed
    np.random.seed(manual_seed)

    if partition == "uniform":
        np.random.shuffle(X)  # Shuffle the rows of X randomly

    if n_machines > 1:
        print(f"Splitting the dataset according to partition type: {partition}")

    # Sort and part (by default)
    n_samples = len(X)
    subset_size = n_samples // n_machines

    subsets = []
    for i in range(n_machines):
        start = i * subset_size
        end = (i + 1) * subset_size
        subsets.append(X[start:end])

    return subsets


def kmeans_plus_plus(X, k, manual_seed=42, metric="euclidean"):
    """Initialize the centroids using k-means++ algorithm.
    :param X: Data matrix of shape (num_samples, num_features)
    :param k: Number of clusters in k-means clustering
    :param manual_seed: Random seed to do a different initial clustering
    :param metric: Distance metric used for clustering
    Return initial centroids of the k clusters
    """

    # Manually initialize seed
    np.random.seed(manual_seed)

    # Initialize the first centroid uniformly at random
    centroids = [X[np.random.choice(len(X))]]
    sigma = 1.0             # Width of the Gaussian kernel

    # Choose the remaining centroids using the K-means++ initialization algorithm
    for i in range(1, k):
        # Compute distance of each point from the existing centroids
        distances = np.zeros((len(X), len(centroids)))

        if metric == "euclidean":
            for i in range(len(X)):
                for j in range(len(centroids)):
                    distances[i][j] = np.linalg.norm(X[i] - centroids[j])

        elif metric == "gaussian-kernel":
            for i in range(len(X)):
                for j in range(len(centroids)):
                    distances[i][j] = np.exp(-np.linalg.norm(X[i] - centroids[j]) ** 2 / (2 * sigma ** 2))

        # Compute the minimum distance of each datapoint and all the previously chosen centroids
        min_distances = np.min(distances, axis=1)

        # Add a small constant to the min_distances array to avoid dividing by zero
        min_distances += np.finfo(float).eps

        # Compute the probability of selecting each data point as the next centroid
        probabilities = min_distances / np.sum(min_distances)

        # Choose the next centroid
        index = np.random.choice(len(X), p=probabilities)
        centroids.append(X[index])

    return np.array(centroids)


def run_local_kmeans(data, centroids, k, max_iterations, metric):
    """Run k-means locally on each machine
    :param data: Local data of the machine
    :param centroids: Initial centroids broadcasted from the parameter server
    :param k: Number of clusters
    :param max_iterations: Number of local k-means iteration
    :param metric: Distance metric used for clustering
    Return: List of updated centroids
    """

    num_datapoints = data.shape[0]
    updated_centroids = np.zeros([k, data.shape[1],])
    sigma = 1.0                     # Width of the Gaussian kernel

    distances = np.zeros([num_datapoints, k])

    for i in range(max_iterations):
        # print(f"Local k-means iteration: {i}/{max_iterations}")

        # Compute the distance of each datapoint to each of the centroids
        if metric == "euclidean":
            for j in range(num_datapoints):
                for l in range(k):
                    distances[j][l] = np.linalg.norm(data[j, :] - centroids[l, :])

        elif metric == "gaussian-kernel":
            for j in range(num_datapoints):
                for l in range(k):
                    distances[j][l] = np.exp(-np.linalg.norm(data[j, :] - centroids[l, :]) ** 2 / (2 * sigma ** 2))

        labels = np.argmin(distances, axis=1)

        for l in range(k):
            if np.sum(labels == l) > 0:
                updated_centroids[l, :] = np.mean(data[labels == l], axis=0)

    return updated_centroids


def calculate_inertia(data, centroids):
    """Inertia measures the sum of squared distances of samples to their closest cluster center
    A lower value of inertia indicates better clustering performance.
    """
    # Number of clusters
    k = len(centroids)

    # Calculate distances between data points and centroids
    distances = np.zeros((len(data), k))
    for i in range(k):
        distances[:, i] = np.sum((data - centroids[i]) ** 2, axis=1)

    # Assign each data point to the nearest centroid
    labels = np.argmin(distances, axis=1)

    # Calculate inertia
    inertia = 0
    for i in range(k):
        inertia += np.sum(distances[labels == i, i])

    return inertia


def assign_to_nearest_centroid(data, centroids, metric="euclidean"):

    assert data.shape[1] == centroids.shape[1], "Dimension mismatch between data and centroids!"
    num_datapoints = data.shape[0]
    k = centroids.shape[0]
    distances = np.zeros([num_datapoints, k])
    sigma = 1.0             # Width of a Gaussian kernel

    # Calculate distances between each datapoint and each centroid depnding on the type of metric
    if metric == "euclidean":
        for j in range(num_datapoints):
            for l in range(k):
                distances[j][l] = np.linalg.norm(data[j, :] - centroids[l, :])

    elif metric == "gaussian-kernel":
        for j in range(num_datapoints):
            for l in range(k):
                distances[j][l] = np.exp(-np.linalg.norm(data[j, :] - centroids[l, :]) ** 2 / (2 * sigma ** 2))

    # Find the index of the nearest centroid for each datapoint
    assigned_centroids = np.argmin(distances, axis=1)

    return assigned_centroids


def assign_labels_to_centroids(train_data, train_labels, centroids):
    """Assign labels to centroids based on majority vote of all training datapoints assigned to that centroid
    """

    assert train_data.shape[0] == len(train_labels), "Dimension mismatch between data and labels!"

    # Find the index of the nearest centroid for each datapoint
    assigned_centroids = assign_to_nearest_centroid(train_data, centroids)

    # Initialize an array to store the labels assigned to each centroid
    assigned_labels = np.zeros(centroids.shape[0])

    # Assign labels to centroids based on the majority of labels of assigned datapoints
    for i in range(centroids.shape[0]):
        datapoints_indices_for_centroid = np.where(assigned_centroids == i)[0]
        labels_for_centroid = train_labels[datapoints_indices_for_centroid]
        majority_label = np.argmax(np.bincount(labels_for_centroid))
        assigned_labels[i] = majority_label

    return assigned_labels


def classify_test_data(test_data, test_labels, centroids, centroid_labels):
    """Assign labels to test datapoints based on closest centroid
    """

    # Find the index of the nearest centroid for each test datapoint
    predicted_centroids = assign_to_nearest_centroid(test_data, centroids)

    # Assign the labels of the nearest centroids as the predicted labels for test datapoints
    predicted_labels = centroid_labels[predicted_centroids]

    # Compare predicted labels with ground truth labels
    correct_predictions = np.sum(predicted_labels == test_labels)

    # Calculate accuracy as the ratio of correct predictions to the total number of datapoints
    accuracy = correct_predictions / len(test_labels)

    return accuracy


def relative_clustering_mismatch(data, centroids_dist, centroids_cent, metric="euclidean"):
    """ Computes the clustering mismatch between distributed and centralized k-means clustering on a dataset -- measure of similarity in semantic search
        test_data: Test data to be clustered
        centroids_dist: Centroids computed using distriubted k-means clustering
        centroids_cent: Centroids computed using centralized k-means clustering
    """

    assert data.shape[1] == centroids_dist.shape[1] == centroids_cent.shape[1], "Mismatch in dimensions of datapoints and centroids!"
    assert centroids_dist.shape[0] == centroids_cent.shape[0], "No. of centroids for deistributed and centralized is not the same!"

    assignment_dist = assign_to_nearest_centroid(data, centroids_dist, metric)
    assignment_cent = assign_to_nearest_centroid(data, centroids_cent, metric)

    assert len(assignment_dist) == len(assignment_cent), "Length of distributed and centralized assignments must be the same!"

    differ_count = sum(1 for dist, cent in zip(assignment_dist, assignment_cent) if dist != cent)
    relative_mismatch = differ_count / len(assignment_cent)

    return relative_mismatch


def distributed_kmeans(
    X: np.ndarray = None,
    n_machines: int = 1,
    k: int = 1,
    max_inner_iters: int = 100,
    max_outer_iters: int = 10,
    partition: str = "uniform",
    consensus_type: str = "perfect",
    tx_probs_PS: np.ndarray = None,
    tx_probs_colab: np.ndarray = None,
    weights: np.ndarray = None,
    noise: np.ndarray = None,
    manual_seed=42,
    metric="euclidean"
):
    """Performs distributed kmeans clustering on by splitting the dataset
    :param X: Complete dataset
    :param n_machines: Number of machines
    :param k: Number of clusters
    :param max_iterations: Number of inner iterations for k-means clustering
    :param max_outer_iters: Number of outer iterations (i.e., consensus at PS) for k-means clustering
    :param partition: Dataset split type
    :param consensus_type: Topology parameters and algorithm used for computing global centroid
    :param tx_probs_PS: Connection probabilities to the PS
    :param tx_probs_colab: Connection probabilities for collaboration amongst clients
    :param weights: Collaboration weights (after optimization)
    :param noise: Standard deviation of privacy pertubation noise
    :param manual_seed: Random seed to do a different initial clustering
    :param metric: Distance metric used for clustering
    Return inertia
    """

    # Check dimension consistency
    if consensus_type in ["intermittent_sole_good_client_naive", "cluster_no_colab"]:
        assert tx_probs_PS.shape[0] == n_machines, "Dimension mismatch: tx_prob_ps and n_machines!"

    elif consensus_type in ["intermittent_sole_good_client_pricer", "cluster_pricer"]:
        assert tx_probs_PS.shape[0] == n_machines, "Dimension mismatch: tx_prob_ps and n_machines!"
        assert tx_probs_colab.shape[0] == tx_probs_colab.shape[1] == n_machines, "Dimension mismatch: tx_prob_colab and n_machines!"
        assert weights.shape[0] == weights.shape[1] == n_machines, "Dimension mismatch: collaboration weights and n_machines!"
        assert noise.shape[0] == noise.shape[1] == n_machines, "Dimension mismatch: privacy noise and n_machines!"

    # Split the dataset across different machines
    dim = X.shape[1]
    X_splits = split_dataset(X, n_machines, partition=partition, manual_seed=manual_seed)

    # Initialize the local centroids on each machine using k-means++
    print(f"Initializing local centroids")
    local_centroids = [kmeans_plus_plus(X_i, k, manual_seed=manual_seed, metric=metric) for X_i in X_splits]

    # Run local k-means cluster on each machine
    updated_local_centroids = [None for _ in range(n_machines)]

    for outer_iter in range(max_outer_iters):

        # Setting new seeds for every outer iteration so that a fresh intermittent connectivity can be realized
        np.random.seed((outer_iter + 1) * 1024)
        torch.manual_seed((outer_iter + 1) * 1024)

        print(f"Running outer iteration: {outer_iter}")

        for i in range(n_machines):
            print(f"Running local k-means on machine {i}/{n_machines}")
            updated_local_centroids[i] = run_local_kmeans(
                data=X_splits[i],
                centroids=local_centroids[i],
                k=k,
                max_iterations=max_inner_iters,
                metric=metric
            )

        # Compute the average of the new centroids to get the global centroid
        if consensus_type == "perfect":
            global_centroid = np.mean(np.array(updated_local_centroids), axis=0)

        elif consensus_type in ["intermittent_sole_good_client_naive", "cluster_no_colab"]:
            # Intermittent connectivity of clients to PS.
            # No collaboration amongst clients
            print(
                f"Computing global centroid with intermittent connectivity and no collaboration amongst nodes."
            )

            global_centroid = np.zeros([k, dim])

            for c_idx in range(k):
                T = np.zeros([n_machines, dim])

                for mc_idx in range(n_machines):
                    T[mc_idx, :] = updated_local_centroids[mc_idx][c_idx, :]

                print(f"Communications from nodes to the PS for each centroid:")
                global_centroid[c_idx, :] = dme_intermittent_naive(
                    client_data=T, transmit_probs=tx_probs_PS
                )

        elif consensus_type in ["intermittent_sole_good_client_pricer", "cluster_pricer"]:
            # Intermittent connectivity of clients to PS and with each other
            # Full collaboration amongst clients
            print(
                f"Computing global centroid in presence of intermittent connectivity with PriCER"
            )

            global_centroid = np.zeros([k, dim])

            for c_idx in range(k):
                T = np.zeros([n_machines, dim])

                for mc_idx in range(n_machines):
                    T[mc_idx, :] = updated_local_centroids[mc_idx][c_idx, :]

                global_centroid[c_idx, :] = dme_pricer(
                    transmit_probs=tx_probs_PS,
                    prob_ngbrs=tx_probs_colab,
                    client_data=T,
                    weights=weights,
                    priv_noise_var=noise,
                )

        else:
            print(f"Consensus type not implemented! Defaulting to perfect consensus")
            global_centroid = np.mean(np.array(updated_local_centroids), axis=0)

        # Broadcast global centroids to all machines.
        for i in range(n_machines):
            local_centroids[i] = global_centroid.copy()

    # Evaluate clustering accuracy
    inertia = calculate_inertia(X, global_centroid)

    return inertia, global_centroid


def compute_relative_inertia(
    X: np.ndarray = None,
    n_machines: int = 1,
    k: int = 1,
    max_iterations: int = 10,
    partition: str = "uniform",
    consensus_type: str = "perfect",
):
    """Compute inertia relative to clustering on a single machine
    :param X: Complete dataset
    :param n_machines: Number of machines
    :param k: Number of clusters
    :param max_iterations: Number of iterations for k-means clustering
    :param partition: Dataset split type
    :param consensus_type: Topology parameters and algorithm used for computing global centroid
    Return relative inertia
    """

    logger.info(f"Doing distributed kmeans clustering...")
    inertia_distr = distributed_kmeans(
        X=X,
        n_machines=n_machines,
        k=k,
        max_iterations=max_iterations,
        partition=partition,
        consensus_type=consensus_type,
    )

    logger.info(f"Doing centralized kmeans clustering...")
    inertia_central = distributed_kmeans(
        X=X, n_machines=1, k=k, max_iterations=max_iterations
    )

    return inertia_distr / inertia_central













