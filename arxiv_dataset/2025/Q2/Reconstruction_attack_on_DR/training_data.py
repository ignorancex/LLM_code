import numpy as np
import keras
from sklearn.manifold import Isomap
from config import s_size, train_size, img_size, n_compo

def create_S(X, y, s_size):
    """
    Randomly selects a known-member set S from the dataset (X, y),
    removes these samples from X and y, and returns the updated datasets.

    Parameters:
    - X (np.ndarray): Array of input data of shape (n_samples, height, width)
    - y (np.ndarray): Array of labels corresponding to X
    - s_size (int): Number of samples to include in the known-member set

    Returns:
    - X (np.ndarray): Remaining data after removing known-member set
    - y (np.ndarray): Remaining labels after removing known-member set
    - S (np.ndarray): known-member set of selected data
    """
    # Set random seed for reproducibility
    np.random.seed(123)    
    # Initialize an empty array to hold the known-member set
    S = np.zeros((s_size, img_size, img_size))

    # Randomly choose unique indices for creating known-member set
    idx = np.random.choice(len(X), s_size, replace=False)

    # Select the samples from X corresponding to the chosen indices
    S = X[idx]

    # Remove the selected samples from X and y
    X = np.delete(X, idx, axis=0)
    y = np.delete(y, idx, axis=0)

    # Return the modified dataset and the known-member set
    return X, y, S


def create_train_and_test(X, y, train_size):
    """
    Randomly selects a training set and test set from entire dataset of size train_size from X and y,
    removes the selected samples from X and y, and returns both training and remaining test sets.
    This function returns both the training set and the test set, which are combined initially.
    You are encouraged to specify the total number of data points (i.e., training + test) using a single value.
    For example, if you want 4,000 samples for training and 500 for testing, you should set train_size = 4500.
    You can then split the data later using 'train_test_split' to separate the training and test sets as needed.

    Parameters:
    - X (np.ndarray): Input data, shape (n_samples, height, width)
    - y (np.ndarray): Labels corresponding to X (can also be image labels of same shape)
    - train_size (int): Number of training samples to select

    Returns:
    - x_head (np.ndarray): Selected training input samples
    - y_head (np.ndarray): Selected training labels
    - X (np.ndarray): Remaining input samples after training set removal
    - y (np.ndarray): Remaining labels after training set removal
    """

    # Initialize a random number generator with a fixed seed for reproducibility
    rng = np.random.default_rng(seed=123)

    # Randomly select indices for the training set
    idx = rng.choice(len(X), train_size, replace=False)

    # Initialize arrays to store training inputs and labels
    x_head = np.zeros((train_size, img_size, img_size))
    y_head = np.zeros((train_size, img_size, img_size))

    # Assign the selected training samples to x_head and y_head
    x_head = X[idx]
    y_head = y[idx]

    # Remove the selected samples from the original dataset
    X = np.delete(X, idx, axis=0)
    y = np.delete(y, idx, axis=0)

    # Return training set and remaining test set
    return x_head, y_head, X, y


def embedding_function(x_head, y_head, S, embedding, train_size, s_size):
    """
    Applies an embedding method to each training sample concatenated with the known-member set.

    Parameters:
    - x_head (np.ndarray): Training input samples, shape (train_size, 28, 28)
    - y_head (np.ndarray): Training labels (images), shape (train_size, 28, 28)
    - S (np.ndarray): known-member set, shape (s_size, 28, 28)
    - embedding: An embedding object with fit_transform() method (e.g., PCA, t-SNE, UMAP)
    - train_size (int): Number of training samples
    - s_size (int): Size of the known-member set

    Returns:
    - X_recon (np.ndarray): Embedded training representations
    - y_recon (np.ndarray): Reconstructed labels (same as y_head)
    """
    # For reproducibility
    np.random.seed(123)

    # Create empty arrays storing embedded data and its original
    X_recon = np.zeros((train_size, n_compo * (s_size + 1)))  # Embedded data output
    y_recon = np.zeros((train_size, img_size, img_size)) # Output is original data

    for i in range(train_size):
        # Stack the current sample with the known-member set 
        tmp = np.vstack(
            (x_head[i].reshape(1, img_size, img_size), S)).reshape(-1, img_size*img_size)
        embedding_results = embedding.fit_transform(tmp)

        # Flatten each image in the stack to shape (s_size + 1, 28*28)
        X_recon[i] = embedding_results.reshape(-1, n_compo * (s_size + 1))

        # Store the corresponding original data
        y_recon[i] = y_head[i]

    return X_recon, y_recon

def begin_create_data():
    
    # Load the dataset
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data() 

    # Concatenate training and test sets into a single dataset
    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((X_train, X_test), axis=0)

    # Create the training set of size 'train_size', and keep the remaining data in X_tmp and y_tmp
    x_head, y_head, X_tmp, y_tmp = create_train_and_test(X, y, train_size)

    # From the remaining data, create the known-member set (S) corresponding to the size of 's_size'
    X_tmp, y_tmp, s = create_S(X_tmp, y_tmp, s_size=s_size)

    # Initialize the embedding with the specified number of components (e.g., Isomap, PCA, t-SNE)
    embedding = Isomap(n_components=n_compo)

    # Apply the embedding function to generate embedded data
    # Each sample is concatenated with the known-member set and transformed using defined embedding
    X_recon, y_recon = embedding_function(x_head, y_head, s, embedding, train_size, s_size)
    np.save("X_recon.npy", X_recon)
    np.save("y_recon.npy", y_recon)