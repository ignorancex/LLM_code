import os
import pickle

import numpy as np
import requests
import tensorflow as tf

# Constants
FEATURE_DIMENSION = 2000
SAMPLE_SIZE = 2**12
CIFAR_URL = 'https://storage.googleapis.com/gresearch/cifar5m/part0.npz'
CACHE_DIR = 'cache'


def download_file(url, save_path):
    """Download a file from a specified URL to a local path."""
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"File downloaded successfully: {save_path}")
    else:
        print(f"Failed to download file, status code: {response.status_code}")


def ensure_raw_data():
    """Ensure the raw CIFAR data is available locally."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    save_path = os.path.join(CACHE_DIR, 'part0.npz')
    if not os.path.exists(save_path):
        download_file(CIFAR_URL, save_path)
    return save_path


def filter_binary_classes(npz_path):
    """Extract and filter data for binary classification (classes 0 and 1)."""
    cache_path = os.path.join(CACHE_DIR, 'data2.pkl')

    if os.path.exists(cache_path):
        return pickle.load(open(cache_path, "rb"))

    with np.load(npz_path) as data:
        X, Y = data['X'], data['Y']

    # Filter for classes 0 and 1
    mask = np.isin(Y, [0, 1])
    X = X[mask]
    Y = Y[mask]

    pickle.dump((X, Y), open(cache_path, "wb"))
    return X, Y


def process_images(images):
    """Convert images to normalized grayscale features."""
    greyscale = tf.image.rgb_to_grayscale(images)
    greyscale = tf.cast(greyscale, tf.float32) / 255.0
    flattened = tf.reshape(greyscale, [greyscale.shape[0], -1])
    normalized = tf.linalg.normalize(
        flattened - tf.reduce_mean(flattened), axis=-1)[0]
    return normalized


def compute_features(normalized_images, feature_dim):
    """Project normalized images into feature space using ReLU."""
    W = tf.Variable(tf.random.normal(
        shape=(normalized_images.shape[1], feature_dim))) / np.sqrt(feature_dim)
    return tf.nn.relu(tf.matmul(normalized_images, W))


def prepare_dataset():
    """Main function to prepare the dataset and compute kernel matrix."""
    # 1. Get raw data
    npz_path = ensure_raw_data()

    # 2. Load and filter binary classes
    cache_path = os.path.join(CACHE_DIR, f'features_{FEATURE_DIMENSION}.pkl')
    if os.path.exists(cache_path):
        return pickle.load(open(cache_path, "rb"))

    # 3. Process images and compute features
    X, Y = filter_binary_classes(npz_path)
    normalized_images = process_images(X)
    features = compute_features(normalized_images, FEATURE_DIMENSION)

    # 4. Prepare labels
    labels = tf.convert_to_tensor(Y, dtype='float32')
    labels = labels - tf.reduce_mean(labels)

    # Cache results
    pickle.dump((features, labels), open(cache_path, "wb"))
    return features, labels


def get_real_data(size=SAMPLE_SIZE):
    """Public interface to get the processed data and kernel matrix."""
    features, labels = prepare_dataset()
    F = features[:size].numpy()
    b = labels[:size].numpy()
    K = (F.T @ F) / size
    return F, b, K
