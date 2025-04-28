import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_small

import os
import tqdm
from tqdm import tqdm


def find_max_l2_norm(arrays):
    max_norm = 0.0

    for array in arrays:
        norms = np.linalg.norm(array, axis=1)  # Calculate L2-norm along rows
        max_norm = max(max_norm, np.max(norms))

    return max_norm


def evaluate_bias_at_clients(
    p: torch.Tensor = None,
    A: torch.Tensor = None,
    P: torch.Tensor = None,
):
    """Evaluate the bias at every node for monitoring and logging
    :param p: Array of transmission probabilities from each of the clients to the PS.
    :param A: Matrix of weights
    :param P: Matrix of probabilities for intermittent connectivity amongst clients
    """

    num_clients = len(p)
    A_dim = A.shape
    neighbors_dim = P.shape

    # Validate inputs
    assert num_clients == A_dim[0] == A_dim[1]
    assert num_clients == neighbors_dim[0] == neighbors_dim[1]

    # Compute the bias terms
    bias = torch.zeros(num_clients)
    for i in range(num_clients):
        for j in range(num_clients):
            bias[i] += p[j] * P[i][j] * A[i][j]
        bias[i] -= 1

    return bias

def load_mnist_data(dataset_path="../datasets/mnist"):
    # Define the transformation, including normalization to unit norm
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x.flatten() / torch.norm(x.flatten()))
    ])

    # Load MNIST dataset with the specified transformations
    mnist_train = datasets.MNIST(root=dataset_path, train=True, download=True, transform=transform)

    # Concatenate and convert the data to a NumPy array
    train_data = torch.cat([sample[0].unsqueeze(0) for sample in mnist_train], dim=0)

    return train_data.numpy()


def load_cifar10_data(dataset_path="../datasets/cifar10"):
    """Load the CIFAR-10 dataset
    :param dataset_path: Location to download the dataset
    :return: Downloaded dataset as a tensor
    """
    # Define the transformation, including normalization to unit norm
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x.flatten() / torch.norm(x.flatten()))
    ])

    cifar10_train = datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transform)
    train_data = torch.cat([sample[0].flatten().unsqueeze(0) for sample in cifar10_train], dim=0)

    return train_data.numpy()


def load_fashion_mnist(dataset_path="../datasets/fashion-mnist"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.FashionMNIST(root=dataset_path, train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root=dataset_path, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader


def create_mobilenetv3_model():
    model = mobilenet_v3_small(pretrained=True)
    model.features[0][0] = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    # Remove the last fully connected layer (classifier)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    return model


def compute_and_save_embeddings(loader, model, file_path):
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for data, target in tqdm(loader, desc="Computing embeddings", unit="batch", leave=False):
            output = model(data)
            normalized_output = nn.functional.normalize(output, p=2, dim=1)
            embeddings.append(normalized_output.numpy())
            labels.append(target.numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    embeddings = np.squeeze(embeddings, axis=(2, 3))
    labels = np.concatenate(labels, axis=0)
    np.savez(file_path, embeddings=embeddings, labels=labels)


def load_embeddings(file_path):
    data = np.load(file_path)
    return data['embeddings'], data['labels']


def extract_image_embeddings():
    # Load Fashion MNIST data
    train_loader, test_loader = load_fashion_mnist()

    # Create MobileNetV3 model
    mobilenetv3_model = create_mobilenetv3_model()

    # Compute and save normalized embeddings for training data
    compute_and_save_embeddings(train_loader, mobilenetv3_model, 'train_embeddings.npz')

    # Compute and save normalized embeddings for test data
    compute_and_save_embeddings(test_loader, mobilenetv3_model, 'test_embeddings.npz')

    # Load embeddings later if needed
    loaded_train_embeddings, train_labels = load_embeddings('train_embeddings.npz')
    loaded_test_embeddings, test_labels = load_embeddings('test_embeddings.npz')

    # Print the shape of embeddings and labels
    print("Shape of loaded_train_embeddings:", loaded_train_embeddings.shape)
    print("Shape of train_labels:", train_labels.shape)

    print("Shape of loaded_test_embeddings:", loaded_test_embeddings.shape)
    print("Shape of test_labels:", test_labels.shape)

    # Print some example embeddings and labels
    print("\nExample embeddings and labels:")
    print("First three embeddings in loaded_train_embeddings:")
    print(loaded_train_embeddings[:3])
    print("Corresponding labels:")
    print(train_labels[:3])

    print("\nFirst three embeddings in loaded_test_embeddings:")
    print(loaded_test_embeddings[:3])
    print("Corresponding labels:")
    print(test_labels[:3])

    # Print the shape of the first embedding in loaded_train_embeddings
    first_embedding_shape = loaded_train_embeddings[0].shape
    print(f"Shape of the first embedding in loaded_train_embeddings: {first_embedding_shape}")

def load_fashion_mnist_embs():

    # Check if embeddings are already extracted, and extract if not.
    train_embs_file = "train_embeddings.npz"
    test_embs_file = "test_embeddings.npz"

    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    datasets_dir = os.path.join(current_dir, '..', 'datasets', 'fashion-mnist-mobilenetv3-embeddings')

    files_in_directory = os.listdir(datasets_dir)

    if train_embs_file in files_in_directory and test_embs_file in files_in_directory:
        print(f"Fashion-MNIST embeddings already extracted!")
        pass
    else:
        print(f"Extracting Fashion-MNIST embeddings using MobileNet-v3")
        extract_image_embeddings()

    train_embs, train_labels = load_embeddings(os.path.join(datasets_dir, 'train_embeddings.npz'))
    test_embs, test_labels = load_embeddings(os.path.join(datasets_dir, 'test_embeddings.npz'))

    return train_embs, train_labels, test_embs, test_labels



    






    