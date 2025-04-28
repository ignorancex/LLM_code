import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from sklearn.datasets import load_wine
from sklearn.decomposition import KernelPCA
from qiskit.compiler import transpile, assemble
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, QuantumError, ReadoutError, pauli_error, depolarizing_error, thermal_relaxation_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
import pickle
import numpy as np
from matplotlib import style
from mpi4py import MPI
import math
import random
import time
import cv2
from sklearn import datasets

def save_pca_results(X_pca, labels, filename):
    """ Saves PCA results as a scatter plot image and a pickle file. """
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', edgecolor='k', s=50)
    plt.colorbar(scatter)
    plt.title(f'PCA Results: {filename}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.savefig(f'{filename}.png')
    plt.close()
    with open(f'{filename}.pkl', 'wb') as f:
        pickle.dump(X_pca, f)

def plot_image(img, title: str):
    """ Plots a grayscale image. """
    plt.figure(figsize=(4, 4))
    plt.title(title)
    plt.imshow(img, extent=[0, img.shape[0], img.shape[1], 0], cmap='gray')
    plt.show()

def nearest_power_of_two(n):
    """ Returns the nearest power of two greater than or equal to n. """
    return 1 << (n-1).bit_length()

def pad_array_to_power_of_two(arr):
    """ Pads an array with zeros to the nearest power of two in length. """
    target_length = nearest_power_of_two(len(arr))
    extended_array = np.pad(arr, (0, target_length - len(arr)), mode='constant')
    return extended_array

def permute_arrays(arr1, arr2, seed=42):
    """ Permutes two given arrays using random seed. """
    if len(arr1) != len(arr2):
        raise ValueError("The arrays must have the same size.")
    if (len(arr1) & (len(arr1) - 1)) != 0:
        arr1 = pad_array_to_power_of_two(arr1)
        arr2 = pad_array_to_power_of_two(arr2)
    random.seed(seed)
    indices = list(range(len(arr1)))
    random.shuffle(indices)
    permuted_arr1 = [arr1[i] for i in indices]
    permuted_arr2 = [arr2[i] for i in indices]
    return permuted_arr1, permuted_arr2

def amplitude_encode_image(image):
    """ Amplitude encodes an image. """    
    norm = np.linalg.norm(image)
    if norm == 0:
        raise ValueError("Image norm is zero, cannot encode a zero image.")
    return image.flatten() / norm

def amplitude_encode_vector(vector):
    """ Amplitude encodes a vector. """
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Vector norm is zero, cannot encode a zero vector.")
    return vector / norm

def setup_noise_model(noise_level_code):
    """
    Sets up a noise model based on the given noise level code.
    Parameters:
    noise_level_code (int): Code indicating the noise level (1 for 0.1%, 2 for 1%).  
    Returns:
    NoiseModel: A Qiskit noise model configured with depolarizing errors.
    """
    noise_model = NoiseModel()
        if noise_level_code == 1:
        noise_level = 0.001
    elif noise_level_code == 2:
        noise_level = 0.01
    else:
        raise ValueError("Invalid noise level code. Use 1 for 0.1% and 2 for 1%.")
    single_qubit_error = depolarizing_error(noise_level, 1)
    noise_model.add_all_qubit_quantum_error(single_qubit_error, ['u1', 'u2', 'u3'])
    two_qubit_error = depolarizing_error(noise_level, 2)
    noise_model.add_all_qubit_quantum_error(two_qubit_error, 'cx')
    return noise_model

def teleport_and_inner_product(data1, data2, noise_level=0):
    """ Quantum encodes two data points, teleports them and computes the inner product of the encodings, given a noise level."""
    if isinstance(data1, np.ndarray) and len(data1.shape) == 1:
        if len(data1) != len(data2):
            raise ValueError("Error: Vectors do not have the same length.")
        qubits_per_data = int(np.ceil(np.log2(len(data1))))
        amplitudes_1 = amplitude_encode_vector(data1)
        amplitudes_2 = amplitude_encode_vector(data2)
    elif isinstance(data1, np.ndarray) and len(data1.shape) == 2:
        if data1.shape != data2.shape:
            raise ValueError("Error: Images do not have the same dimensions.")
        rows, cols = data1.shape
        qubits_per_data = int(np.ceil(np.log2(rows * cols)))
        amplitudes_1 = amplitude_encode_image(data1)
        amplitudes_2 = amplitude_encode_image(data2)
        print('Got Image')
    else:
        raise TypeError("Unsupported data type. Provide either two vectors or two images.")
    amplitudes_1,amplitudes_2 = permute_arrays(amplitudes_1,amplitudes_2)
    qubits_per_image=qubits_per_data
    total_qubits= (4*qubits_per_image)+1
    total_bits=(4*qubits_per_image)+1
    circuit = QuantumCircuit(total_qubits,total_bits)
    qubits_1=np.arange(0,qubits_per_image)
    qubits_2=np.arange(3*qubits_per_image,4*qubits_per_image)
    circuit.initialize(amplitudes_1, qubits_1.tolist())
    circuit.initialize(amplitudes_2, qubits_2.tolist())
    for _ in range(qubits_per_image):
        circuit.h(_+qubits_per_image)
        circuit.cx(_+qubits_per_image,_+(2*qubits_per_image))
        circuit.h(_)
        circuit.cx(_,_+qubits_per_image)
        circuit.h(_)
        circuit.measure(_,_)
        circuit.measure(_+qubits_per_image,_+qubits_per_image)
        circuit.z(_+(2*qubits_per_image)).c_if(circuit.clbits[_], 1)
        circuit.x(_+(2*qubits_per_image)).c_if(circuit.clbits[_+qubits_per_image], 1)
        circuit.reset(_)
        circuit.reset(_+qubits_per_image)
    for _ in range(qubits_per_image):
        circuit.h(_+(1*qubits_per_image))
        circuit.cx(_+(1*qubits_per_image),_+(0*qubits_per_image))
        circuit.h(_+(3*qubits_per_image))
        circuit.cx(_+(3*qubits_per_image),_+(1*qubits_per_image))
        circuit.h(_+(3*qubits_per_image))
        circuit.measure(_+(3*qubits_per_image),_+(2*qubits_per_image))
        circuit.measure(_+(1*qubits_per_image),_+(3*qubits_per_image))
        circuit.z(_+(0*qubits_per_image)).c_if(circuit.clbits[_+(2*qubits_per_image)], 1)
        circuit.x(_+(0*qubits_per_image)).c_if(circuit.clbits[_+(3*qubits_per_image)], 1)
    circuit.h(total_qubits-1)
    for _ in range(qubits_per_image):
        circuit.cswap(total_qubits-1, _+(2*qubits_per_image), _+(0*qubits_per_image))
    circuit.h(total_qubits-1)
    circuit.measure(total_qubits-1, total_bits-1)
    simulator = AerSimulator()
    if noise_level > 0:
        noise_model = setup_noise_model(noise_level)
        simulator.set_options(noise_model=noise_model)
    else:
        noise_model = None
    if noise_model:
        simulator.set_options(noise_model=noise_model)
    new_circuit = transpile(circuit, simulator)
    shots=1024
    job=simulator.run(new_circuit, shots=shots)
    result = job.result()
    counts = result.get_counts()
    #print(counts)
    start_with_1 = 0
    for key, value in counts.items():
        if key.startswith('1'):
            start_with_1 += value
    inner_product=1-((2/shots)*(start_with_1))
#   print(f'inner product is {inner_product}')
    if inner_product<0:
        inner_product=0
    return inner_product

def only_inner_product(data1, data2):
    """ Comptutes the inner product of two data points by encoding them to quantum states. """
    if isinstance(data1, np.ndarray) and len(data1.shape) == 1:
        if len(data1) != len(data2):
            raise ValueError("Error: Vectors do not have the same length.")
        qubits_per_data = int(np.ceil(np.log2(len(data1))))
        amplitudes_1 = amplitude_encode_vector(data1)
        amplitudes_2 = amplitude_encode_vector(data2)
    elif isinstance(data1, np.ndarray) and len(data1.shape) == 2:
        if data1.shape != data2.shape:
            raise ValueError("Error: Images do not have the same dimensions.")
        rows, cols = data1.shape
        qubits_per_data = int(np.ceil(np.log2(rows * cols)))
        amplitudes_1 = amplitude_encode_image(data1)
        amplitudes_2 = amplitude_encode_image(data2)
        print('Got Image')
    else:
        raise TypeError("Unsupported data type. Provide either two vectors or two images.")
    amplitudes_1,amplitudes_2 = permute_arrays(amplitudes_1,amplitudes_2)
    qubits_per_image=qubits_per_data
    total_qubits= (2*qubits_per_image)+1
    total_bits=+1
    circuit = QuantumCircuit(total_qubits,total_bits)
    qubits_1=np.arange(0,qubits_per_image)
    qubits_2=np.arange(qubits_per_image,2*qubits_per_image)
    circuit.initialize(amplitudes_1, qubits_1.tolist())
    circuit.initialize(amplitudes_2, qubits_2.tolist())
    circuit.h(total_qubits-1)
    for _ in range(qubits_per_image):
        circuit.cswap(total_qubits-1, _+(qubits_per_image), _)
    circuit.h(total_qubits-1)
    circuit.measure(total_qubits-1, total_bits-1)
    #display(circuit.draw(output='mpl'))
    simulator = AerSimulator()
    new_circuit = transpile(circuit, simulator)
    shots=1024
    job=simulator.run(new_circuit, shots=shots)
    result = job.result()
    counts = result.get_counts()
    #print(counts)
    start_with_1 = 0
    for key, value in counts.items():
        if key.startswith('1'):
            start_with_1 += value
    inner_product=1-((2/shots)*(start_with_1))
    print(f'inner product is {inner_product}')
    if inner_product<0:
        inner_product=0
    return inner_product

def resize_image(image, size=(11, 11)):
  """ Resizes data so it can be represented using 7 qubits. """
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

def compute_partial_kernel(X, tasks, noise_level):
    """
    Computes the partial kernel matrix for a given set of tasks, which are pairs of indices.
    X (array-like): Input data.
    tasks (list of tuples): Pairs of indices (i, j) for which to compute the kernel.
    noise_level (float): The level of noise to apply during the quantum teleportation process.
    dict: A nested dictionary representing the partial kernel matrix.
    """
    K_partial = {}
    count = 0
    starting_time = time.time()
    for (i, j) in tasks:
        if i not in K_partial:
            K_partial[i] = {}
        K_partial[i][j] = teleport_and_inner_product(X[i], X[j], noise_level)
        if count == 0:
            print(f'Time taken for {rank} is {time.time() - starting_time}.')
        count += 1
        if i != j:
            if j not in K_partial:
                K_partial[j] = {}
            K_partial[j][i] = K_partial[i][j]
    print(f"Process {rank} called the similarity function {count} times.")
    return K_partial

def center_kernel_matrix(K):
   """ Centers a given kernel matrix using the double-centering formula. """
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n
    return K_centered

def compute_pca_metrics_and_svm(kernel_matrix, y, n_components_list):
    """ Computes PCA metrics and SVM accuracy for a list of different PCA component counts. Returns cumulative percentage variance (CPV), reconstruction error, and SVM accuracy and its standard deviation for each number of components. """
    metrics = {}
    kernel_matrix_centered = center_kernel_matrix(kernel_matrix)
    
    for n_components in n_components_list:
        kpca = KernelPCA(n_components=n_components, kernel='precomputed')
        X_kpca = kpca.fit_transform(kernel_matrix_centered)
        explained_variance = np.var(X_kpca, axis=0)
        cpv = np.cumsum(explained_variance) / np.sum(explained_variance)
        K_fit = X_kpca @ X_kpca.T
        reconstruction_error = np.mean((kernel_matrix_centered - K_fit) ** 2)
        svm = SVC(kernel='linear')
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        scores_svm = cross_val_score(svm, X_kpca, y, cv=skf, scoring='accuracy')
        mean_score_svm = scores_svm.mean()
        std_score_svm = scores_svm.std()
        metrics[n_components] = {
            'CPV': cpv[-1],
            'Reconstruction Error': reconstruction_error,
            'SVM Accuracy': mean_score_svm,
            'SVM Accuracy Std': std_score_svm
        }
    return metrics
