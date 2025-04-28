import os
import sys
import numpy as np
import random
import time
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from pathlib import Path
import pandas as pd
from scipy import stats, linalg
from sklearn.metrics.pairwise import euclidean_distances
from scipy.optimize import fsolve, least_squares, minimize, curve_fit
from mpl_toolkits.mplot3d import Axes3D
from array import array
from itertools import product
import cvxpy as cp
from sympy import symbols, Eq, solve
from os.path import join
from plotly.subplots import make_subplots
from plotly.validators.scatter.marker import SymbolValidator
from sklearn.manifold import TSNE
from joblib import Parallel, delayed
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import h5py
import tensorflow as tf

current_dir = os.getcwd()
sys.path.append(os.path.dirname(current_dir))
from EOT_tools import *

random.seed = 1

# Data processing

class LabeledData:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

def map_labels(labels, dictionary):
    return np.array([dictionary[int(i[0])] for i in labels])

def find_indices(numbers, target):
    return [index for index, value in enumerate(numbers) if value == target]

def load_and_process_data(file_path, label_list):
    with h5py.File(file_path, 'r') as f:
        data_file = f['data']
        label_file = f['label']
        dataset = data_file[:]
        all_labels = label_file[:]
    
    reduced_data_indices = []
    reduced_data_labels = []
    for label in label_list:
        indices = find_indices(all_labels, label)
        reduced_data_indices.append(indices)
        reduced_data_labels.append(np.dot(np.ones(len(indices)), label))
    
    reduced_labeled_data = []
    for indices in reduced_data_indices:
        labeled_data_list = [LabeledData(dataset[j], all_labels[j]) for j in indices]
        reduced_labeled_data.append(np.array(labeled_data_list))
    
    return np.transpose(np.array(reduced_labeled_data))

# Constants
LABEL_OBJ_DICT = {0: 'plane', 2: 'bed', 17: 'guitar', 22: 'television', 37: 'vases'}
LABEL_LIST = [0, 2, 17, 22, 37]

# Load and process data
batch_list = load_and_process_data('classification_example/pointcloud-c/clean.h5', LABEL_LIST)
dropout_local_1_batches = load_and_process_data('classification_example/pointcloud-c/dropout_local_1.h5', LABEL_LIST)
dropout_local_4_batches = load_and_process_data('classification_example/pointcloud-c/dropout_local_4.h5', LABEL_LIST)
dropout_global_4_batches = load_and_process_data('classification_example/pointcloud-c/dropout_global_4.h5', LABEL_LIST)

# Point cloud completion tool
def completion(source, references, regularization, method, dim, support_size):
    if method == 'Entropy':
        analysis_method = ent_analysis
        synthesis_method = entropic_synthesis
    elif method == 'Sinkhorn':
        analysis_method = sink_analysis
        synthesis_method = sinkhorn_synthesis
    
    source_points = source.points
    reference_points = [ref.points for ref in references]
    coefficients = analysis_method(reference_points, source_points, regularization)
    
    iterations = 60
    stepsize = 0.08
    uniform_points = np.random.rand(support_size, dim)
    uniform_masses = np.dot(np.ones(support_size), 1 / support_size)
    base_measure = measure(uniform_points, uniform_masses)
    completed_pointcloud = synthesis_method(regularization, references, base_measure, coefficients, stepsize, iterations)
    
    return completed_pointcloud

# Prepare measures
planes = np.transpose(batch_list)[0]
dropout_local_4_planes = np.transpose(dropout_local_4_batches)[0]
dropout_global_4_planes = np.transpose(dropout_global_4_batches)[0]

def prepare_measures(data_array):
    measures = []
    for data in data_array:
        num_points = np.shape(data.data)[0]
        masses = np.dot(np.ones(num_points), 1 / num_points)
        measures.append(measure(data.data, masses))
    return measures

plane_measures = prepare_measures(planes)
dropout_local_4_plane_measures = prepare_measures(dropout_local_4_planes)
dropout_global_4_plane_measures = prepare_measures(dropout_global_4_planes)

references = plane_measures[0:5]

def compute_and_plot_completion(uncorrupted_data_array, corrupted_data_array, index, method, reference_measures, regularization, dim, support_size):
    recon = completion(corrupted_data_array[index], reference_measures, regularization, method, dim, support_size)
    A = recon.points
    A_mass = np.dot(np.ones(len(A)), 1 / len(A))
    original = uncorrupted_data_array[index].points
    original_mass = np.dot(np.ones(len(original)), 1 / len(original))
    M = euclidean_distances(A, original)
    OT_cost = ot.emd2(A_mass, original_mass, M)
    
    x_min, x_max = A[:, 0].min(), A[:, 0].max()
    y_min, y_max = A[:, 1].min(), A[:, 1].max()
    z_min, z_max = A[:, 2].min(), A[:, 2].max()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(A[:, 0], A[:, 1], A[:, 2], s=5, c='blue', alpha=0.8)
    ax.set_title(f'Reconstructed Pointcloud with Local Dropout \n ({method}, $\\epsilon={regularization}$, $OT_2$-cost={OT_cost:.4f})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    
    stoptime = time.time()
    plt.savefig(f"completion_{method}_reg{regularization}_index{stoptime}.png")
    plt.close(fig)

# Compute and plot completions
compute_and_plot_completion(plane_measures, dropout_global_4_plane_measures, 12, 'Entropy', references, 0.003, 3, 1024)
compute_and_plot_completion(plane_measures, dropout_global_4_plane_measures, 12, 'Sinkhorn', references, 0.003, 3, 1024)
compute_and_plot_completion(plane_measures, dropout_local_4_plane_measures, 12, 'Entropy', references, 0.003, 3, 1024)
compute_and_plot_completion(plane_measures, dropout_local_4_plane_measures, 12, 'Sinkhorn', references, 0.003, 3, 1024)