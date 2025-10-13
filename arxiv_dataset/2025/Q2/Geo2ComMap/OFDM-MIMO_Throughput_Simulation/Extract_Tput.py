"""
This script extracts the throughput data from the .mat files and saves the data as numpy arrays.
This script also creates a sparse version of the throughput map by randomly sampling a subset of the data points.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.io as sio
import cv2

# Set the building name
building = 'building29'

# Load the .mat file
for index in range(1, 101):
    print(f"index = {index}")
    data = sio.loadmat(f'C:\\Users\\CTLAB\\Desktop\\Random 4\\{building}_{index}')

    # Display the keys in the dictionary (these are the variable names in the .mat file)
    # print(data.keys())

    # Accessing a specific variable from the .mat file
    variable_name = 'Throughput_ALL'

    if variable_name in data:
        Throughput_data = data[variable_name]
        # print(Throughput_data)
    else:
        print(f"{variable_name} not found in the .mat file.")
    
    # Load the .mat file
    data = sio.loadmat(f'C:\\Users\\CTLAB\\Desktop\\finished data\\{building}_{index}\\position.mat')

    # Display the keys in the dictionary (these are the variable names in the .mat file)
    # print(data.keys())

    # Accessing a specific variable from the .mat file
    variable_name = 'agent'  # Replace with your actual variable name
    if variable_name in data:
        coordinates = data[variable_name]
        # print(data[variable_name])
    else:
        print(f"{variable_name} not found in the .mat file.")
    
    x_coords = coordinates[:, 0]  # All rows, first column (x values)
    y_coords = coordinates[:, 1]  # All rows, second column (y values)

    # Ensure the inputs are 1D arrays
    x_coords = np.ravel(x_coords)
    y_coords = np.ravel(y_coords)
    Throughput_data = np.ravel(Throughput_data)

    # Ensure that all arrays have the same length
    assert len(x_coords) == len(y_coords) == len(Throughput_data)

    # Define the number of bins
    num_bins = 100

    # Create 2D histogram for weighted sum
    H, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=num_bins, weights=Throughput_data)

    # Count the number of points in each bin (without weights)
    H_counts, _, _ = np.histogram2d(x_coords, y_coords, bins=num_bins)

    # Calculate the mean, setting bins with no data points to a very small value
    Throughput_map = np.divide(H, H_counts, out=np.zeros_like(H), where=H_counts != 0)
    # np.save(f'Throughput_map_1e4_depth3_100.npy', Throughput_map.T)

    Throughput_map = cv2.resize(Throughput_map, (128, 128), interpolation=cv2.INTER_AREA)

    # Randomly sample a sparse throughput map
    Sparse_throughput_map = np.zeros(shape=(128, 128))

    # Set the number of points to sample
    num_points = 200
    print(f"Number of sampled points: {num_points}")

    rows = np.random.randint(0, Throughput_map.T.shape[0], num_points)
    cols = np.random.randint(0, Throughput_map.T.shape[1], num_points)

    for row, col in zip(rows, cols):
        Sparse_throughput_map[row, col] = Throughput_map.T[row, col]

    # Optional: Apply interpolation or smoothing to fill in gaps in the sparse map
    Sparse_throughput_map = cv2.resize(Sparse_throughput_map, (128, 128), interpolation=cv2.INTER_AREA)

    print(f"Sparse throughput map shape: {Sparse_throughput_map.shape}")
    np.save(f'Data/Sparse_TP/Sparse_throughput_map_{building}_{index}.npy', Sparse_throughput_map)

    print(f"Throughput map shape: {Throughput_map.shape}")
    np.save(f'Data/TP/Throughput_map_{building}_{index}.npy', Throughput_map.T)

    # Calculate the extents to place the origin at the center
    height, width = Throughput_map.shape
    extent = (-width // 2, width // 2, -height // 2, height // 2)
