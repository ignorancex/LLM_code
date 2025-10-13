import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.io as sio
import cv2

building = 'building29'

# Load the .mat file
for index in range(1, 101):
    print(f"index = {index}")
    data = sio.loadmat(f'C:\\Users\\CTLAB\\Desktop\\Random 4\\{building}_{index}')

    # Display the keys in the dictionary (these are the variable names in the .mat file)
    # print(data.keys())

    # Accessing a specific variable from the .mat file
    variable_name = 'CQI_ALL'  # Replace with your actual variable name
    if variable_name in data:
        CQI_data = data[variable_name]
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
    CQI_data = np.ravel(CQI_data)

    # Ensure that all arrays have the same length
    assert len(x_coords) == len(y_coords) == len(CQI_data)

    # Define the number of bins
    num_bins = 100

    # Create 2D histogram for weighted sum
    H, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=num_bins, weights=CQI_data)

    # Count the number of points in each bin (without weights)
    H_counts, _, _ = np.histogram2d(x_coords, y_coords, bins=num_bins)

    # Calculate the mean, setting bins with no data points to a very small value
    CQI_map = np.divide(H, H_counts, out=np.zeros_like(H), where=H_counts != 0)

    print(f"CQI index map shape: {CQI_map.shape}")
    np.save(f'Data/CQI/CQI_{building}_{index}.npy', CQI_map.T)

    # Calculate the extents to place the origin at the center
    height, width = CQI_map.shape
    extent = (-width // 2, width // 2, -height // 2, height // 2)
