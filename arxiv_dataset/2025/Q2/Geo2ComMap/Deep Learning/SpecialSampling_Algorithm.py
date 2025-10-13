import numpy as np
import os
import random

# Set the number of special points to sample
num_special_points = 150


# Source directories
PGmap_folder = 'Data_all/PGIso'
TP_folder = 'Data_all/TP'
RI_folder = 'Data_all/RI'
CQI_folder = 'Data_all/CQI'

# Target directories
Sparse_TP_folder = 'Data_all/Sparse_TP'
Sparse_RI_folder = 'Data_all/Sparse_RI'
Sparse_CQI_folder = 'Data_all/Sparse_CQI'

os.makedirs(Sparse_TP_folder, exist_ok=True)
os.makedirs(Sparse_RI_folder, exist_ok=True)
os.makedirs(Sparse_CQI_folder, exist_ok=True)

# Sort files in each folder
PGmap_files = sorted([f for f in os.listdir(PGmap_folder) if f.endswith('.npy')])
TP_files = sorted([f for f in os.listdir(TP_folder) if f.endswith('.npy')])
RI_files = sorted([f for f in os.listdir(RI_folder) if f.endswith('.npy')])
CQI_files = sorted([f for f in os.listdir(CQI_folder) if f.endswith('.npy')])

# Check if the number of files in each folder is the same
assert len(PGmap_files) == len(TP_files) == len(RI_files) == len(CQI_files) 

# Restrict the sampling range to the 120x120 area in the middle of the 128x128 matrix
start_idx = 4  
end_idx = 124 

# Iterate over all files
for i in range(len(PGmap_files)):
    # Load PG map
    PGmap_path = os.path.join(PGmap_folder, PGmap_files[i])
    PGmap_matrix = np.load(PGmap_path)
    PGmap_matrix_cropped = PGmap_matrix[start_idx:end_idx, start_idx:end_idx]
        
    # Find 2*num_special_points points with the highest gradient sum
    gradient_matrix = np.gradient(PGmap_matrix_cropped)  # Calculate the gradient
    grad_sum = np.sum(np.abs(gradient_matrix), axis=0)  # Compute the sum of the absolute values of the gradient components
    sorted_indices_flat = np.argsort(grad_sum.ravel())  # Sort the indices based on the gradient sum
    top_indices_flat = sorted_indices_flat[-(2*num_special_points):]  # Select the top 2*num_special_points indices
 
    
    # Randomly select num_special_points points from the 2*num_special_points points
    random_top_indices_flat = np.random.choice(top_indices_flat, size=num_special_points, replace=False)
    random_top_indices = np.unravel_index(random_top_indices_flat, PGmap_matrix_cropped.shape)  

    # Randomly select num_special_points points from the remaining points
    all_indices_flat = np.arange(PGmap_matrix_cropped.size)  
    available_indices_flat = np.setdiff1d(all_indices_flat, top_indices_flat)  
    random_indices_flat = np.random.choice(available_indices_flat, size=num_special_points, replace=False)  
    random_indices = np.unravel_index(random_indices_flat, PGmap_matrix_cropped.shape)  

    # Combine the two sets of indices
    combined_indices = (np.concatenate((random_top_indices[0], random_indices[0])),
                        np.concatenate((random_top_indices[1], random_indices[1])))

    # Sample the TP, RI, and CQI matrices using the combined indices
    TP_matrix = np.load(os.path.join(TP_folder, TP_files[i]))
    RI_matrix = np.load(os.path.join(RI_folder, RI_files[i]))
    CQI_matrix = np.load(os.path.join(CQI_folder, CQI_files[i]))

    # Create sparse matrices
    Sparse_TP_matrix = np.zeros_like(TP_matrix)
    Sparse_RI_matrix = np.zeros_like(RI_matrix)
    Sparse_CQI_matrix = np.zeros_like(CQI_matrix)

    Sparse_TP_matrix[combined_indices] = TP_matrix[combined_indices]
    Sparse_RI_matrix[combined_indices] = RI_matrix[combined_indices]
    Sparse_CQI_matrix[combined_indices] = CQI_matrix[combined_indices]

    # Save the sparse matrices
    np.save(os.path.join(Sparse_TP_folder, TP_files[i]), Sparse_TP_matrix)
    np.save(os.path.join(Sparse_RI_folder, RI_files[i]), Sparse_RI_matrix)
    np.save(os.path.join(Sparse_CQI_folder, CQI_files[i]), Sparse_CQI_matrix)

print("Sparse matrices created successfully.")
