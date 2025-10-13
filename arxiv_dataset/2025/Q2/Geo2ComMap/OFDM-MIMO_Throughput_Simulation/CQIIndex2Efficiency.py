import numpy as np
import os

# 38.214 - Table 5.2.2.1-2: 4-bit CQI Table 1
cqi_to_efficiency = {
    0: 0.0,
    1: 0.1523,
    2: 0.2344,
    3: 0.3770,
    4: 0.6016,
    5: 0.8770,
    6: 1.1758,
    7: 1.4766,
    8: 1.9141,
    9: 2.4063,
    10: 2.7305,
    11: 3.3223,
    12: 3.9023,
    13: 4.5234,
    14: 5.1152,
    15: 5.5547
}

input_directory = 'Data/CQI_Index'  # Replace with the path to your directory
output_directory = 'Data/CQI'  # Replace with the path to save the modified files

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Loop through all .npy files in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith('.npy'):
        # Load the .npy file
        file_path = os.path.join(input_directory, filename)
        matrix = np.load(file_path)

        # Convert the CQI index to efficiency
        efficiency_matrix = np.vectorize(cqi_to_efficiency.get)(matrix)

        # Save the new matrix to the output directory
        output_file_path = os.path.join(output_directory, filename)
        np.save(output_file_path, efficiency_matrix)

        print(f"Converted {filename} and saved to {output_file_path}")
