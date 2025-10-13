import numpy as np
# Create the individual arrays
zeros = np.zeros(7, dtype=int)
ones = np.ones(3, dtype=int)
twos = np.full(100, 2, dtype=int)

# Concatenate the arrays
array_to_save = np.concatenate([zeros, ones, twos])
print(array_to_save)

file_path = 'cholec/cholec_labels_index.npy'

np.save(file_path, array_to_save)
print(f"Array saved to {file_path}")