# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 21:49:52 2025

@author: seema
"""

import os

# Base output directory
output_base_path = <path/to/combined/wav/files>
output_list_file = os.path.join(output_base_path, "processed_files.txt") # Path where text file will be written. Modify for alternative path.

#%%
# Ensure the output directory exists
if not os.path.exists(output_base_path):
    print("Output directory does not exist. Exiting.")
    exit()
#%%
# List to store file paths
processed_files = []

# Traverse the output directory and collect .wav files
for subfolder in ["ad", "cn"]:
    subfolder_path = os.path.join(output_base_path, subfolder)
    
    if os.path.exists(subfolder_path):
        for file in os.listdir(subfolder_path):
            if file.endswith(".wav"):
                file_path = os.path.join(subfolder_path, file)
                processed_files.append(file_path)
#%%
# Write the list to a text file
with open(output_list_file, "w") as f:
    for file_path in processed_files:
        f.write(file_path + "\n")

print(f"List of processed files saved to: {output_list_file}")
#%%