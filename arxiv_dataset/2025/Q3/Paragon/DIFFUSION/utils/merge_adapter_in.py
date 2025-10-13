import torch
from collections import OrderedDict
import os
import argparse

def merge_pt_files(file1_path, file2_path, output_path):
    # Load two .pt files
    state_dict1 = torch.load(file1_path, map_location=torch.device('cpu'))
    state_dict2 = torch.load(file2_path, map_location=torch.device('cpu'))
    
    # Create a new state_dict, first copying all contents of file1_path
    new_state_dict = OrderedDict(state_dict1)
    
    # Get the last 4 layers from file2_path (or fewer if total layers are less than 40)
    layers_to_replace = list(state_dict2.keys())[-4:]
    
    # Replace corresponding layers in file1_path with the last 40 layers from file2_path
    for key in layers_to_replace:
        if key in new_state_dict:
            new_state_dict[key] = state_dict2[key]
        else:
            print(f"Warning: {key} does not exist in file1_path, skipped")
    
    # Save the new state_dict to file
    torch.save(new_state_dict, output_path)
    print(f"Merged model has been saved to {output_path}")
    print(f"Replaced layers: {layers_to_replace}")

