# Copyright (C) 2025 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Marius Kästingschäfer and Théo Gieruc
# ==============================================================================

'''
Convert the data into a format suitable e.g., for 6Img-to-3D: https://6img-to-3d.github.io/
'''

import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image


def replace_word_in_yaml(data, target, replacement):
    """
    Recursively replace a target word with a replacement in a nested YAML structure.
    """
    if isinstance(data, dict):
        return {k: replace_word_in_yaml(v, target, replacement) for k, v in data.items()}
    elif isinstance(data, list):
        return [replace_word_in_yaml(item, target, replacement) for item in data]
    elif isinstance(data, str):
        return data.replace(target, replacement)
    else:
        return data

def transform_transforms_file(data, category):
    """
    Transform the old JSON structure into a new format.

    Parameters:
    - input_json_path: str, path to the input JSON file.
    - category: str, ego_vehicle or sphere.
    """

    # Process each frame to adjust file paths and keep only the necessary information
    if category == 'sphere':
        transformed_data = replace_word_in_yaml(data, 'sensors', 'images')
    elif category == 'ego_vehicle':
        
        # Initialize the transformed data structure
        transformed_data = {
            "camera_model": data["camera_model"],
            "fl_x": data["frames"][0]["fl_x"],
            "fl_y": data["frames"][0]["fl_y"],
            "cx": data["frames"][0]["cx"],
            "cy": data["frames"][0]["cy"],  # Retain original 'cy' from the first frame
            "w": data["frames"][0]["w"],    # Retain original width from the first frame
            "h": data["frames"][0]["h"],    # Retain original height from the first frame
            "k1": data["k1"],
            "k2": data["k2"],
            "p1": data["p1"],
            "p2": data["p2"],
            "frames": []
        }
        
        ego_indexes = [0,1,2,3,5,6]
        for idx, frame in enumerate(data["frames"]):
            if idx in ego_indexes:
                if idx == 5: idx = 4
                if idx == 6: idx = 5
                new_frame = {
                    "file_path": f"../images/{idx}_rgb.png",  # Updated file path
                    "transform_matrix": frame["transform_matrix"]  # Keep the transform matrix
                }
                transformed_data["frames"].append(new_frame)
    else:
        print(f'{category} is not defined.')
            
    return transformed_data

def main(args):
    
    input_path = args.input_path
    origin_output_path = args.output_origin_path
    intermediate_output_path = args.output_intermediate_path
    spawnpoints = args.spawnpoints
    ego_indexes = [0,1,2,3,5,6]
    sensors = ['rgb', 'depth']
    sphere_indexes = range(0, 100)
    
    # prcoess all spawnpoints
    for idx in tqdm(range(1, spawnpoints+1), desc=f"Processing spawnpoints"):
        
        nuscenes_origin = f'{input_path}/spawn_point_{idx}/step_0/ego_vehicle/nuscenes_invisible'
        sphere_origin = f'{input_path}/spawn_point_{idx}/step_0/ego_vehicle/sphere_invisible'
        
        # convert images of ego_vehicle
        for ego_idx in ego_indexes:
            ego_path = f'{nuscenes_origin}/sensors/{ego_idx}_rgb.png'
            # load image
            ego_image = Image.open(ego_path) if os.path.exists(ego_path) else print(f"Image not found: {ego_path}")
            # save image
            if ego_idx == 5: ego_idx = 4
            if ego_idx == 6: ego_idx = 5
            ego_save_to = f'{origin_output_path}/{intermediate_output_path}/spawn_point_{idx}/step_0/nuscenes/images/{ego_idx}_rgb.png'
            os.makedirs(os.path.dirname(ego_save_to), exist_ok=True)
            ego_image.save(ego_save_to)

        # convert images of sphere
        for sensor in sensors: 
            for sphere_idx in sphere_indexes:
                sphere_path = f'{sphere_origin}/sensors/{sphere_idx}_{sensor}.png'
                # load image
                sphere_image = (Image.open(sphere_path) if os.path.exists(sphere_path) else print(f"Image not found: {sphere_path}")) 
                # save image
                sphere_save_to = f'{origin_output_path}/{intermediate_output_path}/spawn_point_{idx}/step_0/sphere/images/{sphere_idx}_{sensor}.png'
                os.makedirs(os.path.dirname(sphere_save_to), exist_ok=True)
                sphere_image.save(sphere_save_to)

        # convert transforms of ego_vehicle
        file_endings = ['transforms_ego.json', 'transforms_ego_train.json', 'transforms_ego_test.json', 'transforms.json', 'transforms_train.json', 'transforms_test.json']
        for file_ending in file_endings:
        
            transforms_ego_path = f'{nuscenes_origin}/transforms/{file_ending}'
            transform_sphere_save_to = f'{origin_output_path}/{intermediate_output_path}/spawn_point_{idx}/step_0/nuscenes/transforms/{file_ending}'
            # load json
            with open(transforms_ego_path, 'r') as file: transforms_file = json.load(file)
            processed_transforms_ego = transform_transforms_file(transforms_file, category='ego_vehicle')
            # save to disk
            os.makedirs(os.path.dirname(transform_sphere_save_to), exist_ok=True)
            with open(transform_sphere_save_to, 'w') as file: json.dump(processed_transforms_ego, file, indent=4)

            # convert transforms of sphere (does not need processing per timestep)
            transforms_sphere_path = f'{sphere_origin}/transforms/{file_ending}'
            transform_sphere_save_to = f'{origin_output_path}/{intermediate_output_path}/spawn_point_{idx}/step_0/sphere/transforms/{file_ending}'
            # load json
            with open(transforms_sphere_path, 'r') as file: transforms_file = json.load(file)
            #processed_transforms_sphere = transforms_file
            processed_transforms_sphere = transform_transforms_file(transforms_file, category='sphere')
            # save to disk
            os.makedirs(os.path.dirname(transform_sphere_save_to), exist_ok=True)
            with open(transform_sphere_save_to, 'w') as file: json.dump(processed_transforms_sphere, file, indent=4)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default='/seed4d/data/static/Town02/ClearNoon/vehicle.mini.cooper_s')
    parser.add_argument("--output_origin_path", type=str, default='/seed4d_xz/Town02/')
    parser.add_argument("--output_intermediate_path", type=str, default='ClearNoon/vehicle.mini.cooper_s')
    parser.add_argument("--spawnpoints", type=int, default=101)

    args = parser.parse_args()
    main(args)

# Example: python3.8 utils/converter.py