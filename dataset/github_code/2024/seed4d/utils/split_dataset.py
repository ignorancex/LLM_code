# Copyright (C) 2025 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Marius Kästingschäfer and Théo Gieruc
# ==============================================================================

import sys, os
import numpy as np
import argparse
import json
from tqdm import tqdm


def get_transform_files(data_dir):
    transform_files = []
    for root, dirs, files in os.walk(data_dir):
        transform_files += [
            os.path.join(root, file)
            for file in files
            if file.endswith("transforms.json") # or file.endswith("transforms_ego.json")
        ]
    return transform_files


def split_dataset(data_dir, split_ratio):
    transform_files = get_transform_files(data_dir)

    num_files = len(transform_files)
    pbar = tqdm(transform_files, total=num_files)
    
    relevant_files = ['transforms_ego.json', 'transforms.json']

    for file in pbar:
        
        if 'lidar' in file or 'invisible' in file:
            continue
        
        if 'nuscenes' in file:
            num_frames = 7
        elif 'sphere' in file:
            num_frames = 100
            
        indices = np.arange(num_frames)
        np.random.shuffle(indices)
        num_train_frames = int(num_frames * split_ratio)
        train_indices = indices[:num_train_frames]
        test_indices = indices[num_train_frames:]

        paths = []
        # get path for transforms, transforms_ego, invisible transforms, invisible transforms_ego
        for relevant_file in relevant_files:
            path = file.replace("transforms.json", relevant_file)
            paths.append(path)
            path = path.replace("nuscenes", "nuscenes_invisible").replace("sphere", "sphere_invisible")
            paths.append(path)
        
        for path in paths: 
            with open(path, "r") as f:
                transforms = json.load(f)
            frames = transforms["frames"]

            train = transforms.copy()
            test = transforms.copy()
            train["frames"] = [frames[i] for i in train_indices]
            test["frames"] = [frames[i] for i in test_indices]
            train_file = path.replace(".json", "_train.json")
            test_file = path.replace(".json", "_test.json")
        
            with open(train_file, "w") as f:
                json.dump(train, f, indent=4)

            with open(test_file, "w") as f:
                json.dump(test, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--split_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    np.random.seed(args.seed)

    split_dataset(args.data_dir, args.split_ratio)
    
# Example: python3.8 utils/split_dataset.py --data_dir /seed4d/data/static
