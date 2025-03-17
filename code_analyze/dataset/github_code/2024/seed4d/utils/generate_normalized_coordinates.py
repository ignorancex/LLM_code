# Copyright (C) 2025 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Marius Kästingschäfer and Théo Gieruc
# ==============================================================================

"""
Mean 0 all coordinates and normalize with largest radius (such that values are centered at 0 and have size -1 to +1)
"""

import os
import json
import argparse
from matplotlib import pyplot as plt
import numpy as np


def normalize(arr):
    return np.array(arr) - np.mean(arr), np.mean(arr)


def max_radius(values):
    return max(abs(min(values)), abs(max(values)))


def main(args):

    max_steps = (
        max(
            (
                int(item[5:])
                for item in os.listdir(args.data_dir)
                if item.startswith("step_")
            ),
            default=0,
        )
        + 1
    )

    values = {"x": [], "y": [], "z": []}

    # elements_mapping = {'trajectory_ego': ['nuscenes'], 'sphere': ['sphere'], 'both': ['nuscenes', 'sphere']}
    # elements = elements_mapping.get(args.elements, [])

    for step in range(max_steps):

        folder_path = args.data_dir + "/step_" + str(step) + "/"
        elements = [
            f
            for f in os.listdir(folder_path)
            if os.path.isdir(os.path.join(folder_path, f))
        ]

        for element in elements:

            # load jsons
            if element.isdigit() or element == 'ego_vehicle':
                json_file_path = (
                    args.data_dir
                    + "/step_"
                    + str(step)
                    + "/"
                    + element
                    + "/nuscenes/transforms/transforms.json"
                )
            else:
                print('Unknown element ', element, ' found')

            with open(json_file_path, "r") as json_file:
                json_data = json.load(json_file)

            # collect all x, y, z values
            for frame in json_data["frames"]:
                transform_matrix = np.array(frame["transform_matrix"])
                for i, dim in enumerate(["x", "y", "z"]):
                    values[dim].append(transform_matrix[i, 3])

    # Convert lists to NumPy arrays
    values_x = np.array(values["x"])
    values_y = np.array(values["y"])
    values_z = np.array(values["z"])

    values_x, avg_x = normalize(values_x)
    values_y, avg_y = normalize(values_y)
    values_z, avg_z = normalize(values_z)

    max_radius_x = max_radius(values_x)
    max_radius_y = max_radius(values_y)
    max_radius_z = max_radius(values_z)
    radius = max([max_radius_x, max_radius_y, max_radius_z])

    # load all config files and replace relevant values
    max_steps = (
        max(
            (
                int(item[5:])
                for item in os.listdir(args.data_dir)
                if item.startswith("step_")
            ),
            default=0,
        )
        + 1
    )
    all_folders = [
        item
        for item in os.listdir(args.data_dir + "/step_0/")
        if os.path.isdir(os.path.join(args.data_dir + "/step_0/", item))
    ]

    for element in all_folders:

        for step in range(max_steps):

            if element == "nuscenes" or element == "sphere":
                json_file_path = (
                    args.data_dir
                    + "/step_"
                    + str(step)
                    + "/"
                    + element
                    + "/transforms/transforms.json"
                )
            else:
                json_file_path = (
                    args.data_dir
                    + "/step_"
                    + str(step)
                    + "/"
                    + element
                    + "/nuscenes/transforms/transforms.json"
                )

            with open(json_file_path, "r") as json_file:
                json_data = json.load(json_file)

            for idx in range(len(json_data["frames"])):
                json_data["frames"][idx]["transform_matrix"][0][3] = (
                    json_data["frames"][idx]["transform_matrix"][0][3] - avg_x
                ) / radius
                json_data["frames"][idx]["transform_matrix"][1][3] = (
                    json_data["frames"][idx]["transform_matrix"][1][3] - avg_y
                ) / radius
                json_data["frames"][idx]["transform_matrix"][2][3] = (
                    json_data["frames"][idx]["transform_matrix"][2][3] - avg_z
                ) / radius

            with open(json_file_path[:-5] + "_normalized.json", "w") as json_file:
                json.dump(json_data, json_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Execute to normalize the existing poses."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="../Generator/config/",
        default="../Generator/config/",
    )
    parser.add_argument(
        "--elements",
        type=str,
        default="trajectory_ego",
        help="Over which values to normalize: trajectory_ego, sphere or both",
    )
    args = parser.parse_args()

    main(args)
