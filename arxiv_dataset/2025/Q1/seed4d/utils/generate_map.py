# Copyright (C) 2025 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Marius Kästingschäfer and Théo Gieruc
# ==============================================================================

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import json
import glob
import yaml
import os
import argparse


def get_files_and_vehilce_ids(folder_path):

    file_pattern = "*_combined_transforms.json"
    matching_files = glob.glob(folder_path + file_pattern)

    file_names = []
    vehicle_ids = []

    for file_path in matching_files:
        file_name = os.path.basename(file_path)
        first_underscore_index = file_name.index("_")
        vehicle_id = file_name[:first_underscore_index]

        file_names.append(file_name)
        vehicle_ids.append(vehicle_id)

    return file_names, vehicle_ids


def extract_trajectory(path):

    with open(path, "r") as json_file:
        data = json.load(json_file)

    x_trajectory, y_trajectory, z_trajectory = [], [], []

    for frame in data["frames"]:

        # using only the first camera for plotting
        file_path = frame["file_path"]
        if "0_rgb.png" in file_path:

            # Extract the transformation matrix
            transform_matrix = frame["transform_matrix"]

            # Extract the translation values (x, y, z)
            x_translation = transform_matrix[0][3]
            y_translation = transform_matrix[1][3]
            z_translation = transform_matrix[2][3]

            # Append translation values to respective lists
            x_trajectory.append(x_translation)
            y_trajectory.append(y_translation)
            z_trajectory.append(z_translation)

    return [x_trajectory, y_trajectory, z_trajectory]


def plot_positions(trajectories, vehicle_ids, data_dir):

    colors = plt.cm.tab10(np.linspace(0, 1, len(vehicle_ids)))
    time = np.linspace(0, 1, len(trajectories[0][0]))

    plt.figure(figsize=(8, 8))
    for idx, trajectory in enumerate(trajectories):
        darkened_color = tuple(np.array(colors[idx]) * (time[:, np.newaxis]))
        _, y_trajectory, z_trajectory = trajectory
        plt.scatter(
            y_trajectory,
            z_trajectory,
            c=darkened_color,
            alpha=0.8,
            marker="o",
            linestyle="-",
        )
        plt.text(
            np.mean(y_trajectory),
            np.mean(z_trajectory) * 0.9993,
            vehicle_ids[idx],
            fontsize=12,
            ha="right",
            va="bottom",
        )

    plt.title("Vehicle Paths")
    plt.xlabel("Y-Axis (in Meters)")
    plt.ylabel("Z-Axis (in Meters)")
    plt.grid(True)
    plt.gca().invert_yaxis()  # Invert y-axis to match typical Cartesian coordinates
    plt.savefig(data_dir + "/overview_map.png")


def write_positions(trajectories, vehicle_ids, times, data_dir):
    timesteps = np.diff(times)

    vehicle_ids_df = []
    times_df = []
    positions_df = []
    speeds_df = []

    for idx, trajectory in enumerate(trajectories):

        # trajectory per vehicle
        vehicle_id = vehicle_ids[idx]
        _, y_trajectory, z_trajectory = trajectory

        # distance traveled
        diff_y = np.diff(y_trajectory)
        diff_z = np.diff(z_trajectory)
        distances = np.sqrt(diff_y**2 + diff_z**2)

        # compute speed
        speed_kmh = (distances / (timesteps + 1e-9)) * 3.6

        vehicle_ids_df.extend([vehicle_id] * len(times))
        times_df.extend(times)
        positions_df.extend([value for value in zip(y_trajectory, z_trajectory)])
        speeds_df.extend(np.insert(speed_kmh, 0, 0))

    data = {
        "vehicle_id": vehicle_ids_df,
        "timepoint": times_df,
        "position": positions_df,
        "speed": speeds_df,
    }

    df = pd.DataFrame(data)
    df.to_json(data_dir + "/positions.json", orient="records")


def main(args):

    data_dir = args.data_dir
    file_names, vehicle_ids = get_files_and_vehilce_ids(data_dir)

    trajectories = []
    for file_name in file_names:
        path = os.path.join(data_dir, file_name)
        trajectories.append(extract_trajectory(path))

    plot_positions(trajectories, vehicle_ids, data_dir)

    with open(data_dir + "/timesteps.json", "r") as file:
        times = yaml.load(file, Loader=yaml.FullLoader)

    times = list(times.values())
    write_positions(trajectories, vehicle_ids, times, data_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Write a map containing all vehicle trajectories."
    )
    parser.add_argument("--data_dir", type=str, help="Path to the data directory.")
    args = parser.parse_args()
    main(args)
