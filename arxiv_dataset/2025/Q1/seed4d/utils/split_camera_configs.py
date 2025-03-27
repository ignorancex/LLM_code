# Copyright (C) 2025 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Marius Kästingschäfer and Théo Gieruc
# ==============================================================================

import json
import argparse
import os

def get_elements_n_to_m(data, n, m):
    result = {}
    for key, value in data.items():
        result[key] = value[n:m]

    return result


def save_split_data(split_data, iteration, name):
    os.makedirs(name, exist_ok=True)
    file_name = f"{name}/split_{name}_iteration-{iteration}.json"

    with open(file_name, "w") as output_file:
        json.dump(split_data, output_file, indent=4)


def main(json_file_path):
    # Load the JSON data
    name = os.path.basename(os.path.splitext(json_file_path)[0])
    with open(json_file_path, "r") as file:
        data = json.load(file)

    split_into = 25

    for key, value in data.items():
        total_elements = len(value)
    N_per_split = total_elements // split_into

    # Splitting the data and saving after each iteration
    for idx in range(split_into):
        split_data = get_elements_n_to_m(data, idx, idx + N_per_split)
        save_split_data(split_data, idx, name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split JSON data into 10 smaller files"
    )
    parser.add_argument("--json_file", type=str, help="Path to the JSON file")

    args = parser.parse_args()
    main(args.json_file)
