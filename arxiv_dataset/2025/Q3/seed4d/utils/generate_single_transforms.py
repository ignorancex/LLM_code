# Copyright (C) 2025 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Marius Kästingschäfer and Théo Gieruc
# ==============================================================================

"""
Write a single transform for nuscenes data (append folder names)
Instead of having a transform file containing the pose information per timestep, this script merges all of them into on file!

"""

import json
import argparse
import os
from matplotlib import pyplot as plt
import numpy as np


def load_json(path):
    with open(path, "r") as file:
        data = json.load(file)
    return data


def is_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


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
    vehicles = [
        f
        for f in os.listdir(args.data_dir + "step_0")
        if os.path.isdir(os.path.join(args.data_dir + "step_0", f))
    ]

    transforms = [
        "transforms.json",
        "transforms_background.json",
        "transforms_normalized.json",
    ]

    for transform in transforms:
        for vehicle in vehicles:
            for step in range(max_steps):
                
                #print(args.data_dir + "step_"+ str(step)+ '' + "/nuscenes/transforms/"+ transform)

                # processing dependents on sphere or non-sphere
                subfolder = "/nuscenes"
                if vehicle == "sphere":
                    number = ""
                    json_path = (
                        args.data_dir
                        + "step_"
                        + str(step)
                        + number
                        + "/sphere/transforms/"
                        + transform
                    )
                    subfolder = "/sphere"
                elif is_integer(vehicle) or 'ego_vehicle':
                    number = "/" + vehicle
                    json_path = (
                        args.data_dir
                        + "step_"
                        + str(step)
                        + number
                        + "/nuscenes/transforms/"
                        + transform
                    )
                else:
                    print(
                        "Vehicle input not allowed! Vehicle should be ego or vehicle number"
                    )

                data = load_json(json_path)

                for idx in range(len(data["frames"])):
                    data["frames"][idx]["file_path"] = (
                        "../step_"
                        + str(step)
                        + number
                        + subfolder
                        + data["frames"][idx]["file_path"][2:]
                    )
                    if args.depth:
                        data["frames"][idx]["depth_file_path"] = (
                            "../step_"
                            + str(step)
                            + number
                            + subfolder
                            + data["frames"][idx]["depth_file_path"][2:]
                        )
                    if args.semantic_segmentation:
                        data["frames"][idx]["semantic_segmentation_file_path"] = (
                            "../step_"
                            + str(step)
                            + number
                            + subfolder
                            + data["frames"][idx]["semantic_segmentation_file_path"][2:]
                        )
                    if args.instance_segmentation:
                        data["frames"][idx]["instance_segmentation_file_path"] = (
                            "../step_"
                            + str(step)
                            + number
                            + subfolder
                            + data["frames"][idx]["instance_segmentation_file_path"][2:]
                        )
                    if args.optical_flow:
                        data["frames"][idx]["optical_flow_file_path"] = (
                            "../step_"
                            + str(step)
                            + number
                            + subfolder
                            + data["frames"][idx]["optical_flow_file_path"][2:]
                        )
                    if transforms == "transforms_background.json" and args.masks == True:
                        data["frames"][idx]["mask_path"] = (
                            "../step_"
                            + str(step)
                            + number
                            + subfolder
                            + data["frames"][idx]["mask_path"][2:]
                        )

                if step == 0:
                    full_data = data
                else:
                    full_data["frames"] += data["frames"]

            with open(
                args.data_dir + "/" + str(vehicle) + "_combined_" + transform, "w"
            ) as file:
                json.dump(full_data, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Write a single transform across multiple steps across normalize and unnormalized transforms."
    )
    parser.add_argument("--data_dir", type=str, help="Path to the data directory.")
    parser.add_argument("--depth", type=bool, default=True)
    parser.add_argument("--semantic_segmentation", type=bool, default=True)
    parser.add_argument("--instance_segmentation", type=bool, default=True)
    parser.add_argument("--optical_flow", type=bool, default=True)
    parser.add_argument("--mask", type=bool, default=True)
    args = parser.parse_args()

    main(args)
