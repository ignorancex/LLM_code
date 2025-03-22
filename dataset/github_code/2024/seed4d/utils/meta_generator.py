# Copyright (C) 2025 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Marius Kästingschäfer and Théo Gieruc
# ==============================================================================

"""
Exectues: generate_masks.py, generate_normalized_coordinates.py, generate_single_transforms.py
"""

import sys, os

cwd = "/mariustheo/Generator/"
sys.path.append(cwd)
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import subprocess
import logging
import yaml
import argparse
import time
from time import sleep
from datetime import datetime


def main(args):

    start_time = time.time()

    # configure logging
    (
        logging.basicConfig(level=logging.ERROR)
        if args.quiet
        else logging.basicConfig(level=logging.INFO)
    )
    logger = logging.getLogger(__name__)

    if args.add_date:
        time_string = datetime.now().strftime("%Y%m%d%H%M%S")
        args.data_dir = args.data_dir + "/" + time_string
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    # generate data
    command = [
        "python3.8",
        "generator.py",
        "--config",
        args.config,
        "--data_dir",
        args.data_dir,
        "--carla_executable",
        args.carla_executable,
    ]
    process = subprocess.Popen(command, cwd=cwd)
    process.wait()

    # obtain file path
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    save_path = os.path.join(
        cwd,
        args.data_dir,
        config["map"],
        config["weather"],
        config["vehicle"],
        "spawn_point_" + str(config["spawn_point"][0]),
    )
    args.data_dir = save_path
    logger.info(
        f" Congratulations! Your data has been written to this location: {args.data_dir}"
    )

    if args.add_date:
        logger.info(" Generate normalize coordinates ...")
        command = [
            "python3.8",
            "generate_normalized_coordinates.py",
            "--data_dir",
            args.data_dir,
            "--elements",
            "nuscenes",
        ]
        process = subprocess.Popen(command, cwd=cwd + "/utils")
        process.wait()

    if args.vehicle_masks:
        logger.info(" Generate vehicle masks ...")
        args.data_dir = save_path + "/"
        command = [
            "python3.8",
            "generate_masks.py",
            "--data_dir",
            args.data_dir,
        ]
        process = subprocess.Popen(command, cwd=cwd + "/utils")
        process.wait()

    if args.combine_transforms:
        logger.info(" Combining transform files across timepoints ...")
        command = [
            "python3.8",
            "generate_single_transforms.py",
            "--data_dir",
            args.data_dir,
        ]
        process = subprocess.Popen(command, cwd=cwd + "utils")
        process.wait()

    if args.map:
        logger.info(" Generate overview map and single positions file ...")
        command = ["python3.8", "generate_map.py", "--data_dir", args.data_dir]
        process = subprocess.Popen(command, cwd=cwd + "utils")
        process.wait()

    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)

    # show elapse time
    elapsed_time = time.time() - start_time
    logger.info(
        f" Total elapsed time: {int(elapsed_time/60)} minutes {round((elapsed_time - int(elapsed_time/60)*60), 0)} seconds"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Execute to perform generator, generate_mask, generate_normalized_coordinates and generate_single_transforms in one script."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config.yaml",
        help="Path to the config file",
    )
    parser.add_argument(
        "--data_dir", type=str, default="data", help="Path to the data directory"
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Disable progress bar and all logging except for errors",
    )
    parser.add_argument(
        "--carla_executable",
        type=str,
        default="CarlaUE4.sh",
        help="Path to the CARLA executable",
    )
    parser.add_argument(
        "--add_date",
        type=bool,
        default=False,
        help="Whether the daytime is added in front of the folder path",
    )
    parser.add_argument(
        "--normalize_coords",
        type=bool,
        default=True,
        help="Whether to generate normalized coordinates",
    )
    parser.add_argument(
        "--vehicle_masks",
        type=bool,
        default=True,
        help="Generate vehicle masks + vehicle only images",
    )
    parser.add_argument(
        "--combine_transforms",	
        type=bool,
        default=False,
        help="Write a single transform across timepoints",
    )
    parser.add_argument(
        "--map",	
        type=bool,
        default=False,
        help="Creates a map overview and a single file containing all positions",
    )
    
    args = parser.parse_args()

    main(args)
