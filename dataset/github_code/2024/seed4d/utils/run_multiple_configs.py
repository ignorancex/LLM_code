# Copyright (C) 2025 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Marius Kästingschäfer and Théo Gieruc
# ==============================================================================

"""
Generates data for all configs within a folder
"""

import argparse
import os, sys

cwd = "/mariustheo/Generator/"
sys.path.append(cwd)
import glob
import subprocess
import logging
import argparse
import re


def main(args):

    # configure logging
    (
        logging.basicConfig(level=logging.ERROR)
        if args.quiet
        else logging.basicConfig(level=logging.INFO)
    )
    logger = logging.getLogger(__name__)

    yaml_file_paths = glob.glob(f"{args.config_dir}/*.yaml")

    for idx, yaml_file_path in enumerate(yaml_file_paths):

        logger.info(
            f"#########  Creating scene {idx} out of {len(yaml_file_paths)}  ###"
        )

        file_name = os.path.basename(yaml_file_path)
        cleaned_file_name = re.sub("_config.yaml", "", file_name)
        logger.info(f"Generating data for {cleaned_file_name}...")

        # generate data
        logger.info("Generate data...")
        args.config = yaml_file_path
        command = [
            "python3.8",
            "meta_generator.py",
            "--config",
            args.config,
            "--data_dir",
            args.data_dir,
            "--carla_executable",
            args.carla_executable,
        ]
        process = subprocess.Popen(command, cwd=cwd + "utils/")
        process.wait()

    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Execute to generate data for all configs within a folder."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config.yaml",
        help="Path to the config file",
    )
    parser.add_argument(
        "--config_dir", type=str, default="data", help="Path to the config directory"
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
    args = parser.parse_args()

    main(args)
