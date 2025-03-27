# Copyright (C) 2025 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Marius Kästingschäfer and Théo Gieruc
# ==============================================================================

import argparse
import yaml
import os
import psutil
from time import sleep
from tqdm import tqdm
import sys, os
import subprocess
import logging

print(" Imports successfull")


def data_exists(config, data_dir):
    with open(config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return os.path.exists(
        os.path.join(
            data_dir,
            config["map"],
            config["weather"],
            config["vehicle"],
            f"spawn_point_{config['spawn_point'][0]}",
        )
    )


def main(args):
    '''
    Run the generator for all the config files in the specified directory.
    
    Parameters:
        args (argparse.Namespace): The parsed arguments from the command line.
    '''
    
    fails = []
    if args.config:
        config_files = [args.config]
    elif args.config_dir:
        config_files = [
            os.path.join(args.config_dir, f)
            for f in sorted(os.listdir(args.config_dir))
            if f.endswith(".yaml")
        ]

    print(" Loaded {} config files".format(len(config_files)))
    
    # configure logging
    (
        logging.basicConfig(level=logging.ERROR)
        if args.quiet
        else logging.basicConfig(level=logging.INFO)
    )
    logger = logging.getLogger(__name__)
    
    cwd = "/seed4d/utils"

    for i, config in tqdm(enumerate(config_files), total=len(config_files)):
                
        if args.only_missing:
            if data_exists(config, args.data_dir):
                continue

        # "--quiet",
        process = psutil.Popen(
            [
                "python3.8",
                "generator.py",
                "--config",
                config,
                "--data_dir",
                args.data_dir,
                "--carla_executable",
                args.carla_executable,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        while (process.status() != psutil.STATUS_DEAD) and (
            process.status() != psutil.STATUS_ZOMBIE
        ):
            sleep(1)

        if process.status() == psutil.STATUS_DEAD:
            fails.append(config)
        process.kill()
        
        # obtain file path
        with open(config, "r") as file:
            config = yaml.safe_load(file)
        save_path = os.path.join(
            args.data_dir,
            config["map"],
            config["weather"],
            config["vehicle"],
            "spawn_point_" + str(config["spawn_point"][0]),
        )
        
        # normalize coordinates
        if args.normalize_coords:
            logger.info(" Normalize coordinates ...")
            command = [
                "python3.8",
                "/seed4d/utils/generate_normalized_coordinates.py",
                "--data_dir",
                save_path,
                "--elements",
                "nuscenes",
            ]
            process = subprocess.Popen(command, cwd=cwd)
            process.wait()
            
        # vehicle masks
        if args.vehicle_masks:
            logger.info(" Generate vehicle masks ...")
            command = [
                "python3.8",
                "/seed4d/utils/generate_masks.py",
                "--data_dir",
                save_path + "/",
            ]
            process = subprocess.Popen(command, cwd=cwd)
            process.wait()
        
        # combine transfroms across timesteps
        if args.combine_transforms:
            logger.info(" Combining transform files across timepoints ...")
            command = [
                "python3.8",
                "/seed4d/utils/generate_single_transforms.py",
                "--data_dir",
                save_path + "/",
            ]
            process = subprocess.Popen(command, cwd=cwd)
            process.wait()

        # create single overview map
        if args.map:
            logger.info(" Generate overview map and single position file ...")
            command = ["python3.8", "/seed4d/utils/generate_map.py", "--data_dir", save_path + "/"]
            process = subprocess.Popen(command, cwd=cwd)
            process.wait()
        
        for handler in logger.handlers:
            handler.close()
            logger.removeHandler(handler)

    if len(fails) > 0:
        print(f"Failed to generate data for {len(fails)} configs")

        os.makedirs(os.path.join(args.data_dir, "failed_configs"), exist_ok=True)
        for i, config in enumerate(fails):
            with open(
                os.path.join(args.data_dir, "failed_configs", f"config_{i}.yaml"), "w"
            ) as f:
                yaml.dump(config, f)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help=".yaml config file path")
    parser.add_argument("--config_dir", type=str, default=None, help="Directory of .yaml config files")
    parser.add_argument(
        "--only_missing",
        action="store_true",
        default=False,
        help="Only generate data for configs that don't already have data",
    )
    parser.add_argument("--carla_executable", type=str, default="/home/carla/CarlaUE4.sh", help="Location of the CarlaUE4.sh executable")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument(
        "--normalize_coords",
        type=str2bool,
        default=True,
        help="Whether to generate normalized coordinates",
    )
    parser.add_argument(
        "--vehicle_masks",
        type=str2bool,
        default=True,
        help="Generate vehicle masks + vehicle only images",
    )
    parser.add_argument(
        "--combine_transforms",	
        type=str2bool,
        default=True,
        help="Write a single transform across timepoints",
    )
    parser.add_argument(
        "--map",	
        type=str2bool,
        default=True,
        help="Creates a map overview and a single file containing all positions",
    )
    parser.add_argument(
        "--quiet",
        type=str2bool,
        default=True,
        help="Disable progress bar and all logging except for errors",
    )
    
    args = parser.parse_args()
    print(args)

    assert (
        args.config or args.config_dir
    ), "Please specify either a config file or a directory of config files"

    main(args)
