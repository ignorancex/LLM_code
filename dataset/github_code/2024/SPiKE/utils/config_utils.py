"""
Module for configuration utilities.
"""

import sys
import os
import yaml
import numpy as np
import torch
from const import path


def load_yaml_file(file_path):
    """Load a YAML file and return its content."""
    try:
        with open(file_path, "r", encoding="utf-8") as stream:
            return yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
        sys.exit(1)


def convert_to_float(config, keys):
    """Convert specific values in the config to floats."""
    for key in keys:
        config[key] = float(config[key])


def set_dataset_paths(config):
    """Set the dataset paths in the config based on the dataset name."""
    dataset_paths = {"ITOP-SIDE": path.ITOP_SIDE_PATH}
    if config["dataset"] in dataset_paths:
        config["dataset_root"] = dataset_paths[config["dataset"]]
        config["dataset_path"] = dataset_paths[config["dataset"]]
    else:
        raise ValueError(f"Dataset {config['dataset']} not found.")


def set_output_dir(config, config_file):
    """Set the output directory in the config and create it if it doesn't exist."""
    if config.get("log_dir"):
        config["output_dir"] = os.path.join(
            path.EXPERIMENTS_PATH, config_file, config["log_dir"]
        )
        print(f"Log dir: {config['output_dir']}")
        os.makedirs(config["output_dir"], exist_ok=True)
        if os.listdir(config["output_dir"]) and config.get("mode_train"):
            raise FileExistsError(
                f"Directory {config['output_dir']} already exists and is not empty"
            )


def load_config(config_file):
    """Load a config file and return its content."""
    path_config_file = os.path.join(path.EXPERIMENTS_PATH, config_file, "config.yaml")
    config = load_yaml_file(path_config_file)
    convert_to_float(config, ["momentum", "weight_decay"])
    set_dataset_paths(config)
    set_output_dir(config, config_file)
    return config


def set_random_seed(seed):
    """Set random seeds"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
