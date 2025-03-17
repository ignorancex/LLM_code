# Copyright (C) 2025 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Marius Kästingschäfer and Théo Gieruc
# ==============================================================================

"""
From one config file, create a batch of config files with each one spawnpoint
"""

import argparse
import yaml
import os

towns_N_spawnpoints = {
    "Town01": 255,
    "Town02": 101,
    "Town03": 265,
    "Town04": 372,
    "Town05": 302,
}


def main(args):
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    os.makedirs(args.output_dir, exist_ok=True)
    town = config["map"]
    for spawnpoint in range(towns_N_spawnpoints[town]):
        config["spawn_point"] = [spawnpoint]
        with open(
            os.path.join(args.output_dir, "{}_{}.yaml".format(town, spawnpoint)), "w"
        ) as f:
            yaml.dump(config, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--output_dir", type=str, default="configs")

    args = parser.parse_args()

    main(args)
