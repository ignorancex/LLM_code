# Copyright (C) 2025 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Marius Kästingschäfer and Théo Gieruc
# ==============================================================================

"""
Obtain mask(s) to learn single dyn. obj. and background
"""

import os
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
from common import mask

thickness = 1


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
    
    items = [f for f in os.listdir(args.data_dir + "/step_0/") if os.path.isdir(os.path.join(args.data_dir + "/step_0/", f))]
    datasets = ["nuscenes", "sphere"]
    
    for idx in range(max_steps):
        for itm in items:
            for dataset in datasets:
                file_path = args.data_dir + "/step_" + str(idx) + "/" + itm + "/" + dataset
                mask.obtain_wb_mask_and_object_only(file_path, thickness)
                mask.write_transform_jsons(file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Execute to generate masks and associated transform.jsons of vehicles."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="../Town0X/Weather/vehicle.X/spawn_point_x/",
        default="data",
    )
    args = parser.parse_args()

    main(args)
