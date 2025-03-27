# Copyright (C) 2025 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Marius Kästingschäfer and Théo Gieruc
# ==============================================================================

import argparse
import codecs
import json
import os
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common.pose as pose


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def main(args):
    if args.type == "sphere":
        config = pose.generate_sphere_transforms(
            origin=args.origin,
            z_offset=args.z_offset,
            radius=args.radius,
            N=args.N,
        )
    elif args.type == "nuscenes":
        config = pose.generate_nuscenes_transforms()
    else:
        raise ValueError("Unsupported camera type: {}".format(args.type))

    for key, value in config.items():
        if isinstance(value, np.ndarray):
            config[key] = config[key].tolist()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    json.dump(
        config,
        codecs.open(args.output, "w", encoding="utf-8"),
        separators=(",", ":"),
        sort_keys=True,
        indent=4,
        cls=NumpyEncoder,
    )
    # with open(args.output, "w") as f:
    #     json.dump(config, f, indent=4)

    print("Wrote camera config to {}".format(args.output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--type",
        type=str,
        default="sphere",
        help="Type of camera to create. Options: sphere, nuscenes",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="camera_configs/sphere.json",
        help="Where to write the camera config",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=100,
        help="Number of cameras to create. Only used for sphere cameras",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=1.0,
        help="Radius of sphere. Only used for sphere cameras",
    )
    parser.add_argument(
        "--origin",
        type=float,
        nargs="+",
        default=[0.0, 0.0, 0.0],
        help="Origin of sphere. Only used for sphere cameras",
    )
    parser.add_argument(
        "--z_offset",
        type=float,
        default=0.1,
        help="Offset of z-axis. Only used for sphere cameras",
    )

    args = parser.parse_args()
    main(args)
