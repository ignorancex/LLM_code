# Copyright (C) 2025 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Marius Kästingschäfer and Théo Gieruc
# ==============================================================================

"""
Generates configs where vehicle is centered, single timestep and surrounded by cameras
"""

import os
import argparse
import yaml


def load_yaml_file(file_path):
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    return data


def main(args):

    cars = [
        "audi.a2",
        "audi.tt",
        "citroen.c3",
        "bmw.grandtourer",
        "ford.mustang",
        "ford.ambulance",
        "carlamotors.firetruck",
        "tesla.model3",
        "diamondback.century",
        "vespa.zx125",
        "lincoln.mkz_2017",
        "chevrolet.impala",
        "micro.microlino",
        "dodge.charger_2020",
        "jeep.wrangler_rubicon",
        "volkswagen.t2_2021",
        "volkswagen.t2",
        "mercedes.sprinter",
        "tesla.cybertruck",
        "kawasaki.ninja",
        "nissan.patrol_2021",
        "ford.crown",
        "mini.cooper_s",
        "audi.etron",
        "mitsubishi.fusorosa",
        "mini.cooper_s_2021",
        "gazelle.omafiets",
        "mercedes.coupe_2020",
        "dodge.charger_police_2020",
        "bh.crossbike",
        "harley-davidson.low_rider",
        "toyota.prius",
        "seat.leon",
        "nissan.patrol",
        "nissan.micra",
        "yamaha.yzf",
        "mercedes.coupe",
        "dodge.charger_police",
        "lincoln.mkz_2020",
        "carlamotors.carlacola",
    ]

    radius_10_vehicles = ["carlamotors.firetruck", "mitsubishi.fusorosa"]

    folder_path_radius_5 = (
        "/mariustheo/Generator/config/camera_configs/sphere-2500-radius-5"
    )
    sphere_file_names_radius_5 = os.listdir(folder_path_radius_5)
    sphere_paths_radius_5 = [
        os.path.join(folder_path_radius_5, file) for file in sphere_file_names_radius_5
    ]

    folder_path_radius_10 = (
        "/mariustheo/Generator/config/camera_configs/sphere-2500-radius-10"
    )
    sphere_file_names_radius_10 = os.listdir(folder_path_radius_10)
    sphere_paths_radius_10 = [
        os.path.join(folder_path_radius_10, file)
        for file in sphere_file_names_radius_10
    ]

    for car in cars:
        if car in sphere_paths_radius_10:
            paths = sphere_paths_radius_10
        else:
            paths = sphere_paths_radius_5
        for idx, path in enumerate(paths):
            config_file = load_yaml_file(args.read_dir)
            config_file["vehicle"] = "vehicle." + car
            config_file["dataset"]["sphere"]["transform_file"] = path

            with open(
                args.write_dir
                + "/configs/"
                + str(car)
                + "_config_"
                + str(idx)
                + ".yaml",
                "w",
            ) as file:
                yaml.dump(config_file, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write config ")
    parser.add_argument(
        "--read_dir",
        type=str,
        help="../Generator/config/defaul.yaml",
        default="/mariustheo/Generator/config/vehicle_configs/vehicle_default.yaml",
    )
    parser.add_argument(
        "--write_dir",
        type=str,
        help="../Generator/config/vehicle_configs",
        default="/mariustheo/Generator/config/vehicle_configs",
    )
    args = parser.parse_args()

    main(args)
