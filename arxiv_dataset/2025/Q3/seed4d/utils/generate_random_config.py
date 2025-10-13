# Copyright (C) 2025 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Marius Kästingschäfer and Théo Gieruc
# ==============================================================================

"""
Create a random config file (with random town, spawnpoint, weather, ego vehicle)
"""

import random
import yaml
import argparse


def get_random_paramter():

    towns = ["Town01", "Town02", "Town03", "Town04", "Town05"]
    towns_N_spawnpoints = [255, 101, 265, 372, 302]
    weathers = [
        "ClearNoon",
        "CloudyNoon",
        "WetNoon",
        "WetCloudyNoon",
        "MidRainyNoon",
        "HardRainNoon",
        "SoftRainNoon",
        "ClearSunset",
        "CloudySunset",
        "WetSunset",
        "WetCloudySunset",
        "MidRainSunset",
        "HardRainSunset",
        "SoftRainSunset",
    ]
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

    towns_N_spawnpoints = dict(zip(towns, towns_N_spawnpoints))

    TOWN = random.choice(towns)
    SPAWNPOINT = random.randint(0, towns_N_spawnpoints[TOWN])
    WEATHER = random.choice(weathers)
    VEHICLE = random.choice(cars)

    return TOWN, SPAWNPOINT, WEATHER, VEHICLE


def generate_config(file_path):

    TOWN, SPAWNPOINT, WEATHER, VEHICLE = get_random_paramter()

    config = {
        "carla": {
            "host": "localhost",
            "port": 2000,
            "timeout": 20.0,
            "synchronous_mode": True,
            "fixed_delta_seconds": 0.1,
        },
        "data_dir": "data",
        "vehicle": "vehicle." + VEHICLE,  # "vehicle.tesla.model3",
        "map": TOWN,
        "spawn_point": [SPAWNPOINT],
        "weather": WEATHER,
        "number_of_vehicles": 40,
        "number_of_walkers": 40,
        "steps": 2,
        "min_distance": 0.5,
        "BEVCamera": True,
        "dataset": {
            "sphere": {
                "camera_info": {
                    "type": [
                        "sensor.camera.rgb",
                        # "sensor.camera.depth",
                        "sensor.camera.semantic_segmentation",
                    ],
                    "width": 800,
                    "height": 600,
                    "fov": 90,
                },
                "transform_info": {
                    "type": "sphere",
                    "origin": [0, 0, 0],
                    "z_offset": 0.5,
                    "radius": 5,
                    "N": 10,
                },
                "attached_to_vehicle": True,
            },
            "nuscenes": {
                "camera_info": {
                    "type": [
                        "sensor.camera.rgb",
                        "sensor.camera.depth",
                        "sensor.camera.semantic_segmentation",
                    ],
                    "width": 800,
                    "height": 600,
                    "fov": 90,
                },
                "transform_info": {
                    "type": "nuscenes",
                },
                "attached_to_vehicle": True,
            },
        },
    }

    file_name = TOWN + "_" + str(SPAWNPOINT) + "_" + VEHICLE + "_" + WEATHER + ".yaml"
    with open(file_path + file_name, "w") as file:
        yaml.dump(config, file)

    return file_path + "/" + file_name


def load_yaml_file(file_path):
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    return data


def main(args):
    for i in range(args.number):
        config_path = generate_config(args.data_dir)
        if args.output:
            print(config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Execute to generate random config(s)."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="../Generator/config/random/",
        default="../Generator/config/",
    )
    parser.add_argument("--number", type=int, help="e.g., 1 or 100", default=1)
    parser.add_argument("--output", type=bool, default=True)
    args = parser.parse_args()

    main(args)
