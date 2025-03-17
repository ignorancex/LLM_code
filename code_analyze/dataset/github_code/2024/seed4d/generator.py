# Copyright (C) 2025 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Marius Kästingschäfer and Théo Gieruc
# ==============================================================================

import argparse
import logging
import os
import subprocess
import json
from time import sleep
from typing import Dict

import carla
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

import common.generate_traffic as generate_traffic
import common.pose as pose
import common.sensor as sensor
from common.vehicle import Vehicle
from contextlib import contextmanager
import sys, os
import random


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


class Generator:
    """
    Class responsible for generating data by capturing sensor outputs from multiple spawn points in CARLA simulation.

    Args:
        config (dict): Configuration dictionary specifying the parameters for data generation.

    Attributes:
        config (dict): Configuration dictionary specifying the parameters for data generation.
        client (carla.Client): CARLA client instance.
        world (carla.World): CARLA world instance.
        blueprint_library (carla.BlueprintLibrary): CARLA blueprint library instance.
        map (carla.Map): CARLA map instance.
        spawn_points (list): List of CARLA spawn points filtered for data generation.

    """

    def __init__(
        self,
        config_path: str,
        config: dict,
        data_dir: str,
        carla_executable: str,
        logger: logging.Logger = None,
        quiet: bool = False,
    ):
        self.config_path = config_path
        self.config = config
        self.client = None
        self.world = None
        self.blueprint_library = None
        self.map = None
        self.spawn_points = None
        self.traffic_manager = None
        self.carla_process = None
        self.data_dir = data_dir
        self.carla_executable = carla_executable
        self.quiet = quiet
        self.logger = logger if logger is not None else logging.getLogger(__name__)

    def __enter__(self):
        """
        Enter method for using Generator as a context manager.

        Returns:
            Generator: Initialized Generator instance.

        """

        try:
            self._launch_carla()
            self._setup_world()
            self._filter_spawn_points()
        except Exception as e:
            # logger.exception("Error occurred during initialization")
            raise e

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit method for cleaning up resources when exiting the context manager.
        """
        self.kill_carla()

    def _launch_carla(self):
        """
        Launch the CARLA server.
        """

        self.kill_carla()
        sleep(1)

        self.logger.info(" Launching CARLA server ...")

        if not os.path.exists(self.carla_executable):
            self.logger.error(
                "CARLA executable not found at {}".format(self.carla_executable)
            )
            raise Exception(
                "CARLA executable not found at {}".format(self.carla_executable)
            )

        flag = "-carla-server -RenderOffScreen -nosound -quality-level=Epic "
        command = [self.carla_executable, flag]
        # potentially add: stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.carla_process = subprocess.Popen(command)  
        # this timer might cause problems - if not starting up increase the value!
        sleep(15) #previous 15

        while self.carla_process.poll() == None:
            try:
                self.client = carla.Client(
                    self.config["carla"]["host"], self.config["carla"]["port"]
                )
                self.client.set_timeout(self.config["carla"]["timeout"])
                self.world = self.client.get_world()
                self._setup_world()
                self._set_weather()
                self.logger.info(" Connected to the CARLA server")
                return
            except Exception as e:
                # logger.exception("Error occurred while connecting to the CARLA server")
                sleep(1)

        self._launch_carla()

    def kill_carla(self):

        subprocess.call(
            ["killall", "CarlaUE4-Linux-Shipping"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.call(
            ["killall", "CarlaUE4.sh"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _setup_world(self):
        """
        Set up the CARLA world by creating a client, connecting to the server, and configuring the world settings.
        """

        self.logger.info(" Setting up CARLA world ...")

        self.blueprint_library = self.world.get_blueprint_library()

        if self.world.get_map().name != self.config["map"]:
            self.world = self.client.load_world(self.config["map"])

        self.map = self.world.get_map()

        self.spawn_points = self.map.get_spawn_points()

        self._set_weather()

        # Set synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = self.config["carla"]["synchronous_mode"]
        settings.fixed_delta_seconds = self.config["carla"]["fixed_delta_seconds"]
        self.world.apply_settings(settings) 

        for actor in self.world.get_actors():
            actor.destroy()

        if self.config["number_of_vehicles"] > 0:
            self.traffic_manager = self.client.get_trafficmanager(8000)
            self.traffic_manager.set_synchronous_mode(self.config["carla"]["synchronous_mode"])
            #self.traffic_manager.set_global_distance_to_leading_vehicle(2.5)
            

        

    def _set_weather(self):
        """
        Set the weather conditions in the CARLA world based on the specified weather parameter in the configuration.
        https://carla.org/Doxygen/html/db/ddb/classcarla_1_1rpc_1_1WeatherParameters.html
        """
        presets = {
            "Default": carla.WeatherParameters.Default,
            "ClearNoon": carla.WeatherParameters.ClearNoon,
            "CloudyNoon": carla.WeatherParameters.CloudyNoon,
            "WetNoon": carla.WeatherParameters.WetNoon,
            "WetCloudyNoon": carla.WeatherParameters.WetCloudyNoon,
            "MidRainyNoon": carla.WeatherParameters.MidRainyNoon,
            "HardRainNoon": carla.WeatherParameters.HardRainNoon,
            "SoftRainNoon": carla.WeatherParameters.SoftRainNoon,
            "ClearSunset": carla.WeatherParameters.ClearSunset,
            "CloudySunset": carla.WeatherParameters.CloudySunset,
            "WetSunset": carla.WeatherParameters.WetSunset,
            "WetCloudySunset": carla.WeatherParameters.WetCloudySunset,
            "MidRainSunset": carla.WeatherParameters.MidRainSunset,
            "HardRainSunset": carla.WeatherParameters.HardRainSunset,
            "SoftRainSunset": carla.WeatherParameters.SoftRainSunset,
        }

        self.world.set_weather(presets[self.config["weather"]])

    def _filter_spawn_points(self):
        if self.config["spawn_point"] is not None:
            spawn_points = []
            for n_spawn_point in self.config["spawn_point"]:
                # n_spawn_point from 1-N and spawn_points from 0-N
                spawn_points.append(self.spawn_points[n_spawn_point-1])
            self.spawn_points = spawn_points

        else:
            raise Exception("No spawn point specified")

    def generate(self):
        """
        Generate data by capturing sensor outputs from each spawn point and saving them to the specified directory.
        """

        self.logger.info(" Generating data ...")
        progress_bar = tqdm(
            total=len(self.spawn_points) * self.config["steps"],
            desc="Generating data",
            disable=self.quiet,
        )

        for n_spawnpoint, spawn_point in zip(
            self.config["spawn_point"], self.spawn_points
        ):

            logger.info(
                f" Starting data generation for spawn point {n_spawnpoint} of map {self.config['map']}."
            )

            # Spawn ego vehicle, traffic vehicles and pedestrians
            ego_blueprint = self.blueprint_library.filter(self.config["vehicle"])[0]
            ego_vehicle = Vehicle(ego_blueprint, spawn_point, self.world, self.traffic_manager, self.logger)
            
            traffic_vehicles = generate_traffic.spawn_cars(
                self.client,
                self.world,
                self.config["number_of_vehicles"],
                self.blueprint_library.filter("vehicle.*"),
                spawn_point,
                self.config["large_vehicles"],
                self.config["sort_spawnpoints"],
                self.traffic_manager,
                self.logger,
            )

            generate_traffic.spawn_pedestrians(
                self.client,
                self.world,
                self.config["number_of_walkers"],
                self.blueprint_library.filter("walker.*"),
                self.logger,
            )

            # Disable autopilot for all vehicles
            
            ego_vehicle.set_autopilot(True)
            for vehicle in traffic_vehicles:
                vehicle.set_autopilot(True)

            # Wait for vehicles to touch the ground
            # add random ticks to somewhat randomize starting time (vehicle speed, etc.)
            if self.config["steps"] == 1:
                random_offset = 24
            else:
                random_offset = random.randint(0, 20)
            for _ in range(random_offset):
                self._tick()

                    
            # Set up sensors
            ego_vehicle.set_sensors(
                self.config["dataset"], self.config.get("invisible_ego", False)
            )
            if self.config.get("other_vehicles_have_sensors", False):
                for vehicle in traffic_vehicles:
                    vehicle.set_sensors(self.config["dataset"], False)

            # Set up BEV cameras
            if self.config.get("BEVCamera", False):
                ego_vehicle.set_BEV()
                for vehicle in traffic_vehicles:
                    vehicle.set_BEV()

            timesteps = {}

            for step in range(self.config["steps"]):

                self.logger.info(
                    f" Step {step} of {self.config['steps']} for spawn point {n_spawnpoint} of map {self.config['map']}"
                )
                progress_bar.set_description(
                    f"Step {step + 1}/{self.config['steps']} | spawn point {n_spawnpoint} of map {self.config['map']}"
                )

                # Save 3D bounding box data
                if self.config.get("3Dboundingbox", False):
                    self._write_3Dboundingbox_data(ego_vehicle.id, n_spawnpoint, step)

                actors = self.world.get_actors()
                sens = [actor for actor in actors if actor.type_id.startswith("sensor")]
                self.logger.info(
                    " Amount actors: %d, Amount sensors: %d, Amount vehicles: %d",
                    len(actors),
                    len(sens),
                    len(self.world.get_actors().filter("vehicle.*")),
                )

                # Reset sensors before capturing data
                ego_vehicle.reset_sensors()
                if self.config.get("other_vehicles_have_sensors", False):
                    for vehicle in traffic_vehicles:
                        vehicle.reset_sensors()

                # Capture sensor data
                self._tick()

                timesteps[int(step)] = float(
                    self.world.get_snapshot().timestamp.elapsed_seconds
                )

                # Save sensor data
                self.logger.info("saving data")
                path = os.path.join(
                    self.data_dir,
                    self.config["map"],
                    self.config["weather"],
                    self.config["vehicle"],
                    f"spawn_point_{n_spawnpoint}",
                    f"step_{step}",
                    "ego_vehicle",
                )
                ego_vehicle.save_data(path)
                if self.config.get("other_vehicles_have_sensors", False):
                    for vehicle in traffic_vehicles:
                        path = os.path.join(
                            self.data_dir,
                            self.config["map"],
                            self.config["weather"],
                            self.config["vehicle"],
                            f"spawn_point_{n_spawnpoint}",
                            f"step_{step}",
                            str(vehicle.id),
                        )
                        vehicle.save_data(path)

                # If invisible, save invisible data
                if self.config.get("invisible_ego", False):
                    ego_vehicle.go_down()
                    if self.config.get("invisible_all", False):
                        [vehicle.go_down() for vehicle in traffic_vehicles]
                    ego_vehicle.reset_invisible_sensors()
                    self._tick()
                    ego_vehicle.save_invisible_data(path)
                    ego_vehicle.go_up()
                    self._tick()

                # If more than one step, activate autopilot and move vehicles forward
                if self.config["steps"] > 1:
                    if step == 0:
                        ego_vehicle.set_autopilot(True)
                        [vehicle.set_autopilot(True) for vehicle in traffic_vehicles]
                        # Move vehicle of minimum distance
                    # self._tick()
                    # self._tick()
                    while (
                        self._birdeye_distance(
                            ego_vehicle.get_location(), spawn_point.location
                        )
                        < self.config["min_distance"]
                    ):
                        self._tick()

                    if self.config.get("invisible_ego", False):
                        ego_vehicle.set_autopilot(False)
                        [vehicle.set_autopilot(False) for vehicle in traffic_vehicles]

                    spawn_point = ego_vehicle.get_transform()

                progress_bar.update(1)

            # Save BEV images
            save_path = os.path.join(
                self.data_dir,
                self.config["map"],
                self.config["weather"],
                self.config["vehicle"],
                f"spawn_point_{n_spawnpoint}",
                "BEV_ego.gif",
            )
            if self.config.get("BEVCamera", False):
                ego_vehicle.save_bev(save_path)
                if self.config.get("other_vehicles_have_sensors", False):
                    for vehicle in traffic_vehicles:
                        save_path_traffic = os.path.join(
                            self.data_dir,
                            self.config["map"],
                            self.config["weather"],
                            self.config["vehicle"],
                            f"spawn_point_{n_spawnpoint}",
                            f"BEV_{vehicle.id}.gif",
                        )
                        vehicle.save_bev(save_path_traffic)

            save_path = os.path.join(
                self.data_dir,
                self.config["map"],
                self.config["weather"],
                self.config["vehicle"],
                f"spawn_point_{n_spawnpoint}",
            )
            with open(os.path.join(save_path, "timesteps.json"), "w") as file:
                json.dump(timesteps, file)

            self._dump_config(save_path)

            vehicle_types = {}
            try:
                for actors in self.world.get_actors().filter("vehicle.*"):
                    vehicle_types[actors.id] = actors.type_id
                    actors.destroy()
                for actors in self.world.get_actors().filter("walker.*"):
                    actors.destroy()
            except:
                pass

            with open(save_path + "/vehicles.json", "w") as file:
                json.dump(vehicle_types, file)

            self._tick()

    def _dump_config(self, save_path):
        # write original yaml config into folder
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        with open(save_path + "/config.yaml", "w") as file:
            yaml.dump(config, file)

    def _write_3Dboundingbox_data(self, ego_id, n_spawnpoint, step):
        """
        Write 3D bounding box data to a json file with the following architecture
        {
            id: {
            transform:
            3d bounding box: xz
            ego:
            }
        }
        """
        data = {}
        for npc in self.world.get_actors().filter("vehicle.*"):
            bb = npc.bounding_box

            ego = npc.id == ego_id

            data[npc.id] = dict(
                transform=pose.carla_to_nerf_unnormalized(npc.get_transform()),
                bb=[
                    [
                        localization.x,
                        localization.y,
                        localization.z,
                    ]
                    for localization in bb.get_world_vertices(npc.get_transform())
                ],
                ego=ego,
            )

        save_path = os.path.join(
            # self.config["data_dir"],
            self.data_dir,
            self.config["map"],
            self.config["weather"],
            self.config["vehicle"],
            f"spawn_point_{n_spawnpoint}",
            f"step_{step}",
            "3Dboundingbox.json",
        )
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        with open(save_path, "w") as f:
            json.dump(data, f, indent=4)

    def _destroy_sensors(
        self,
        sensor_managers: Dict[str, sensor.SensorManager] = None,
        traffic_sensor_managers: Dict[str, Dict[str, sensor.SensorManager]] = None,
    ):
        if sensor_managers is not None:
            for _, sensor_manager in sensor_managers.items():
                sensor_manager.destroy()
        if traffic_sensor_managers is not None:
            for _, traffic_sensor_manager in traffic_sensor_managers.items():
                for _, sensor_manager in traffic_sensor_manager.items():
                    sensor_manager.destroy()

    def _birdeye_distance(
        self, location1: carla.Location, location2: carla.Location
    ) -> float:
        """
        Calculate the Euclidean distance between two locations in the x-y plane.
        """
        return np.sqrt(
            (location1.x - location2.x) ** 2 + (location1.y - location2.y) ** 2
        )

    def _tick(self):
        if self.config["carla"]["synchronous_mode"]:
            self.world.tick()
        else:
            self.world.wait_for_tick()

    def _spawn_vehicle(self, spawn_point: carla.Transform) -> carla.Actor:
        blueprint = self.blueprint_library.find(self.config["vehicle"])
        vehicle = self.world.spawn_actor(blueprint, spawn_point)
        vehicle.set_autopilot(False)

        # wait for vehicle to touch the ground
        for i in range(20):
            self._tick()

        return vehicle

    def _setup_sensor_managers(
        self, vehicle: carla.Actor
    ) -> Dict[str, sensor.SensorManager]:
        sensor_managers = {}

        for setup_name, sensor_config in self.config["dataset"].items():
            sensor_managers[setup_name] = sensor.SensorManager(
                world=self.world,
                blueprint_library=self.blueprint_library,
                sensor_info=sensor_config["sensor_info"],
                transform_file_cams=sensor_config["transform_file_cams"],
                transform_file_lidar=sensor_config.get("transform_file_lidar", None),
                vehicle=vehicle if sensor_config["attached_to_vehicle"] else None,
                logger=logger,
            )

        return sensor_managers

    def _setup_temporary_managers(
        self, vehicle_cam2world: dict
    ) -> Dict[str, sensor.SensorManager]:
        """
        Run through all vehicle_cam2world and spawn a camera for each (using cam configs from ego-vehicle)
        """
        sensor_managers = {}

        # for setup_name, sensor_config in self.config["dataset"].items():

        for vehicle_id in self.vehicle_cam2world:

            if vehicle_id == "nuscenes":
                sensor_info = self.config["dataset"]["nuscenes"]["sensor_info"]
            else:
                sensor_info = self.config["traffic_vehicles"]["dataset"]["nuscenes"][
                    "sensor_info"
                ]

            sensor_managers[vehicle_id] = dict()

            for idx, cam2world in enumerate(self.vehicle_cam2world[vehicle_id]):
                sensor_managers[vehicle_id][str(idx)] = sensor.SensorManager(
                    world=self.world,
                    blueprint_library=self.blueprint_library,
                    sensor_info=sensor_info,
                    transform_file_cams=cam2world,
                    transform_file_lidar=None,
                    vehicle=None,
                    logger=logger,
                    temporary=True,
                )

        return sensor_managers

    def _setup_traffic_sensor_managers(
        self, vehicle_id_list: list
    ) -> Dict[str, sensor.SensorManager]:
        sensor_managers = {}
        for vehicle_id in vehicle_id_list:
            vehicle = self.world.get_actor(vehicle_id)

            sensor_managers[vehicle_id] = dict()

            for setup_name, sensor_config in self.config["traffic_vehicles"][
                "dataset"
            ].items():
                sensor_managers[vehicle_id][setup_name] = sensor.SensorManager(
                    world=self.world,
                    blueprint_library=self.blueprint_library,
                    sensor_info=sensor_config["sensor_info"],
                    transform_file_cams=sensor_config["transform_file_cams"],
                    transform_file_lidar=sensor_config.get(
                        "transform_file_lidar", None
                    ),
                    vehicle=vehicle,
                    logger=logger,
                )

        return sensor_managers

    def _save_sensor_data(
        self,
        sensor_managers: Dict[str, sensor.SensorManager],
        n_spawnpoint: int,
        step: int,
        traffic_sensor_managers: Dict[str, Dict[str, sensor.SensorManager]] = None,
        temporary: bool = False,
    ):

        save_path = os.path.join(
            # self.config["data_dir"],
            self.data_dir,
            self.config["map"],
            self.config["weather"],
            self.config["vehicle"],
            f"spawn_point_{n_spawnpoint}",
            f"step_{step}",
        )

        if sensor_managers is not None:
            for setup_name, sensor_manager in sensor_managers.items():
                sensor_manager.save_data(os.path.join(save_path, setup_name))

        if traffic_sensor_managers is not None:

            for (
                traffic_vehicle_id,
                traffic_sensor_manager,
            ) in traffic_sensor_managers.items():
                path = os.path.join(save_path, str(traffic_vehicle_id))
                for setup_name, sensor_manager in traffic_sensor_manager.items():
                    if sensor_manager.temporary:
                        # within the temporary sensor manager, the path is combined slightly different
                        # to maintain the camera relations
                        sensor_manager.save_data(path, setup_name)
                    else:
                        sensor_manager.save_data(os.path.join(path, setup_name))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate data for the CARLA dataset")
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
    args = parser.parse_args()

    # Configure logging
    (
        logging.basicConfig(level=logging.ERROR)
        if args.quiet
        else logging.basicConfig(level=logging.INFO)
    )
    logger = logging.getLogger(__name__)

    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        with Generator(
            args.config,
            config,
            args.data_dir,
            args.carla_executable,
            logger,
            args.quiet,
        ) as generator:
            generator.generate()

    except Exception as e:
        logger.exception("An error occurred during data generation")

    finally:
        for handler in logger.handlers:
            handler.close()
            logger.removeHandler(handler)

        subprocess.call(
            ["killall", "CarlaUE4-Linux-Shipping"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.call(
            ["killall", "CarlaUE4.sh"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
