# Copyright (C) 2025 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Marius Kästingschäfer and Théo Gieruc
# ==============================================================================

from common.sensor import SensorManager, BEVCamera
import os
from PIL import Image
import copy
import carla

# from concurrent.futures import ThreadPoolExecutor, as_completed


class Vehicle:
    """A class for a CARLA vehicle."""

    def __init__(self, blueprint, spawn_point, world, traffic_manager, logger):
        self.vehicle = None
        self.blueprint = blueprint
        self.blueprint_library = world.get_blueprint_library()
        self.spawn_point = spawn_point
        self.world = world
        self.logger = logger
        self.sensors = {}
        self.bev_camera = None
        self.bev_imgs = []
        self.status = "up"
        self.traffic_manager = traffic_manager

        self.spawn()
        self.id = self.vehicle.id

    def spawn(self):
        self.vehicle = self.world.spawn_actor(self.blueprint, self.spawn_point)

    def set_sensors(self, sensor_configs, invisible=False):
        self.sensors = {}
        for setup_name, sensor_config in sensor_configs.items():
            self.sensors[setup_name] = SensorManager(
                world=self.world,
                blueprint_library=self.blueprint_library,
                sensor_info=sensor_config["sensor_info"],
                transform_file_cams=sensor_config.get("transform_file_cams", None),
                transform_file_lidar=sensor_config.get("transform_file_lidar", None),
                vehicle=self.vehicle,
                logger=self.logger,
            )

        if invisible:
            # sets same sensors but 5m above the current ones
            self.invisible_sensors = {}
            for setup_name, sensor_config in sensor_configs.items():
                self.invisible_sensors[setup_name] = SensorManager(
                    world=self.world,
                    blueprint_library=self.blueprint_library,
                    sensor_info=sensor_config["sensor_info"],
                    transform_file_cams=sensor_config.get("transform_file_cams", None),
                    transform_file_lidar=sensor_config.get(
                        "transform_file_lidar", None
                    ),
                    vehicle=self.vehicle,
                    logger=self.logger,
                    z_offset=5,
                )

    def set_BEV(self):
        self.bev_camera = BEVCamera(self.world, self.vehicle, self.logger)
        self.bev_imgs = []

    def save_data(self, data_dir):
        for setup_name, sensor_manager in self.sensors.items():
            sensor_manager.save_data(os.path.join(data_dir, setup_name))

        if self.bev_camera:
            self.bev_imgs.append(self.bev_camera.get_sensor_data().copy())

    def save_invisible_data(self, data_dir):
        for setup_name, sensor_manager in self.invisible_sensors.items():
            sensor_manager.save_data(os.path.join(data_dir, f"{setup_name}_invisible"))

    def get_location(self):
        return self.vehicle.get_location()

    def get_transform(self):
        return self.vehicle.get_transform()

    def save_bev(self, save_path):
        bev_PIL = [
            Image.fromarray(image[:, :, ::-1]).convert("RGB") for image in self.bev_imgs
        ]
        if len(bev_PIL) > 1:
            bev_PIL[0].save(
                save_path,
                save_all=True,
                append_images=bev_PIL[1:],
                duration=100,
                optimize=True,
                loop=0,
            )
        else:
            bev_PIL[0].save(save_path)

    def destroy(self):
        for setup_name, sensor_manager in self.sensors.items():
            sensor_manager.destroy()
        if self.bev_camera:
            self.bev_camera.destroy()
        if self.invisible_sensors:
            for setup_name, sensor_manager in self.invisible_sensors.items():
                sensor_manager.destroy()
        self.vehicle.destroy()

    def go_down(self):
        self.vehicle.set_simulate_physics(False)
        self.vehicle.set_enable_gravity(False)
        if self.status == "up":
            up_transform = self.vehicle.get_transform()
            self.x_up = up_transform.location.x
            self.y_up = up_transform.location.y
            self.z_up = up_transform.location.z
            self.pitch_up = up_transform.rotation.pitch
            self.roll_up = up_transform.rotation.roll
            self.yaw_up = up_transform.rotation.yaw
            self.status = "down"

        down_location = self.vehicle.get_location()
        down_location.z -= 5
        self.vehicle.set_location(down_location)

    def go_up(self):
        self.vehicle.set_simulate_physics(True)
        self.vehicle.set_enable_gravity(True)
        up_transform = carla.Transform(
            location=carla.Location(x=self.x_up, y=self.y_up, z=self.z_up + 0.01),
            rotation=carla.Rotation(
                pitch=self.pitch_up, roll=self.roll_up, yaw=self.yaw_up
            ),
        )

        self.vehicle.set_transform(up_transform)
        self.status = "up"

    def reset_invisible_sensors(self):
        for setup_name, sensor_manager in self.invisible_sensors.items():
            sensor_manager.reset()

    def reset_sensors(self):
        for setup_name, sensor_manager in self.sensors.items():
            sensor_manager.reset()

    def set_autopilot(self, enable):
        self.vehicle.set_autopilot(enable, self.traffic_manager.get_port())
