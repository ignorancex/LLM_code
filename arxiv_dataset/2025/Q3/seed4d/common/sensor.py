# Copyright (C) 2025 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Marius Kästingschäfer and Théo Gieruc
# ==============================================================================

import json
import logging
import math
import os
import re
from time import sleep, time

import carla
import numpy as np
import png
from PIL import Image

import common.pose as pose


class Sensor:
    '''
    Class for handling Carla sensors.
    
    Attributes:
        world (carla.World): The Carla world object.
        blueprint (carla.SensorBlueprint): The Carla sensor blueprint.
        carla_transform (carla.Transform): The Carla transform of the sensor.
        sensor_type (str): The type of sensor.
        vehicle (carla.Vehicle): The Carla vehicle object.
        sensor (carla.Sensor): The Carla sensor object.
        output (carla.Image): The sensor output.
        ready (bool): Whether the sensor is ready.
    
    Methods:
        get_pixel_angles: Get the pixel angles of the sensor.
        init_sensor: Initialize the sensor.
        callback: Callback function for the sensor.
        get_sensor_data: Get the sensor data.
        save_sensor_data: Save the sensor data to a file.
        destroy: Destroy the sensor.
        get_carla_transform: Get the Carla transform of the sensor.
        get_nerf_transform: Get the Nerf transform of the sensor.
        get_nerf_transform_ego: Get the Nerf transform of the sensor relative to the ego.
    '''
    
    def __init__(self, world, blueprint, carla_transform, sensor_type, vehicle=None, z_offset=0):
        self.world = world
        self.blueprint = blueprint
        self.carla_transform = carla_transform
        self.sensor_type = sensor_type

        self.vehicle = vehicle
        self.sensor = None
        self.output = None
        self.ready = False
        self.z_offset = z_offset

        if not self.sensor_type == "sensor.lidar.ray_cast":
            self.get_pixel_angles()

        self.init_sensor()

    def get_pixel_angles(self):
        H = self.blueprint.get_attribute("image_size_y").as_int()
        W = self.blueprint.get_attribute("image_size_x").as_int()
        focal = W / (
            2 * math.tan(self.blueprint.get_attribute("fov").as_float() * math.pi / 360)
        )
        grid = np.meshgrid(np.arange(W), np.arange(H))
        i, j = grid[0], grid[1]
        directions = np.stack(
            [(i - W / 2) / focal, -(j - H / 2) / focal, -np.ones_like(i)], -1
        )  # (H, W, 3)

        depth_offset = np.linalg.norm(directions[:, :, :2], axis=2)
        self.pixel_angles = np.arctan(depth_offset)

    def init_sensor(self):

        self.sensor = self.world.spawn_actor(
            self.blueprint,
            self.carla_transform,
            attach_to=self.vehicle,
        )
        self.sensor.listen(self.callback)

    def callback(self, out):
        self.output = out
        self.ready = True

    def get_sensor_data(self):
        return np.frombuffer(self.output.raw_data, dtype=np.uint8).reshape(
            (self.output.height, self.output.width, 4)
        )[:, :, :3]

    def save_sensor_data(self, path):

        if self.sensor_type == "sensor.camera.depth":
            rgb = np.array(self.output.raw_data, dtype=np.float32).reshape(
                (self.output.height, self.output.width, 4)
            )[:, :, :3]
            B = rgb[:, :, 0]
            G = rgb[:, :, 1]
            R = rgb[:, :, 2]

            normalized = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
            np.clip(normalized, 0, 65535 / 1e6, out=normalized)
            depth_mm = np.array(normalized * 1e6, dtype=np.uint16)

            with open(path, "wb") as f:
                writer = png.Writer(
                    width=depth_mm.shape[1],
                    height=depth_mm.shape[0],
                    bitdepth=16,
                    greyscale=True,
                )
                zgray2list = depth_mm.tolist()
                writer.write(f, zgray2list)

        elif self.sensor_type == "sensor.camera.semantic_segmentation":
            semantic_segmentation = np.array(
                self.output.raw_data, dtype=np.uint8
            ).reshape((self.output.height, self.output.width, 4))[:, :, 2]

            with open(path, "wb") as f:
                writer = png.Writer(
                    width=semantic_segmentation.shape[1],
                    height=semantic_segmentation.shape[0],
                    bitdepth=8,
                    greyscale=True,
                )
                writer.write(f, semantic_segmentation.tolist())

        elif self.sensor_type == "sensor.camera.optical_flow":
            optical_flow = (
                self.output.get_color_coded_flow()
            )  # see https://carla.readthedocs.io/en/latest/python_api/#carla.Image
            optical_flow = np.frombuffer(optical_flow.raw_data, dtype=np.uint8).reshape(
                (self.output.height, self.output.width, 4)
            )[
                ..., [2, 1, 0]
            ]  # BGRA to RGB

            Image.fromarray(optical_flow).save(path)

        else:
            self.output.save_to_disk(path)

    def destroy(self):
        self.sensor.destroy()

    def get_carla_transform(self):
        return self.sensor.get_transform()

    def get_nerf_transform(self):
        # carla_transform = self.sensor.get_transform()
        carla_transform = (np.array(self.vehicle.get_transform().get_matrix()) @ np.array(self.carla_transform.get_matrix()) )  
        x,y,z,yaw, roll,pitch = pose.extract_xyz_yaw_pitch_roll(carla_transform)
        carla_transform = carla.Transform(
            carla.Location(x=x, y=y, z=z), carla.Rotation(pitch=pitch, yaw=yaw, roll=roll)
        )
        nerf_transform = pose.carla_to_nerf_unnormalized(carla_transform)
        return nerf_transform

    def get_nerf_transform_ego(self):
        vehicle_location = self.carla_transform
        vehicle_location.location.z -= self.z_offset
        nerf_ego_transform = pose.carla_to_nerf_unnormalized(vehicle_location)
        # nerf_ego_transform = pose.carla_to_nerf_unnormalized(self.carla_transform)
        #nerf_ego_transform = np.array(nerf_ego_transform)
        #x,y,z,yaw, roll,pitch = pose.extract_xyz_yaw_pitch_roll(nerf_ego_transform)
        #nerf_ego_transform = carla.Transform(
        #    carla.Location(x=x, y=y, z=z), carla.Rotation(pitch=pitch, yaw=yaw, roll=roll)
        #).get_matrix()
        return nerf_ego_transform


class BEVCamera:
    '''
    Class for handling a BEV camera.
    
    Attributes:
        vehicle (carla.Vehicle): The Carla vehicle object to which the BEVCamera is attached.
        logger (logging.Logger): The logger object.
        world (carla.World): The Carla world object.
        transform (carla.Transform): The Carla transform of the BEVCamera.
        sensor (Sensor): The Sensor object.
    
    Methods:
        init_sensor: Initialize the sensor. 
        get_sensor_data: Get the sensor data.
        save_sensor_data: Save the sensor data to a file.
        destroy: Destroy the sensor.
    '''
    
    def __init__(self, world, vehicle, logger=logging.getLogger(__name__)):
        self.vehicle = vehicle
        self.logger = logger
        self.world = world

        self.transform = carla.Transform(
            carla.Location(x=0, y=0, z=10), carla.Rotation(pitch=-90, yaw=0, roll=0)
        )
        blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
        blueprint.set_attribute("image_size_x", str(512))
        blueprint.set_attribute("image_size_y", str(512))
        blueprint.set_attribute("fov", str(120))

        self.sensor = Sensor(
            self.world, blueprint, self.transform, "sensor.camera.rgb", self.vehicle
        )

    def init_sensor(self):
        self.sensor.init_sensor()

    def get_sensor_data(self):
        while not self.sensor.ready:
            sleep(0.05)
        self.sensor.ready = False
        return self.sensor.get_sensor_data()

    def save_sensor_data(self, path):
        self.sensor.save_sensor_data(path)

    def destroy(self):
        self.sensor.destroy()


class SensorManager:
    '''
    Class for handling multiple sensors.
    
    Attributes:
        world (carla.World): The Carla world object.
        blueprint_library (carla.BlueprintLibrary): The Carla blueprint library object.
        sensor_info (dict): The sensor information.
        transform_file_cams (str): The file containing the camera transforms.
        transform_file_lidar (str): The file containing the lidar transform.
        vehicle (carla.Vehicle): The Carla vehicle object.
        logger (logging.Logger): The logger object.
        temporary (bool): Whether the sensor is temporary.
        z_offset (float): The z offset of the sensor.
        carla_transforms_cams (list[carla.Transform]): The Carla transforms of the cameras.
        carla_transforms_lidar (list[carla.Transform]): The Carla transforms of the lidar.
        blueprint (dict): The blueprint of the sensors.
        sensors (dict[str, list[Sensor]]): The sensors.
        
    Methods:
        _get_cam_properties: Get the camera properties.
        _get_carla_transforms: Get the Carla transforms.
        _get_blueprint: Get the blueprint of the sensor.
        _get_lidar_blueprint: Get the blueprint of the lidar sensor.
        _get_sensors: Get the sensors.
        _save_sensor_data: Save the sensor data.
        _check_sensor_ready: Check if the sensor is ready.
        _reset_sensor_ready: Reset the sensor ready status.
        get_poses: Get the poses of the sensors.
        get_camera_intrinsics: Get the camera intrinsics.
        get_camera_intrinsics_from_fov: Get the camera intrinsics from the field of view.
        destroy: Destroy the sensors.
        save_data: Save the sensor data.
        _save_metadata: Save the sensor information.
        _save_image_transforms: Save the image transforms.
        _save_lidar_transforms: Save the lidar transforms.
        reset: Reset the sensors.
    '''
    def __init__(
        self,
        world: carla.World,
        blueprint_library: carla.BlueprintLibrary,
        sensor_info: dict,
        transform_file_cams: str,
        transform_file_lidar: str,
        vehicle: carla.Vehicle = None,
        logger=logging.getLogger(__name__),
        temporary: bool = False,
        z_offset: float = 0.0,
    ):
        self.world = world
        self.blueprint_library = blueprint_library
        self.sensor_info = sensor_info
        self.temporary = temporary
        self.z_offset = z_offset

        self.carla_transforms_cams = []
        self.carla_transforms_lidar = []
        self.blueprint = {}
        self.sensors: dict[str, list[Sensor]] = {}

        self.vehicle = vehicle
        self.logger = logger
        # self._get_blueprints()

        if transform_file_cams:
            with open(transform_file_cams, "r") as f:
                self.spawn_transforms_cams = json.load(f)

        if transform_file_lidar:
            with open(transform_file_lidar, "r") as f:
                self.spawn_transforms_lidar = json.load(f)

        self._get_carla_transforms()

        if transform_file_cams:
            self._get_cam_properties()
        self._get_sensors()

    def _get_cam_properties(self):
        if "fov" not in self.spawn_transforms_cams:
            self.spawn_transforms_cams["fov"] = [self.sensor_info["fov"]] * len(
                self.spawn_transforms_cams["coordinates"]
            )
        if "width" not in self.spawn_transforms_cams:
            self.spawn_transforms_cams["width"] = [self.sensor_info["width"]] * len(
                self.spawn_transforms_cams["coordinates"]
            )
        if "height" not in self.spawn_transforms_cams:
            self.spawn_transforms_cams["height"] = [self.sensor_info["height"]] * len(
                self.spawn_transforms_cams["coordinates"]
            )

    def _get_carla_transforms(self):

        yaw_correction = +90
        
        cams = ["sensor.camera.rgb", "sensor.camera.semantic_segmentation", 
                "sensor.camera.instance_segmentation", "sensor.camera.depth",
                "sensor.camera.optical_flow"]
        
        if any(cam in self.sensor_info["type"] for cam in cams):

            for coord, pitch, yaw in zip(
                self.spawn_transforms_cams["coordinates"],
                self.spawn_transforms_cams["pitchs"],
                self.spawn_transforms_cams["yaws"],
            ):
                pitch = -pitch * (180 / math.pi)
                yaw = -(yaw * (180 / math.pi) + yaw_correction)

                transform_cam = carla.Transform(
                    carla.Location(*coord), carla.Rotation(pitch=pitch, yaw=yaw)
                )

                transform_cam.location.z += self.z_offset

                self.carla_transforms_cams.append(transform_cam)

        if "sensor.lidar.ray_cast" in self.sensor_info["type"]:

            for coord, pitch, yaw in zip(
                self.spawn_transforms_lidar["coordinates"],
                self.spawn_transforms_lidar["pitchs"],
                self.spawn_transforms_lidar["yaws"],
            ):
                pitch = -pitch * (180 / math.pi)
                yaw = -(yaw * (180 / math.pi) + yaw_correction)

                transform_cam = carla.Transform(
                    carla.Location(*coord), carla.Rotation(pitch=pitch, yaw=yaw)
                )
                self.carla_transforms_lidar.append(transform_cam)

    def _get_blueprint(self, sensor_type, fov, height, width):

        blueprint = self.blueprint_library.find(sensor_type)

        blueprint.set_attribute("image_size_x", str(width))
        blueprint.set_attribute("image_size_y", str(height))
        blueprint.set_attribute("fov", str(fov))
        if sensor_type == "sensor.camera.rgb":
            if not self.temporary:
                blueprint.set_attribute("blur_amount", str(0.0))
                blueprint.set_attribute("motion_blur_max_distortion", "0")
                blueprint.set_attribute("motion_blur_intensity", "0")
                blueprint.set_attribute("motion_blur_min_object_screen_size", "0")

        return blueprint

    def _get_lidar_blueprint(self):
        blueprint = self.blueprint_library.find("sensor.lidar.ray_cast")
        blueprint.set_attribute("channels", str(self.sensor_info["channels"]))
        blueprint.set_attribute(
            "points_per_second", str(self.sensor_info["points_per_second"])
        )
        blueprint.set_attribute(
            "rotation_frequency", str(self.sensor_info["rotation_frequency"])
        )
        blueprint.set_attribute("range", str(self.sensor_info["range"]))
        return blueprint

    def _get_sensors(self):
        for sensor_type in self.sensor_info["type"]:
            self.sensors[sensor_type] = []

            if sensor_type != "sensor.lidar.ray_cast":
                for transform, fov, height, width in zip(
                    self.carla_transforms_cams,
                    self.spawn_transforms_cams["fov"],
                    self.spawn_transforms_cams["height"],
                    self.spawn_transforms_cams["width"],
                ):
                    blueprint = self._get_blueprint(
                        sensor_type=sensor_type, fov=fov, height=height, width=width
                    )
                    sensor = Sensor(
                        self.world, blueprint, transform, sensor_type, self.vehicle, self.z_offset
                    )
                    self.sensors[sensor_type].append(sensor)
            else:
                blueprint = self._get_lidar_blueprint()
                for transform in self.carla_transforms_lidar:
                    sensor = Sensor(
                        self.world, blueprint, transform, sensor_type, self.vehicle
                    )
                    self.sensors[sensor_type].append(sensor)

    def _save_sensor_data(self, path, setup_name=None):
        """
        Save sensor outputs from all sensors to path.
        """
        if not os.path.exists(path):
            os.makedirs(path)

        def save_sensor_data(sensor, sensor_path, sensor_type):
            if sensor_type == "sensor.lidar.ray_cast":
                sensor.save_sensor_data(sensor_path + "_lidar.ply")
            else:
                sensor.save_sensor_data(
                    sensor_path + "_" + sensor_type.split(".")[-1] + ".png"
                )

        t0 = time()
        while not self._check_sensor_ready():
            sleep(0.1)
        self.logger.info(
            " Sensor ready, saving sensor output. Took {} seconds.".format(time() - t0)
        )

        if not self.temporary:
            for sensor_type, sensors in self.sensors.items():
                for i, sensor in enumerate(sensors):
                    if sensor_type == "sensor.lidar.ray_cast":
                        sensor.save_sensor_data(
                            os.path.join(path, str(i) + "_lidar.ply")
                        )
                    else:
                        sensor.save_sensor_data(
                            os.path.join(
                                path, str(i) + "_" + sensor_type.split(".")[-1] + ".png"
                            )
                        )
        else:
            for sensor_type, sensors in self.sensors.items():
                for i, sensor in enumerate(sensors):
                    if sensor_type == "sensor.camera.rgb":
                        sensor.save_sensor_data(
                            os.path.join(
                                path,
                                str(setup_name)
                                + "_no_vehicles_"
                                + sensor_type.split(".")[-1]
                                + ".png",
                            )
                        )

        self._reset_sensor_ready()

    def _check_sensor_ready(self):
        for sensor_type, sensors in self.sensors.items():
            for sensor in sensors:
                if not sensor.ready:
                    return False
        return True

    def _reset_sensor_ready(self):
        for sensor_type, sensors in self.sensors.items():
            for sensor in sensors:
                sensor.ready = False

    def get_poses(self, pose_type="carla"):
        sensor_type = list(self.sensors.keys())[0]
        if pose_type == "carla":
            return [
                sensor.get_carla_transform() for sensor in self.sensors[sensor_type]
            ]
        elif pose_type == "nerf":
            return [sensor.get_nerf_transform() for sensor in self.sensors[sensor_type]]
        elif pose_type == "nerf_ego":
            return [
                sensor.get_nerf_transform_ego() for sensor in self.sensors[sensor_type]
            ]
        else:
            raise Exception("Pose type not supported.")

    def get_camera_intrinsics(self):

        focal = math.tan(math.radians(self.sensor_info["fov"] / 2))
        fx = (0.5 * self.sensor_info["width"]) / focal
        fy = fx
        cx = self.sensor_info["width"] / 2
        cy = self.sensor_info["height"] / 2
        return {"fx": fx, "fy": fy, "cx": cx, "cy": cy}

    @staticmethod
    def get_camera_intrinsics_from_fov(fov, width, height):
        focal = math.tan(math.radians(fov / 2))
        fx = (0.5 * width) / focal
        fy = fx
        cx = width / 2
        cy = height / 2
        return {"fx": fx, "fy": fy, "cx": cx, "cy": cy}

    def destroy(self):
        for sensor_type, sensors in self.sensors.items():
            for sensor in sensors:
                sensor.destroy()

    def save_data(self, save_path, setup_name=None):
        if not self.temporary:
            self._save_sensor_data(os.path.join(save_path, "sensors"))
            if hasattr(self, 'spawn_transforms_cams'):
                self._save_image_transforms(
                    save_path, self.get_poses(pose_type="nerf"), "transforms.json"
                )
                self._save_image_transforms(
                    save_path, self.get_poses(pose_type="nerf_ego"), "transforms_ego.json"
                )
            if hasattr(self, 'spawn_transforms_lidar'):
                self._save_lidar_transforms(save_path, "lidar_transforms.json")
            self._save_metadata(save_path)
        else:
            if save_path.split("/")[9] == "nuscenes":
                self._save_sensor_data(os.path.join(save_path, "sensors"), setup_name)
            else:
                self._save_sensor_data(
                    os.path.join(save_path, "nuscenes/sensors"), setup_name
                )

    def _save_metadata(self, path):
        with open(os.path.join(path, "sensor_info.json"), "w") as f:
            json.dump(self.sensor_info, f, indent=4)

    def _save_image_transforms(self, path, transforms, filename):
        camera_intrinsics = self.get_camera_intrinsics()

        json_file = {
            "camera_model": "OPENCV",
            "k1": 0,
            "k2": 0,
            "p1": 0,
            "p2": 0,
            "frames": [],
        }
        if (
            len(set(self.spawn_transforms_cams["fov"])) == 1
            and len(set(self.spawn_transforms_cams["height"])) == 1
            and len(set(self.spawn_transforms_cams["width"])) == 1
        ):
            json_file["fl_x"] = camera_intrinsics["fx"]
            json_file["fl_y"] = camera_intrinsics["fy"]
            json_file["cx"] = camera_intrinsics["cx"]
            json_file["cy"] = camera_intrinsics["cy"]
            json_file["w"] = self.sensor_info["width"]
            json_file["h"] = self.sensor_info["height"]

        sensor_types = list(self.sensors.keys())

        save_path = os.path.join(path, "transforms")

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for i, transform in enumerate(transforms):
            frame = {}
            if "sensor.camera.rgb" in sensor_types:
                frame["file_path"] = os.path.join("..", "sensors", str(i) + "_rgb.png")
            if "sensor.camera.depth" in sensor_types:
                frame["depth_file_path"] = os.path.join(
                    "..", "sensors", str(i) + "_depth.png"
                )
            if "sensor.camera.semantic_segmentation" in sensor_types:
                frame["semantic_segmentation_file_path"] = os.path.join(
                    "..", "sensors", str(i) + "_semantic_segmentation.png"
                )
            if "sensor.camera.instance_segmentation" in sensor_types:
                frame["instance_segmentation_file_path"] = os.path.join(
                    "..", "sensors", str(i) + "_instance_segmentation.png"
                )
            if "sensor.camera.optical_flow" in sensor_types:
                frame["optical_flow_file_path"] = os.path.join(
                    "..", "sensors", str(i) + "_optical_flow.png"
                )
            frame["transform_matrix"] = transform
            if (
                len(set(self.spawn_transforms_cams["fov"])) > 1
                or len(set(self.spawn_transforms_cams["height"])) > 1
                or len(set(self.spawn_transforms_cams["width"])) > 1
            ):
                intrinsics = self.get_camera_intrinsics_from_fov(
                    self.spawn_transforms_cams["fov"][i],
                    self.spawn_transforms_cams["width"][i],
                    self.spawn_transforms_cams["height"][i],
                )
                frame["fl_x"] = intrinsics["fx"]
                frame["fl_y"] = intrinsics["fy"]
                frame["cx"] = intrinsics["cx"]
                frame["cy"] = intrinsics["cy"]
                frame["w"] = self.spawn_transforms_cams["width"][i]
                frame["h"] = self.spawn_transforms_cams["height"][i]

            json_file["frames"].append(frame)

        with open(os.path.join(save_path, filename), "w") as f:
            json.dump(json_file, f, indent=4)

    def _save_lidar_transforms(self, path, filename):

        sensor_types = list(self.sensors.keys())

        if "sensor.lidar.ray_cast" in sensor_types:

            save_path = os.path.join(path, "transforms")

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            json_file = {
                "channels": int(self.sensor_info["channels"]),
                "points_per_second": int(self.sensor_info["points_per_second"]),
                "rotation_frequency": int(self.sensor_info["rotation_frequency"]),
                "range": self.sensor_info["range"],
                "frames": [],
            }

            for i, transform in enumerate(self.get_poses(pose_type="nerf")):
                frame = {}
                frame["file_path"] = os.path.join(
                    "..", "sensors", str(i) + "_lidar.ply"
                )
                frame["transform_matrix"] = transform
                json_file["frames"].append(frame)

            with open(os.path.join(save_path, filename), "w") as f:
                json.dump(json_file, f, indent=4)

    def reset(self):
        for sensor_type, sensors in self.sensors.items():
            for sensor in sensors:
                sensor.ready = False
