# Copyright (C) 2025 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Marius Kästingschäfer and Théo Gieruc
# ==============================================================================

import math
import os
import json
import carla
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R


def fibonacci_sphere(span: float, N: int) -> list:
    """
    Generates `N` points on a half sphere using the Fibonacci spiral method, with each point
    separated by the golden angle, and scales the result by `span`.

    Parameters:
        span (float): a scaling factor for the output points
        N (int): the number of points to generate

    Returns:
        A list of `N` points in three-dimensional Cartesian coordinates, as a list of lists [x, y, z]
    """

    phi = math.pi * (3.0 - math.sqrt(5.0))
    ys = np.linspace(0, 1, N)

    points = [
        [
            span * math.cos(phi * i) * math.sqrt(1 - y * y),
            span * math.sin(phi * i) * math.sqrt(1 - y * y),
            span * y,
        ]
        for i, y in enumerate(ys)
    ]

    return points


def generate_sphere_transforms(
    origin: list, z_offset: float, radius: float, N: int
) -> list:
    """
    Generate 3D coordinates, pitch angles and yaw angles of N evenly distributed points on a sphere centered at origin with
    radius 'radius' and z-offset 'z_offset'.

    Parameters:
        origin (list): A list of 3 floats representing the x, y, and z coordinates of the center of the sphere.
        z_offset (float): A float representing the amount by which all the generated points will be lifted up along the z-axis.
        radius (float): A float representing the radius of the sphere.
        N (int): An integer representing the number of points to generate.

    Returns:
        A list of 3 numpy arrays containing the generated 3D coordinates, pitch angles, and yaw angles respectively.
        Each numpy array has N elements.
    """

    z_correction = [0, 0, z_offset]

    coordinates = np.array(fibonacci_sphere(1.0, N))
    pitchs = np.arcsin(coordinates[:, 2])
    yaws = np.sign(coordinates[:, 0]) * np.arccos(
        coordinates[:, 1]
        / (coordinates[:, 0] ** 2 + coordinates[:, 1] ** 2 + 1e-8) ** 0.5
    )
    coordinates = radius * coordinates + origin + z_correction

    return {"coordinates": coordinates, "pitchs": pitchs, "yaws": yaws}


def generate_nuscenes_transforms():
    transforms = {
        "CAM_FRONT": {
            "translation": [1.72200568, 0.00475453, 1.49491292],
            "rotation": {
                "pitch": -90.5879208855,
                "yaw": -0.7007173473950344,
                "roll": -89.50957761160001,
            },
        },
        "CAM_FRONT_RIGHT": {
            "translation": [1.58082566, -0.49907871, 1.51749368],
            "rotation": {
                "pitch": -32.42591154099997,
                "yaw": -0.9071318409999901,
                "roll": -89.66956817790008,
            },
        },
        "CAM_FRONT_LEFT": {
            "translation": [1.57525595, 0.50051938, 1.50696033],
            "rotation": {
                "pitch": -145.34114947130004,
                "yaw": -0.23019817163994902,
                "roll": -89.0143474659,
            },
        },
        "CAM_BACK": {
            "translation": [0.05524611, 0.01078824, 1.56794287],
            "rotation": {
                "pitch": 90.52804473049999,
                "yaw": 0.6418405371349465,
                "roll": -89.6044253498,
            },
        },
        "CAM_BACK_LEFT": {
            "translation": [1.04852048, 0.48305813, 1.56210154],
            "rotation": {
                "pitch": 161.42034397320006,
                "yaw": -0.14162822446094572,
                "roll": -88.81655348310001,
            },
        },
        "CAM_BACK_RIGHT": {
            "translation": [1.05945173, -0.46720295, 1.55050858],
            "rotation": {
                "pitch": 22.521190830999938,
                "yaw": -0.11343093253300289,
                "roll": -89.33706258359996,
            },
        },
    }

    coordinates = []
    pitchs = []
    yaws = []

    for key in transforms:
        coordinates.append(
            np.array(transforms[key]["translation"]) * np.array([1, -1, 1])
        )
        pitchs.append(transforms[key]["rotation"]["yaw"] / 180 * math.pi)
        yaws.append(-transforms[key]["rotation"]["pitch"] / 180 * math.pi + math.pi)

    return {"coordinates": coordinates, "pitchs": pitchs, "yaws": yaws}


def carla_to_nerf_normalized(camera_transform: carla.Transform, origin, RADIUS):
    """
    Convert a carla.Transform to a 4x4 matrix that can be used in Nerfstudio.
    """
    unreal_location = camera_transform.location
    unreal_rotation = camera_transform.rotation

    # center coordinates around [0,0,0] and normalize coordinates [betw. -1 and 1]
    normalized_x = (unreal_location.x - origin.location.x) / RADIUS
    normalized_y = (unreal_location.y - origin.location.y) / RADIUS
    normalized_z = (unreal_location.z - origin.location.z) / RADIUS

    # Convert from Unreal Engine to Blender coordinate system
    # From Carla Docs: Warning: The declaration order is different in CARLA (pitch,yaw,roll), and in the Unreal Engine Editor (roll,pitch,yaw). When working in a build from source, don't mix up the axes' rotations.
    openGL_matrix = carla.Transform(
        # carla.Location(x=-unreal_location.z/normalize_val, y=unreal_location.x/normalize_val, z=unreal_location.y/normalize_val),
        carla.Location(x=-normalized_z, y=normalized_x, z=normalized_y),
        carla.Rotation(
            pitch=unreal_rotation.yaw + 90,
            yaw=unreal_rotation.roll + 90,
            roll=-unreal_rotation.pitch,
        ),
        # carla.Rotation(pitch=unreal_rotation.yaw + 90, yaw=unreal_rotation.roll + 90, roll=unreal_rotation.pitch)
    )

    return openGL_matrix.get_matrix()


def carla_to_nerf_unnormalized(camera_transform: carla.Transform):
    """
    Convert a carla.Transform to a 4x4 matrix that can be used in Nerfstudio.
    """
    unreal_location = camera_transform.location
    unreal_rotation = camera_transform.rotation

    normalized_x = unreal_location.x
    normalized_y = unreal_location.y
    normalized_z = unreal_location.z

    # Convert from Unreal Engine to Blender coordinate system
    # From Carla Docs: Warning: The declaration order is different in CARLA (pitch,yaw,roll), and in the Unreal Engine Editor (roll,pitch,yaw). When working in a build from source, don't mix up the axes' rotations.
    openGL_matrix = carla.Transform(
        carla.Location(x=-normalized_z, y=normalized_x, z=normalized_y),
        carla.Rotation(
            pitch=unreal_rotation.yaw + 90,
            yaw=unreal_rotation.roll + 90,
            roll=-unreal_rotation.pitch,
        ),
    )

    return openGL_matrix.get_matrix()

def extract_xyz_yaw_pitch_roll(matrix: np.ndarray):
    """
    Extract x, y, z, yaw, pitch, and roll from a 4x4 Rt matrix.
    """
    # Extract translation components
    x = matrix[0, 3]
    y = matrix[1, 3]
    z = matrix[2, 3]

    # Extract rotation components
    rotmat = R.from_matrix(matrix[:3, :3])
    yaw, pitch, roll = rotmat.as_euler("ZYX", degrees=True)
    pitch = -pitch

    return x, y, z, yaw, roll, pitch

def carlarpy_to_nerf(spawn_transforms: dict):
    """
    Convert a carla.Transform to a 4x4 matrix that can be used in Nerfstudio.
    """

    coordinates = spawn_transforms["coordinates"]
    pitchs = spawn_transforms["pitchs"]
    yaws = spawn_transforms["yaws"]

    nerf_matrices = []
    yaw_correction = +90

    for coord, pitch, yaw in zip(coordinates, pitchs, yaws):
        pitch = -pitch * (180 / math.pi)
        yaw = -(yaw * (180 / math.pi) + yaw_correction)

        transform_cam = carla.Transform(
            carla.Location(*coord), carla.Rotation(pitch=pitch, yaw=yaw)
        )

        nerf_matrix = carla_to_nerf_normalized(transform_cam)

        nerf_matrices.append(nerf_matrix)

    return nerf_matrices


def get_OpenGL_matrices_normalized(coordinates, pitchs, yaws, origin, RADIUS):
    nerf_matrices = []
    yaw_correction = +90

    for coord, pitch, yaw in zip(coordinates, pitchs, yaws):
        pitch = -pitch * (180 / math.pi)
        yaw = -(yaw * (180 / math.pi) + yaw_correction)

        transform_cam = carla.Transform(
            carla.Location(*coord), carla.Rotation(pitch=pitch, yaw=yaw)
        )

        nerf_matrix = carla_to_nerf_normalized(transform_cam, origin, RADIUS)

        nerf_matrices.append(nerf_matrix)

    return nerf_matrices


def get_OpenGL_matrices_unnormalized(coordinates, pitchs, yaws, origin, RADIUS):
    nerf_matrices = []
    yaw_correction = +90

    for coord, pitch, yaw in zip(coordinates, pitchs, yaws):
        pitch = -pitch * (180 / math.pi)
        yaw = -(yaw * (180 / math.pi) + yaw_correction)

        transform_cam = carla.Transform(
            carla.Location(*coord), carla.Rotation(pitch=pitch, yaw=yaw)
        )

        nerf_matrix = carla_to_nerf_unnormalized(transform_cam, origin, RADIUS)

        nerf_matrices.append(nerf_matrix)

    return nerf_matrices
