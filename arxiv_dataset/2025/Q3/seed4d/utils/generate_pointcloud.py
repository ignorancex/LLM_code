# Copyright (C) 2025 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Marius Kästingschäfer and Théo Gieruc
# ==============================================================================

import argparse
import json
import os
import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm


def get_single_point_cloud(
    rgb_image_path, depth_map_path, camera_intrinsics, transform
):
    """
    Convert RGB and depth images to point cloud

    Args:
        rgb_image_path (str): Path to RGB image
        depth_map_path (str): Path to depth map
        camera_intrinsics (np.array): Camera intrinsics

    Returns:
        o3d.geometry.PointCloud: Point cloud
    """

    rotation = transform[:3, :3]
    translation = transform[:3, 3] / 100.0

    transform[:3, :3] = rotation.T
    # transform[0,:3] = transform[0,:3] * -1
    # transform[1,:3] = transform[1,:3] * -1
    transform[2, :3] = transform[2, :3] * -1
    transform[:3, 3] = translation

    depth_raw = o3d.io.read_image(depth_map_path)

    color_raw_cv = cv2.imread(rgb_image_path, cv2.COLOR_RGB2BGR)
    color_raw = o3d.geometry.Image(color_raw_cv)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw,
        depth_raw,
        convert_rgb_to_intensity=False,
        depth_trunc=65 * 1e3,
        depth_scale=1.0,
    )

    (height, width, _) = np.array(color_raw).shape

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width,
        height,
        camera_intrinsics["fx"],
        camera_intrinsics["fy"],
        camera_intrinsics["cx"],
        camera_intrinsics["cy"],
    )

    return o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsic, transform
    )


def get_pointcloud_from_transform(root, file):
    with open(os.path.join(root, file), "r") as f:
        transform_info = json.load(f)

    fx = transform_info["fl_x"]
    fy = transform_info["fl_y"]
    cx = transform_info["cx"]
    cy = transform_info["cy"]

    camera_intrinsics = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}

    point_clouds = []
    for frame in transform_info["frames"]:
        if ("depth_file_path" in frame.keys()) and ("file_path" in frame.keys()):
            point_cloud = get_single_point_cloud(
                os.path.join(root, frame["file_path"]),
                os.path.join(root, frame["depth_file_path"]),
                camera_intrinsics,
                np.array(frame["transform_matrix"]),
            )
            point_clouds.append(point_cloud)
        else:
            return None

    pcd_combined = o3d.geometry.PointCloud()
    for point_cloud in point_clouds:
        pcd_combined += point_cloud

    return pcd_combined


def main(args):
    if args.pointcloud:

        # walk through the directory and get all transforms.json files
        for root, dirs, files in tqdm(os.walk(args.data_dir)):
            if "transforms.json" in files:
                pcd = get_pointcloud_from_transform(root, "transforms.json")
                if pcd:
                    # o3d.visualization.draw_geometries([pcd])
                    o3d.io.write_point_cloud(os.path.join(root, "point_cloud.ply"), pcd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument(
        "--pointcloud", action="store_true", help="Generate point clouds"
    )
    args = parser.parse_args()

    main(args)
