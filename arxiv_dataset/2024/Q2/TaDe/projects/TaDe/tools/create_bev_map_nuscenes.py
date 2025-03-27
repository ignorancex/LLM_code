import os
import argparse
from typing import Optional
from collections import OrderedDict

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from shapely import affinity, geometry
from nuscenes import NuScenes
from pyquaternion import Quaternion
from shapely.strtree import STRtree
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.map_expansion.map_api import NuScenesMap

CAMERA_NAMES = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]
LAYOUT_NAMES = [
    "drivable_area",
    "ped_crossing",
    "walkway",
    "carpark_area",
]
OBJECT_NAMES = [
    "car",
    "truck",
    "trailer",
    "bus",
    "construction_vehicle",
    "bicycle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "barrier",
]
LOCATIONS = [
    "boston-seaport",
    "singapore-onenorth",
    "singapore-queenstown",
    "singapore-hollandvillage",
]
PON_PALETTE = {
    "drivable_area": (0, 48, 117),
    "ped_crossing": (0, 72, 172),
    "walkway": (0, 61, 147),
    "carpark_area": (0, 39, 91),
    "car": (179, 0, 27),
    "truck": (108, 13, 27),
    "trailer": (69, 8, 17),
    "bus": (141, 2, 136),
    "construction_vehicle": (179, 85, 0),
    "bicycle": (0, 133, 41),
    "motorcycle": (38, 125, 128),
    "pedestrian": (204, 201, 41),
    "traffic_cone": (204, 133, 41),
    "barrier": (128, 128, 128),
    "mask": (0, 0, 0),
}
NUSCENES_PALETTE = {
    "drivable_area": (166, 206, 227),  # '#a6cee3'
    "ped_crossing": (251, 154, 153),  # '#fb9a99'
    "walkway": (227, 26, 28),  # '#e31a1c'
    "carpark_area": (255, 127, 0),  # '#ff7f00'
    "car": (255, 158, 0),  # Orange
    "truck": (255, 99, 71),  # Tomato
    "trailer": (255, 140, 0),  # Darkorange
    "bus": (255, 127, 80),  # Coral
    "construction_vehicle": (233, 150, 70),  # Darksalmon
    "bicycle": (220, 20, 60),  # Crimson
    "motorcycle": (255, 61, 99),  # Red
    "pedestrian": (0, 0, 230),  # Blue
    "traffic_cone": (47, 79, 79),  # Darkslategrey
    "barrier": (112, 128, 144),  # Slategrey
    "mask": (63, 63, 63),  # Grey
}  # RGB

colormap = PON_PALETTE
map_extents = [-25.0, 0.0, 25.0, 50.0]
map_resolution = 0.25


def iterate_samples(nuscenes, start_token):
    sample_token = start_token
    while sample_token != "":
        sample = nuscenes.get("sample", sample_token)
        yield sample
        sample_token = sample["next"]


def load_point_cloud(nuscenes, sample_data):
    # Load point cloud
    lidar_path = os.path.join(nuscenes.dataroot, sample_data["filename"])
    pcl = LidarPointCloud.from_file(lidar_path)
    return pcl.points[:3, :].T


def make_transform_matrix(record):
    """
    Create a 4x4 transform matrix from a calibrated_sensor or ego_pose record
    """
    transform = np.eye(4)
    transform[:3, :3] = Quaternion(record["rotation"]).rotation_matrix
    transform[:3, 3] = np.array(record["translation"])
    return transform


def get_sensor_transform(nuscenes, sample_data):
    # Load sensor transform data
    sensor = nuscenes.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
    sensor_tfm = make_transform_matrix(sensor)

    # Load ego pose data
    pose = nuscenes.get("ego_pose", sample_data["ego_pose_token"])
    pose_tfm = make_transform_matrix(pose)

    return np.dot(pose_tfm, sensor_tfm)


def transform(matrix, vectors):
    vectors = np.dot(matrix[:-1, :-1], vectors.T)
    vectors = vectors.T + matrix[:-1, -1]
    return vectors


def transform_polygon(polygon, affine):
    """
    Transform a 2D polygon
    """
    a, b, tx, c, d, ty = affine.flatten()[:6]
    return affinity.affine_transform(polygon, [a, b, c, d, tx, ty])


def render_polygon(mask, polygon, extents, resolution, value=1):
    if len(polygon) == 0:
        return
    polygon = (polygon - np.array(extents[:2])) / resolution
    polygon = np.ascontiguousarray(polygon).round().astype(np.int32)
    cv2.fillConvexPoly(mask, polygon, value)


def render_shapely_polygon(mask, polygon, extents, resolution):
    if polygon.geom_type == "Polygon":
        # Render exteriors
        render_polygon(mask, polygon.exterior.coords, extents, resolution, 1)

        # Render interiors
        for hole in polygon.interiors:
            render_polygon(mask, hole.coords, extents, resolution, 0)

    # Handle the case of compound shapes
    else:
        for poly in polygon:
            render_shapely_polygon(mask, poly, extents, resolution)


def category_to_detection_name(category_name: str) -> Optional[str]:
    """
    Default label mapping from nuScenes to nuScenes detection classes.
    Note that pedestrian does not include personal_mobility, stroller and wheelchair.
    :param category_name: Generic nuScenes class.
    :return: nuScenes detection class.
    """
    detection_mapping = {
        "movable_object.barrier": "barrier",
        "vehicle.bicycle": "bicycle",
        "vehicle.bus.bendy": "bus",
        "vehicle.bus.rigid": "bus",
        "vehicle.car": "car",
        "vehicle.construction": "construction_vehicle",
        "vehicle.motorcycle": "motorcycle",
        "human.pedestrian.adult": "pedestrian",
        "human.pedestrian.child": "pedestrian",
        "human.pedestrian.construction_worker": "pedestrian",
        "human.pedestrian.police_officer": "pedestrian",
        "movable_object.trafficcone": "traffic_cone",
        "vehicle.trailer": "trailer",
        "vehicle.truck": "truck",
    }

    if category_name in detection_mapping:
        return detection_mapping[category_name]
    else:
        return None


def get_map_masks(nuscenes, map_data, sample_data, extents, resolution):
    # Render each layer sequentially
    layers = [get_layer_mask(nuscenes, polys, sample_data, extents, resolution) for layer, polys in map_data.items()]

    return np.stack(layers, axis=0)


def get_layer_mask(nuscenes, polygons, sample_data, extents, resolution):
    # Get the 2D affine transform from bev coords to map coords
    tfm = get_sensor_transform(nuscenes, sample_data)[[0, 1, 3]][:, [0, 2, 3]]
    inv_tfm = np.linalg.inv(tfm)

    # Create a patch representing the birds-eye-view region in map coordinates
    map_patch = geometry.box(*extents)
    map_patch = transform_polygon(map_patch, tfm)

    # Initialise the map mask
    x1, z1, x2, z2 = extents
    mask = np.zeros((int((z2 - z1) / resolution), int((x2 - x1) / resolution)), dtype=np.uint8)

    # Find all polygons which intersect with the area of interest
    for polygon in polygons.query(map_patch):
        polygon = polygon.intersection(map_patch)

        # Transform into map coordinates
        polygon = transform_polygon(polygon, inv_tfm)

        # Render the polygon to the mask
        render_shapely_polygon(mask, polygon, extents, resolution)

    return mask.astype(np.bool_)


def get_object_masks(nuscenes, sample_data, extents, resolution):
    # Initialize object masks
    nclass = len(OBJECT_NAMES) + 1
    grid_width = int((extents[2] - extents[0]) / resolution)
    grid_height = int((extents[3] - extents[1]) / resolution)
    masks = np.zeros((nclass, grid_height, grid_width), dtype=np.uint8)

    # Get the 2D affine transform from bev coords to map coords
    tfm = get_sensor_transform(nuscenes, sample_data)[[0, 1, 3]][:, [0, 2, 3]]
    inv_tfm = np.linalg.inv(tfm)

    for box in nuscenes.get_boxes(sample_data["token"]):
        # Get the index of the class
        det_name = category_to_detection_name(box.name)
        if det_name not in OBJECT_NAMES:
            class_id = -1
        else:
            class_id = OBJECT_NAMES.index(det_name)

        # Get bounding box coordinates in the grid coordinate frame
        bbox = box.bottom_corners()[:2]
        local_bbox = np.dot(inv_tfm[:2, :2], bbox).T + inv_tfm[:2, 2]

        # Render the rotated bounding box to the mask
        render_polygon(masks[class_id], local_bbox, extents, resolution)

    return masks.astype(np.bool_)


def get_visible_mask(instrinsics, image_width, extents, resolution):
    # Get calibration parameters
    fu, cu = instrinsics[0, 0], instrinsics[0, 2]

    # Construct a grid of image coordinates
    x1, z1, x2, z2 = extents
    x, z = np.arange(x1, x2, resolution), np.maximum(np.arange(z1, z2, resolution), 0.01)
    ucoords = x / z[:, None] * fu + cu

    # Return all points which lie within the camera bounds
    return (ucoords >= 0) & (ucoords < image_width)


def get_occlusion_mask(points, extents, resolution):
    x1, z1, x2, z2 = extents

    # A 'ray' is defined by the ratio between x and z coordinates
    ray_width = resolution / z2
    ray_offset = x1 / ray_width
    max_rays = int((x2 - x1) / ray_width)

    # Group LiDAR points into bins
    rayid = np.round(points[:, 0] / points[:, 2] / ray_width - ray_offset)
    depth = points[:, 2]

    # Ignore rays which do not correspond to any grid cells in the BEV
    valid = (rayid > 0) & (rayid < max_rays) & (depth > 0)
    rayid = rayid[valid]
    depth = depth[valid]

    # Find the LiDAR point with maximum depth within each bin
    max_depth = np.zeros((max_rays,))
    np.maximum.at(max_depth, rayid.astype(np.int32), depth)

    # For each bev grid point, sample the max depth along the corresponding ray
    x = np.arange(x1, x2, resolution)
    z = np.maximum(np.arange(z1, z2, resolution)[:, None], 0.01)
    grid_rayid = np.round(x / z / ray_width - ray_offset).astype(np.int32)
    grid_max_depth = max_depth[np.clip(grid_rayid, a_min=0, a_max=9999)]

    # A grid position is considered occluded if the there are no LiDAR points
    # passing through it
    occluded = grid_max_depth < z
    return occluded


def process_scene(nuscenes, map_data, scene, dataroot):
    # Get the map corresponding to the current sample data
    log = nuscenes.get("log", scene["log_token"])
    scene_map_data = map_data[log["location"]]

    # Iterate over samples
    first_sample_token = scene["first_sample_token"]
    for sample in iterate_samples(nuscenes, first_sample_token):
        process_sample(nuscenes, scene_map_data, sample, dataroot)


def process_sample(nuscenes, map_data, sample, dataroot):
    # Load the lidar point cloud associated with this sample
    lidar_data = nuscenes.get("sample_data", sample["data"]["LIDAR_TOP"])
    lidar_pcl = load_point_cloud(nuscenes, lidar_data)

    # Transform points into world coordinate system
    lidar_transform = get_sensor_transform(nuscenes, lidar_data)
    lidar_pcl = transform(lidar_transform, lidar_pcl)

    # Iterate over sample data
    for camera in CAMERA_NAMES:
        sample_data = nuscenes.get("sample_data", sample["data"][camera])
        process_sample_data(nuscenes, map_data, sample_data, lidar_pcl, dataroot)


def process_sample_data(nuscenes, map_data, sample_data, lidar, dataroot):
    # Render static road geometry masks
    map_masks = get_map_masks(nuscenes, map_data, sample_data, map_extents, map_resolution)

    # Render dynamic object masks
    obj_masks = get_object_masks(nuscenes, sample_data, map_extents, map_resolution)

    mask = np.zeros_like(obj_masks[-1])
    # Ignore regions of the BEV which are outside the image
    sensor = nuscenes.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
    intrinsics = np.array(sensor["camera_intrinsic"])
    mask |= ~get_visible_mask(intrinsics, sample_data["width"], map_extents, map_resolution)

    # Transform lidar points into camera coordinates
    cam_transform = get_sensor_transform(nuscenes, sample_data)
    cam_points = transform(np.linalg.inv(cam_transform), lidar)
    mask |= get_occlusion_mask(cam_points, map_extents, map_resolution)

    bev_map = np.concatenate([map_masks, obj_masks[:-1], mask[None, :]], axis=0)

    """
    fpath = os.path.join(
        dataroot,
        "bev_maps_demo",
        sample_data["channel"],
        os.path.basename(sample_data["filename"]).split(".")[0],
    )

    image = Image.open(os.path.join(dataroot, sample_data["filename"]))

    # Save bev segmentation maps in "bev_maps_demo"
    visualize_bev_map(bev_map, fpath, image=image)
    """

    fpath = os.path.join(dataroot, "bev_maps", sample_data["channel"])

    os.makedirs(fpath, exist_ok=True)

    # len([*LAYOUT_NAMES, *OBJECT_NAMES, "mask"]) == 15
    np.savez_compressed(
        os.path.join(fpath, sample_data["token"]),
        bev_map=bev_map,
        bev_shape=bev_map.shape,
        resolution=map_resolution,
        extents=map_extents,
        cam2img=sensor["camera_intrinsic"],
        image_width=sample_data["width"],
        names=LAYOUT_NAMES + OBJECT_NAMES,
        colormap=colormap,
    )


def load_map_data(dataroot, location):
    # Load the NuScenes map object
    nusc_map = NuScenesMap(dataroot, location)

    map_data = OrderedDict()
    for layer in LAYOUT_NAMES:
        # Retrieve all data associated with the current layer
        records = getattr(nusc_map, layer)
        polygons = list()

        # Drivable area records can contain multiple polygons
        if layer == "drivable_area":
            for record in records:
                # Convert each entry in the record into a shapely object
                for token in record["polygon_tokens"]:
                    poly = nusc_map.extract_polygon(token)
                    if poly.is_valid:
                        polygons.append(poly)
        else:
            for record in records:
                # Convert each entry in the record into a shapely object
                poly = nusc_map.extract_polygon(record["polygon_token"])
                if poly.is_valid:
                    polygons.append(poly)

        # Store as an R-Tree for fast intersection queries
        map_data[layer] = STRtree(polygons)

    return map_data


def create_bev_map(dataroot):
    # Load NuScenes dataset
    nuscenes = NuScenes("v1.0-trainval", dataroot)
    # Preload NuScenes map data
    map_data = {location: load_map_data(dataroot, location) for location in LOCATIONS}
    for scene in tqdm(nuscenes.scene):
        process_scene(nuscenes, map_data, scene, dataroot)


def visualize_bev_map(bev_map, out_dir, image=None):
    os.makedirs(out_dir, exist_ok=True)
    if image:
        image.save(os.path.join(out_dir, "image.png"))

    bev_map = np.flip(bev_map, axis=1)

    # 200, 200, 3, RGB
    canvas_all = np.zeros((*bev_map.shape[-2:], 3), dtype=np.uint8)
    for i, (name, color) in enumerate(colormap.items()):
        if np.any(bev_map[i]):
            canvas = np.zeros((*bev_map.shape[-2:], 3), dtype=np.uint8)
            canvas[bev_map[i], :] = color
            canvas_all[bev_map[i], :] = color
            Image.fromarray(canvas).save(os.path.join(out_dir, name + ".png"))
        if i == 13:
            Image.fromarray(canvas_all).save(os.path.join(out_dir, "all.png"))
        elif i == 14:
            Image.fromarray(canvas_all).save(os.path.join(out_dir, "all_masked.png"))


def load_bev_map(fpath, sample_data_token):
    return np.load(os.path.join(fpath, sample_data_token + ".npz"))["bev_map"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default="data/nuscenes")
    args = parser.parse_args()
    create_bev_map(args.dataroot)
