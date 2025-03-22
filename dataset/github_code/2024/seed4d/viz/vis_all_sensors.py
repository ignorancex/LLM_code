# Copyright (C) 2025 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Marius Kästingschäfer and Théo Gieruc
# ==============================================================================

import os
import cv2
import json
import datetime
import argparse
import numpy as np
import open3d as o3d
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def load_image(sensor_name, path, num_image):
    file_path = path + "/" + str(num_image) + "_" + sensor_name + ".png"
    
    if sensor_name == "semantic_segmentation":
        img = Image.open(file_path)
        img = labels_to_cityscapes_palette(img)
        img = Image.fromarray(img.astype('uint8'))
    else:
        img = mpimg.imread(file_path)
    return img

def labels_to_cityscapes_palette(image):
    """
    Convert an image containing CARLA semantic segmentation labels to
    Cityscapes palette.
    """
    classes = {
    0: [0, 0, 0],             # Unlabeled
    1: [128, 64, 128],        # Roads
    2: [244, 35, 232],        # SideWalks
    3: [70, 70, 70],          # Building
    4: [102, 102, 156],       # Wall
    5: [190, 153, 153],       # Fence
    6: [153, 153, 153],       # Pole
    7: [250, 170, 30],        # TrafficLight
    8: [220, 220, 0],         # TrafficSign
    9: [107, 142, 35],        # Vegetation
    10: [152, 251, 152],      # Terrain
    11: [70, 130, 180],       # Sky
    12: [220, 20, 60],        # Pedestrian
    13: [255, 0, 0],          # Rider
    14: [0, 0, 142],          # Car
    15: [0, 0, 70],           # Truck
    16: [0, 60, 100],         # Bus
    17: [0, 60, 100],         # Train
    18: [0, 0, 230],          # Motorcycle
    19: [119, 11, 32],        # Bicycle
    20: [110, 190, 160],      # Static
    21: [170, 120, 50],       # Dynamic
    22: [55, 90, 80],         # Other
    23: [45, 60, 150],        # Water
    24: [157, 234, 50],       # RoadLine
    25: [81, 0, 81],          # Ground
    26: [150, 100, 100],      # Bridge
    27: [230, 150, 140],      # RailTrack
    28: [180, 165, 180]       # GuardRail
}
    image = np.array(image)
    result = np.zeros((image.shape[0], image.shape[1], 3))
    for key, value in classes.items():
        result[np.where(image == key)] = value
    return result

def write_lidar_img(lidar_path, saving_path):
    # Load the PLY file
    point_cloud = o3d.io.read_point_cloud(lidar_path)
    points = np.array(point_cloud.points)
    # Calculate distance from origin for each point
    distances = np.linalg.norm(points, axis=1)
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=distances, cmap='jet', s=25)
    ax.view_init(elev=45, azim=45)  # z rotation, x rotation
    ax.set_xlim3d(-10 , 12)
    ax.set_ylim3d(-12 , 10)
    ax.axis('off')
    plt.subplots_adjust(wspace=-0.3, hspace=-0.6)
    plt.savefig(saving_path, bbox_inches='tight')
    plt.close()

def get_image_point(loc, K, w2c):
        # Calculate 2D projection of 3D coordinate

        # Format the input coordinate (loc is a 3D point)
        point = np.concatenate((loc, np.array([1.0])))
        # transform to camera coordinates
        point_camera = np.matmul(w2c, point)
        point_camera = np.array(point_camera)[:3]
        # now project 3D->2D using the camera matrix
        point_img = np.matmul(K, point_camera)
        # normalize
        if point_img[2] == 0:
            point_img[0] /= point_img[2]+0.000001
            point_img[1] /= point_img[2]+0.000001
        else: 
            point_img[0] /= point_img[2]
            point_img[1] /= point_img[2]

        if point_img[2] > 0:
            return [-1000,-1000]
        
        return point_img[0:2]
    
def get_bbox_images(folder):
    
    with open(folder + "3Dboundingbox.json") as json_data:
        bboxes = json.load(json_data)
    
    with open(folder + "ego_vehicle/nuscenes/transforms/transforms.json") as json_data:
        data = json.load(json_data)

    # bbox
    bbox_array = []
    for key, bbox in bboxes.items():
        bbox_array.append(bbox["bb"])
    edges_bbox = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]

    bbox_array = np.array(bbox_array)
            
    cam_mesh = np.array([
        [0,0,0],
        [1,1,1],
        [1,1,-1],
        [1,-1,-1],
        [1,-1,1],
    ], dtype=np.float32)[:,[1,2,0]]  * -np.array([0.3,0.2,0.2])

    edges_cam =[[0,1], [0,2], [0,3], [0,4], [1,2], [2,3], [3,4], [4,1]]
    cam_mesh = np.concatenate((cam_mesh, np.ones((5,1))), axis=1)

    c2w_nuscenes = np.asarray([frame["transform_matrix"] for frame in data["frames"]])

    images = []
    for cam in range(6):
        
        # intrinsics
        c2w = np.array(data['frames'][cam]["transform_matrix"])
        w2c = np.linalg.inv(c2w)
        intrinsics = data['frames'][cam]
        fl_x, fl_y, cx, cy, w, h = intrinsics["fl_x"], intrinsics["fl_y"], intrinsics["cx"], intrinsics["cy"], intrinsics["w"], intrinsics["h"]
        K = np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]]) # camera intrinsics 3x4
        rgb_path = os.path.join(folder, 'ego_vehicle/nuscenes', data['frames'][cam]['file_path'][3:])
        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        # bbobx
        collector = []
        for i in range(len(list(bboxes.keys()))):
            ego_bbox = bboxes[list(bboxes.keys())[i]]['bb']
            ego_center = np.asarray(ego_bbox).mean(0)
            rgb_bbox = rgb.copy()
            for i, bbox in enumerate(bbox_array[:-1]):
                if  np.linalg.norm(ego_bbox - bbox.mean(0)) > 150:
                    continue
                bbox = np.array(bbox) # 8x3
                img_points = []
                for point in bbox:
                    point = [-point[2], point[0], point[1]]
                    img_points.append(get_image_point(point, K, w2c))
        
                img_points = np.array(img_points)
                img_points[:,0] = w - img_points[:,0]
                collector.append(img_points)

        # plotting
        for img_points in collector:
            for edge in edges_bbox:
                if np.linalg.norm(img_points[edge[0]] - img_points[edge[1]]) < 400:
                    cv2.line(rgb_bbox, (int(img_points[edge[0]][0]), int(img_points[edge[0]][1])), (int(img_points[edge[1]][0]), int(img_points[edge[1]][1])), (255,0,0), 5)
        
        images.append(rgb_bbox)
    
    return images

def plot_sensors_full(args, bbox_images):
    
    if args.category == 'static':
        sensor_path_names = {0: "rgb", 1: "obj", 2: "depth", 3: "semantic_segmentation", 4: "instance_segmentation",5: "rgb"}
        labels_row = ["Full Image \n(RGB)", "vehicles only \n(RGB)", "Depth map", "Semantic \nSegmentation", "Instance \nSegmentation", "Bounding boxes"]
    elif args.category == 'dynamic':
        sensor_path_names = {0: "rgb", 1: "obj", 2: "depth", 3: "optical_flow", 4: "semantic_segmentation", 5: "instance_segmentation", 6: "rgb"}
        labels_row = ["Full Image \n(RGB)", "vehicles only \n(RGB)", "Depth map", "Optical Flow", "Semantic \nSegmentation", "Instance \nSegmentation", "Bounding boxes"]
    else:
        assert False, "Invalid dataset type"

    reorder ={0: 2, 1: 0, 2: 1, 3: 5, 4: 3, 5: 4}
    labels_column = ["Front left", "Front center", "Front right", "Rear left", "Rear center", "Rear right"]
        
    saving_path = f'{args.save_at}_sensors_full.png'
    path = f'{args.path}ego_vehicle/nuscenes/sensors'
        
    fig, axes = plt.subplots(nrows=len(sensor_path_names), ncols=6, figsize=(12, 5))
        
    indices = {
        'static': {0, 1, 3, 4},
        'dynamic': {0, 1, 3, 4, 5}
    }

    for i in tqdm(range(len(sensor_path_names)), desc="Processing full sensors"):
        for j in range(6):
            sensor_name = sensor_path_names[i]
            jdx = reorder[j]
            img = load_image(sensor_name, path, jdx)
            # rgb image
            if i in indices.get(args.category, set()):
                axes[i, j].imshow(img)
            # depth maps
            elif i == 2:
                axes[i, j].imshow(img, cmap='jet')
            # bounding boxes
            elif i == 5:
                axes[i, j].imshow(bbox_images[jdx])
            elif i == 6:
                axes[i, j].imshow(bbox_images[jdx])
            axes[i, j].axis('off')
                
            if i == 0:
                axes[i, j].set_title(f'{labels_column[j]}', fontsize=args.fontsize_full)
            if j == 0:
                axes[i, j].text(-1.5, 0.5, f'{labels_row[i]}', fontsize=args.fontsize_full, transform=axes[i, j].transAxes)
                
    plt.subplots_adjust(wspace=-0.3, hspace=-0.2)
    plt.tight_layout()
    plt.savefig(saving_path, bbox_inches='tight')
    plt.close()
    
    
def plot_sensors(args, bbox_images):
    
    if args.category == 'static':
        sensor_path_names = {0: "rgb", 1: "obj", 2: "depth", 3: "semantic_segmentation", 4: "instance_segmentation",5: "rgb"}
        labels_row = ["Full Image (RGB)", "Vehicles Image (RGB)", "Depth Map", "Semantic Segmentation", "Instance Segmentation", "Bounding Boxes", "LiDAR", ""]
    elif args.category == 'dynamic':
        sensor_path_names = {0: "rgb", 1: "obj", 2: "depth", 3: "optical_flow", 4: "semantic_segmentation", 5: "instance_segmentation", 6: "rgb"}
        labels_row = ["Full Image (RGB)", "Vehicles Image (RGB)", "Depth Map", "Optical Flow", "Semantic Segmentation", "Instance Segmentation", "Bounding Boxes", "LiDAR"]
    else:
        assert False, "Invalid dataset type"
    
    lidar_path = args.path + "ego_vehicle/nuscenes_lidar/sensors" + '/0_lidar.ply'
    pc_image_path = f'{args.save_at}_lidar.png'
    write_lidar_img(lidar_path, pc_image_path)
    saving_path = f'{args.save_at}_sensors.png'
    path = f'{args.path}ego_vehicle/nuscenes/sensors'
    
    indices = {
        'static': {0, 1, 3, 4},
        'dynamic': {0, 1, 3, 4, 5, 6}
    }
    
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 6))
    for i in tqdm(range(2), desc="Processing single sensor output"): # rows
        for j in range(4): # colums
            index = i*4 + j
            
            if args.category == 'static':
                if index < len(sensor_path_names):
                    sensor_name = sensor_path_names[index]
                    jdx = 0
                    img = load_image(sensor_name, path, jdx)
                    if i in indices.get(args.category, set()):
                        axes[i, j].imshow(img)
                    if i == 2:
                        axes[i, j].imshow(img, cmap='jet')
                    elif index == 5:
                        axes[i, j].imshow(bbox_images[0])
                elif index == 6:
                    # LiDAR
                    resized_img = Image.open(pc_image_path).resize((800, 450))
                    axes[i, j].imshow(resized_img)
                
            if args.category == 'dynamic':
                if index < len(sensor_path_names):
                    sensor_name = sensor_path_names[index]
                    jdx = 0
                    img = load_image(sensor_name, path, jdx)
                    if i in indices.get(args.category, set()):
                        axes[i, j].imshow(img)
                    if i == 2:
                        axes[i, j].imshow(img, cmap='jet')
                    elif index == 6:
                        axes[i, j].imshow(bbox_images[0])
                elif index == 7:
                    # LiDAR
                    resized_img = Image.open(pc_image_path).resize((800, 450))
                    axes[i, j].imshow(resized_img)
                        
            axes[i, j].axis('off')
            axes[i, j].set_title(labels_row[index], fontsize=args.fontsize_single, va='bottom', y=-0.35)

    plt.subplots_adjust(wspace=-0.65, hspace=-1.03)
    plt.tight_layout()
    plt.savefig(saving_path, bbox_inches='tight')
    plt.close()

def main(args):

    bbox_images = get_bbox_images(args.path)
    plot_sensors_full(args, bbox_images)
    plot_sensors(args, bbox_images)

if __name__ == "__main__":
    os.makedirs('/seed4d/logs', exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="path to the step")
    parser.add_argument("--category", type=str, default=None, help="static or dynamic") 
    current_date = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    parser.add_argument("--save_at", type=str, default=f'/seed4d/logs/{current_date}')
    parser.add_argument("--fontsize_full", type=int, default=10)
    parser.add_argument("--fontsize_single", type=int, default=16)
    
    args = parser.parse_args()
    
    main(args)

# IMPORTANT: only implemented currently for the ego vehicle

# Example static: python3.8 vis_all_sensors.py --path /seed4d/data/static/Town02/ClearNoon/vehicle.mini.cooper_s/spawn_point_1/step_0/ --category static
# Example dynamic: python3.8 vis_all_sensors.py --path /seed4d/data/dynamic/Town01/ClearNoon/vehicle.mini.cooper_s/spawn_point_4/step_0/ --category dynamic