# Copyright (C) 2025 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Marius Kästingschäfer and Théo Gieruc
# ==============================================================================

import os
import json
import argparse
import datetime
import numpy as np
from PIL import Image
import open3d as o3d 
from matplotlib import pyplot as pltc

def path_to_info(transforms_path, index):
    pass

def img_to_pointcloud(img, depth, K, Rt, voxel_size=0.01):
    
    img = (img*256).astype(np.int8)
    rgb = o3d.geometry.Image(np.ascontiguousarray(img))
    depth = o3d.geometry.Image(np.ascontiguousarray(depth))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth, depth_scale=1.0, depth_trunc=50.0, convert_rgb_to_intensity=False)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(int(cx*2), int(cy*2), fx, fy, cx, cy)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, Rt)
    downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    # print(f"Created point cloud with {len(pcd.points)} points, saved {len(downpcd.points)}")
    return downpcd
    
def transform_file_reader(transform_path, voxel_size=0.1, viewpoint='sphere'):
    
    
    with open(transform_path, 'r') as f: jsonData = json.load(f)
    
    if viewpoint == 'sphere':
        # global intrinsics
        camera_dict = {}
        camera_dict['fl_x'] = jsonData['fl_x']
        camera_dict['fl_y'] = jsonData['fl_y']
        camera_dict['cx'] = jsonData['cx']
        camera_dict['cy'] = jsonData['cy']
        camera_dict['h'] = jsonData['h']
        camera_dict['w'] = jsonData['w']
        K = np.zeros((3,4)) # (C,3,4)
        K[0,0] = jsonData['fl_x']
        K[1,1] = jsonData['fl_y']
        K[2,2] = 1
        K[0,2] = jsonData['cx']
        K[1,2] = jsonData['cy']
        frames = jsonData['frames']
        
    elif viewpoint == 'nuscenes':
        frames = jsonData['frames']
    
    base_image_dir = transform_path[:transform_path.rfind('/', 0, transform_path.rfind('/'))]
    
    pcd_combined = o3d.geometry.PointCloud()
    w2c_combined = []

    for frame_dicts in frames:
        # separating image and depth_map information
        image_path = base_image_dir + '/' + frame_dicts['file_path'][2:]
        depth_map_path = base_image_dir + '/' + frame_dicts['depth_file_path'][2:]
        
        if viewpoint == 'nuscenes':
            # per image intrinsics
            camera_dict = {}
            camera_dict['fl_x'] = frame_dicts['fl_x']
            camera_dict['fl_y'] = frame_dicts['fl_y']
            camera_dict['cx'] = frame_dicts['cx']
            camera_dict['cy'] = frame_dicts['cy']
            camera_dict['h'] = frame_dicts['h']
            camera_dict['w'] = frame_dicts['w']
            K = np.zeros((3,4)) # (C,3,4)
            K[0,0] = frame_dicts['fl_x']
            K[1,1] = frame_dicts['fl_y']
            K[2,2] = 1
            K[0,2] = frame_dicts['cx']
            K[1,2] = frame_dicts['cy']
        
        # reading images and depth 
        image = Image.open(image_path).resize((camera_dict['w'], camera_dict['h']))
        image = np.array(image, dtype=float)[:, :, :-1]/255.0
        
        depth_map = Image.open(depth_map_path)
        depth_map = depth_map.resize((camera_dict['w'], camera_dict['h']))
        depth_map = np.array(depth_map, dtype=float)/1000.0
        c2w = np.array(frame_dicts['transform_matrix'])
        c2w[0:3, 1] = -c2w[0:3, 1]
        c2w[0:3, 2] = -c2w[0:3, 2]
        
        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        
        # Combining with all points
        pcd_combined += img_to_pointcloud(image.astype(np.float32), depth_map.astype(np.float32), K, w2c, voxel_size)
        w2c_combined.append(w2c)
        
    return pcd_combined, K, w2c_combined

def main(args):
    
    pcd_combined = o3d.geometry.PointCloud()
    K_collector = []
    w2cs_collector = []
    
    for viewpoint in args.viewpoints:
        for vehicle in args.vehicles:
            path = f'{args.path}/{vehicle}/{viewpoint}/transforms/transforms.json'
            pcd_single, K, w2cs = transform_file_reader(path, args.voxel_size, viewpoint)
            if args.rotate:
                pcd_single = pcd_single.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, np.radians(270)]))
            print(f'Number of points for vehicle {vehicle} and view {viewpoint}: {len(pcd_single.points)}')
            pcd_combined += pcd_single
            K_collector.append(K)
            w2cs_collector.append(w2cs)
        
    o3d.io.write_point_cloud(args.save_at, pcd_combined)
    print(f'Total number of points: {len(pcd_combined.points)}')

    #if args.viewpoint == 'sphere':
    #    K_collector = [K_collector]*100
    #    w2cs_collector = [[file] for file in w2cs_collector]

if __name__ == "__main__":
    os.makedirs('/seed4d/logs', exist_ok=True)
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--path", type=str)
    parser.add_argument("--vehicles", nargs='+', help='Name vehicles to inlcude', default=['ego_vehicle'])
    parser.add_argument("--voxel_size", type=float, default=0.2, help="Voxel size for 3D processing. Higher values results in fewer points.")
    parser.add_argument("--viewpoints", nargs='+', help='Name viewpoint(s) to inlcude default is sphere. possible are nuscenes, sphere or nuscenes sphere', default=['nuscenes'])
    parser.add_argument("--rotate", default=False, help="Enable rotation (default: False)")
    
    current_date = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    parser.add_argument("--save_at", type=str, default=f'/seed4d/logs/{current_date}_pointcloud.ply')
    
    args = parser.parse_args()
    
    main(args)

# Example static: python3.8 get_point_cloud.py --path /seed4d/data/static/Town01/ClearNoon/vehicle.mini.cooper_s/spawn_point_1/step_0 --voxel_size 0.1 --viewpoint nuscenes sphere
# Example dynamic: python3.8 get_point_cloud.py --path /seed4d/data/dynamic/Town01/ClearNoon/vehicle.mini.cooper_s/spawn_point_12/step_0 --vehicles 371 372 373 ego_vehicle --voxel_size 0.1 --viewpoint nuscenes sphere