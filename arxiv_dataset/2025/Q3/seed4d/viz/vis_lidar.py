# Copyright (C) 2025 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Marius Kästingschäfer and Théo Gieruc
# ==============================================================================

import os
import datetime
import argparse
import numpy as np
import open3d as o3d
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main(args):
    
    # Load the PLY file
    point_cloud = o3d.io.read_point_cloud(args.lidar_path)
    points = np.array(point_cloud.points)

    # Calculate distance from origin for each point
    distances = np.linalg.norm(points, axis=1)
    fig = plt.figure(figsize=(15, 5))
    # Front view
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c=distances, cmap='jet', s=1)
    ax1.view_init(elev=0, azim=0)  # Front view
    ax1.set_title('Front View')
    # Side view
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.scatter(points[:, 0], points[:, 1], points[:, 2], c=distances, cmap='jet', s=1)
    ax2.view_init(elev=0, azim=90)  # Side view
    ax2.set_title('Side View')
    # Top view
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.scatter(points[:, 0], points[:, 1], points[:, 2], c=distances, cmap='jet', s=1)
    ax3.view_init(elev=90, azim=0)  # Top view
    ax3.set_title('Top View')
    plt.subplots_adjust(wspace=-0.3, hspace=-0.6)
    plt.savefig(f'{args.save_at}_lidar_views.png', bbox_inches='tight')
    plt.close()


    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=distances, cmap='jet', s=25)
    ax.view_init(elev=50, azim=45)  # z rotation, x rotation
    ax.set_xlim3d(-10 , 12)
    ax.set_ylim3d(-12 , 10)
    ax.axis('off')
    plt.subplots_adjust(wspace=-0.3, hspace=-0.6)
    plt.savefig(f'{args.save_at}_lidar_single_view.png', bbox_inches='tight')
    plt.close()
    
    img = Image.open(f'{args.save_at}_lidar_single_view.png')
    resized_img = img.resize((800, 450))
    plt.imshow(resized_img)
    plt.axis('off') 
    plt.savefig(f'{args.save_at}_lidar_single_view.png', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    os.makedirs('/seed4d/logs', exist_ok=True)
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--lidar_path", type=str, help='Path to a lidar .ply file')    
    current_date = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    parser.add_argument("--save_at", type=str, default=f'/seed4d/logs/{current_date}')
    
    args = parser.parse_args()
    
    main(args)

# Example static: python3.8 vis_lidar.py --lidar_path /seed4d/data/static/Town01/ClearNoon/vehicle.mini.cooper_s/spawn_point_1/step_0/ego_vehicle/nuscenes_lidar/sensors/0_lidar.ply