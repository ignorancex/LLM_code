# Copyright (C) 2025 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Marius Kästingschäfer and Théo Gieruc
# ==============================================================================

import os
import argparse
import datetime
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def load_image(timestep, vehicle_name, path):
    if vehicle_name == "nuscenes":
        file_path = path + "/step_" + str(timestep) + "/"+ vehicle_name + "/sensors/0_rgb.png"
    else:
        file_path = path + "/step_" + str(timestep) + "/"+ vehicle_name + "/nuscenes/sensors/0_rgb.png"
    
    img = mpimg.imread(file_path)
    return img


def main(args):

    labels_column = [rf'$t_{{{i}}}$' for i in args.timesteps]
    labels_row = [f'Vehicle {i}' if i != 'ego_vehicle' else 'Vehicle Ego' for i in args.vehicles]
    fontsize = 10

    fig, axes = plt.subplots(nrows=len(args.vehicles), ncols=len(args.timesteps), figsize=(len(args.timesteps)*1.8, len(args.vehicles)*0.8))

    for i in range(len(args.vehicles)):
        for j in range(len(args.timesteps)):
            vehicle_name = args.vehicles[i]
            timestep = args.timesteps[j]
            img = load_image(timestep, vehicle_name, args.path)
            if i == 2:
                axes[i, j].imshow(img)
            else: 
                axes[i, j].imshow(img)
            axes[i, j].axis('off') 
            if i == 0:
                axes[i, j].set_title(f'{labels_column[j]}', fontsize=fontsize)
            if j == 0:
                axes[i, j].text(-0.8, 0.5, f'{labels_row[i]}', fontsize=fontsize, transform=axes[i, j].transAxes)
            
    plt.subplots_adjust(wspace=-0.3, hspace=-0.2)
    plt.tight_layout()
    saving_path = f"{args.save_at}_timesteps.png"
    plt.savefig(saving_path, bbox_inches='tight' , pad_inches=0)
    plt.close()
    
if __name__ == "__main__":
    os.makedirs('/seed4d/logs', exist_ok=True)
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--path", type=str, help='Path to a lidar .ply file')    
    current_date = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    parser.add_argument("--save_at", type=str, default=f'/seed4d/logs/{current_date}')
    parser.add_argument("--timesteps", nargs='+', help='List of timesteps to plot', default=['4', '8', '15', '16', '27'])
    parser.add_argument("--vehicles", nargs='+', help='Name vehicles to include. Must be int or ego_vehicles', default=['71', '74', '79', '82', '87', 'ego_vehicle'])
    args = parser.parse_args()
    
    main(args)
    
# Example: python3.8 vis_timesteps.py --path /seed4d/data/dynamic/Town10HD/ClearNoon/vehicle.mini.cooper_s/spawn_point_12
