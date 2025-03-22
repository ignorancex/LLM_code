# Copyright (C) 2025 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Marius Kästingschäfer and Théo Gieruc
# ==============================================================================

import os
import re
import imageio.v2 as imageio
import argparse
import numpy as np
from PIL import Image
import datetime
from matplotlib import pyplot as plt

def load_image(path):
    if "semantic_segmentation" in path:
        img = Image.open(path)
        img = labels_to_cityscapes_palette(img)
        img = Image.fromarray(img.astype('uint8'))
    else:
        img = Image.open(path)
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

def create_Nx3_subplot(args, folder_path):
    
    
    image_files = []
    for sensor in args.sensors:
        image_files.extend([f'{idx}_{sensor}.png' for idx in [2, 0, 1]])
    
    images = [load_image(os.path.join(folder_path, filename)) for filename in image_files]
    
    name_mapping = {
            'rgb': 'rgb',
            'obj': 'obj',
            'optical_flow': 'flow',
            'semantic_segmentation': 'sem.',
            'instance_segmentation': 'inst.',
            'depth': 'depth'
        }
    
    row_labels = [name_mapping.get(sensor, sensor) for sensor in args.sensors]
    column_labels = ['Front Left', 'Front', 'Front Right']
    
    n = len(args.sensors)
    fig, axs = plt.subplots(n, 3, figsize=(8, n))
    
    for i, ax in enumerate(axs.flat):
        if 'depth' in row_labels[int(i/3)]:
            ax.imshow(images[i], cmap='jet')
        else:
            ax.imshow(images[i])
        ax.axis('off')
        
    if args.labels:
        # Add row labels
        for i, ax in enumerate(axs[:, 0]):
            ax.annotate(row_labels[i], xy=(-0.05, 0.5), xytext=(-0, 0), xycoords='axes fraction', rotation=90, textcoords='offset points', ha='right', va='center')
        # Add column labels
        for j, ax in enumerate(axs[0]):
            ax.annotate(column_labels[j], xy=(0.5, 0.9), xytext=(0, 10), xycoords='axes fraction', textcoords='offset points', ha='center', va='bottom')
    plt.subplots_adjust(wspace=-0.5, hspace=0.05)  # Adjust the spacing between subplots 
    return fig

def save_as_gif(args):

    for vehicle in args.vehicles:
        folder_path_vehicle = args.path + f'/{vehicle}/nuscenes/sensors/'

        with imageio.get_writer(f'{args.save_at}_{vehicle}.gif', mode='I', duration=0.5) as writer:
            for timestep in range(args.timesteps):
                folder_path_vehicle = re.sub(r'step_\d+', f'step_{timestep}', folder_path_vehicle)
                fig = create_Nx3_subplot(args, folder_path_vehicle)
                fig.savefig(f'{args.save_at}_temp.png', bbox_inches='tight')
                plt.close()
                image = imageio.imread(f'{args.save_at}_temp.png')
                writer.append_data(image)
                
        print(f"Saved gif to {args.save_at}_{vehicle}.gif")
        
def main(args):
    
    save_as_gif(args)

if __name__ == "__main__":
    os.makedirs('/seed4d/logs', exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="path to the step")
    current_date = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    parser.add_argument("--save_at", type=str, default=f'/seed4d/logs/{current_date}')
    parser.add_argument("--vehicles", nargs='+', help='Name vehicles to include. Must be int or ego_vehicles', default=['ego_vehicle'])
    parser.add_argument("--sensors", nargs='+', default=['rgb', 'obj', 'optical_flow', 'semantic_segmentation', 'instance_segmentation','depth'])
    parser.add_argument("--labels", action="store_true", help="Enable labels")
    parser.add_argument("--timesteps", type=int, default=99)

    args = parser.parse_args()

    main(args)
    
# Example: python3.8 vis_gifs.py --path /seed4d/data/dynamic/Town10HD/ClearNoon/vehicle.mini.cooper_s/spawn_point_16/step_0/ --vehicles 71 72 73 76 ego_vehicle
# Example: python3.8 vis_gifs.py --path /seed4d/data/dynamic/Town10HD/ClearNoon/vehicle.mini.cooper_s/spawn_point_4/step_0/ --vehicles 71