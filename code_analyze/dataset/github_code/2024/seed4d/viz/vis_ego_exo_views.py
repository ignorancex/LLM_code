# Copyright (C) 2025 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Marius Kästingschäfer and Théo Gieruc
# ==============================================================================

'''
simple script for visualizing the views
'''

import os
import cv2
import random
import datetime
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def get_ego_exo_paths(folder):
    ego_paths = os.listdir(folder + '/nuscenes/sensors/')
    ego_paths = sorted([[folder + '/nuscenes/sensors/' + file] for file in ego_paths if file.endswith('rgb.png')])
    del ego_paths[3:] # remove the one that is double

    exo_paths = os.listdir(folder + '/sphere/sensors/')
    exo_paths = [[folder + '/sphere/sensors/' + file] for file in exo_paths if file.endswith('rgb.png')]
    random.shuffle(exo_paths)
    
    return ego_paths, exo_paths

def main(args):
    
    reorder = {0: 2, 1: 0, 2: 1, 3: 5, 4: 3, 5: 4}
    width, height = 900, 800
    columns = 7
    
    for town in args.towns:
        fig, axes = plt.subplots(len(args.spawnpoints), columns, figsize=(12, 6), dpi=300)
        for jdx, spawnpoint in enumerate(args.spawnpoints):
            folder = f'{args.path}/Town{town}/ClearNoon/vehicle.mini.cooper_s/spawn_point_{spawnpoint}/step_0/ego_vehicle/'
            ego_paths, exo_paths = get_ego_exo_paths(folder)
            for idx in range(columns):
                if idx < 3:
                    path = ego_paths[reorder[idx]][0]
                else:
                    path = exo_paths[idx][0]
                reshaped_img = cv2.resize(np.array(plt.imread(path)), (width, height))
                axes[jdx, idx].imshow(reshaped_img)
                axes[jdx, idx].axis('off')  # Hide axes

        saving_path = f"{args.save_at}_town{town}.png"
        plt.subplots_adjust(wspace=0.1, hspace=-0.5)
        #plt.tight_layout() 
        plt.savefig(saving_path, bbox_inches='tight' , pad_inches=0)
        plt.close()
        print(f'File succesfully saved to {saving_path}')
        
if __name__ == "__main__":
    os.makedirs('/seed4d/logs', exist_ok=True)
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--path", type=str, help="Base directory")
    parser.add_argument("--towns", nargs='+', help='Name town(s) to plot', default=['01'])
    parser.add_argument("--spawnpoints", nargs='+', help='Put 3 spawnpoints', default=['5 15 30'])
    
    current_date = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    parser.add_argument("--save_at", type=str, default=f'/seed4d/logs/{current_date}_')
    
    args = parser.parse_args()
    
    main(args)
    
# example: python3.8 vis_ego_exo_views.py --path /seed4d/data/static --towns 01 02 --spawnpoints 4 8 15