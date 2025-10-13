# Copyright (C) 2025 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Marius Kästingschäfer and Théo Gieruc
# ==============================================================================

import os
import argparse
from tqdm import tqdm
from tabulate import tabulate

def count_png_files(data_path):
    count = 0
    for root, dirs, files in os.walk(data_path):
        for filename in files:
            if filename.endswith('.png'):
                count += 1
    return count

def check_towns(data_path, towns, category):
    incomplete_spawnpoints =  []
    
    img_amount = 1070 if category == 'static' else 249901 if category == 'dynamic' else None
    total_scenes = 2002 if category == 'static' else 498 if category == 'dynamic' else None
    if img_amount is None:
        raise ValueError(f"Unknown category: {category}")
    
    incomplete_towns = []
    table_data = []
    total_pngs = 0
    total_per_scene = 0
    total_per_town = 0 
    for town in tqdm(towns, desc="Processing towns"):
        spawnpoints = towns[town]
        for spawnpoint in  tqdm(range(1, spawnpoints+1), desc=f"Processing spawnpoints in {town}", leave=False):
            
            if category == 'dynamic': spawnpoint *= 4
            
            if category == 'dynamic': data_path_ending = "ClearNoon/vehicle.mini.cooper_s/spawn_point_" + str(spawnpoint)
            if category == 'static': data_path_ending = "ClearNoon/vehicle.audi.tt/spawn_point_" + str(spawnpoint)
            
            png_count = count_png_files(os.path.join(data_path, town, data_path_ending))
            total_per_town += png_count
            total_pngs += png_count
            if png_count != img_amount:
                incomplete_spawnpoints.append((town, spawnpoint, png_count))
        
        png_per_scene = total_per_town / towns[town]
        table_data.append([town, total_per_town, towns[town], png_per_scene])
        total_per_town = 0
                
    table_data.append(["Total", total_pngs, sum(towns.values()), total_pngs / total_scenes])
                
    return table_data, incomplete_spawnpoints

def main(args):
    
    data_path = args.data_dir
    category = args.category 
    
    if category == 'static':
        towns = {'Town01': 255, 'Town02': 101, 'Town03': 265, 'Town04': 372, 'Town05': 302, 'Town06': 436, 'Town07': 116, 'Town10HD': 155}
    elif category == 'dynamic':
        towns = {'Town01': 63, 'Town02': 25, 'Town03': 66, 'Town04': 93, 'Town05': 75, 'Town06': 109, 'Town07': 29, 'Town10HD': 38}
    else:
        raise ValueError(f"Unknown category: {category}")
    
    table_data, incomplete_spawnpoints = check_towns(data_path, towns, category)
    print(tabulate(table_data, headers=["Town", "PNG Count", "Scenes", "PNGs per Scene"], tablefmt="grid"))

    if incomplete_spawnpoints == []:
        print('All spawnpoints are complete.')
    else:
        print("\nIncomplete spawnpoints:")
        for incomplete_spawnpoint in incomplete_spawnpoints:
            print(f"{incomplete_spawnpoint[0]} spawnpoint {incomplete_spawnpoint[1]} contains {incomplete_spawnpoint[2]} pngs.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None,  help="Folder to check. Example: '/seed4d/data/static'")
    parser.add_argument("--category", type=str, default=None, help="static or dynamic")
    
    args = parser.parse_args()
    main(args)
    
# RUN: python3.8 check_dataset.py --data_dir /seed4d/data/static --category static