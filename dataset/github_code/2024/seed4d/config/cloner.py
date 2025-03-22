# Copyright (C) 2025 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Marius Kästingschäfer and Théo Gieruc
# ==============================================================================


import yaml
import os
import argparse

'''
writes yaml file for each Town and for each spawnpoint in the Town
'''

def count_json_files(directory):
    json_count = 0
    
    # Walk through all files and directories within the given directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.yaml'):
                json_count += 1
                
    return json_count

def main(args):
    
    yaml_path = args.yaml_file
    folder = args.output_dir
    
    with open(yaml_path, 'r') as file:
        yaml_file = yaml.safe_load(file)
        
    # https://github.com/carla-simulator/carla/issues/4199
    towns = {'Town01': 255, 'Town02': 101, 'Town03': 265, 'Town04': 372, 'Town05': 302, 'Town06': 436, 'Town07': 116, 'Town10HD': 155}

    for town in towns.keys():
        
        folder_path = os.path.join(folder, town)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        else:
            pass
        
        spawnpoints = towns[town]
        
        for spawnpoint in range(1, spawnpoints+1):
            
            # for the dynamic dataset only every 4th spawnpoint is used
            if args.type == 'dynamic':
                if spawnpoint % 4 == 0:
                    yaml_file['map'] = town
                    yaml_file['spawn_point'] = [spawnpoint]
                    
                    with open(os.path.join(folder_path, f'{args.type}_{town}_Spawnpoint{spawnpoint}.yaml'), 'w') as file:
                        yaml.dump(yaml_file, file)
            # every spawnpoint is used for static and custom datasets
            elif args.type == 'static' or args.type == 'custom':
                yaml_file['map'] = town
                yaml_file['spawn_point'] = [spawnpoint]
                
                with open(os.path.join(folder_path, f'{args.type}_{town}_Spawnpoint{spawnpoint}.yaml'), 'w') as file:
                    yaml.dump(yaml_file, file)
            else:
                print('Please specify a type: static, dynamic or custom')
                break
    
    json_file_count = count_json_files(folder)
    print(f"Total number of .yaml files: {json_file_count}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_file", type=str, default=None,  help="Example: '/seed4d/config/dynamic.yaml'")
    parser.add_argument("--output_dir", type=str, default=None, help="Example: '/seed4d/config/dynamic'")
    parser.add_argument("--type", type=str, default='custom', help="Example: 'static', 'dynamic' or 'custom'")
    
    args = parser.parse_args()
    main(args)
