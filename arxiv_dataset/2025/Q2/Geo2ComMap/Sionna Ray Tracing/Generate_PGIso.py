import numpy as np
import tensorflow as tf
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Paths, Camera
# %matplotlib widget
import matplotlib.pyplot as plt
import math
from random import randint
import re
import os

# building = 'building1'
# function to read antenna positions from a file
def read_antenna_positions(file_path):
    antenna_positions = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # regular expression to extract x, y, and z values
            match = re.search(r'Tx at \[([-0-9.]+), ([-0-9.]+), ([-0-9.]+)\]', line)
            if match:
                x = float(match.group(1))
                y = float(match.group(2))
                z = float(match.group(3))  
                antenna_positions.append([x, y, z])
    return antenna_positions


# generating PGIso for building map
def generate_IsoMap(index, antenna_positions):
    num = 1

    # load the building map
    scene_building = load_scene(f'C:\\Users\\CTLAB\\Desktop\\RT2\\building_xml\\{building}.xml')

    for i, (x, y, z) in enumerate(antenna_positions):  
        
        # set the transmitter and receiver arrays
        scene_building.tx_array = PlanarArray(num_rows=1,
                                              num_cols=1,
                                              vertical_spacing=0.5,
                                              horizontal_spacing=0.5,
                                              pattern="iso",
                                              polarization="VH")

        scene_building.rx_array = PlanarArray(num_rows=1,
                                              num_cols=1,
                                              vertical_spacing=0.5,
                                              horizontal_spacing=0.5,
                                              pattern="iso",
                                              polarization="VH")

        # set the transmitter
        tx_b = Transmitter(name=f"tx_b_{i}",
                           position=[x, y, z],  
                           orientation=[0,0,0],
                           # color=(1, 0, 0)
                           )
        
        scene_building.add(tx_b)
        scene_building.frequency = 3.5e9  # in Hz; implicitly updates RadioMaterials
        scene_building.synthetic_array = True
        
        # compute the coverage map
        cm = scene_building.coverage_map(max_depth=8, 
                                         los=True, 
                                         reflection=True, 
                                         diffraction=True, 
                                         check_scene=False)
        
        # turn the coverage map into a 2D numpy array
        cm_tensor = cm.as_tensor()
        cm_2D = cm_tensor.numpy()[0, :, :]
        cm_2D = np.flip(cm_2D[::-1])
        
        # change the values to dB
        cm_db = 10 * np.log10(cm_2D)
        
        # save the coverage map
        np.save(f'Data/PGIso/PGmap{index}_{num}.npy', cm_db)
        print(f"saved PGmap_{num}.npy")

        num += 1
        scene_building.remove(f"tx_b_{i}")


# select the buildings to generate PGIso
Index_list = ['27', '28', '31', '32']

# iterate through the buildings
for index in Index_list:
    # set the building map
    building = f"building{index}"

    # read the antenna positions from the file
    antenna_positions = read_antenna_positions(f'C:\\Users\\CTLAB\\Desktop\\RT2\\Data\\BS_info\\{building}.txt')
    print(f"Processing {building}...")
    generate_IsoMap(index=index, antenna_positions=antenna_positions)

