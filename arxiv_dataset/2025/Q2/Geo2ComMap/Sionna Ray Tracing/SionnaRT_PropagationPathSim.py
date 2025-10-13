import matplotlib.pyplot as plt
import numpy as np
import time

# Import Sionna RT components
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera

# For link-level simulations
from sionna.channel import cir_to_ofdm_channel, subcarrier_frequencies, OFDMChannel, ApplyOFDMChannel, CIRDataset
from sionna.nr import PUSCHConfig, PUSCHTransmitter, PUSCHReceiver
from sionna.utils import compute_ber, ebnodb2no, PlotBER
from sionna.ofdm import KBestDetector, LinearDetector
from sionna.mimo import StreamManagement
import math
import scipy.io as sio
import random
import os
import cv2
import random
import re


pi_value = math.pi
def Ray_Tracing(building, tx_pos = [0, 0, 25], batch_size = 10000, index = 1):
    
    ant_dir = np.random.uniform(0, 2*pi_value)
    print(f"ant dir : {ant_dir}")
    
    ant_dir_file_name = f'Data/BS_info/{building}.txt'
    with open(ant_dir_file_name, 'a') as file:  
        file.write(f"Index {index} : {building} Tx at [{tx_pos[0]}, {tx_pos[1]}, {tx_pos[2]}] with dir : {ant_dir} \n")
        
    
    scene = load_scene(f'building_xml/{building}.xml')
    scene.frequency = 3.5e9 # in Hz
            
    ######### scene_building settings ###############################
    scene.tx_array = PlanarArray(num_rows=1,
                                            num_cols=1,
                                            vertical_spacing=0.5,
                                            horizontal_spacing=0.5,
                                            pattern="tr38901",
                                            polarization="VH")

    # Configure antenna array for all receivers
    scene.rx_array = PlanarArray(num_rows=1,
                                            num_cols=1,
                                            vertical_spacing=0.5,
                                            horizontal_spacing=0.5,
                                            pattern="iso",
                                            polarization="VH")

    # Create transmitter
    tx = Transmitter(name="tx",
                        position=tx_pos,
                        orientation=[ant_dir,0,0],
                        color=(1, 0, 0))
                
    scene.add(tx)
    scene.synthetic_array = True
    cm = scene.coverage_map(max_depth=3, 
                                    los=True, 
                                    reflection=True, 
                                    diffraction=True, 
                                    check_scene=False)
    ##################################################################

       
    # Open the file and read its contents
    with open(f'Data/UE_loc/UE_pos_{building}.txt', 'r') as file:
        lines = file.readlines()

    # Initialize an empty list to store the coordinates
    ue_pos = []

    # Loop through each line in the file
    for line in lines:
        # Extract the numbers inside the brackets and convert them to floats
        coord = [float(num) for num in line.strip().strip('[]').split()]
        # Add the coordinate to the list
        ue_pos.append(coord)

            
    # change coverage map into tensor
    cm_tensor = cm.as_tensor()
    cm_2D = cm_tensor.numpy()[0, :, :]
    cm_2D = np.flip(cm_2D[::-1])
                
    # cm_2D = np.resize(cm_2D, (128, 128))
            
    # change W into dB
    cm_db = 10 * np.log10(cm_2D)
    # shape = cm_db.shape
    # print(shape)

    # save dirctional coverage map                
    np.save(f'Data/PGmap/PGmap_{building}_{index}.npy', cm_db)                               


    ############################### ray tracing #####################################################
    UE_num = 0
    UE_position_list = []
    for i in range(batch_size):
        rx = Receiver(name=f"rx-{i}",
                    position=ue_pos[i], # Random position sampled from coverage map
                    )
        scene.add(rx)
        
        paths = scene.compute_paths(max_depth=3,
                            method='fibonacci',
                            num_samples=1e5,
                            los = True,
                            reflection=True,
                            diffraction=True,
                            check_scene=True,
                            # scattering=True,
                            edge_diffraction=True,
                            # scat_keep_prob=0.5
                            )
    
        ray_num = len(paths.types.numpy()[0])
        # print(f"ray number : {ray_num}")

        # Save the data to a .mat file with specific format          
        path_gain_list = []
        path_phase_list = []
        path_delay_list = []
        path_AOD_hor_list = []
        path_AOD_ver_list = []
        path_AOA_hor_list = []
        path_AOA_ver_list = []
    
        count = 0
        for j in range(ray_num):
            path_idx = j 
            # For a detailed overview of the dimensions of all properties, have a look at the API documentation
            # print(f"\n--- Detailed results for path {path_idx} ---")

            # print(f"Channel coefficient: {paths.a[0,0,0,0,0,path_idx, 0].numpy()}")
            path_coefficients = paths.a[0,0,0,0,0,path_idx, 0].numpy()
            if path_coefficients == 0:
                continue
            path_gain = np.abs(path_coefficients)
            path_phase = np.angle(path_coefficients)
            path_gain_list.append(path_gain)
            path_phase_list.append(path_phase)

            # print(f"Propagation delay: {paths.tau[0,0,0,path_idx].numpy()*1e6:.8f} us")
            path_delay = paths.tau[0,0,0,path_idx].numpy()
            path_delay_list.append(path_delay)

            # print(f"Zenith angle of departure: {paths.theta_t[0,0,0,path_idx]:.4f} rad")
            path_AOD_ver = paths.theta_t[0,0,0,path_idx]
            path_AOD_ver_list.append(path_AOD_ver)

            # print(f"Azimuth angle of departure: {paths.phi_t[0,0,0,path_idx]:.4f} rad")
            path_AOD_hor = paths.phi_t[0,0,0,path_idx]
            path_AOD_hor_list.append(path_AOD_hor)

            # print(f"Zenith angle of arrival: {paths.theta_r[0,0,0,path_idx]:.4f} rad")
            path_AOA_ver = paths.theta_r[0,0,0,path_idx]
            path_AOA_ver_list.append(path_AOA_ver)

            # print(f"Azimuth angle of arrival: {paths.phi_r[0,0,0,path_idx]:.4f} rad")
            path_AOA_hor = paths.phi_r[0,0,0,path_idx]
            path_AOA_hor_list.append(path_AOA_hor)
            count += 1
            if count >= 297:
                break
        k = len(path_gain_list)    
        
        def pad_to_297(arr_list):
            arr = np.array(arr_list).reshape(-1, 1)
            return np.pad(arr, ((0, 297 - arr.shape[0]), (0, 0)), 'constant', constant_values=0)

        path_gains = pad_to_297(path_gain_list)
        path_phases = pad_to_297(path_phase_list)
        path_delays = pad_to_297(path_delay_list)
        path_AOA_hors = pad_to_297(path_AOA_hor_list)
        path_AOA_vers = pad_to_297(path_AOA_ver_list)
        path_AOD_hors = pad_to_297(path_AOD_hor_list)
        path_AOD_vers = pad_to_297(path_AOD_ver_list)

        if np.all(path_gains == 0):
            scene.remove(f"rx-{i}")
            # print(f"skip data{i}")  
            continue
    
        else:
            UE_position_list.append(ue_pos[i])
            UE_postion_arr = np.array(UE_position_list).reshape(-1, 3)
            UE_num += 1
        
            data_dict = {
            'path_gain': path_gains,
            'path_phase': path_phases,
            'path_delay': path_delays,
            'path_AOA_hor': path_AOA_hors,
            'path_AOA_ver': path_AOA_vers,
            'path_AOD_hor': path_AOD_hors,
            'path_AOD_ver': path_AOD_vers,
            }

            mat_struct = {'sim': data_dict}
            folder_path = f'Data/data_all'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                sio.savemat(f'Data/data_all/data{i+1}.mat', mat_struct)
            else:
                sio.savemat(f'Data/data_all/data{i+1}.mat', mat_struct)

            # print(f"data{i+1} file created successfully.")
            # print("##############################################")
            scene.remove(f"rx-{i}")
            # print(f"rx-{i} removed")
        
    pos_dict = {
    'agent_num' : UE_num,
    'agent' : UE_postion_arr,
    'anchor_num' : 1,
    'anchor' : tx_pos
    }
    os.makedirs(f'C:\\Users\\CTLAB\\Desktop\\MIMO_Throughput_Sim-V3.2\\{building}_{index}')
    sio.savemat(f'C:\\Users\\CTLAB\\Desktop\\MIMO_Throughput_Sim-V3.2\\{building}_{index}\\position.mat', pos_dict)



# start ray tracing for a building map
for index in range(1, 101):
    # set the building map
    building = 'building1'
    print(f"Index = {index}")

    
    # set the range of BS position
    x_min, x_max = -512, 512
    y_min, y_max = -512, 512

    # random select position of BS
    random_x = random.uniform(x_min, x_max)
    random_y = random.uniform(y_min, y_max)

    print(f"Tx x coordinate : {random_x}")
    print(f"Tx y coordinate : {random_y}")

    Ray_Tracing(building = building, 
                tx_pos = [random_x, random_y, 25],  # change the height of Tx
                batch_size = 10000, 
                index = index)
    

    """ ReIndex the files if needed
    # set directory path
    folder_path = f'Data/data_all'
    output_folder = f'C:\\Users\\CTLAB\\Desktop\\MIMO_Throughput_Sim-V3.2\\{building}_{index}\\data_all'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.mat')], key=lambda x: int(re.findall(r'\d+', x)[0]))
    # print(files)

    new_index = 1
    for file_name in files:
        current_index = int(re.findall(r'\d+', file_name)[0])
        new_file_name = f"data{new_index}.mat"
        old_file_path = os.path.join(folder_path, file_name)
        new_file_path = os.path.join(output_folder, new_file_name)

        os.rename(old_file_path, new_file_path)
        # print(f"Rename: {file_name} -> {new_file_name}")
        new_index += 1
    """
