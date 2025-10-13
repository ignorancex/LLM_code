import numpy as np
import os
import cv2

def replace_nan_inf_in_npy(input_folder_path, output_folder_path):
    
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    
    
    npy_files = [f for f in os.listdir(input_folder_path) if f.endswith('.npy')]
    
    for npy_file in npy_files:
        input_file_path = os.path.join(input_folder_path, npy_file)
        output_file_path = os.path.join(output_folder_path, npy_file)
         
        data = np.load(input_file_path)
        
        data = np.where(np.isnan(data) | np.isinf(data), -160, data)
        data = np.clip(data, -160, None)
        
        np.save(output_file_path, data)
        print(f"Processed {npy_file}")


def resize_npy_files(input_folder, output_folder, target_size=(128, 128)):
    # check if the output folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # get all .npy files in the input folder
    files = [f for f in os.listdir(input_folder) if f.endswith('.npy')]

    for file in files:
        # get the full path of the input and output files
        input_file_path = os.path.join(input_folder, file)
        output_file_path = os.path.join(output_folder, file)
        
        # load the .npy file
        data = np.load(input_file_path)
        
        # check if the data is None
        if data is None:
            print(f"Failed to load .npy file {input_file_path}")
            continue
        
        # resize the data
        resized_data = cv2.resize(data, target_size, interpolation=cv2.INTER_LINEAR)
        
        # save the resized data
        np.save(output_file_path, resized_data)
        print(f"Processed and saved {file} as {output_file_path}")


# change the input_folder_path and output_folder_path to the folder path where the .npy files are stored
input_folder_path = 'PGmap'
output_folder_path = 'PGmap'

replace_nan_inf_in_npy(input_folder_path, output_folder_path)
resize_npy_files(input_folder_path, output_folder_path)