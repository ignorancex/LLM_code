"""
@author: hdsullivan
"""

import numpy as np
import os
import time
import yaml
import argparse

from ResSR.ressr import res_sr
from ResSR.utils import metrics_utils, decimation_utils, image_utils

# ######
# Load yaml file specified by user
# ######
parser = argparse.ArgumentParser()
parser.add_argument('-opt', '--options', type=str, default='parameters.yaml', help='Path to the YAML file with parameters')
args = parser.parse_args()
yaml_file = args.options

with open(yaml_file, 'r') as f:
    params = yaml.safe_load(f)

# #####
# If simulated data, initialize dictionaries to store SSIM and NRMSE values
# #####
if params["simulated"]:
    ssim_dict = {}
    nrmse_dict = {}

# #####
# Loop through all images in the specified directory
# #####
image_idx = 0
time_arr = []
img_name_list = []

for img_name in os.listdir(params['main_dir']):
    params["img_name"] = img_name
    params["img_dir"] = params['main_dir'] + img_name

    if img_name == '.DS_Store':
        continue

    print("Super-resolving ", params["img_name"])

    img_name_list.append(params["img_name"])

    # #####
    # Make directory to save results based on parameters
    # #####
    if params["cropped_during_recon"]:
        cropped_str = "cropped_" + str(params["x_crop_range"][0]) + "_" + str(params["x_crop_range"][1]) + "," + str(params["y_crop_range"][0]) + "_" + str(params["y_crop_range"][1]) + "/"
    else:
        cropped_str = "uncropped/"

    if params["simulated"] and params['L_synth'] == 2:
        results_dir = "../demo/results/" + params['save_dir_name'] + "/" + cropped_str + params["img_name"] + "/simulated_L=2/"
    elif params["simulated"] and params['L_synth'] == 6:
        results_dir = "../demo/results/" + params['save_dir_name'] + "/" + cropped_str + params["img_name"] + "/simulated_L=6/"
    else:
        results_dir = "../demo/results/" + params['save_dir_name'] + "/" + cropped_str + params["img_name"] + "/"

    save_dir = results_dir

    os.makedirs(save_dir, exist_ok=True) 

    # #####
    # Load 10m, 20m, and 60m measured images from given directory
    # #####
    (
        y_10m,
        y_20m,
        y_60m,
        band_name_list_10m,
        band_name_list_20m,
        band_name_list_60m,
    ) = image_utils.load_S2_images(params["img_dir"])

    # #####
    # Crop images if needed
    # #####
    if params["cropped_during_recon"]:
        x_min = params["x_crop_range"][0]
        x_max = params["x_crop_range"][1]
        y_min = params["y_crop_range"][0]
        y_max = params["y_crop_range"][1]

        y_10m = y_10m[x_min:x_max, y_min:y_max, :]
        y_20m = y_20m[x_min // 2 : x_max // 2, y_min // 2 : y_max // 2, :]
        y_60m = y_60m[x_min // 6 : x_max // 6, y_min // 6 : y_max // 6, :]

    # #####
    # Create simulated data if needed
    # #####
    if params["simulated"]:
        y_10m_gt = y_10m
        y_20m_gt = y_20m
        y_60m_gt = y_60m
        y_10m_lr = np.zeros((y_10m.shape[0] // params["L_synth"], y_10m.shape[1] // params["L_synth"], y_10m.shape[2]))
        y_20m_lr = np.zeros((y_20m.shape[0] // params["L_synth"], y_20m.shape[1] // params["L_synth"], y_20m.shape[2]))
        y_60m_lr = np.zeros((y_60m.shape[0] // params["L_synth"], y_60m.shape[1] // params["L_synth"], y_60m.shape[2]))
        for i in range(y_10m.shape[2]):
            y_10m_lr[:, :, i] = decimation_utils.simulated_data_decimation(y_10m[:, :, i], params["L_synth"])
        for i in range(y_20m.shape[2]):
            y_20m_lr[:, :, i] = decimation_utils.simulated_data_decimation(y_20m[:, :, i], params["L_synth"])
        for i in range(y_60m.shape[2]):
            y_60m_lr[:, :, i] = decimation_utils.simulated_data_decimation(y_60m[:, :, i], params["L_synth"])
        y_10m = y_10m_lr
        y_20m = y_20m_lr
        y_60m = y_60m_lr

    if "APEX" in params["main_dir"]:
        y_20m_gt = y_20m
        y_60m_gt = y_60m
        y_20m_lr = np.zeros((y_20m.shape[0] // 2, y_20m.shape[1] // 2, y_20m.shape[2]))
        y_60m_lr = np.zeros((y_60m.shape[0] // 6, y_60m.shape[1] // 6, y_60m.shape[2]))
        for i in range(y_20m.shape[2]):
            y_20m_lr[:, :, i] = decimation_utils.simulated_data_decimation(y_20m[:, :, i], 2)
        for i in range(y_60m.shape[2]):
            y_60m_lr[:, :, i] = decimation_utils.simulated_data_decimation(y_60m[:, :, i], 6)
        y_20m = y_20m_lr
        y_60m = y_60m_lr

    # #####
    # Super-resolve MSI with ResSR
    # #####
    tic = time.time()
    y = [y_10m, y_20m, y_60m]
    params['N_s'] = int(np.sqrt(y_10m.shape[0] * y_10m.shape[1]))
    x = res_sr(y, params['sigma'], params['N_s'], params['K'], params['gamma_hr'], params['lam'])
    toc = time.time()
    time_arr.append(toc - tic)

    # #####
    # Save super-resolved images and compute image quality metrics
    # #####            
    metrics_dict_list = []
    idx = 0
    for band in band_name_list_10m:
        save_name = save_dir + "/" + band.replace('jp2', 'tif')
        image_utils.im_save(x[:, :, idx], save_name)
        idx += 1

    ssim_list = []
    nrmse_list = []
    for band in band_name_list_20m:
        save_name = save_dir + "/" + band.replace('jp2', 'tif')
        image_utils.im_save(x[:, :, idx], save_name)
        if params["simulated"] and params["L_synth"] == 2:
            metrics_dict = metrics_utils.get_recon_metrics(y_20m_gt[:, :, idx - y_10m.shape[2]], x[:, :, idx])
            metrics_dict_list.append(metrics_dict)
            ssim_list.append(metrics_dict['SSIM'])
            nrmse_list.append(metrics_dict['NRMSE'])
        idx += 1
    if params["simulated"] and params["L_synth"] == 2:
        ssim_dict[img_name] = ssim_list
        nrmse_dict[img_name] = nrmse_list

    ssim_list = []
    nrmse_list = []
    for band in band_name_list_60m:
        save_name = save_dir + "/" + band.replace('jp2', 'tif')
        image_utils.im_save(x[:, :, idx], save_name)
        if params["simulated"] and params["L_synth"] == 6:
            metrics_dict = metrics_utils.get_recon_metrics(
                y_60m_gt[:, :, idx - y_10m.shape[2] - y_20m.shape[2]], x[:, :, idx]
            )
            ssim_list.append(metrics_dict['SSIM'])
            nrmse_list.append(metrics_dict['NRMSE'])
            metrics_dict_list.append(metrics_dict)
        idx += 1
    if params["simulated"] and params["L_synth"] == 6:
        ssim_dict[img_name] = ssim_list
        nrmse_dict[img_name] = nrmse_list

    image_idx += 1

# #####
# Print SSIM and NRMSE values
# #####
print("\nSuper-resolved " + str(image_idx) + " test images...\n")

if params["simulated"] and params["L_synth"] == 2:
    ssim_arr = np.zeros((len(ssim_dict.keys()), 6))
    nrmse_arr = np.zeros((len(ssim_dict.keys()), 6))
elif params["simulated"] and params["L_synth"] == 6:
    ssim_arr = np.zeros((len(ssim_dict.keys()), 2))
    nrmse_arr = np.zeros((len(ssim_dict.keys()), 2))

if params["simulated"]:
    idx = 0
    for key in ssim_dict:
        print("\nSSIM for " + key + ": " + str(ssim_dict[key]))
        print("NRMSE for " + key + ": " + str(nrmse_dict[key]))
        ssim_arr[idx, :] = np.array(ssim_dict[key])
        nrmse_arr[idx, :] = np.array(nrmse_dict[key])
        idx += 1
        
    print("\nAvg SSIM: ", np.mean(ssim_arr,axis=0))
    print("Std SSIM: ", np.std(ssim_arr,axis=0))
    print("Avg NRMSE: ", np.mean(nrmse_arr,axis=0))
    print("Std NRMSE: ", np.std(nrmse_arr,axis=0))

print("Average Runtime: ", np.average(time_arr))
