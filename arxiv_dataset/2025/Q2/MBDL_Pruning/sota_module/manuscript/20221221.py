import os
from skimage.io import imread
import pandas as pd
import numpy as np
import tabulate


ROOT_PATH = '/opt/weijiegan/Downloads'

BASELINE_PATH = {

    'Groundtruth': {
        'X4': '20221221/TEST_12210303_ray_tune_run_9fb2f_00008_8_tau=0.0010_2022-12-21_02-38-31/',
        'X8': '20221221/TEST_12210148_ray_tune_run_5bbf8_00005_5_tau=0.0025_2022-12-21_01-25-08/',
    },

    'Zero-filled': {
        'X4': '20221221/TEST_12210303_ray_tune_run_9fb2f_00008_8_tau=0.0010_2022-12-21_02-38-31/',
        'X8': '20221221/TEST_12210148_ray_tune_run_5bbf8_00005_5_tau=0.0025_2022-12-21_01-25-08/',
    },

    'Total Variation': {
        'X4': '20221221/TEST_12210303_ray_tune_run_9fb2f_00008_8_tau=0.0010_2022-12-21_02-38-31/',
        'X8': '20221221/TEST_12210148_ray_tune_run_5bbf8_00005_5_tau=0.0025_2022-12-21_01-25-08/',
    },

    'UNet (MICCAI 2017)': {
        'X4': '20221221/TEST_12210508_ray_tune_run_26ad6_00000_0_acceleration_rate=4,method=unet_2022-12-21_04-15-49_BEST_val_loss',
        'X8': '20221221/TEST_12210506_ray_tune_run_26ad6_00001_1_acceleration_rate=8,method=unet_2022-12-21_04-15-54_BEST_val_loss',
    },

    'ISTA-Net+ (CVPR 2017)': {
        'X4': '20221221/TEST_12212006_ray_tune_run_82c49_00001_1_method=istanetplus_2022-12-21_15-24-11_BEST_val_loss',
        'X8': '20221221/TEST_12210945_ray_tune_run_26ad6_00007_7_acceleration_rate=8,method=istanetplus_2022-12-21_05-08-54_BEST_val_loss',
    },

    'MoDL (TMI 2018)': {
        'X4': None,
        'X8': None,
    },

    'UNet-PnP': {
        'X4': "20221221/TEST_12212013_ray_tune_run_7395e_00024_24_alpha=0.2500,gamma=1,x_sigma=7_2022-12-21_20-12-37",
        'X8': '20221221/TEST_12212040_ray_tune_run_fb58b_00024_24_alpha=0.2500,gamma=1,x_sigma=7_2022-12-21_20-37-48',
    },

    'Prox-PnP (ICML 2022)': {
        'X4': "20221221/TEST_12220343_ray_tune_run_68224_00048_48_alpha=0.2500,gamma=1,x_sigma=5_2022-12-22_03-38-58",
        'X8': '20221221/TEST_12220003_ray_tune_run_f2c0c_00084_84_alpha=0.2500,gamma=1,x_sigma=8_2022-12-21_23-54-12',
    },

    'E2EVarNet (MICCAI 2020)': {
        'X4': '20221221/TEST_12211636_ray_tune_run_82c49_00000_0_method=e2evarnet_2022-12-21_15-24-07_BEST_val_loss',
        'X8': '20221221/TEST_12210620_ray_tune_run_26ad6_00005_5_acceleration_rate=8,method=e2evarnet_2022-12-21_05-06-01_BEST_val_loss',
    },

    'Joint-ICNet (CVPR 2021)': {
        'X4': None,
        'X8': None,
    },

    "newline_0": {},

    'DU w/o calibration (Unet)': {
        'X4': "20221221/TEST_12222213_20221222_pmri_modl_x4_deq_cal_iteration10_no_calibration_ssim_loss_unet_BEST_val_loss",
        'X8': "20221221/TEST_12250116_ray_tune_run_a2a29_00000_0_iterations=10,lr=0.0000_2022-12-24_20-36-25_BEST_val_loss",
    },

    'DU w/ calibration (Unet SPICE-like)': {
        'X4': "20221221/TEST_12240256_20221223_pmri_modl_x4_deq_cal_iteration10_calibration_ssim_loss_unet_BEST_val_loss",
        'X8': "20221221/TEST_12241307_ray_tune_run_e0cad_00000_0_iterations=10,lr=0.0000_2022-12-24_04-10-19_BEST_val_loss",
    },

    'DEQ w/o calibration (Potential)': {
        'X4': None,
        'X8': None,
    },

    'DEQ w/o calibration (UNet)': {
        'X4': "20221221/TEST_12231055_20221222_pmri_modl_x4_deq_cal_no_calibration_ssim_loss_unet_BEST_val_loss",
        'X8': "20221221/TEST_12261117_ray_tune_run_a2a29_00001_1_iterations=-1,lr=0.0000_2022-12-24_20-36-30_BEST_val_loss",
    },

    'Ours (Unet SPICE-like)': {
        'X4': "20221221/TEST_12240151_20221222_pmri_modl_x4_deq_cal_calibration_ssim_loss_unet_BEST_val_loss",
        'X8': '20221221/TEST_12252253_ray_tune_run_e0cad_00001_1_iterations=-1,lr=0.0000_2022-12-24_04-10-23_BEST_val_loss',
    },

    'Ours (Potential SPICE-like)':{
        'X4': None,
        'X8': '20221221/TEST_01020042_20221231_pmri_modl_x8_deq_cal_deq_potential_SSIM_jacobian0.001_with_warmup_RERUN_is_update_theta_iteratively[false]'
    }

    # "newline_1": {},
    #
    # 'DEQ w/ GT theta (UNet)': {
    #     'X4': None,
    #     'X8': None,
    # },
    #
    # 'DEQ w/ GT theta (Potential)': {
    #     'X4': None,
    #     'X8': None,
    # },
}

HEADER = ['Method', 'X4', 'X8']


table = []
for method in BASELINE_PATH:

    if method == 'Groundtruth':
        continue

    elif method == 'Zero-filled':
        psnr_key = 'tst_psnr_init'
        ssim_key = 'tst_ssim_init'

    elif "newline" in method:
        table.append(tabulate.SEPARATING_LINE)
        continue

    else:
        psnr_key = 'tst_psnr_x_hat'
        ssim_key = 'tst_ssim_x_hat'

    tmp = [method]

    for k in ['X4', 'X8']:
        if BASELINE_PATH[method][k] is not None:
            metrics = pd.read_csv(os.path.join(ROOT_PATH, BASELINE_PATH[method][k], 'metrics.csv'))
            psnr = np.array(metrics[psnr_key])
            ssim = np.array(metrics[ssim_key])

            tmp.append("%.2f" % psnr.mean() + u"\u00B1" + "%.3f" % psnr.std() + " / " "%.3f" % ssim.mean() + u"\u00B1" + "%.3f" % ssim.std())

        else:
            tmp.append('')

    table.append(tmp)

# print(tabulate(table, headers=HEADER, tablefmt="latex"))
# print("\n")
print(tabulate.tabulate(table, headers=HEADER, tablefmt="rst"))
