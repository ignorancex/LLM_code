import tabulate
from tabulate import SEPARATING_LINE
from collections import defaultdict

import numpy as np
import pandas as pd
import os
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


ROOT_PATH = '/Users/weijiegan/Downloads'

FOLDER = 'NonconvexBCPnP_20230201'

BASELINE_PATH = {
    'ground-truth': 'TEST_02012100_20230202_deq_cal_alter_pmri_inference_TST_is_joint_cal[false]_is_use_gt_theta[false]_acc_rate[4]_smps_hat_method[low_k]',

    'init (low-k)':
        {
            'unet':
                {
                    'X4': 'TEST_02012100_20230202_deq_cal_alter_pmri_inference_TST_is_joint_cal[false]_is_use_gt_theta[false]_acc_rate[4]_smps_hat_method[low_k]',
                    # x_gamma=1, x_alpha=0.1, x_sigma=10, theta_gamma=1, theta_alpha=0.25, theta_sigma=3
                    'X8': 'TEST_02061906_20230205_deq_cal_alter_pmri_inference_TST_is_joint_cal[false]_is_use_gt_theta[false]_acc_rate[8]_smps_hat_method[low_k]',
                },
            'unet-v':
                {
                    'X4': 'TEST_02012100_20230202_deq_cal_alter_pmri_inference_TST_is_joint_cal[false]_is_use_gt_theta[false]_acc_rate[4]_smps_hat_method[low_k]',
                    'X8': 'TEST_02061906_20230205_deq_cal_alter_pmri_inference_TST_is_joint_cal[false]_is_use_gt_theta[false]_acc_rate[8]_smps_hat_method[low_k]',
                }
        },

    'pre-estimate theta (low-k)':
        {
            'unet':
                {
                    # x_gamma=1, x_alpha=0.1, x_sigma=10
                    'X4': 'TEST_02012100_20230202_deq_cal_alter_pmri_inference_TST_is_joint_cal[false]_is_use_gt_theta[false]_acc_rate[4]_smps_hat_method[low_k]',
                    # x_gamma=1, x_alpha=0.1, x_sigma=10, theta_gamma=1, theta_alpha=0.25, theta_sigma=3
                    'X8': 'TEST_02061906_20230205_deq_cal_alter_pmri_inference_TST_is_joint_cal[false]_is_use_gt_theta[false]_acc_rate[8]_smps_hat_method[low_k]',
                },
            'unet-v':
                {
                    'X4': 'unet_v/TEST_02151903_20230212_deq_cal_alter_pmri_inference_tst_is_joint_cal[false]_is_use_gt_theta[false]_smps_hat_method[low_k]_g_denoise_unet',
                    'X8': 'unet_v/TEST_02151925_20230212_deq_cal_alter_pmri_inference_tst_is_joint_cal[false]_is_use_gt_theta[false]_smps_hat_method[low_k]_g_denoise_unet_acc[8]',
                }
        },

    'joint-estimate theta(no-theta-prior, low-k)':
        {
            'unet':
                {
                    # x_gamma=1, x_alpha=0.1, x_sigma=3, theta_gamma=1
                    'X4': 'TEST_02031824_tmp_20230203_deq_cal_alter_pmri_inference_TST_is_joint_cal[true]_is_use_gt_theta[false]_acc_rate[4]_smps_hat_method[low_k]_alpha[0.1]',
                    # x_gamma=1, x_alpha=0.1, x_sigma=10, theta_gamma=1, theta_alpha=0.25, theta_sigma=3
                    'X8': 'TEST_02061909_20230205_deq_cal_alter_pmri_inference_TST_is_joint_cal[true]_is_use_gt_theta[false]_acc_rate[8]_smps_hat_method[low_k]_theta_alpha[0]',
                },
            'unet-v':
                {
                    'X4': 'unet_v/TEST_02151907_20230212_deq_cal_alter_pmri_inference_tst_is_joint_cal[true]_is_use_gt_theta[false]_smps_hat_method[low_k]_g_denoise_unet_theta_alpha[0]',
                    'X8': 'unet_v/TEST_02151931_20230212_deq_cal_alter_pmri_inference_tst_is_joint_cal[true]_is_use_gt_theta[false]_smps_hat_method[low_k]_g_denoise_unet_acc[8]_theta_alpha[0]',
                }
        },

    'joint-estimate theta(low-k)':
        {
            'unet':
                {
                    'X4': 'TEST_02030634_ray_tune_run_420f0_00467_467_theta_gamma=1,theta_sigma=5,x_sigma=3,x_gamma=1_2023-02-03_06-30-29',
                    # x_gamma=1, x_alpha=0.1, x_sigma=10, theta_gamma=1, theta_alpha=0.25, theta_sigma=3
                    'X8': 'TEST_02061905_20230205_deq_cal_alter_pmri_inference_TST_is_joint_cal[true]_is_use_gt_theta[false]_acc_rate[8]_smps_hat_method[low_k]',
                },
            'unet-v':
                {
                    # {'x_gamma': 1, 'x_alpha': 0.1, 'theta_gamma': 1, 'theta_alpha': 0.25, 'warmup': {'x_sigma': 5, 'theta_sigma': 1}}}}
                    'X4': 'unet_v/TEST_02151859_20230212_deq_cal_alter_pmri_inference_tst_is_joint_cal[true]_is_use_gt_theta[false]_smps_hat_method[low_k]_g_denoise_unet',
                    # {'x_gamma': 1, 'x_alpha': 0.1, 'theta_gamma': 1, 'theta_alpha': 0.25, 'warmup': {'x_sigma': 10, 'theta_sigma': 10}}}}}}
                    'X8': 'unet_v/TEST_02151923_20230212_deq_cal_alter_pmri_inference_tst_is_joint_cal[true]_is_use_gt_theta[false]_smps_hat_method[low_k]_g_denoise_unet_acc[8]',
                }
        },

    "SEPARATING_LINE_0": {

    },

    'init (esp)':
        {
            'unet':
                {
                    'X4': 'TEST_02021746_20230203_deq_cal_alter_pmri_inference_TST_is_joint_cal[false]_is_use_gt_theta[false]_acc_rate[4]_smps_hat_method[esp]',
                    'X8': 'TEST_02041948_20230204_deq_cal_alter_pmri_inference_TST_is_joint_cal[true]_is_use_gt_theta[false]_acc_rate[8]_smps_hat_method[esp]',
                },
            'unet-v':
                {
                    'X4': 'unet_v/TEST_02181815_20230215_deq_cal_alter_pmri_inference_TST_is_joint_cal[false]_is_use_gt_theta[false]_smps_hat_method[esp]_g_denoise_unet_acc[4]',
                    # x_gamma=1, x_alpha=0.1, x_sigma=7
                    'X8': 'unet_v/TEST_02120412_20230209_deq_cal_alter_pmri_inference_TST_is_joint_cal[false]_is_use_gt_theta[false]_acc_rate[8]_smps_hat_method[esp]_g_denoise_unet_RERUN',
                }
        },

    'pre-estimate theta (esp)':
        {
            'unet':
                {
                    # x_gamma=1, x_alpha=0.1, x_sigma=3
                    'X4': 'TEST_02021746_20230203_deq_cal_alter_pmri_inference_TST_is_joint_cal[false]_is_use_gt_theta[false]_acc_rate[4]_smps_hat_method[esp]',
                    # x_gamma=1, x_alpha=0.1, x_sigma=7, theta_gamma=0.5, theta_alpha=0.1, theta_sigma=3
                    'X8': 'TEST_02041946_20230204_deq_cal_alter_pmri_inference_TST_is_joint_cal[false]_is_use_gt_theta[false]_acc_rate[8]_smps_hat_method[esp]'
                },
            'unet-v':
                {
                    'X4': 'unet_v/TEST_02181815_20230215_deq_cal_alter_pmri_inference_TST_is_joint_cal[false]_is_use_gt_theta[false]_smps_hat_method[esp]_g_denoise_unet_acc[4]',
                    # x_gamma=1, x_alpha=0.1, x_sigma=7
                    'X8': 'unet_v/TEST_02120412_20230209_deq_cal_alter_pmri_inference_TST_is_joint_cal[false]_is_use_gt_theta[false]_acc_rate[8]_smps_hat_method[esp]_g_denoise_unet_RERUN',
                }
        },

    'joint-estimate theta(no-prior, esp)':
        {
            'unet':
                {
                    'X4': 'TEST_02030237_tmp_20230203_deq_cal_alter_pmri_inference_TST_is_joint_cal[true]_is_use_gt_theta[false]_acc_rate[4]_smps_hat_method[esp]_alpha[0.1]',
                    # x_gamma=1, x_alpha=0.1, x_sigma=7, theta_gamma=0.5, theta_alpha=0, theta_sigma=3
                    'X8': 'TEST_02041953_20230204_deq_cal_alter_pmri_inference_TST_is_joint_cal[true]_is_use_gt_theta[false]_acc_rate[8]_smps_hat_method[esp]_theta_alpha[0]',
                },
            'unet-v':
                {
                    'X4': 'unet_v/TEST_02181819_20230215_deq_cal_alter_pmri_inference_TST_is_joint_cal[true]_is_use_gt_theta[false]_smps_hat_method[esp]_g_denoise_unet_acc[4]_theta_alpha[0]',
                    # x_gamma=1, x_alpha=0.1, x_sigma=7, theta_gamma=0.5, theta_alpha=0
                    'X8': 'unet_v/TEST_02120614_20230209_deq_cal_alter_pmri_inference_TST_is_joint_cal[true]_is_use_gt_theta[false]_acc_rate[8]_smps_hat_method[esp]_g_denoise_unet_RERUN_theta_alpha[0]',
                }
        },


    'joint-estimate theta(esp)':
        {
            'unet':
                {
                    'X4': 'TEST_02022304_ray_tune_run_ab37a_00457_457_theta_gamma=0.1000,theta_sigma=1,x_sigma=3,x_gamma=1_2023-02-02_23-00-53',
                    # x_gamma=1, x_alpha=0.1, x_sigma=7, theta_gamma=0.5, theta_alpha=0.1, theta_sigma=3
                    'X8': 'TEST_02041948_20230204_deq_cal_alter_pmri_inference_TST_is_joint_cal[true]_is_use_gt_theta[false]_acc_rate[8]_smps_hat_method[esp]'
                },
            'unet-v':
                {
                    # x_gamma=1, x_alpha=0.1, x_sigma=5, theta_gamma=0.5, theta_alpha=0.1, theta_sigma=1
                    'X4': 'unet_v/TEST_02181815_20230215_deq_cal_alter_pmri_inference_TST_is_joint_cal[true]_is_use_gt_theta[false]_smps_hat_method[esp]_g_denoise_unet_acc[4]',
                    # x_gamma=1, x_alpha=0.1, x_sigma=7, theta_gamma=0.5, theta_alpha=0.1, theta_sigma=10
                    'X8': 'unet_v/TEST_02120407_20230209_deq_cal_alter_pmri_inference_TST_is_joint_cal[true]_is_use_gt_theta[false]_acc_rate[8]_smps_hat_method[esp]_g_denoise_unet_RERUN',
                }
        },

}

HEADER = ['Method', 'X4', 'X8']
print("pMRI results (image PSNR / image SSIM / coil sensitivity NMSE )\n")


for cnn in ['unet', 'unet-v']:
    print("\nnetwork=[%s]" % cnn)

    table = []
    for method in BASELINE_PATH:

        if 'SEPARATING_LINE' in method:
            tmp = SEPARATING_LINE

        else:
            tmp = [method]

            for k in ['X4', 'X8']:

                if method == 'ground-truth':
                    x_gt = imread(os.path.join(ROOT_PATH, FOLDER, BASELINE_PATH[method], 'x_gt_real.tiff')) + \
                           imread(os.path.join(ROOT_PATH, FOLDER, BASELINE_PATH[method], 'x_gt_imag.tiff')) * 1j

                    theta_gt = imread(os.path.join(ROOT_PATH, FOLDER, BASELINE_PATH[method], 'theta_gt_real.tiff')) + \
                        imread(os.path.join(ROOT_PATH, FOLDER, BASELINE_PATH[method], 'theta_gt_imag.tiff')) * 1j

                    continue

                if BASELINE_PATH[method][cnn][k] == "":
                    tmp.append("")
                    continue

                if 'init' in method:
                    x = imread(os.path.join(ROOT_PATH, FOLDER, BASELINE_PATH[method][cnn][k], 'x_input_real.tiff')) + \
                           imread(os.path.join(ROOT_PATH, FOLDER, BASELINE_PATH[method][cnn][k], 'x_input_imag.tiff')) * 1j

                    theta = imread(os.path.join(ROOT_PATH, FOLDER, BASELINE_PATH[method][cnn][k], 'theta_input_real.tiff')) + \
                        imread(os.path.join(ROOT_PATH, FOLDER, BASELINE_PATH[method][cnn][k], 'theta_input_imag.tiff')) * 1j

                else:
                    x = imread(os.path.join(ROOT_PATH, FOLDER, BASELINE_PATH[method][cnn][k], 'x_hat_real.tiff')) + \
                        imread(os.path.join(ROOT_PATH, FOLDER, BASELINE_PATH[method][cnn][k], 'x_hat_imag.tiff')) * 1j

                    theta = imread(os.path.join(ROOT_PATH, FOLDER, BASELINE_PATH[method][cnn][k], 'theta_hat_real.tiff')) + \
                        imread(os.path.join(ROOT_PATH, FOLDER, BASELINE_PATH[method][cnn][k], 'theta_hat_imag.tiff')) * 1j

                num_batch = x.shape[0]
                psnr, ssim, nmse = [], [], []
                for i in range(num_batch):
                    tmp_x_gt = abs(x_gt[i])
                    tmp_x = abs(x[i])
                    tmp_x[tmp_x_gt == 0] = 0

                    psnr.append(peak_signal_noise_ratio(tmp_x_gt, tmp_x, data_range=1))
                    ssim.append(structural_similarity(tmp_x_gt, tmp_x, data_range=1))

                    tmp_theta_gt = theta_gt[i]
                    tmp_theta = theta[i]

                    num_coil = tmp_theta_gt.shape[0]
                    for j in range(num_coil):
                        tmp_theta_gt[j][tmp_x_gt == 0] = 0
                        tmp_theta[j][tmp_x_gt == 0] = 0

                    # nmse.append(np.sqrt(np.mean(np.abs(tmp_theta - tmp_theta_gt) ** 2)) / np.sqrt(np.mean(np.abs(tmp_theta_gt) ** 2)))
                    nmse.append(np.sqrt(np.sum(np.abs(tmp_theta - tmp_theta_gt) ** 2)) / np.sqrt(np.sum(np.abs(tmp_theta_gt) ** 2)))

                tmp.append(u"%.2f \u00B1 %.3f / %.3f \u00B1 %.3f / %.5f \u00B1 %.3f" % (np.array(psnr).mean(), np.array(psnr).std(), np.array(ssim).mean(), np.array(ssim).std(), np.array(nmse).mean(), np.array(nmse).std()))

        if method != 'ground-truth':
            table.append(tmp)

    print(tabulate.tabulate(table, headers=HEADER, tablefmt="rst"))
