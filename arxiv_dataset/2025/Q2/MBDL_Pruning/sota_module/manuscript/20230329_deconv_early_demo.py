import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial
from skimage.io import imread
import os
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
import tqdm


# ROOT_PATH = '/Users/weijiegan/ExperimentsLog/'
ROOT_PATH = '/Users/weijiegan/Downloads/'
FOLDER = '20230329_early_deconv_set12'

ITEM = 7

img_dict = {
    'Oracle': 'TEST_03291941_20230329_demo_deq_cal_alter_natural_inference_input_idx_9_TST_oracle_TST',
    'Joint': 'TEST_03291934_20230329_demo_deq_cal_alter_natural_inference_input_idx_9_TST_joint_TST',
    'w/o Calibration': 'TEST_03291944_20230329_demo_deq_cal_alter_natural_inference_input_idx_9_TST_no_cal_TST',
}

x_list = {}
theta_list = {}
x_gt_list = {}
theta_gt_list = {}

x_res_list = {}
theta_res_list = {}
x_psnr_list = {}
x_ssim_list = {}
theta_mse_list = {}

for k in img_dict:

    x_list[k] = imread(os.path.join(ROOT_PATH, FOLDER, img_dict[k], "item_%d" % ITEM, 'x_pre_list.tiff'))
    theta_list[k] = imread(os.path.join(ROOT_PATH, FOLDER, img_dict[k], "item_%d" % ITEM, 'theta_pre_list.tiff'))

    x_gt_list[k] = imread(os.path.join(ROOT_PATH, FOLDER, img_dict[k], "item_%d" % ITEM, 'x_gt.tiff'))
    theta_gt_list[k] = imread(os.path.join(ROOT_PATH, FOLDER, img_dict[k], "item_%d" % ITEM, 'theta_gt.tiff'))

    x_res, theta_res, x_psnr, x_ssim, theta_mse = [], [], [], [], []
    for i in tqdm.tqdm(range(1, x_list[k].shape[0]), desc='computing metrics of %s' % k):
        x_res.append(np.linalg.norm(x_list[k][i] - x_list[k][i-1]) ** 2)
        theta_res.append(np.linalg.norm(theta_list[k][i] - theta_list[k][i - 1]) ** 2)

        x_psnr.append(peak_signal_noise_ratio(x_gt_list[k], x_list[k][i]))
        x_ssim.append(structural_similarity(x_gt_list[k], x_list[k][i]))
        theta_mse.append(np.linalg.norm(theta_gt_list[k] - theta_list[k][i]) / np.linalg.norm(theta_gt_list[k]))

    x_res_list[k] = x_res
    theta_res_list[k] = theta_res
    x_psnr_list[k] = x_psnr
    x_ssim_list[k] = x_ssim
    theta_mse_list[k] = theta_mse

fig, ax = plt.subplots(4, 3, figsize=[9, 12])


def init():

    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.03, hspace=0.1)

    return ax


def update(frame):

    if frame < x_list['Oracle'].shape[0]:
        for sub_ax in ax:
            sub_ax[0].clear()

        ax[0][0].imshow(x_list['Oracle'][frame], cmap='gray')
        ax[0][0].axis('off')
        ax[0][0].text(5, 15, "PSNR / SSIM: %.2f / %.2f" % (x_psnr_list['Oracle'][frame - 1], x_ssim_list['Oracle'][frame - 1]))

        ax[0][0].set_title('Oracle')

        ax[1][0].imshow(theta_list['Oracle'][frame], cmap='gray')
        ax[1][0].axis('off')
        ax[1][0].text(0, 1, "NRMSE: %.5f" % theta_mse_list['Oracle'][frame - 1], color='white')

        ax[2][0].plot(x_res_list['Oracle'][:frame])
        ax[2][0].set_yscale('log')

        ax[3][0].plot(theta_res_list['Oracle'][:frame])

    if frame < x_list['Joint'].shape[0]:
        for sub_ax in ax:
            sub_ax[1].clear()

        ax[0][1].imshow(x_list['Joint'][frame], cmap='gray')
        ax[0][1].axis('off')
        ax[0][1].text(5, 15, "PSNR / SSIM: %.2f / %.2f" % (x_psnr_list['Joint'][frame - 1], x_ssim_list['Joint'][frame - 1]))

        ax[0][1].set_title('Joint')

        # ax[1][1].imshow(theta_list['Joint'][frame], cmap='gray', vmin=np.amin(theta_gt_list['Joint']), vmax=np.amax(theta_gt_list['Joint']))
        ax[1][1].imshow(theta_list['Joint'][frame], cmap='gray')
        ax[1][1].axis('off')
        ax[1][1].text(0, 1, "NRMSE: %.5f" % theta_mse_list['Joint'][frame - 1], color='white')

        ax[2][1].plot(x_res_list['Joint'][:frame])
        ax[2][1].set_yscale('log')

        ax[3][1].plot(theta_res_list['Joint'][:frame])
        ax[3][1].set_yscale('log')

    if frame < x_list['w/o Calibration'].shape[0]:
        for sub_ax in ax:
            sub_ax[2].clear()

        ax[0][2].imshow(x_list['w/o Calibration'][frame], cmap='gray')
        ax[0][2].axis('off')
        ax[0][2].text(5, 15, "PSNR / SSIM: %.2f / %.2f" % (x_psnr_list['w/o Calibration'][frame - 1], x_ssim_list['w/o Calibration'][frame - 1]))

        ax[0][2].set_title('w/o Calibration')

        ax[1][2].imshow(theta_list['w/o Calibration'][frame], cmap='gray')
        ax[1][2].axis('off')
        ax[1][2].text(0, 1, "NRMSE: %.5f" % theta_mse_list['w/o Calibration'][frame - 1], color='white')

        ax[2][2].plot(x_res_list['w/o Calibration'][:frame])
        ax[2][2].set_yscale('log')
        ax[3][2].plot(theta_res_list['w/o Calibration'][:frame])


ani = FuncAnimation(
    fig, update, init_func=init,
    frames=range(0, 499, 10),
    # frames=range(499),
    repeat=False,
    interval=50
)

# ani.save(os.path.join(ROOT_PATH, FOLDER, 'demo.mp4'))
plt.show()