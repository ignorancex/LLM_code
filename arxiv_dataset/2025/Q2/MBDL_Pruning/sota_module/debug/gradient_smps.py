import torch

from get_from_config import get_dataset_from_config
from method.dep_cal import DEQCalibration
from tqdm import tqdm
from fwd.pmri_modl import ftran, fmult, gradient_smps, divided_by_rss
from tifffile import imwrite
from sigpy.mri.app import EspiritCalib
from sigpy import Device
import cupy


def run(config):
    _, _, tst_dataset = get_dataset_from_config(config)

    x_input, theta_input, y, fwd_para, x_gt, theta_gt = tst_dataset[50]
    mask = fwd_para['mask']

    x_input, theta_input, y, mask, x_gt, theta_gt = [
        i.unsqueeze(0) for i in [x_input, theta_input, y, mask, x_gt, theta_gt]]

    y = fmult(x_gt, theta_gt, mask)
    tmp = EspiritCalib(y[0].numpy(), device=Device(0), show_pbar=False).run()
    tmp = cupy.asnumpy(tmp)
    theta_input = torch.from_numpy(tmp).unsqueeze(0)

    for ii in [x_input, theta_input, y, mask, x_gt, theta_gt]:
        print(ii.shape, ii.dtype)

    print("x input:", DEQCalibration.psnr_ssim_helper(x_input, x_gt, 1))

    # x_hat = torch.zeros_like(x_input)
    # for i in tqdm(range(500)):
    #     x_hat = x_hat - 0.5 * ftran(fmult(x_hat, theta_input, mask) - y, theta_input, mask)
    #
    # print("x hat (theta_input):", DEQCalibration.psnr_ssim_helper(x_hat, x_gt, 1))
    #
    # x_hat = torch.zeros_like(x_input)
    # for i in tqdm(range(500)):
    #     x_hat = x_hat - 0.5 * ftran(fmult(x_hat, theta_gt, mask) - y, theta_gt, mask)
    #
    # print("x hat (theta_gt):", DEQCalibration.psnr_ssim_helper(x_hat, x_gt, 1))
    #
    # smps_hat = torch.zeros_like(theta_input)
    # for i in tqdm(range(500)):
    #     smps_hat = smps_hat - 0.1 * gradient_smps(smps_hat, x_gt, y, mask)
    #     smps_hat = divided_by_rss(smps_hat)
    #
    #     obj = torch.norm((fmult(x_gt, smps_hat, mask) - y).norm())
    #     print(i, "obj", obj.item())
    #
    # x_input = ftran(y, smps_hat, mask)
    # print("x input:", DEQCalibration.psnr_ssim_helper(x_input, x_gt, 1))
    #
    # imwrite('/opt/experiment/smps_hat.tiff', smps_hat.abs().numpy(), imagej=True)

    x_hat = x_input
    smps_hat = theta_input
    # x_hat = torch.zeros_like(x_input)
    # smps_hat = torch.zeros_like(theta_input)
    for i in tqdm(range(100)):
        x_hat = x_hat - 0.5 * ftran(fmult(x_hat, smps_hat, mask) - y, smps_hat, mask)
        smps_hat = smps_hat - 0.5 * gradient_smps(smps_hat, x_hat, y, mask)
        # smps_hat = divided_by_rss(smps_hat)

        obj = torch.norm((fmult(x_hat, smps_hat, mask) - y).norm())
        print(i, "obj", obj.item())

    print("x hat:", DEQCalibration.psnr_ssim_helper(x_hat, x_gt, 1))
