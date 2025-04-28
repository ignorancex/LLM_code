import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


def PSNR(img, imclean, data_range):
    Img = img.squeeze(0).data.cpu().numpy().astype(np.float64)
    Iclean = imclean.squeeze(0).data.cpu().numpy().astype(np.float64)
    PSNR = peak_signal_noise_ratio(Iclean, Img, data_range=data_range)
    return PSNR


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += peak_signal_noise_ratio(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)
    return (PSNR / Img.shape[0])


def SSIM(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float64)
    Iclean = imclean.data.cpu().numpy().astype(np.float64)
    SSIM = 0
    for i in range(Img.shape[0]):
        SSIM += structural_similarity(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range, channel_axis=0)
    return (SSIM / Img.shape[0])


def A_operator(z, Phi):
    y = torch.sum(Phi * z, 1, keepdim=True)
    return y


def At_operator(z, Phi):
    y = z * Phi
    return y


def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
