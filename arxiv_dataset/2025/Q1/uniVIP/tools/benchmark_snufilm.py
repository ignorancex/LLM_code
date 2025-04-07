import os
import math
import numpy as np
import argparse
import warnings
from distutils.util import strtobool

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader


from core.pipeline import Pipeline
from core.dataset import SnuFilm
from core.utils.pytorch_msssim import ssim_matlab

warnings.filterwarnings("ignore")


def eval_subset(ppl, val_data, subset_name="easy", pred_type="interpolation"):
    psnr_list = []
    ssim_list = []
    nr_val = val_data.__len__()
    for i, data in enumerate(val_data):
        data_gpu = data[0] if isinstance(data, list) else data
        data_gpu = data_gpu.to(DEVICE, non_blocking=True) / 255.

        if pred_type == "interpolation":
            img0 = data_gpu[:, :3]
            img1 = data_gpu[:, 3:6]
            gt = data_gpu[:, 6:9]
            time_period = 0.5
        else:
            img0 = data_gpu[:, :3]
            gt = data_gpu[:, 3:6]
            img1 = data_gpu[:, 6:9]
            time_period = 2.0

        n, c, h, w = img0.shape
        divisor = 64
        if (h % divisor != 0) or (w % divisor != 0):
            has_pad = True
            ph = ((h - 1) // divisor + 1) * divisor
            pw = ((w - 1) // divisor + 1) * divisor
            padding = (0, pw - w, 0, ph - h)
            img0 = F.pad(img0, padding, "constant", 0.5)
            img1 = F.pad(img1, padding, "constant", 0.5)

        # use fixed pyramid level for testing, as suggested in our paper
        with torch.no_grad():
            pred, _ = ppl.model(img0, img1, time_period=time_period, skip_pad=True, fixed_pyr_level=5)
            pred = pred[:, :, :h, :w]

        batch_psnr = []
        batch_ssim = []
        for j in range(gt.shape[0]):
            this_gt = gt[j]
            this_pred = pred[j]
            ssim = float(ssim_matlab(this_pred.unsqueeze(0), this_gt.unsqueeze(0)).cpu().numpy())
            ssim_list.append(ssim)
            batch_ssim.append(ssim)
            psnr = -10 * math.log10(
                    torch.mean((this_gt - this_pred) * (this_gt - this_pred)).cpu().data)
            psnr_list.append(psnr)
            batch_psnr.append(psnr)

        print('subset: {}; batch: {}/{}; psnr: {:.4f}; ssim: {:.4f}'
                .format(subset_name, i, nr_val,
                    np.mean(batch_psnr), np.mean(batch_ssim)))

    avg_psnr = np.array(psnr_list).mean()
    print('subset: {}, average psnr: {:.4f}'.format(subset_name, avg_psnr))
    avg_ssim = np.array(ssim_list).mean()
    print('subset: {}, average ssim: {:.4f}'.format(subset_name, avg_ssim))

    return avg_psnr, avg_ssim



def evaluate(ppl, data_root, batch_size,
        nr_data_worker=1,
        pred_type="interpolation"):
    print('start to evaluate the easy subset ......')
    dataset_val = SnuFilm(data_root=data_root, data_type="easy")
    val_data = DataLoader(dataset_val, batch_size=batch_size,
            num_workers=nr_data_worker, pin_memory=True)
    easy_avg_psnr, easy_avg_ssim = \
            eval_subset(ppl, val_data, subset_name="easy", pred_type=pred_type)

    print('start to evaluate the medium subset ......')
    dataset_val = SnuFilm(data_root=data_root, data_type="medium")
    val_data = DataLoader(dataset_val, batch_size=batch_size,
            num_workers=nr_data_worker, pin_memory=True)
    medium_avg_psnr, medium_avg_ssim = \
            eval_subset(ppl, val_data, subset_name="medium", pred_type=pred_type)

    print('start to evaluate the hard subset ......')
    dataset_val = SnuFilm(data_root=data_root, data_type="hard")
    val_data = DataLoader(dataset_val, batch_size=batch_size,
            num_workers=nr_data_worker, pin_memory=True)
    hard_avg_psnr, hard_avg_ssim = \
            eval_subset( ppl, val_data, subset_name="hard", pred_type=pred_type)

    print('start to evaluate the extreme subset ......')
    dataset_val = SnuFilm(data_root=data_root, data_type="extreme")
    val_data = DataLoader(dataset_val, batch_size=batch_size,
            num_workers=nr_data_worker, pin_memory=True)
    extreme_avg_psnr, extreme_avg_ssim = \
            eval_subset(ppl, val_data, subset_name="extreme",
                    pred_type=pred_type)

    print('easy subset: avg psnr: {:.4f}'.format(easy_avg_psnr))
    print('easy subset: avg ssim: {:.4f}'.format(easy_avg_ssim))

    print('medium subset: avg psnr: {:.4f}'.format(medium_avg_psnr))
    print('medium subset: avg ssim: {:.4f}'.format(medium_avg_ssim))

    print('hard subset: avg psnr: {:.4f}'.format(hard_avg_psnr))
    print('hard subset: avg ssim: {:.4f}'.format(hard_avg_ssim))

    print('extreme subset: avg psnr: {:.4f}'.format(extreme_avg_psnr))
    print('extreme subset: avg ssim: {:.4f}'.format(extreme_avg_ssim))




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='benchmarking on snu-film')

    #**********************************************************#
    # => args for data loader
    parser.add_argument('--data_root', type=str, required=True,
            help='root dir of snu-film')
    parser.add_argument('--batch_size', type=int, default=1,
            help='batch size for data loader')
    parser.add_argument('--nr_data_worker', type=int, default=1,
            help='number of the worker for data loader')

    #**********************************************************#
    # => args for model
    parser.add_argument('--pred_type', type=str, default="interpolation",
            help='interpolation or prediction')

    ## load uniVIP-B, by default
    parser.add_argument('--large_model', type=strtobool, default=False,
            help='if False, load uniVIP-B; if True, load uniVIP-L model')
    parser.add_argument('--model_file', type=str,
            default="./checkpoints/uniVIP-B.pkl",
            help='weight of uniVIP model')

    ## load uniVIP-L
    # parser.add_argument('--large_model', type=strtobool, default=True,
    #         help='if False, load uniVIP-B; if True, load uniVIP-L model')
    # parser.add_argument('--model_file', type=str,
    #         default="./checkpoints/uniVIP-L.pkl",
    #         help='weight of uniVIP model')

    #**********************************************************#
    # => init the benchmarking environment
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.demo = True
    torch.backends.cudnn.benchmark = True

    #**********************************************************#
    # => init the pipeline and start to benchmark
    args = parser.parse_args()

    model_cfg_dict = dict(
            load_pretrain = True,
            large_model = args.large_model,
            model_file = args.model_file
            )
    ppl = Pipeline(model_cfg_dict)

    print("benchmarking on SNU-FILM...")
    evaluate(ppl, args.data_root, args.batch_size, args.nr_data_worker, args.pred_type)
