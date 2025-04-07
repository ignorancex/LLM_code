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
from core.dataset import VimeoDataset
from core.utils.pytorch_msssim import ssim_matlab

warnings.filterwarnings("ignore")


def evaluate(ppl, data_root, batch_size, nr_data_worker=1, pred_type="interpolation"):
    dataset = VimeoDataset(dataset_name='validation', data_root=data_root)
    val_data = DataLoader(dataset, batch_size=batch_size,
            num_workers=nr_data_worker, pin_memory=True)

    psnr_list = []
    ssim_list = []
    nr_val = val_data.__len__()
    for i, data in enumerate(val_data):
        data_gpu = data[0] if isinstance(data, list) else data
        data_gpu = data_gpu.to(DEVICE, non_blocking=True) / 255.
        if pred_type == "interpolation":
            img0 = data_gpu[:, :3]
            gt = data_gpu[:, 3:6]
            img1 = data_gpu[:, 6:9]
            time_period = 0.5
        else:
            img0 = data_gpu[:, :3]
            img1 = data_gpu[:, 3:6]
            gt = data_gpu[:, 6:9]
            time_period = 2.0

        with torch.no_grad():
            pred, _ = ppl.model(img0, img1, time_period=time_period)

        batch_psnr = []
        batch_ssim = []
        for j in range(gt.shape[0]):
            this_gt = gt[j]
            this_pred = pred[j]
            ssim = ssim_matlab(
                    this_pred.unsqueeze(0),
                    this_gt.unsqueeze(0)).cpu().numpy()
            ssim = float(ssim)
            ssim_list.append(ssim)
            batch_ssim.append(ssim)
            psnr = -10 * math.log10(
                    torch.mean(
                        (this_gt - this_pred) * (this_gt - this_pred)
                        ).cpu().data)
            psnr_list.append(psnr)
            batch_psnr.append(psnr)
        print('batch: {}/{}; psnr: {:.4f}; ssim: {:.4f}'.format(i+1, nr_val,
            np.mean(batch_psnr), np.mean(batch_ssim)))

    psnr = np.array(psnr_list).mean()
    print('average psnr: {:.4f}'.format(psnr))
    ssim = np.array(ssim_list).mean()
    print('average ssim: {:.4f}'.format(ssim))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='benchmark on vimeo90k')

    #**********************************************************#
    # => args for dataset and data loader
    parser.add_argument('--data_root', type=str, required=True,
            help='root dir of vimeo_triplet')
    parser.add_argument('--batch_size', type=int, default=8,
            help='batch size for data loader')
    parser.add_argument('--nr_data_worker', type=int, default=2,
            help='number of the worker for data loader')

    #**********************************************************#
    # => set the task type: "interpolation" or "prediction"
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

    print("benchmarking on Vimeo90K...")
    evaluate(ppl, args.data_root, args.batch_size, args.nr_data_worker, args.pred_type)
