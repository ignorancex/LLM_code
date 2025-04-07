import cv2
import os
import math
import numpy as np
import argparse
import warnings
from distutils.util import strtobool
from glob import glob

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from core.pipeline import Pipeline
from core.dataset import X_Test
from core.utils.pytorch_msssim import ssim_matlab

warnings.filterwarnings("ignore")


def RGBframes_np2Tensor(imgIn, channel=3):
    ## input : T, H, W, C
    if channel == 1:
        # rgb --> Y (gray)
        imgIn = np.sum(
                imgIn * np.reshape(
                    [65.481, 128.553, 24.966], [1, 1, 1, 3]
                    ) / 255.0,
                axis=3,
                keepdims=True) + 16.0

    # to Tensor
    ts = (3, 0, 1, 2)  ############# dimension order should be [C, T, H, W]
    imgIn = torch.Tensor(imgIn.transpose(ts).astype(float)).mul_(1.0)
    return imgIn


def evaluate_prediction(ppl, test_data_path, multiple=8, pred_prev=False):
    tvalue2score = {}
    for type_folder in sorted(glob(os.path.join(test_data_path, '*', ''))):  # [type1,type2,type3,...]
        print("start to process type_folder: %s" % type_folder)
        for scene_folder in sorted(glob(os.path.join(type_folder, '*', ''))):  # [scene1,scene2,..]
            torch.cuda.empty_cache()
            print("start to process scene_folder: %s" % scene_folder)
            frame_folder = sorted(glob(scene_folder + '*.png'))  # 32 frames, ['00000.png',...,'00032.png']
            # previous frame prediction is converted into future frame prediction by switching input frames.
            if pred_prev:
                frame_folder = sorted(frame_folder, reverse=True)
            frames = []
            for ind in range(33):
                path = frame_folder[ind]
                frame = cv2.imread(path)
                frames.append(frame)
            frames = np.stack(frames, axis=0)  # (T, H, W, 3)
            """ np2Tensor [-1,1] normalized """
            frames = RGBframes_np2Tensor(frames)
            frames = frames.to(DEVICE, non_blocking=True) / 255.
            img0 = frames[:, 0, :, :].unsqueeze(0)
            img1 = frames[:, multiple, :, :].unsqueeze(0)

            n, c, h, w = img1.size()
            divisor = 256
            if (h % divisor != 0) or (w % divisor != 0):
                ph = ((h - 1) // divisor + 1) * divisor
                pw = ((w - 1) // divisor + 1) * divisor
                divisor = (0, pw - w, 0, ph - h)
                img0 = F.pad(img0, divisor, "constant", 0.5)
                img1 = F.pad(img1, divisor, "constant", 0.5)

            for i in range(multiple+1, 33):
                gt = frames[:, i, :, :]
                tvalue = i / multiple
                with torch.no_grad():
                    pred, _ = ppl.model(img0, img1, time_period=tvalue, skip_pad=True, fixed_pyr_level=7)
                pred = pred[0, :, :h, :w]

                ssim = ssim_matlab(pred.unsqueeze(0), gt.unsqueeze(0)).cpu().numpy()
                ssim = float(ssim)
                psnr = -10 * math.log10(
                        torch.mean((gt - pred) * (gt - pred)
                            ).cpu().data)
                if tvalue not in tvalue2score:
                    tvalue2score[tvalue] = []
                tvalue2score[tvalue].append([psnr, ssim])
                print('tvalue: {:.4f}; psnr: {:.4f}; ssim: {:.4f}'.format(tvalue, psnr, ssim))


    psnr_list = []
    for tvalue in tvalue2score:
        score = np.array(tvalue2score[tvalue]).mean(axis=0)
        psnr_list.append(score[0])
        print("previous frame time step: %.4f;  average psnr: %.4f; average ssim: %.4f" % (tvalue, score[0], score[1]))
    print("avg psnr for frame prediction: %.4f" % (np.mean(psnr_list)))


def evaluate_interp(ppl, test_data_path):
    dataset = X_Test(test_data_path=test_data_path, multiple=8)
    val_data = DataLoader(dataset, batch_size=1, num_workers=1, pin_memory=True)

    psnr_list = []
    ssim_list = []
    nr_val = val_data.__len__()
    for i, (frames, t_value, scene_name, frame_range) in enumerate(val_data):
        torch.cuda.empty_cache()
        frames = frames.to(DEVICE, non_blocking=True) / 255.
        B, C, T, h, w = frames.size()
        t_value = t_value.to(DEVICE, non_blocking=True)
        img0 = frames[:, :, 0, :, :]
        img1 = frames[:, :, 1, :, :]
        gt = frames[:, :, 2, :, :]

        divisor = 256
        if (h % divisor != 0) or (w % divisor != 0):
            ph = ((h - 1) // divisor + 1) * divisor
            pw = ((w - 1) // divisor + 1) * divisor
            divisor = (0, pw - w, 0, ph - h)
            img0 = F.pad(img0, divisor, "constant", 0.5)
            img1 = F.pad(img1, divisor, "constant", 0.5)
        # use fixed pyramid level for testing, as suggested in our paper
        with torch.no_grad():
            pred, _ = ppl.model(img0, img1, time_period=t_value, skip_pad=True, fixed_pyr_level=7)
        pred = pred[:, :, :h, :w]

        batch_psnr = []
        batch_ssim = []
        for j in range(gt.shape[0]):
            this_gt = gt[j]
            this_pred = pred[j]
            ssim = ssim_matlab(
                    this_pred.unsqueeze(0), this_gt.unsqueeze(0)
                    ).cpu().numpy()
            ssim = float(ssim)
            ssim_list.append(ssim)
            batch_ssim.append(ssim)
            psnr = -10 * math.log10(
                    torch.mean((this_gt - this_pred) * (this_gt - this_pred)
                        ).cpu().data)
            psnr_list.append(psnr)
            batch_psnr.append(psnr)

        print('batch: {}/{}; psnr: {:.4f}; ssim: {:.4f}'.format(i, nr_val,
            np.mean(batch_psnr), np.mean(batch_ssim)))

    psnr = np.array(psnr_list).mean()
    print('average psnr: {:.4f}'.format(psnr))
    ssim = np.array(ssim_list).mean()
    print('average ssim: {:.4f}'.format(ssim))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='benchmarking on 4k1000fps' +\
            'dataset for multiple interpolation or prediction')

    #**********************************************************#
    # => args for data loader
    parser.add_argument('--test_data_path', type=str, required=True,
            help='the path of 4k10000fps benchmark')

    #**********************************************************#
    # => set the task type: "interpolation" or "prediction"
    parser.add_argument('--pred_type', type=str, default="interpolation",
            help='interpolation or prediction')
    parser.add_argument('--multiple', type=int, default=8,
            help='multiple used in evaluating multi-frame prediction')
    parser.add_argument('--pred_prev', type=strtobool, default=False,
            help='predict previous (rather than future) frames')

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

    print("benchmarking on 4K1000FPS...")
    if args.pred_type == "interpolation":
        evaluate_interp(ppl, args.test_data_path)
    elif args.pred_type == "prediction":
        # note that multiple should be less than 32, since each test sequence only contains 32 frames.
        if args.multiple >= 32:
            raise ValueError("Please set multiple less than 32 for frame prediciton!")
        evaluate_prediction(ppl, args.test_data_path, multiple=args.multiple, pred_prev=args.pred_prev)
    else:
        raise ValueError("Please set `pred_type` as 'interpolation' or 'prediction'!")
