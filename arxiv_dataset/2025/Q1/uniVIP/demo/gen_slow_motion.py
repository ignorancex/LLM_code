import os
import sys
import shutil
import cv2
import torch
import argparse
import numpy as np
import math
from imageio import mimsave
from datetime import datetime
from importlib import import_module
from distutils.util import strtobool
from torch.nn import functional as F

from core.utils import flow_viz
from core.pipeline import Pipeline

import warnings
warnings.filterwarnings("ignore")

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="interpolate or predict frame, based on a given pair of images")
    parser.add_argument("--frame0", type=str, required=True,
            help="file path of the first input frame")
    parser.add_argument("--frame1", type=str, required=True,
            help="file path of the second input frame")
    parser.add_argument("--multiple", type=int, default=8,
            help="increase frame rate by multiple times")
    parser.add_argument("--save_root", type=str,
            default="./demo/output",
            help="root to save synthesized frame")

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
    # => init frame synthesis environment
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.demo = True

    #**********************************************************#
    # => init the pipeline
    args = parser.parse_args()
    model_cfg_dict = dict(
            load_pretrain = True,
            large_model = args.large_model,
            model_file = args.model_file
            )

    ppl = Pipeline(model_cfg_dict)
    ppl.eval()

    #**********************************************************#
    # => synthesize frames
    ori_img0 = cv2.imread(args.frame0)
    ori_img1 = cv2.imread(args.frame1)
    if ori_img0.shape != ori_img1.shape:
        ValueError("Please ensure that the input frames have the same size!")
    img0 = (torch.tensor(ori_img0.transpose(2, 0, 1)).to(DEVICE) / 255.).unsqueeze(0)
    img1 = (torch.tensor(ori_img1.transpose(2, 0, 1)).to(DEVICE) / 255.).unsqueeze(0)

    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)

    print("Start to synthesize %d frames, including %d past, %d in-between, and %d future frames" %\
            ((args.multiple-1)*3, (args.multiple-1), (args.multiple-1), (args.multiple-1)))
    now = str(datetime.now())
    now = now.replace(" ", "-")
    now = now.replace(":", ".")
    save_dir = os.path.join(args.save_root, now)
    os.makedirs(save_dir)

    n, c, h, w = img0.shape
    i = 1
    imgs = []
    for t in np.linspace(-1, 0, args.multiple+1):
        if t == -1: continue
        if t == 0: # the first input frame
            break
        print("synthesize %d-th frame..." % i)
        pred_img, info_dict = ppl.model(img0, img1, time_period=t)
        pred_img = pred_img[:, :, :h, :w]
        pred_img = (pred_img[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)
        imgs.append(pred_img[:,:,::-1])
        cv2.imwrite(os.path.join(save_dir, 'img%d.png' % i), pred_img)
        i += 1

    imgs.append(ori_img0[:,:,::-1])
    cv2.imwrite(os.path.join(save_dir, 'img%d.png' % i), ori_img0)
    print("copy frame0 as the %d-th frame..." % i)
    i += 1

    for t in np.linspace(0, 1, args.multiple+1):
        if t == 0: continue
        if t == 1: # the second input frame
            break
        print("synthesize %d-th frame..." % i)
        pred_img, info_dict = ppl.model(img0, img1, time_period=t)
        pred_img = pred_img[:, :, :h, :w]
        pred_img = (pred_img[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)
        imgs.append(pred_img[:,:,::-1])
        cv2.imwrite(os.path.join(save_dir, 'img%d.png' % i), pred_img)
        i += 1

    imgs.append(ori_img1[:,:,::-1])
    cv2.imwrite(os.path.join(save_dir, 'img%d.png' % i), ori_img1)
    print("copy frame1 as the %d-th frame..." % i)
    i += 1

    for t in np.linspace(1, 2, args.multiple+1):
        if t == 1: continue
        if t == 2:
            break
        print("synthesize %d-th frame..." % i)
        pred_img, info_dict = ppl.model(img0, img1, time_period=t)
        pred_img = pred_img[:, :, :h, :w]
        pred_img = (pred_img[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)
        imgs.append(pred_img[:,:,::-1])
        cv2.imwrite(os.path.join(save_dir, 'img%d.png' % i), pred_img)
        i += 1

    # make gif, and ensure that the shorter side is <= 480
    shorter =  min(h, w)
    ratio = 1
    if shorter > 480:
        ratio = 480 / shorter
        for i in range(len(imgs)):
            imgs[i] = cv2.resize(imgs[i], (int(w * ratio), int(h * ratio)))
    mimsave(os.path.join(save_dir, 'slomo_%dx.gif' % args.multiple), imgs, fps=args.multiple)
    overlay = 0.5 * ori_img0 + 0.5 * ori_img1
    overlay = cv2.resize(overlay, (int(w * ratio), int(h * ratio)))
    cv2.imwrite(os.path.join(save_dir, 'overlay.png'), overlay)
    print("\nConsecutive frames for slow-motion is completed! Please see the results in %s" % (save_dir))
