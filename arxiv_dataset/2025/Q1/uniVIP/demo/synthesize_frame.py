import os
import sys
import shutil
import cv2
import torch
import argparse
import numpy as np
import math
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
    parser.add_argument("--time_period", type=float, default=0.5,
            help="time period to synthesize frame")
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
    # parser.add_argument('--large_model', type=strtobool, default=true,
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
    # => synthesize frame
    ori_img0 = cv2.imread(args.frame0)
    ori_img1 = cv2.imread(args.frame1)
    if ori_img0.shape != ori_img1.shape:
        ValueError("Please ensure that the input frames have the same size!")
    img0 = (torch.tensor(ori_img0.transpose(2, 0, 1)).to(DEVICE) / 255.).unsqueeze(0)
    img1 = (torch.tensor(ori_img1.transpose(2, 0, 1)).to(DEVICE) / 255.).unsqueeze(0)

    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)

    print("\nInitialization is OK! Begin to synthesize images...")
    n, c, h, w = img0.shape

    pred_img, info_dict = ppl.model(img0, img1, time_period=args.time_period)
    pred_img = pred_img[:, :, :h, :w]
    bi_flow = info_dict["bi_flow"][:, :, :h, :w]

    overlay_input = (ori_img0 * 0.5 + ori_img1 * 0.5)
    pred_img = (pred_img[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)
    bi_flow = bi_flow[0].cpu().numpy().transpose(1, 2, 0)

    flow01 = bi_flow[:, :, :2]
    flow10 = bi_flow[:, :, 2:]
    flow01 = flow_viz.flow_to_image(flow01, convert_to_bgr=True)
    flow10 = flow_viz.flow_to_image(flow10, convert_to_bgr=True)
    bi_flow = np.concatenate([flow01, flow10], axis=1)

    now = str(datetime.now())
    now = now.replace(" ", "-")
    now = now.replace(":", ".")
    save_dir = os.path.join(args.save_root, now)
    os.makedirs(save_dir)
    cv2.imwrite(os.path.join(save_dir, '0-img0.png'), ori_img0)
    cv2.imwrite(os.path.join(save_dir, '1-img1.png'), ori_img1)
    cv2.imwrite(os.path.join(save_dir, '2-overlay-input.png'), overlay_input)
    cv2.imwrite(os.path.join(save_dir, '3-synthesized-img.png'), pred_img)
    cv2.imwrite(os.path.join(save_dir, '4-bi-flow.png'), bi_flow)

    print("\nFrame synthesis is completed! Please see the results in %s" % (save_dir))
