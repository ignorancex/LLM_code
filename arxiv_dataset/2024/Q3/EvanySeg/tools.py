import os.path
import torch
import warnings
from torchvision.models import resnet50, resnet101
import cv2
import copy
import timm
import argparse
import numpy as np
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
from model import *
from monai.metrics import compute_hausdorff_distance

from transformers import ViTForImageClassification, SwinBackbone,SwinConfig
from argparse import Namespace

from transformers import ViTForImageClassification, ViTModel


def load_model(model_name):
    if model_name == "SAM":
        from SAM.segment_anything import sam_model_registry, SamPredictor
        sam = sam_model_registry["vit_h"](checkpoint="./checkpoints/sam_with_variants/sam_vit_h_4b8939.pth").cuda()
        predictor = SamPredictor(sam)
    elif model_name == "MedSAM_bbox":
        from SAM.segment_anything import sam_model_registry, SamPredictor
        sam = sam_model_registry["vit_b"](checkpoint="./checkpoints/sam_with_variants/medsam_vit_b.pth").cuda()
        predictor = SamPredictor(sam)
    elif model_name == "MedSAM_point":
        from SAM.segment_anything import sam_model_registry, SamPredictor
        sam = sam_model_registry["vit_b"](checkpoint="./checkpoints/sam_with_variants/medsam_point_prompt_flare22.pth").cuda()
        predictor = SamPredictor(sam)
    elif model_name == "SAM_Med2D":
        from SAM_Med2D.segment_anything import sam_model_registry
        from SAM_Med2D.segment_anything.predictor_sammed import SammedPredictor
        device = "cuda:0"
        opt = argparse.Namespace()
        opt.image_size = 256
        opt.encoder_adapter = True
        opt.sam_checkpoint = "./checkpoints/sam_with_variants/sam-med2d_b.pth"
        model = sam_model_registry["vit_b"](opt).to(device)
        predictor = SammedPredictor(model)
    else:
        raise RuntimeError("model_name must be  SAM 、MedSAM_bbox、MedSAM_point or SAM_Med2D")

    return predictor


def calc_information(mask_rpath):
    mask = cv2.imread(mask_rpath, 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    retval = retval - 1
    ind = stats[:, 4].argsort()
    stats = stats[ind][:-1]
    centroids = centroids[ind][:-1]
    mask = mask/255
    mask = mask[:, :, None]
    return retval, labels, stats, centroids, mask


def binary(rrr):
    raw = copy.deepcopy(rrr)
    raw[raw > 0.5] = 1
    raw[raw <= 0.5] = 0
    return raw


def calc_predict_bbox(predict):
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(predict, connectivity=8)
    retval = retval - 1
    ind = stats[:, 4].argsort()
    stats = stats[ind][:-1]
    centroids = centroids[ind][:-1]
    predict = predict//255
    predict = predict[:, :, None]
    return retval, labels, stats, centroids, predict


def load_classification_model(args):
    if args.classification_model_name == "resnet50":
        model_resnet50 = resnet50(pretrained=True)
        model = ResNet_ImageClassification(model_resnet50, args.flag)
    elif args.classification_model_name == "resnet101":
        model_resnet101 = resnet101(pretrained=True)
        model = ResNet_ImageClassification(model_resnet101, args.flag)
    elif args.classification_model_name == 'vit_base_224':
        model = ViT_ImageClassification("./checkpoints/vit_base", args.flag)
    elif args.classification_model_name == 'vit_large_224':
        model = ViT_ImageClassification("./checkpoints/vit_large", args.flag)
    return model


def create_paths(result_base_path):
    result_path = create_experiment_name(result_base_path=result_base_path, mode="number")
    for p in ["log", "view", "predict", "checkpoints", "test"]:
        sub_path = os.path.join(result_path, p)
        Path(sub_path).mkdir(parents=True, exist_ok=True)
    return result_path


def create_experiment_name(result_base_path, mode="number"):
    for i in range(1, 999):
        result_path = os.path.join(result_base_path, f"exp_{i}")
        if not os.path.exists(result_path):
            Path(result_path).mkdir(parents=True, exist_ok=True)
            return result_path


def create_run_name(result_base_path, mode="number"):
    for i in range(1, 999):
        result_path = os.path.join(result_base_path, f"run_{i}")
        if not os.path.exists(result_path):
            Path(result_path).mkdir(parents=True, exist_ok=True)
            return result_path


def show_args(opt, logger):
    dict = vars(opt)
    for key, value in dict.items():
        logger.info(f"{key}: {value}")


