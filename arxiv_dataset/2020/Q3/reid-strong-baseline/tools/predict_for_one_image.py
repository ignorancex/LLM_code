# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
from os import mkdir

import torch
from torch.backends import cudnn

sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.inference import inference
from modeling import build_model
from utils.logger import setup_logger
from skimage import io
from PIL import Image
import torchvision.transforms as T

from torch.utils.data import DataLoader

from data.collate_batch import train_collate_fn, val_collate_fn
from data.datasets import init_dataset, ImageDataset
from data.samplers import RandomIdentitySampler, RandomIdentitySampler_alignedreid  # New add by gu
from data.transforms import build_transforms

def make_data_loader_for_val_data(cfg):
    val_transforms = build_transforms(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    if len(cfg.DATASETS.NAMES) == 1:
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)
    else:
        # TODO: add multi dataset to train
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)

    num_classes = dataset.num_train_pids

    val_set = ImageDataset(dataset= dataset.query + dataset.gallery,
                           rap_data_=dataset.rap_data,
                           transform=val_transforms,
                           is_train=False,
                           swap_roi_rou=False)
    val_loader = DataLoader(val_set,
                            batch_size=cfg.TEST.IMS_PER_BATCH,
                            shuffle=False,
                            num_workers=num_workers,
                            collate_fn=val_collate_fn)
    return val_loader, len(dataset.query), num_classes

def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument(
        "--config_file", default='../configs/softmax_triplet.yml', help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True

    val_loader, num_query, num_classes = make_data_loader_for_val_data(cfg)
    model = build_model(cfg, num_classes)
    model.load_state_dict(torch.load(cfg.TEST.WEIGHT))


    print("model is built and its parameters are loaded")
    inference(cfg, model, val_loader, num_query)

    # path_to_imgs = "/home/eshan/Desktop/ICIP_Paper/Figures"
    # img_1 = "CAM01-2013-12-23-20131223122515-20131223123103-tarid7-frame1602-line1.png"
    # img_1_path = os.path.join(path_to_imgs,img_1)
    # img_1_array = io.imread(img_1_path)
    # print("single image is read")
    #
    #
    # normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    # transform = T.Compose([T.Resize(cfg.INPUT.SIZE_TEST),T.ToTensor(), normalize_transform])
    # if not isinstance(img_1_array, Image.Image):  # convert the image type to PIL if it is not already
    #     img_1_PIL = Image.fromarray(img_1_array, 'RGB')
    # image_1_transformed = transform(img_1_PIL)
    # print("single image is transformed 'resized, normalized, became tensor'")
    # #single_loaded_img = val_loader.dataset.__getitem__(0)
    #
    # #single_loaded_img = image_1_transformed.to(cfg.MODEL.DEVICE)
    # single_loaded_img = image_1_transformed.unsqueeze(0)
    # #single_loaded_img = single_loaded_img.type('torch.FloatTensor') # instead of DoubleTensor
    # print("single image is ready to be fed to the model ")
    #
    # out_features = model(single_loaded_img)


if __name__ == '__main__':
    main()
