import os

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet50, resnet101
from torch import optim
from torch.optim import lr_scheduler
from dataset import MyDataset 
import torch.distributed as dist
import argparse
import torch.nn as nn
import random
import numpy as np
import pandas as pd
from tools import *
from tqdm import tqdm
from utils.logger import Logger


def test(args, model, test_dataset, test_dataloader):
    true_dices = []
    true_hd = []
    pred_dices = []
    pred_hd = []
    Path(args.result_path).mkdir(parents=True, exist_ok=True)
    for i, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc='test'):
        crop_images = batch['crop_image_trans'].cuda()
        output = model(crop_images)
        if args.flag == 1:
            true_dices.append(batch['dice'].item())
            output_dice = output["logits_dice"].detach().cpu()
            pred_dices.append(output_dice.item())
        elif args.flag == 2:
            true_dices.append(batch['dice'].item())
            true_hd.append(batch['hd'].item())
            output_dice = output["logits_dice"].detach().cpu()
            output_hd = output["logits_hd"].detach().cpu()
            pred_dices.append(output_dice.item())
            pred_hd.append(output_hd.item())

    if args.flag == 1:
        df = pd.DataFrame(dict(true_dice=true_dices, preds_dices=pred_dices))
    elif args.flag == 2:
        df = pd.DataFrame(dict(true_dice=true_dices, preds_dices=pred_dices, true_hd=true_hd, pred_hd=pred_hd))

    csv_name = f"{args.result_path}/{os.path.basename(args.dataset_path)}_{args.classification_model_name}.csv"
    df.to_csv(csv_name, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--img_size', type=int, default=224, help='image size')
    parser.add_argument('--flag', type=int, default=1, help="1 for dice; 2 for dice and hd")
    parser.add_argument("--dataset_path", type=str, default='./datasets/preprocess/test_sam_Polyp', help='path to dataset')
    parser.add_argument("--result_path", type=str, default="./result")
    parser.add_argument("--model_path", type=str, default='./result/exp_1/checkpoints/ViT_base_bbox_dice.pth')
    parser.add_argument("--classification_model_name", type=str, default="vit_base_224", help='[ resnet50 | resnet101 | vit_base_224 | vit_large_224 ]')
    args = parser.parse_args()

    args.result_path = os.path.join(os.path.dirname(os.path.dirname(args.model_path)), "test", os.path.basename(args.model_path).split(".")[0])
    args.result_path = create_run_name(args.result_path)
    logger = Logger(os.path.join(args.result_path, "test_log.txt")).get_logger()
    show_args(args, logger)
    args.result_path = os.path.join(args.result_path, "save")
    Path(args.result_path).mkdir(exist_ok=True, parents=True)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    test_dataset = MyDataset(root_dir=args.dataset_path, transform=transform, flag=args.flag)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = load_classification_model(args)
    model.load_state_dict(torch.load(args.model_path))
    model.cuda()
    model.eval()
    test(args, model, test_dataset, test_dataloader)

