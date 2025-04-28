import json
import os
import torch
import Miou
import cv2
import argparse
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import csv
from PIL import Image
import time
import torch
from torch.utils.data import DataLoader
from sklearn import metrics
from datasets.create_dataset import Mydataset,for_train_transform,test_transform
import argparse
import Miou
import sys
import torchvision.transforms.functional as tf
from datasets.create_dataset import Mydataset,for_train_transform,test_transform
import pandas as pd
from config import get_config
from datasets.create_dataset import Mydataset_test
# from collections import OrderedDict
from model.contrast.unetxt.UNetxt import UNext
# from model.contrast.TransFuse.TransFuse import TransFuse_S
# from model.contrast.cenet import CE_Net_
from model.unet import UNet
# from model.swin_unet.vision_transformer import SwinUnet as Vit
# from model.contrast.TransUnet.vit_seg_modeling import VisionTransformer as ViT_seg
# from model.contrast.TransUnet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg


parser = argparse.ArgumentParser()
parser.add_argument('--imgs_path', type=str,
                    default='/mnt/zrg/dataset/yingxiang/ccm_dataset/ccm_dataste_all/image',
                    help='imgs train data path.')
parser.add_argument('--labels_path', type=str,
                    default='/mnt/zrg/dataset/yingxiang/ccm_dataset/ccm_dataste_all/mask',
                    help='labels train data path.')
parser.add_argument('--csv_dir_test', type=str,
                    default='/mnt/zrg/dataset/yingxiang/ccm_dataset/ccm_dataste_all/val.csv',
                    help='labels val data path.')
parser.add_argument('--pre_save_dir', type=str,
                    default='/mnt/zrg/Image_segmentation/2D_segmentation/FCCM_TBME_1/result/FCCM2/',
                    help='labels pre save path.')
parser.add_argument('--zhibiao_save_txt', default='/mnt/zrg/Image_segmentation/2D_segmentation/FCCM_TBME_1/result/infer_score_test/', type=str, )

# TransUnet
parser.add_argument('--img_size', type=int,
                    default=768, help='input patch size of network input')
parser.add_argument('--num_classes', type=int, default=2,)
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')

parser.add_argument('--batch_size', default=1, type=int, help='batch size')
parser.add_argument('--devicenum', default='0', type=str, )

parser.add_argument('--model_name', default='Unet_effi_b3', type=str, )
parser.add_argument('--weight', default='./model-weight/Unet_effi_b3.pth', type=str, )

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.devicenum
# + '/' + 'predict'
pre_result_path = args.pre_save_dir + args.model_name
if not os.path.exists(pre_result_path):
    os.makedirs(pre_result_path)

if not os.path.exists(args.zhibiao_save_txt):
    os.makedirs(args.zhibiao_save_txt)

# shangse_pre_result_path = args.pre_save_dir + args.model_name + '/' + 'predict_shangse'
# if not os.path.exists(shangse_pre_result_path):
#     os.makedirs(shangse_pre_result_path)

begin_time = time.time()

imgs_dir, masks_dir = args.imgs_path, args.labels_path
df_test = pd.read_csv(args.csv_dir_test)#[0:4]
test_imgs = [''.join([imgs_dir, '/', i]) for i in df_test['image_name']]
test_masks = [''.join([masks_dir, '/', i]) for i in df_test['image_name']]

train_transform = for_train_transform()
test_transform = test_transform

test_number = len(test_imgs)
test_ds = Mydataset_test(test_imgs, test_masks, test_transform)
test_dl = DataLoader(test_ds, batch_size=args.batch_size, pin_memory=False, num_workers=4, )
print('==> Preparing data..')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model
print('==> Building model..')

# model = smp.UnetPlusPlus(encoder_name='vgg16', encoder_weights=None, classes=2).to(device)
# model = smp.DeepLabV3Plus(encoder_name='mobilenet_v2', encoder_weights=None, classes=2).to(device)
# model = UNext(num_classes=2, img_size=768).to(device)
# model = TransFuse_S(num_classes=2, pretrained=True).cuda()
# model = CE_Net_(num_classes=2).cuda()
# model = UNet().cuda()
model = smp.Unet(encoder_name='efficientnet-b3', encoder_weights='imagenet', classes=2).cuda()
# model = smp.Unet(encoder_name='mobilenet_v2', encoder_weights='imagenet', classes=2).cuda()
# model = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', classes=2).cuda()
# model = smp.UnetPlusPlus(encoder_name='efficientnet-b3', encoder_weights='imagenet', classes=2).cuda()
# model = smp.UnetPlusPlus(encoder_name='mobilenet_v2', encoder_weights='imagenet', classes=2).cuda()
# model = smp.UnetPlusPlus(encoder_name='resnet34', encoder_weights='imagenet', classes=2).cuda()
# model = smp.DeepLabV3Plus(encoder_name='efficientnet-b3', encoder_weights='imagenet', classes=2).cuda()
# model = smp.DeepLabV3Plus(encoder_name='mobilenet_v2', encoder_weights='imagenet', classes=2).cuda()
# model = smp.DeepLabV3Plus(encoder_name='resnet34', encoder_weights='imagenet', classes=2).cuda()
# config_vit = CONFIGS_ViT_seg[args.vit_name]
# config_vit.n_classes = args.num_classes
# config_vit.n_skip = args.n_skip
# if args.vit_name.find('R50') != -1:
#     config_vit.patches.grid = (
#     int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
# model = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
# model.load_from(weights=np.load(config_vit.pretrained_path))

state_dict = torch.load(args.weight)
model.load_state_dict(state_dict)
model.eval()

pixel_sum_list, bai_pixel_list, name_list, each_img_lesion_num, each_lesion_Volume, component_dict_list = [], [], [], [], [], []
test_dice, test_miou, test_Pre, test_recall, test_F1score, test_pa = 0, 0, 0, 0, 0, 0
with torch.no_grad():
    for batch_idx, (name, inputs, targets) in enumerate(test_dl):
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        out = model(inputs)
        predicted = out.argmax(1)

        test_dice += Miou.calculate_mdice(predicted, targets, 2).item()
        test_miou += Miou.calculate_miou(predicted, targets, 2).item()
        test_Pre += Miou.pre(predicted, targets).item()
        test_recall += Miou.recall(predicted, targets).item()
        test_F1score += Miou.F1score(predicted, targets).item()
        test_pa += Miou.Pa(predicted, targets).item()

        predict = predicted.squeeze(0)
        mask_np = predict.cpu().numpy()
        mask_np = (mask_np * 255).astype('uint8')
        mask_np[mask_np > 160] = 255
        mask_np[mask_np <= 160] = 0

        cv2.imwrite(pre_result_path + '/' + name[0], mask_np)

average_test_dice = test_dice / test_number
average_test_miou = test_miou / test_number
average_test_Pre = test_Pre / test_number
average_test_recall = test_recall / test_number
average_test_F1score = test_F1score / test_number
average_test_pa = test_pa / test_number

dice, miou, pre, recall, f1_score, pa = \
    '%.4f' % average_test_dice, '%.4f' % average_test_miou, '%.4f' % average_test_Pre, '%.4f' % average_test_recall, '%.4f' % average_test_F1score, '%.4f' % average_test_pa

end_time = time.time()

with open(args.zhibiao_save_txt + args.model_name + '.txt', 'w+') as f:
    f.write("时间:" + str(end_time - begin_time) + '\n' + "dice:" + str(dice) + '\n' + "miou:" + str(miou) + '\n' +
            "pre:" + str(pre) + '\n' + "recall:" + str(recall) + '\n' + "f1_score:" + str(f1_score) + '\n' + "pa:" + str(pa))

print("时间")
print(end_time - begin_time)
print("dice:" + '\t' + str(dice))
print("miou:" + '\t' + str(miou))
print("pre:" + '\t' + str(pre))
print("recall:" + '\t' + str(recall))
print("f1_score:" + '\t' + str(f1_score))
print("pa:" + '\t' + str(pa))

