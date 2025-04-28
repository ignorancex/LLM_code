import argparse
import copy
import time
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.create_dataset import Mydataset, for_train_transform1, test_transform
from test_block import test_mertric_here
from fit import fit, set_seed, write_options
import cv2
import segmentation_models_pytorch as smp


parser = argparse.ArgumentParser()
parser.add_argument('--imgs_path', type=str,
                    default='/mnt/zrg/dataset/yingxiang/ccm_dataset/ccm_dataste_all/image',
                    help='imgs train data path.')
parser.add_argument('--labels_path', type=str,
                    default='/mnt/zrg/dataset/yingxiang/ccm_dataset/ccm_dataste_all/mask',
                    help='labels train data path.')
parser.add_argument('--csv_dir_train', type=str,
                    default='/mnt/zrg/dataset/yingxiang/ccm_dataset/ccm_dataste_all/train.csv',
                    help='labels train data path.')
parser.add_argument('--csv_dir_val', type=str,
                    default='/mnt/zrg/dataset/yingxiang/ccm_dataset/ccm_dataste_all/val.csv',
                    help='labels val data path.')
parser.add_argument('--csv_dir_test', type=str,
                    default='/mnt/zrg/dataset/yingxiang/ccm_dataset/ccm_dataste_all/val.csv',
                    help='labels val data path.')
parser.add_argument('--batch_size', default=2, type=int, help='batchsize')
parser.add_argument('--workers', default=8, type=int, help='batchsize')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--start_epoch', '-s', default=0, type=int, )
parser.add_argument('--warm_epoch', '-w', default=0, type=int, )
parser.add_argument('--end_epoch', '-e', default=50, type=int, )
parser.add_argument('--img_size', type=int,
                    default=768, help='input patch size of network input')
parser.add_argument('--num_classes', type=int, default=2,)
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--seed', type=int,
                    default=2026, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
parser.add_argument('--device', default='cuda', type=str, )
parser.add_argument('--throughput', action='store_true', help='Test throughput only')
parser.add_argument('--checkpoint', type=str, default='checkpoint/', )
parser.add_argument('--save_fold', type=str, default='FCCM', )
parser.add_argument('--model_name', type=str, default='UNet_effi-b3', )
parser.add_argument('--devicenum', default='0', type=str, )

args = parser.parse_args()
# config = get_config(args)
os.environ['CUDA_VISIBLE_DEVICES'] = args.devicenum
begin_time = time.time()

set_seed(seed=args.seed)
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = False
device = args.device
epochs = args.warm_epoch + args.end_epoch

imgs_dir, masks_dir = args.imgs_path, args.labels_path
train_csv = args.csv_dir_train
df_train = pd.read_csv(train_csv)#[0:4]
train_imgs = [''.join([imgs_dir, '/', i]) for i in df_train['image_name']]
train_masks = [''.join([masks_dir, '/', i]) for i in df_train['image_name']]

df_val = pd.read_csv(args.csv_dir_val)#[0:4]
val_imgs = [''.join([imgs_dir, '/', i]) for i in df_val['image_name']]
val_masks = [''.join([masks_dir, '/', i]) for i in df_val['image_name']]

df_test = pd.read_csv(args.csv_dir_test)#[0:4]
test_imgs = [''.join([imgs_dir, '/', i]) for i in df_test['image_name']]
test_masks = [''.join([masks_dir, '/', i]) for i in df_test['image_name']]

imgs_train = [cv2.imread(i)[:, :, ::-1] for i in train_imgs]
masks_train = [cv2.imread(i)[:, :, 0] for i in train_masks]
imgs_val = [cv2.imread(i)[:, :, ::-1] for i in val_imgs]
masks_val = [cv2.imread(i)[:, :, 0] for i in val_masks]
imgs_test = [cv2.imread(i)[:, :, ::-1] for i in test_imgs]
masks_test = [cv2.imread(i)[:, :, 0] for i in test_masks]
print('image done')


train_transform = for_train_transform1()
test_transform = test_transform

best_acc_final = []

def train(model, save_name, model_name):
    model_savedir = args.checkpoint + save_name + '/' + model_name + '/'
    save_name = model_savedir + model_name
    print(model_savedir)
    if not os.path.exists(model_savedir):
        os.makedirs(model_savedir)

    train_ds = Mydataset(imgs_train, masks_train, train_transform)
    val_ds = Mydataset(imgs_val, masks_val, test_transform)

    criterion = nn.CrossEntropyLoss().to('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    CosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-8)
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=args.batch_size, pin_memory=False, num_workers=8,
                          drop_last=True, )
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, pin_memory=False, num_workers=8, )
    best_acc = 0
    with tqdm(total=epochs, ncols=60) as t:
        for epoch in range(epochs):
            epoch_loss, epoch_iou, epoch_val_loss, epoch_val_iou = \
                fit(epoch, epochs, model, train_dl, val_dl, device, criterion, optimizer, CosineLR)

            f = open(model_savedir + 'log' + '.txt', "a")
            f.write('epoch' + str(float(epoch)) +
                    '  _train_loss' + str(epoch_loss) + '  _val_loss' + str(epoch_val_loss) +
                    ' _epoch_acc' + str(epoch_iou) + ' _val_iou' + str(epoch_val_iou) + '\n')

            if epoch_val_iou > best_acc:
                f.write('\n' + 'here' + '\n')
                best_model_wts = copy.deepcopy(model.state_dict())
                best_acc = epoch_val_iou
                torch.save(best_model_wts, ''.join([save_name, '.pth']))
            f.close()
            # torch.cuda.empty_cache()
            t.update(1)
    write_options(model_savedir, args, best_acc)

    dice, miou, pre, recall, f1_score, pa = test_mertric_here(model, imgs_test, masks_test, save_name)
    f = open(model_savedir + 'log1' + '.txt', "a")
    f.write('dice' + str(float(dice)) + '  _miou' + str(miou) +
            '  _pre' + str(pre) + '  _recall' + str(recall) +
            ' _f1_score' + str(f1_score) + ' _pa' + str(pa) + '\n')
    f.close()
    print('Done!')


if __name__ == '__main__':
    model = smp.Unet(encoder_name='efficientnet-b3', encoder_weights='imagenet', classes=2).cuda()
    print('model done')
    train(model, args.save_fold, args.model_name)

