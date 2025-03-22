import argparse
import os, sys
import pandas as pd

#sys.path.append("..")
sys.path.append("/Data/DoDNet/")
import glob
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import pickle
import cv2
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random


import torch.nn as nn
from unet2D_Omni import UNet2D as UNet2D_ns
import imgaug.augmenters as iaa

from torchvision import transforms

from PIL import Image, ImageOps

import kornia
import os.path as osp

from MOTSDataset_2D_Patch_supervise_merge_csv_V4 import MOTSDataSet as MOTSDataSet
from MOTSDataset_2D_Patch_supervise_csv import MOTSValDataSet as MOTSValDataSet

import random
import timeit
from tensorboardX import SummaryWriter
import loss_functions.loss_2D as loss

from engine import Engine
from apex import amp
from apex.parallel import convert_syncbn_model
#from focalloss import FocalLoss2dff
from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data import DataLoader, random_split
start = timeit.default_timer()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from util.image_pool import ImagePool
import math
from skimage.transform import rescale, resize

def one_hot_3D(targets,C = 2):
    targets_extend=targets.clone()
    targets_extend.unsqueeze_(1) # convert to Nx1xHxW
    one_hot = torch.cuda.FloatTensor(targets_extend.size(0), C, targets_extend.size(2), targets_extend.size(3)).zero_()
    one_hot.scatter_(1, targets_extend, 1)
    return one_hot

    # parser.add_argument("--valset_dir", type=str, defau


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


confident_flag = 1
confident_e = 1
simattention_flag = 1
simattention_e = 0
simloss_flag = 0
simloss_e = 0

def get_arguments():

    parser = argparse.ArgumentParser(description="DeepLabV3")
    parser.add_argument("--trainset_dir", type=str, default='/Data3/HumanKidney/cell_segmentation/datasets_patch/After_correct/auto_label_train_set/data_list.csv')
    # parser.add_argument("--trainset_dir", type=str, default='/Data2/Demo_KI_data_trainingset_patch/data_list.csv')

    parser.add_argument("--valset_dir", type=str, default='/Data3/HumanKidney/cell_segmentation/datasets_patch/After_correct/auto_label_val_set/data_list.csv')
    # parser.add_argument("--valset_dir", type=str, default='/Data2/Demo_KI_data_trainingset_patch/data_list.csv')
    parser.add_argument("--psuedo_ratio", type=float, default=0.1)
    parser.add_argument("--psuedo_ratio_ptc", type=float, default=0.1)

    parser.add_argument("--train_list", type=str, default='list/MOTS/MOTS_train.txt')
    parser.add_argument("--val_list", type=str, default='list/MOTS/xx.txt')
    parser.add_argument("--edge_weight", type=float, default=1.0)

    parser.add_argument("--scale", type=str2bool, default=False)
    parser.add_argument("--snapshot_dir", type=str, default='snapshots_2D/Omni-Seg_0212_After_correct_selfloss_weight_1_%s_%s_%s_%s_%s_%s/' % (str(confident_flag), str(confident_e), str(simattention_flag), str(simattention_e), str(simloss_flag), str(simloss_e)))
    parser.add_argument("--reload_path", type=str, default='/Data3/HumanKidney/cell_segmentation/snapshots_2D/Omni-Seg_0212_Before_correct_selfloss_weight_1_1_1_1_0_0_0/MOTS_DynConv_Omni-Seg_0212_Before_correct_selfloss_weight_1_1_1_1_0_0_0_e59.pth')
    parser.add_argument("--reload_from_checkpoint", type=str2bool, default=True)
    parser.add_argument("--input_size", type=str, default='256,256')
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("--FP16", type=str2bool, default=False)
    parser.add_argument("--num_epochs", type=int, default=101)
    parser.add_argument("--itrs_each_epoch", type=int, default=250)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--val_pred_every", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--weight_std", type=str2bool, default=True)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--power", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--ignore_label", type=int, default=255)
    parser.add_argument("--is_training", action="store_true")
    parser.add_argument("--random_mirror", type=str2bool, default=True)
    parser.add_argument("--random_scale", type=str2bool, default=True)
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--gpu", type=str, default='None')
    return parser


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, lr, num_stemps, power):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(lr, i_iter, num_stemps, power)
    optimizer.param_groups[0]['lr'] = lr
    return lr


def mask_to_box(tensor):
    tensor = tensor.permute([0,2,3,1]).cpu().numpy()
    rmin = np.zeros((4))
    rmax = np.zeros((4))
    cmin = np.zeros((4))
    cmax = np.zeros((4))

    for ki in range(len(tensor)):
        rows = np.any(tensor[ki], axis=1)
        cols = np.any(tensor[ki], axis=0)

        try:
            rmin[ki], rmax[ki] = np.where(rows)[0][[0, -1]]
            cmin[ki], cmax[ki] = np.where(cols)[0][[0, -1]]
        except:
            rmin[ki], rmax[ki] = 0, 511
            cmin[ki], cmax[ki] = 0, 511

    # plt.imshow(tensor[0,int(rmin[0]):int(rmax[0]),int(cmin[0]):int(cmax[0]),:])
    return rmin.astype(np.uint32), rmax.astype(np.uint32), cmin.astype(np.uint32), cmax.astype(np.uint32)

def get_scale_tensor(pred, rmin, rmax, cmin, cmax):
    if len(pred.shape) == 3:
        return pred[:,rmin:rmax,cmin:cmax].unsqueeze(0)
    else:
        return pred[rmin:rmax, cmin:cmax].unsqueeze(0)

def count_score(preds, labels, rmin, rmax, cmin, cmax, simscore):

    Val_F1 = 0
    Val_DICE = 0
    Val_TPR = 0
    Val_PPV = 0
    cnt = 0

    for ki in range(len(preds)):
        cnt += 1
        pred = preds[ki,:,rmin[ki]:rmax[ki],cmin[ki]:cmax[ki]]
        label = labels[ki,:,rmin[ki]:rmax[ki],cmin[ki]:cmax[ki]]
        weight = simscore[ki,rmin[ki]:rmax[ki],cmin[ki]:cmax[ki]].unsqueeze(0)

        # preds1 = sum(preds,[])
        # labels1 = sum(labels,[])
        if label.sum() > 0:
            Val_DICE += dice_score(pred, label, weight)
            # preds1 = preds[:,1,...].detach().view(-1).cpu().numpy()
            preds1 = pred[1, ...].flatten().detach().cpu().numpy()
            # labels1 = labels[:,1,...].view(-1).cpu().numpy()
            labels1 = label[1, ...].detach().flatten().detach().cpu().numpy()

            weight1 = weight.detach().flatten().detach().cpu().numpy()

            cnf_matrix = confusion_matrix(preds1 > 0.5, labels1 > 0.5, sample_weight = weight1)

            FP = cnf_matrix[1,0]
            FN = cnf_matrix[0,1]
            TP = cnf_matrix[1,1]
            TN = cnf_matrix[0,0]

            FP = FP.astype(float)
            FN = FN.astype(float)
            TP = TP.astype(float)
            TN = TN.astype(float)

            Val_TPR += TP / (TP + FN)
            Val_PPV += TP / (TP + FP)

            Val_F1 += f1_score(preds1 > 0.5, labels1 > 0.5, average='macro')

        else:

            Val_DICE += 1.
            Val_F1 += 1.
            Val_TPR += 1.
            Val_PPV += 1.

    return Val_F1/cnt, Val_DICE/cnt, Val_TPR/cnt, Val_PPV/cnt

def dice_score(preds, labels, weights):  # on GPU
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    predict = preds.contiguous().view(preds.shape[0], -1)
    target = labels.contiguous().view(labels.shape[0], -1)

    weight = weights.contiguous().view(1, -1)

    num = torch.sum(torch.mul(predict, target) * weight, dim=1)
    den = torch.sum(predict * weight, dim=1) + torch.sum(target * weight, dim=1) + 1

    dice = (2 * num / den)

    return dice.mean()

def get_loss(preds, labels, weight, loss_seg_DICE, loss_seg_CE):

    term_seg_Dice = 0
    term_seg_BCE = 0
    term_all = 0

    term_seg_Dice += loss_seg_DICE.forward(preds, labels, weight)
    term_seg_BCE += loss_seg_CE.forward(preds, labels, weight)
    term_all += (term_seg_Dice + term_seg_BCE)

    return term_seg_Dice, term_seg_BCE, term_all

def supervise_learning(images, labels, batch_size, scales, model, now_task, weight, loss_seg_DICE, loss_seg_CE_1, loss_seg_CE_10, confident_flag, confident_e, simattention_flag, simattention_e, simloss_flag, simloss_e):
    a = labels.clone().detach()
    preds, simscore, conscore = model(images, torch.ones(batch_size).cuda() * now_task, scales, labels)
    labels = one_hot_3D(labels.long())

    term_seg_Dice = 0
    term_seg_BCE = 0
    term_seg_all = 0

    #term_seg_Dice, term_seg_BCE, term_all = get_loss(preds, labels, 1.5 ** simscore, loss_seg_DICE, loss_seg_CE)

    cnt = 0
    if confident_flag > 0:
        if confident_e:
            now_term_seg_Dice, now_term_seg_BCE, now_term_seg_all = get_loss(preds, labels, 1.5 ** (conscore * a), loss_seg_DICE, loss_seg_CE_10)
        else:
            now_term_seg_Dice, now_term_seg_BCE, now_term_seg_all = get_loss(preds, labels, conscore * a, loss_seg_DICE, loss_seg_CE_10)

        term_seg_Dice += confident_flag * now_term_seg_Dice
        term_seg_BCE += confident_flag * now_term_seg_BCE
        term_seg_all += confident_flag * now_term_seg_all
        cnt += confident_flag

    if simattention_flag > 0:
        if simattention_e:
            now_term_seg_Dice, now_term_seg_BCE, now_term_seg_all = get_loss(preds, labels, 1.5 ** (simscore * a), loss_seg_DICE,
                                                                             loss_seg_CE_10)
        else:
            now_term_seg_Dice, now_term_seg_BCE, now_term_seg_all = get_loss(preds, labels, simscore * a, loss_seg_DICE, loss_seg_CE_10)

        term_seg_Dice += simattention_flag * now_term_seg_Dice
        term_seg_BCE += simattention_flag * now_term_seg_BCE
        term_seg_all += simattention_flag * now_term_seg_all
        cnt += simattention_flag

    if simloss_flag > 0:
        final_sim = simscore * a

        sim_score_onehot = torch.zeros((labels.shape)).cuda()
        sim_score_onehot[:, 0] = 1 - final_sim
        sim_score_onehot[:, 1] = final_sim

        if simloss_e:
            now_term_seg_Dice, now_term_seg_BCE, now_term_seg_all = get_loss(preds, sim_score_onehot, 1.5 ** final_sim, loss_seg_DICE, loss_seg_CE_1)
        else:
            now_term_seg_Dice, now_term_seg_BCE, now_term_seg_all = get_loss(preds, sim_score_onehot, final_sim, loss_seg_DICE, loss_seg_CE_1)

        term_seg_Dice += simloss_flag * now_term_seg_Dice
        term_seg_BCE += simloss_flag * now_term_seg_BCE
        term_seg_all += simloss_flag * now_term_seg_all
        cnt += simloss_flag

    return term_seg_Dice / cnt, term_seg_BCE / cnt, term_seg_all / cnt

def supervise_learning_semi_nonzero(images, labels, batch_size, scales, model, now_task, weight, loss_seg_DICE, loss_seg_CE, loss_KL, loss_MSE, data_aug):

    preds,_ = model(images, torch.ones(batch_size).cuda() * now_task, scales)

    labels = one_hot_3D(labels.long())

    'semi part'
    affine_images1 = data_aug(images)
    affine_images2 = data_aug(images)

    _, features_map1 = model(affine_images1.cuda(), torch.ones(batch_size).cuda() * now_task, scales)
    _, features_map2 = model(affine_images2.cuda(), torch.ones(batch_size).cuda() * now_task, scales)

    norm_features_map1 = (features_map1 - torch.min(features_map1, dim = 1, keepdim = True)[0] * torch.ones((features_map1.shape)).cuda()) / ((torch.max(features_map1, dim = 1, keepdim = True)[0] - torch.min(features_map1, dim = 1, keepdim = True)[0]) * torch.ones((features_map1.shape)).cuda())
    norm_features_map2 = (features_map2 - torch.min(features_map2, dim = 1, keepdim = True)[0] * torch.ones((features_map2.shape)).cuda()) / ((torch.max(features_map2, dim = 1, keepdim = True)[0] - torch.min(features_map2, dim = 1, keepdim = True)[0]) * torch.ones((features_map2.shape)).cuda())

    term_KL = loss_KL(norm_features_map1, norm_features_map2)

    term_MSE = loss_MSE(features_map1 + 0.001, features_map2 + 0.001)
    term_all_semi = term_KL + term_MSE

    non_zero_preds = torch.zeros((preds.shape)).cuda()
    non_zero_labels = torch.zeros((labels.shape)).cuda()
    non_zero_weight = torch.ones((weight.shape)).cuda()

    cnt = 0
    for ki in range(batch_size):
        if labels[ki][1].sum() != 0.:
            non_zero_labels[cnt] = labels[ki]
            non_zero_preds[cnt] = preds[ki]
            non_zero_weight[cnt] = weight[ki]
            cnt += 1


    if cnt > 0:
        #print(non_zero_weight[:cnt].max(), non_zero_weight[:cnt].min())
        term_seg_Dice, term_seg_BCE, term_all_psuedo = get_loss(non_zero_preds[:cnt], non_zero_labels[:cnt], non_zero_weight[:cnt], loss_seg_DICE, loss_seg_CE)
        return term_seg_Dice, term_seg_BCE, term_all_psuedo, term_KL, term_MSE, term_all_semi
    else:
        return 0, 0, 0, 0, 0, 0

def semi_learning(images, batch_size, scales, model, now_task, loss_KL, loss_MSE, data_aug, data_return):

    affine_images1 = data_aug(images)
    affine_images2 = data_aug(images)

    _, features_map1 = model(affine_images1.cuda(), torch.ones(batch_size).cuda() * now_task, scales)
    _, features_map2 = model(affine_images2.cuda(), torch.ones(batch_size).cuda() * now_task, scales)

    softmax = nn.Softmax()
    term_KL = loss_KL(softmax(features_map1), softmax(features_map2))

    term_MSE = loss_MSE(features_map1, features_map2)
    term_all = term_KL + term_MSE

    return term_KL, term_MSE, term_all


def data_aug(images):

    'Tensor version data augmentation'

    image_size = 256
    imagenet_norm = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]

    self_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0),
                                     interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=image_size // 20 * 2 + 1, sigma=(0.1, 2.0))],
                               p=0.1),
        #transforms.RandomApply([Solarization()], p=0.2),
        transforms.Normalize(*imagenet_norm)
    ])

    aug_images = self_transform(images)

    return aug_images.cuda()


def kornia_aug(images, label):
    affine_transform = kornia.augmentation.AugmentationSequential(
        kornia.augmentation.ColorJitter(0.2, 0.3, 0.2, 0.3),
        kornia.augmentation.RandomAffine(p=0.75, degrees=(-180, 180), translate=(0.2, 0.2), shear=(-16, 16),
                                         scale=(0.75, 1.5)),
        kornia.augmentation.RandomHorizontalFlip(return_transform=True, p=0.5),
        kornia.augmentation.RandomVerticalFlip(return_transform=True, p=0.5),
        return_transform=False,
        same_on_batch=False,
        data_keys=["input", "mask"]
    )

    augmentation = affine_transform(images, label)
    return augmentation[0], augmentation[1]


def data_return(images, invert_matrix):

    return_images = kornia.geometry.transform.affine(images, invert_matrix)

    return return_images

def MappingLocation(big_image, now_patch):
    if big_image.max() < 10:
        big_image = (255 * big_image).astype(np.uint8)
    big_img_grey = cv2.cvtColor(big_image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255# rgb2gray(big_image) / 255
    patch_grey = cv2.cvtColor(now_patch, cv2.COLOR_RGB2GRAY)#rgb2gray(now_patch)
    w,h = patch_grey.shape
    res = cv2.matchTemplate(big_img_grey.copy(), patch_grey.copy(), cv2.TM_CCOEFF_NORMED)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    if max_val >0.5:
        top_left = max_loc
        return int(top_left[1]), int(top_left[1] + w), int(top_left[0]), int(top_left[0] + h)
    else:
        return 0, 0, 0, 0


def create_batch_10X(imgs, lbls, wts):
    img_size = 256
    img, lbl, wt = imgs[0], lbls[0], wts[0]
    resize_function = transforms.Resize(img_size * 2)
    img_resize = resize_function(img)
    lbl_resize = resize_function(lbl.unsqueeze(0)).squeeze(0)
    wt_resize = resize_function(wt.unsqueeze(0)).squeeze(0)

    img_batch = torch.zeros((4, 3, 256, 256))
    lbl_batch = torch.zeros((4, 256, 256))
    wt_batch = torch.zeros((4, 256, 256))

    cnt = 0
    for ki in range(2):
        for kj in range(2):
            start_x = ki * img_size
            start_y = kj * img_size
            end_x = (ki + 1) * img_size
            end_y = (kj + 1) * img_size

            img_batch[cnt] = img_resize[:, start_x:end_x , start_y:end_y]
            lbl_batch[cnt] = lbl_resize[start_x:end_x , start_y:end_y]
            wt_batch[cnt] = wt_resize[start_x:end_x , start_y:end_y]
            cnt += 1

    s_id_batch = torch.ones((4)) * 1

    return img_batch.cuda(), lbl_batch.cuda(), wt_batch.cuda(), s_id_batch

def create_batch_5X(imgs, lbls, wts):
    img_size = 256
    # img, lbl, wt = imgs[0], lbls[0], wts[0]
    resize_function = transforms.Resize(img_size)
    img_resize = resize_function(imgs)
    lbl_resize = resize_function(lbls)
    wt_resize = resize_function(wts)

    s_id_batch = torch.ones((4)) * 0

    return img_resize.cuda(), lbl_resize.cuda(), wt_resize.cuda(), s_id_batch.cuda()


def create_batch_40X(imgs, lbls, wts):
    img_size = 256
    img, lbl, wt = imgs[0], lbls[0], wts[0]
    resize_function = transforms.Resize(img_size * 8)
    img_resize = img
    lbl_resize = lbl
    wt_resize = wt

    img_batch = torch.zeros((64, 3, 256, 256))
    lbl_batch = torch.zeros((64, 256, 256))
    wt_batch = torch.zeros((64, 256, 256))

    cnt = 0
    for ki in range(8):
        for kj in range(8):
            start_x = ki * img_size
            start_y = kj * img_size
            end_x = (ki + 1) * img_size
            end_y = (kj + 1) * img_size

            img_batch[cnt] = img_resize[:, start_x:end_x, start_y:end_y]
            lbl_batch[cnt] = lbl_resize[start_x:end_x, start_y:end_y]
            wt_batch[cnt] = wt_resize[start_x:end_x, start_y:end_y]
            cnt += 1

    s_id_batch = torch.ones((64)) * 3

    return img_batch.cuda(), lbl_batch.cuda(), wt_batch.cuda(), s_id_batch.cuda()


def psuedo_label_40X_Matching(images, labels, batch_size, scales, model, now_task, scale):
    big_size = 2048

    preds, _ = model(images, torch.ones(batch_size).cuda() * now_task, scales)

    psuedo_preds = preds[:, 1] > preds[:, 0]

    'get dataset patches'
    big_img_name = now_name.replace('-', '/')
    big_img_change = big_img_name.replace('6_1_dt', '0_1_dt').replace('7_1_pt', '1_1_pt').replace('8_1_vessel',
                                                                                                  '4_1_vessel').replace(
        '9_3_ptc', '5_3_ptc')
    patches_list = glob.glob(os.path.join(os.path.dirname(big_img_change),
                                          os.path.basename(big_img_change).replace('.png', '_*.png').replace('.tif',
                                                                                                             '_*.png')))
    patches_list_img = [x for x in patches_list if not 'mask' in x]
    random.shuffle(patches_list_img)

    if scale == 0:
        resize_function = transforms.Resize(big_size)
        big_image = resize_function(images.permute([0, 2, 3, 1]))[0].float()
        big_psuedo = resize_function(psuedo_preds)[0].float()
        big_label = resize_function(labels)[0]
        big_psuedo = torch.maximum((big_psuedo - big_label), torch.zeros((big_psuedo.shape)).cuda())

        big_psuedo_array = big_psuedo.cpu().numpy().astype(int)
        big_weight = torch.from_numpy(scipy.ndimage.morphology.binary_dilation(big_psuedo_array == 1, iterations=2) & ~ big_psuedo_array)

        img_size = 256

        image_batch = torch.zeros((16, 3, img_size, img_size)).cuda()
        psuedo_batch = torch.zeros((16, img_size, img_size))
        weight_batch = torch.ones((16, img_size, img_size))

        cnt = 0
        for ki in range(len(patches_list_img)):
            if cnt == 16:
                break
            now_patch_name = patches_list_img[ki]
            now_patch_label_name = glob.glob(os.path.join(os.path.dirname(now_patch_name),
                                                          os.path.basename(now_patch_name).replace('.png',
                                                                                                   '_mask_*.png').replace(
                                                              '.tif', '_mask_*.png')))[0]

            now_patch = plt.imread(now_patch_name)[:, :, :3]
            now_patch_label = plt.imread(now_patch_label_name)[:, :, :3]

            rmin, rmax, cmin, cmax = MappingLocation(big_image.detach().cpu().numpy(), now_patch)
            
            if rmin + rmax + cmin + cmax > 0:
                # now_pseudo_image = big_image[rmin:rmax, cmin:cmax, :]
                now_pseduo_label = big_psuedo_array[rmin:rmax, cmin:cmax]
                now_pseduo_weight = big_weight[rmin:rmax, cmin:cmax]
    
                now_pseduo_label[now_pseduo_label > 0.5] == 1.
                now_pseduo_label[now_pseduo_label < 0.5] == 0.
    
                assert (now_pseduo_label * now_patch_label[:, :, 0] == 0)
    
                image_batch[cnt] = torch.from_numpy(now_patch).permute([2, 0, 1])
                psuedo_batch[cnt] = torch.from_numpy(now_pseduo_label)
                # weight_batch[cnt] = torch.from_numpy(now_pseduo_weight)
                cnt += 1
            else:
                continue

    else:
        resize_function = transforms.Resize(1024)
        psuedo_preds_resize = resize_function(psuedo_preds)
        big_image_resize = resize_function(images).permute([0,2, 3, 1]).float()
        labels_resize = resize_function(labels)

        big_image = torch.zeros((big_size, big_size, 3))
        big_psuedo = torch.zeros((big_size, big_size))
        big_label = torch.zeros((big_size, big_size))

        img_size = 1024

        cnt = 0
        for ki in range(2):
            for kj in range(2):
                start_x = ki * img_size
                start_y = kj * img_size
                end_x = (ki + 1) * img_size
                end_y = (kj + 1) * img_size

                big_image[start_x:end_x, start_y:end_y, :] = big_image_resize[cnt]
                big_psuedo[start_x:end_x, start_y:end_y] = psuedo_preds_resize[cnt]
                big_label[start_x:end_x, start_y:end_y] = labels_resize[cnt]
                cnt += 1

        big_psuedo = torch.maximum((big_psuedo - big_label), torch.zeros((big_psuedo.shape)))
        big_psuedo_array = big_psuedo.cpu().numpy().astype(int)
        big_weight = torch.from_numpy(scipy.ndimage.morphology.binary_dilation(big_psuedo_array == 1, iterations=2) & ~ big_psuedo_array)

        img_size = 256
        image_batch = torch.zeros((16, 3, img_size, img_size)).cuda()
        psuedo_batch = torch.zeros((16, img_size, img_size)).cuda()
        weight_batch = torch.zeros((16, img_size, img_size)).cuda()

        cnt = 0
        for ki in range(len(patches_list_img)):
            if cnt == 16:
                break
            now_patch_name = patches_list_img[ki]
            now_patch_label_name = glob.glob(os.path.join(os.path.dirname(now_patch_name),
                                                          os.path.basename(now_patch_name).replace('.png',
                                                                                                   '_mask_*.png').replace(
                                                              '.tif', '_mask_*.png')))[0]

            now_patch = plt.imread(now_patch_name)[:, :, :3]
            now_patch_label = plt.imread(now_patch_label_name)[:, :, :3]

            rmin, rmax, cmin, cmax = MappingLocation(big_image.detach().cpu().numpy(), now_patch)
            if rmin + rmax + cmin + cmax > 0:
                # now_pseudo_image = big_image[rmin:rmax, cmin:cmax, :]
                now_pseduo_label = big_psuedo_array[rmin:rmax, cmin:cmax]
                now_pseduo_weight = big_weight[rmin:rmax, cmin:cmax]
    
                now_pseduo_label[now_pseduo_label > 0.5] == 1.
                now_pseduo_label[now_pseduo_label < 0.5] == 0.
    
                assert (now_pseduo_label * now_patch_label[:, :, 0] == 0)
    
                image_batch[cnt] = torch.from_numpy(now_patch).permute([2, 0, 1])
                psuedo_batch[cnt] = torch.from_numpy(now_pseduo_label)
                cnt += 1
            else:
                continue

    # 'augmentation'
    # image_batch, psuedo_batch = kornia_aug(image_batch, psuedo_batch)

    return psuedo_batch.detach().cuda(), weight_batch.detach().cuda()


def psuedo_label_Matching_first(imgs, big_label, now_name, batch_size, scales, model, now_task, scale):
    transform_5X = transforms.Resize(375)
    transform_10X = transforms.Resize(750)
    patch_size = 256

    big_image = imgs[0].permute([1,2,0]).cpu().numpy()

    'get dataset patches'
    big_img_name = now_name.replace('-', '/')
    big_img_change = big_img_name.replace('6_1_dt', '0_1_dt').replace('7_1_pt', '1_1_pt').replace('8_1_vessel',
                                                                                                  '4_1_vessel').replace(
        '9_3_ptc', '5_3_ptc')
    patches_list = glob.glob(os.path.join(os.path.dirname(big_img_change),
                                          os.path.basename(big_img_change).replace('.png', '_*.png').replace('.tif',
                                                                                                             '_*.png')))
    patches_list_img = [x for x in patches_list if not 'mask' in x]
    random.shuffle(patches_list_img)

    if not '5_3_ptc' in big_img_change:
        resize_flag = 1
    else:
        resize_flag = 0

    if scale == 0:
        imgs_5X = transform_5X(imgs)
        big_psuedo, confidence = testing_5X(imgs_5X, now_task, scales, now_name, patch_size, batch_size, model)
    
    else:
        imgs_10X = transform_10X(imgs)
        big_psuedo, confidence = testing_10X(imgs_10X, now_task, scales, now_name, patch_size, batch_size, model)

    big_psuedo_array = big_psuedo

    img_size = 256

    if resize_flag == 1:
        image_batch = torch.zeros((4, 3, img_size, img_size)).cuda()
        psuedo_batch = torch.zeros((4, img_size, img_size)).cuda()
        weight_batch = torch.ones((4, img_size, img_size)).cuda()
        label_batch = torch.zeros((4, img_size, img_size)).cuda()
        loc_cnt = np.zeros((4,4))
    
        cnt = 0
        for ki in range(len(patches_list_img)):
            if cnt == 4:
                break
            now_patch_name = patches_list_img[ki]
            now_patch_label_name = glob.glob(os.path.join(os.path.dirname(now_patch_name),
                                                      os.path.basename(now_patch_name).replace('.png',
                                                                                            '_mask_*.png').replace(
                                                          '.tif', '_mask_*.png')))[0]
    
            now_patch = plt.imread(now_patch_name)[:,:,:3]
            now_patch_label = plt.imread(now_patch_label_name)[:, :, :3]

    
            now_patch_ori = now_patch.copy()
            now_patch = resize(now_patch, (1024, 1024, 3), anti_aliasing=False)

            rmin, rmax, cmin, cmax = MappingLocation(big_image, now_patch)
    
            if rmin + rmax + cmin + cmax > 0:
                loc_cnt[cnt, 0] = rmin
                loc_cnt[cnt, 1] = rmax
                loc_cnt[cnt, 2] = cmin
                loc_cnt[cnt, 3] = cmax
    
                now_pseduo_label = big_psuedo_array[rmin:rmax, cmin:cmax]
                now_confidence = confidence[rmin:rmax, cmin:cmax]

                now_pseduo_label = resize(now_pseduo_label, (256, 256), anti_aliasing=False)
                now_weight = resize(now_confidence, (256, 256), anti_aliasing=False) + now_patch_label[:,:,0]
                now_weight[now_weight > 1.] = 1.


                now_pseduo_label = np.maximum(now_pseduo_label - now_patch_label[:,:,0],
                                              np.zeros(now_patch_label[:,:,0].shape))
    
                now_pseduo_label[now_pseduo_label > 0.5] == 1.
                now_pseduo_label[now_pseduo_label < 0.5] == 0.


                assert ((now_pseduo_label * now_patch_label[:,:,0]).sum() == 0)
    
                image_batch[cnt] = torch.from_numpy(now_patch_ori).permute([2,0,1])
                psuedo_batch[cnt] = torch.from_numpy(now_pseduo_label)
                weight_batch[cnt] = torch.from_numpy(now_weight)
                label_batch[cnt] = torch.from_numpy(now_patch_label[:, :, 0])
                cnt += 1
            else:
                continue

    else:
        image_batch = torch.zeros((16, 3, img_size, img_size)).cuda()
        psuedo_batch = torch.zeros((16, img_size, img_size)).cuda()
        weight_batch = torch.ones((16, img_size, img_size)).cuda()
        label_batch = torch.zeros((16, img_size, img_size)).cuda()
        loc_cnt = np.zeros((16, 4))

        cnt = 0
        for ki in range(len(patches_list_img)):
            if cnt == 16:
                break
            now_patch_name = patches_list_img[ki]
            now_patch_label_name = glob.glob(os.path.join(os.path.dirname(now_patch_name),
                                                          os.path.basename(now_patch_name).replace('.png',
                                                                                                   '_mask_*.png').replace(
                                                              '.tif', '_mask_*.png')))[0]

            now_patch = plt.imread(now_patch_name)[:, :, :3]
            now_patch_label = plt.imread(now_patch_label_name)[:, :, :3]

            now_patch_ori = now_patch.copy()
            rmin, rmax, cmin, cmax = MappingLocation(big_image, now_patch)

            if rmin + rmax + cmin + cmax > 0:
                loc_cnt[cnt, 0] = rmin
                loc_cnt[cnt, 1] = rmax
                loc_cnt[cnt, 2] = cmin
                loc_cnt[cnt, 3] = cmax
                now_pseduo_label = big_psuedo_array[rmin:rmax, cmin:cmax]
                now_confidence = confidence[rmin:rmax, cmin:cmax]

                now_pseduo_label = np.maximum(now_pseduo_label - now_patch_label[:,:,0],
                                              np.zeros(now_patch_label[:,:,0].shape))

                now_weight = now_confidence + now_patch_label[:,:,0]
                now_weight[now_weight > 1.] = 1.

                now_pseduo_label[now_pseduo_label > 0.5] == 1.
                now_pseduo_label[now_pseduo_label < 0.5] == 0.
            
                assert ((now_pseduo_label * now_patch_label[:, :, 0]).sum() == 0)

                image_batch[cnt] = torch.from_numpy(now_patch_ori).permute([2, 0, 1])
                psuedo_batch[cnt] = torch.from_numpy(now_pseduo_label)
                weight_batch[cnt] = torch.from_numpy(now_weight)
                label_batch[cnt] = torch.from_numpy(now_patch_label[:,:,0])
                cnt += 1
            else:
                continue

    return image_batch.detach().cuda(), psuedo_batch.detach().cuda(), weight_batch.detach().cuda(), label_batch.detach().cuda(), loc_cnt


def psuedo_label_Matching_second(imgs, big_label, now_name, batch_size, scales, model, now_task, scale, label_batch, loc_cnt):
    transform_5X = transforms.Resize(375)
    transform_10X = transforms.Resize(750)
    patch_size = 256

    big_image = imgs[0].permute([1, 2, 0]).cpu().numpy()

    'get dataset patches'
    big_img_name = now_name.replace('-', '/')
    big_img_change = big_img_name.replace('6_1_dt', '0_1_dt').replace('7_1_pt', '1_1_pt').replace('8_1_vessel',
                                                                                                  '4_1_vessel').replace(
        '9_3_ptc', '5_3_ptc')


    if not '5_3_ptc' in big_img_change:
        resize_flag = 1
    else:
        resize_flag = 0

    if scale == 0:
        imgs_5X = transform_5X(imgs)
        big_psuedo, confidence = testing_5X(imgs_5X, now_task, scales, now_name, patch_size, batch_size, model)

    else:
        imgs_10X = transform_10X(imgs)
        big_psuedo, confidence = testing_10X(imgs_10X, now_task, scales, now_name, patch_size, batch_size, model)

    big_psuedo_array = big_psuedo

    img_size = 256

    if resize_flag == 1:
        image_batch = torch.zeros((4, 3, img_size, img_size)).cuda()
        psuedo_batch = torch.zeros((4, img_size, img_size)).cuda()
        weight_batch = torch.ones((4, img_size, img_size)).cuda()

        cnt = 0
        for ki in range(len(loc_cnt)):
            rmin, rmax, cmin, cmax = int(loc_cnt[ki, 0]), int(loc_cnt[ki, 1]), int(loc_cnt[ki, 2]), int(loc_cnt[ki, 3])

            if rmin + rmax + cmin + cmax > 0:
                now_patch_label = label_batch[ki].cpu().numpy()
                now_pseduo_label = big_psuedo_array[rmin:rmax, cmin:cmax]

                now_confidence = confidence[rmin:rmax, cmin:cmax]

                now_pseduo_label = resize(now_pseduo_label, (256, 256), anti_aliasing=False)
                now_weight = resize(now_confidence, (256, 256), anti_aliasing=False) + now_patch_label
                now_weight[now_weight > 1.] = 1.

                now_pseduo_label = np.maximum(now_pseduo_label - now_patch_label,
                                              np.zeros(now_patch_label.shape))

                now_pseduo_label[now_pseduo_label > 0.5] == 1.
                now_pseduo_label[now_pseduo_label < 0.5] == 0.

                assert ((now_pseduo_label * now_patch_label).sum() == 0)
                psuedo_batch[cnt] = torch.from_numpy(now_pseduo_label)
                weight_batch[cnt] = torch.from_numpy(now_weight)
                cnt += 1
            else:
                continue

    else:
        image_batch = torch.zeros((16, 3, img_size, img_size)).cuda()
        psuedo_batch = torch.zeros((16, img_size, img_size)).cuda()
        weight_batch = torch.ones((16, img_size, img_size)).cuda()

        cnt = 0
        for ki in range(len(loc_cnt)):
            rmin, rmax, cmin, cmax = int(loc_cnt[ki, 0]), int(loc_cnt[ki, 1]), int(loc_cnt[ki, 2]), int(loc_cnt[ki, 3])

            if rmin + rmax + cmin + cmax > 0:
                now_pseduo_label = big_psuedo_array[rmin:rmax, cmin:cmax]
                now_patch_label = label_batch[ki].cpu().numpy()
                now_pseduo_label = np.maximum(now_pseduo_label - now_patch_label,
                                              np.zeros(now_patch_label.shape))

                now_confidence = confidence[rmin:rmax, cmin:cmax]
                now_weight = now_confidence + now_patch_label
                now_weight[now_weight > 1.] = 1.

                now_pseduo_label[now_pseduo_label > 0.5] == 1.
                now_pseduo_label[now_pseduo_label < 0.5] == 0.

                assert ((now_pseduo_label * now_patch_label).sum() == 0)
                psuedo_batch[cnt] = torch.from_numpy(now_pseduo_label)
                weight_batch[cnt] = torch.from_numpy(now_weight)
                cnt += 1
            else:
                continue

    return image_batch.detach().cuda(), psuedo_batch.detach().cuda(), weight_batch.detach().cuda()

def testing_10X(imgs_10X, now_task, now_scale, volumeName, patch_size, batch_size, model):
    batch1 = torch.zeros([batch_size, 3, patch_size, patch_size])
    batch2 = torch.zeros([batch_size, 3, patch_size, patch_size])
    batch3 = torch.zeros([batch_size, 3, patch_size, patch_size])

    batch1[0] = imgs_10X[0,:,patch_size * 0:patch_size * 1, patch_size * 0:patch_size * 1]
    batch1[1] = imgs_10X[0, :, patch_size * 1:patch_size * 2, patch_size * 0:patch_size * 1]
    batch1[2] = imgs_10X[0, :, -patch_size:, patch_size * 0:patch_size * 1]
    batch1[3] = imgs_10X[0,:,patch_size * 0:patch_size * 1, patch_size * 1:patch_size * 2]

    batch2[0] = imgs_10X[0, :, patch_size * 1:patch_size * 2, patch_size * 1:patch_size * 2]
    batch2[1] = imgs_10X[0, :, -patch_size:, patch_size * 1:patch_size * 2]
    batch2[2] = imgs_10X[0, :,patch_size * 0:patch_size * 1, -patch_size:]
    batch2[3] = imgs_10X[0, :, patch_size * 1:patch_size * 2, -patch_size:]

    batch3[0] = imgs_10X[0, :, -patch_size:, -patch_size:]
    batch3[1] = imgs_10X[0,:,patch_size * 0:patch_size * 1, patch_size * 0:patch_size * 1]
    batch3[2] = imgs_10X[0, :, patch_size * 1:patch_size * 2, patch_size * 0:patch_size * 1]
    batch3[3] = imgs_10X[0, :, -patch_size:, patch_size * 0:patch_size * 1]

    preds_batch1,_ = model(batch1.cuda(), torch.ones(batch_size).cuda() * now_task, torch.ones(batch_size).cuda() * now_scale)
    preds_batch2,_ = model(batch2.cuda(), torch.ones(batch_size).cuda() * now_task, torch.ones(batch_size).cuda() * now_scale)
    preds_batch3,_ = model(batch3.cuda(), torch.ones(batch_size).cuda() * now_task, torch.ones(batch_size).cuda() * now_scale)

    big_preds = torch.zeros(2,imgs_10X.shape[2],imgs_10X.shape[3]).cuda()

    big_preds[:, patch_size * 0:patch_size * 1, patch_size * 0:patch_size * 1] = big_preds[:, patch_size * 0:patch_size * 1, patch_size * 0:patch_size * 1] + preds_batch1[0]
    big_preds[:, patch_size * 1:patch_size * 2, patch_size * 0:patch_size * 1] = big_preds[:, patch_size * 1:patch_size * 2, patch_size * 0:patch_size * 1] + preds_batch1[1]
    big_preds[:, -patch_size:, patch_size * 0:patch_size * 1] = big_preds[:, -patch_size:, patch_size * 0:patch_size * 1] + preds_batch1[2]
    big_preds[:, patch_size * 0:patch_size * 1, patch_size * 1:patch_size * 2] = big_preds[:, patch_size * 0:patch_size * 1, patch_size * 1:patch_size * 2] + preds_batch1[3]
    big_preds[:, patch_size * 1:patch_size * 2, patch_size * 1:patch_size * 2] = big_preds[:, patch_size * 1:patch_size * 2, patch_size * 1:patch_size * 2] + preds_batch2[0]
    big_preds[:, -patch_size:, patch_size * 1:patch_size * 2] = big_preds[:, -patch_size:, patch_size * 1:patch_size * 2] + preds_batch2[1]
    big_preds[:,patch_size * 0:patch_size * 1, -patch_size:] = big_preds[:,patch_size * 0:patch_size * 1, -patch_size:] + preds_batch2[2]
    big_preds[:, patch_size * 1:patch_size * 2, -patch_size:] = big_preds[:, patch_size * 1:patch_size * 2, -patch_size:] + preds_batch2[3]
    big_preds[:, -patch_size:, -patch_size:] = big_preds[:, -patch_size:, -patch_size:] + preds_batch3[0]

    resize_function = transforms.Resize(3000)
    big_preds_resize = resize_function(big_preds)
    # big_img_resize = resize_function(big_img)

    prediction = (big_preds_resize[1, ...] > big_preds_resize[0, ...]).detach().cpu().numpy().astype(np.float32)
    softmax = nn.Softmax(0)
    confidence = torch.abs(softmax(big_preds_resize) - 0.5) * 2

    return prediction, confidence[1].cpu().numpy()

def testing_5X(imgs_5X, now_task, now_scale, volumeName, patch_size, batch_size, model):
    batch = torch.zeros([batch_size, 3, patch_size, patch_size])

    batch[0] = imgs_5X[0, :, patch_size * 0:patch_size * 1, patch_size * 0:patch_size * 1]
    batch[1] = imgs_5X[0, :, -patch_size:, patch_size * 0:patch_size * 1]
    batch[2] = imgs_5X[0, :, patch_size * 0:patch_size * 1, -patch_size:]
    batch[3] = imgs_5X[0, :, -patch_size:, -patch_size:]

    preds_batch,_ = model(batch.cuda(), torch.ones(batch_size).cuda() * now_task, torch.ones(batch_size).cuda() * now_scale)

    big_preds = torch.zeros(2,imgs_5X.shape[2],imgs_5X.shape[3]).cuda()

    big_preds[:,patch_size * 0:patch_size * 1, patch_size * 0:patch_size * 1] = big_preds[:,patch_size * 0:patch_size * 1, patch_size * 0:patch_size * 1] + preds_batch[0]
    big_preds[:, -patch_size:, patch_size * 0:patch_size * 1] = big_preds[:, -patch_size:, patch_size * 0:patch_size * 1] + preds_batch[1]
    big_preds[:, patch_size * 0:patch_size * 1, -patch_size:] = big_preds[:, patch_size * 0:patch_size * 1, -patch_size:] + preds_batch[2]
    big_preds[:, -patch_size:, -patch_size:] = big_preds[:, -patch_size:, -patch_size:] + preds_batch[3]

    resize_function = transforms.Resize(3000)
    big_preds_resize = resize_function(big_preds)
    # big_img_resize = resize_function(big_img)

    prediction = (big_preds_resize[1, ...] > big_preds_resize[0, ...]).detach().cpu().numpy().astype(np.float32)
    softmax = nn.Softmax(0)
    confidence = torch.abs(softmax(big_preds_resize) - 0.5) * 2

    return prediction, confidence[1].cpu().numpy()

def main():
    """Create the model and start the training."""
    parser = get_arguments()
    print(parser)

    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()
        if args.num_gpus > 1:
            torch.cuda.set_device(args.local_rank)

        writer = SummaryWriter(args.snapshot_dir)

        if not args.gpu == 'None':
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        h, w = map(int, args.input_size.split(','))
        input_size = (h, w)

        cudnn.benchmark = True
        seed = args.random_seed
        if engine.distributed:
            seed = args.local_rank
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Create model
        criterion = None
        model = UNet2D_ns(num_classes=args.num_classes, weight_std = False)

        check_wo_gpu = 0

        if not check_wo_gpu:
            device = torch.device('cuda:{}'.format(args.local_rank))
            model.to(device)

        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.99, nesterov=True)

        if not check_wo_gpu:
            if args.FP16:
                print("Note: Using FP16 during training************")
                model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

            if args.num_gpus > 1:
                model = engine.data_parallel(model)

        # load checkpoint...a
        if args.reload_from_checkpoint:
            print('loading from checkpoint: {}'.format(args.reload_path))
            if os.path.exists(args.reload_path):
                if args.FP16:
                    checkpoint = torch.load(args.reload_path, map_location=torch.device('cpu'))
                    model.load_state_dict(checkpoint['model'])
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    amp.load_state_dict(checkpoint['amp'])
                else:
                    model.load_state_dict(torch.load(args.reload_path, map_location=torch.device('cpu')))
            else:
                print('File not exists in the reload path: {}'.format(args.reload_path))

        if not check_wo_gpu:
            weights1 = [1., 1.]
            #class_weights = torch.FloatTensor(weights).cuda()
            loss_seg_DICE = loss.DiceLoss4MOTS(num_classes=args.num_classes).to(device)
            loss_seg_CE_1 = loss.CELoss4MOTS(weight = weights1, num_classes=args.num_classes, ignore_index=255).to(device)
            weights2 = [1., 1.]
            loss_seg_CE_10 = loss.CELoss4MOTS(weight = weights2, num_classes=args.num_classes, ignore_index=255).to(device)
            loss_KL = nn.KLDivLoss().to(device)
            loss_MSE = nn.MSELoss().to(device)


        else:
            weights1 = [1., 1.]
            #class_weights = torch.FloatTensor(weights)
            loss_seg_DICE = loss.DiceLoss4MOTS(num_classes=args.num_classes)
            loss_seg_CE_1 = loss.CELoss4MOTS(weight = weights1, num_classes=args.num_classes, ignore_index=255)
            weights2 = [1., 1.]
            loss_seg_CE_10 = loss.CELoss4MOTS(weight = weights2, num_classes=args.num_classes, ignore_index=255)
            loss_KL = nn.KLDivLoss()
            loss_MSE = nn.MSELoss()


        if not os.path.exists(args.snapshot_dir):
            os.makedirs(args.snapshot_dir)

        edge_weight = args.edge_weight

        num_worker = 8

        trainloader = DataLoader(
            MOTSDataSet(args.trainset_dir, args.train_list, max_iters=args.itrs_each_epoch * args.batch_size,
                        crop_size=input_size, scale=args.random_scale, mirror=args.random_mirror,
                        edge_weight=edge_weight),batch_size=4,shuffle=True,num_workers=num_worker)

        valloader = DataLoader(
            MOTSValDataSet(args.valset_dir, args.val_list, max_iters=args.itrs_each_epoch * args.batch_size,
                           crop_size=input_size, scale=args.random_scale, mirror=args.random_mirror,
                           edge_weight=edge_weight),batch_size=4,shuffle=False,num_workers=num_worker)


        all_tr_loss_supervise = []
        df_loss = pd.DataFrame(columns=['epoch', 'supervise'])
        all_va_loss = []
        train_loss_MA = None
        val_loss_MA = None

        val_best_loss = 999999

        for epoch in range(50,args.num_epochs):
            model.train()

            task0_pool_image = ImagePool(8)
            task0_pool_mask = ImagePool(8)
            task0_pool_weight = ImagePool(8)
            task0_scale = []
            task1_pool_image = ImagePool(8)
            task1_pool_mask = ImagePool(8)
            task1_pool_weight = ImagePool(8)
            task1_scale = []
            task2_pool_image = ImagePool(8)
            task2_pool_mask = ImagePool(8)
            task2_pool_weight = ImagePool(8)
            task2_scale = []
            task3_pool_image = ImagePool(8)
            task3_pool_mask = ImagePool(8)
            task3_pool_weight = ImagePool(8)
            task3_scale = []
            task4_pool_image = ImagePool(8)
            task4_pool_mask = ImagePool(8)
            task4_pool_weight = ImagePool(8)
            task4_scale = []
            task5_pool_image = ImagePool(8)
            task5_pool_mask = ImagePool(8)
            task5_pool_weight = ImagePool(8)
            task5_scale = []
            #
            #
            #
            # task6_pool_image = ImagePool(8)
            # task6_pool_mask = ImagePool(8)
            # task6_pool_weight = ImagePool(8)
            # task6_scale = []
            # task6_name = []
            # task7_pool_image = ImagePool(8)
            # task7_pool_mask = ImagePool(8)
            # task7_pool_weight = ImagePool(8)
            # task7_scale = []
            # task7_name = []
            # task8_pool_image = ImagePool(8)
            # task8_pool_mask = ImagePool(8)
            # task8_pool_weight = ImagePool(8)
            # task8_scale = []
            # task8_name = []
            # task9_pool_image = ImagePool(8)
            # task9_pool_mask = ImagePool(8)
            # task9_pool_weight = ImagePool(8)
            # task9_scale = []
            # task9_name = []

            if epoch < args.start_epoch:
                continue

            if engine.distributed:
                train_sampler.set_epoch(epoch)

            epoch_loss_supervise = []
            epoch_loss_psuedo = []
            epoch_loss_semi = []
            adjust_learning_rate(optimizer, epoch, args.learning_rate, args.num_epochs, args.power)

            batch_size = args.batch_size
            task_num = 6
            each_loss = torch.zeros((task_num)).cuda()
            count_batch = torch.zeros((task_num)).cuda()
            loss_weight = torch.ones((task_num)).cuda()

            scaler = torch.cuda.amp.GradScaler()

            for iter, batch in enumerate(trainloader):

                'dataloader'
                imgs = batch[0].cuda()
                lbls = batch[1].cuda()
                wt = batch[2].cuda().float()
                volumeName = batch[3]
                t_ids = batch[4].cuda()
                s_ids = batch[5]

                sum_loss = 0

                for ki in range(len(imgs)):
                    now_task = t_ids[ki]
                    # if now_task <= 5:
                    #     continue

                    # if now_task > 5 and ((imgs.shape[2] != 3000) or (imgs.shape[3] != 3000)):
                    #     continue

                    if now_task == 0:
                        task0_pool_image.add(imgs[ki].unsqueeze(0))
                        task0_pool_mask.add(lbls[ki].unsqueeze(0))
                        task0_pool_weight.add(wt[ki].unsqueeze(0))
                        task0_scale.append((s_ids[ki]))
                    elif now_task == 1:
                        task1_pool_image.add(imgs[ki].unsqueeze(0))
                        task1_pool_mask.add(lbls[ki].unsqueeze(0))
                        task1_pool_weight.add(wt[ki].unsqueeze(0))
                        task1_scale.append((s_ids[ki]))
                    elif now_task == 2:
                        task2_pool_image.add(imgs[ki].unsqueeze(0))
                        task2_pool_mask.add(lbls[ki].unsqueeze(0))
                        task2_pool_weight.add(wt[ki].unsqueeze(0))
                        task2_scale.append((s_ids[ki]))
                    elif now_task == 3:
                        task3_pool_image.add(imgs[ki].unsqueeze(0))
                        task3_pool_mask.add(lbls[ki].unsqueeze(0))
                        task3_pool_weight.add(wt[ki].unsqueeze(0))
                        task3_scale.append((s_ids[ki]))
                    elif now_task == 4:
                        task4_pool_image.add(imgs[ki].unsqueeze(0))
                        task4_pool_mask.add(lbls[ki].unsqueeze(0))
                        task4_pool_weight.add(wt[ki].unsqueeze(0))
                        task4_scale.append((s_ids[ki]))
                    elif now_task == 5:
                        task5_pool_image.add(imgs[ki].unsqueeze(0))
                        task5_pool_mask.add(lbls[ki].unsqueeze(0))
                        task5_pool_weight.add(wt[ki].unsqueeze(0))
                        task5_scale.append((s_ids[ki]))
                        
                    #
                    # elif now_task == 6:
                    #     task6_pool_image.add(imgs[ki].unsqueeze(0))
                    #     task6_pool_mask.add(lbls[ki].unsqueeze(0))
                    #     task6_pool_weight.add(wt[ki].unsqueeze(0))
                    #     task6_scale.append((s_ids[ki]))
                    #     task6_name.append((volumeName[ki]))
                    # elif now_task == 7:
                    #     task7_pool_image.add(imgs[ki].unsqueeze(0))
                    #     task7_pool_mask.add(lbls[ki].unsqueeze(0))
                    #     task7_pool_weight.add(wt[ki].unsqueeze(0))
                    #     task7_scale.append((s_ids[ki]))
                    #     task7_name.append((volumeName[ki]))
                    # elif now_task == 8:
                    #     task8_pool_image.add(imgs[ki].unsqueeze(0))
                    #     task8_pool_mask.add(lbls[ki].unsqueeze(0))
                    #     task8_pool_weight.add(wt[ki].unsqueeze(0))
                    #     task8_scale.append((s_ids[ki]))
                    #     task8_name.append((volumeName[ki]))
                    # elif now_task == 9:
                    #     task9_pool_image.add(imgs[ki].unsqueeze(0))
                    #     task9_pool_mask.add(lbls[ki].unsqueeze(0))
                    #     task9_pool_weight.add(wt[ki].unsqueeze(0))
                    #     task9_scale.append((s_ids[ki]))
                    #     task9_name.append((volumeName[ki]))

                if task0_pool_image.num_imgs >= batch_size:
                    images = task0_pool_image.query(batch_size)
                    labels = task0_pool_mask.query(batch_size)
                    wts = task0_pool_weight.query(batch_size)
                    scales = torch.ones(batch_size).cuda()
                    for bi in range(len(scales)):
                        scales[bi] = task0_scale.pop(0)

                    now_task = 0
                    weight = 1 ** wts
                    
                    # if seed > semi_ratio:
                    
                    'supervise_learning'
                    term_seg_Dice, term_seg_BCE, Sup_term_all = supervise_learning(images, labels, batch_size, scales,
                                                                                   model, now_task, weight,
                                                                                   loss_seg_DICE, loss_seg_CE_1, loss_seg_CE_10, confident_flag, confident_e, simattention_flag, simattention_e, simloss_flag, simloss_e)

                    reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
                    reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
                    reduce_all = engine.all_reduce_tensor(Sup_term_all)





                    optimizer.zero_grad()
                    # reduce_all.backward()
                    # optimizer.step()

                    scaler.scale(reduce_all).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    print(
                        'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
                            epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
                            reduce_BCE.item(), reduce_all.item()))


                    term_all = reduce_all
                    #sum_loss += term_all
                    each_loss[now_task] += term_all
                    count_batch[now_task] += 1

                    epoch_loss_supervise.append(float(term_all))


                if task1_pool_image.num_imgs >= batch_size:
                    images = task1_pool_image.query(batch_size)
                    labels = task1_pool_mask.query(batch_size)
                    wts = task1_pool_weight.query(batch_size)
                    scales = torch.ones(batch_size).cuda()
                    for bi in range(len(scales)):
                        scales[bi] = task1_scale.pop(0)

                        
                    now_task = 1
                    weight = 1 ** wts

                    seed = np.random.rand(1)

                    'supervise_learning'
                    term_seg_Dice, term_seg_BCE, Sup_term_all = supervise_learning(images, labels, batch_size, scales,
                                                                                   model, now_task, weight,
                                                                                   loss_seg_DICE, loss_seg_CE_1, loss_seg_CE_10, confident_flag, confident_e, simattention_flag, simattention_e, simloss_flag, simloss_e)

                    reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
                    reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
                    reduce_all = engine.all_reduce_tensor(Sup_term_all)

                    optimizer.zero_grad()
                    # reduce_all.backward()
                    # optimizer.step()

                    scaler.scale(reduce_all).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    print(
                        'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
                            epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
                            reduce_BCE.item(), reduce_all.item()))


                    term_all = reduce_all

                    # sum_loss += term_all
                    each_loss[now_task] += term_all
                    count_batch[now_task] += 1

                    epoch_loss_supervise.append(float(term_all))

                if task2_pool_image.num_imgs >= batch_size:
                    images = task2_pool_image.query(batch_size)
                    labels = task2_pool_mask.query(batch_size)
                    wts = task2_pool_weight.query(batch_size)

                    scales = torch.ones(batch_size).cuda()
                    for bi in range(len(scales)):
                        scales[bi] = task2_scale.pop(0)

                    now_task = 2
                    weight = edge_weight ** wts

                    'supervise_learning'
                    term_seg_Dice, term_seg_BCE, Sup_term_all = supervise_learning(images, labels, batch_size, scales,
                                                                                   model, now_task, weight,
                                                                                   loss_seg_DICE, loss_seg_CE)

                    reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
                    reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
                    reduce_all = engine.all_reduce_tensor(Sup_term_all)

                    optimizer.zero_grad()
                    # reduce_all.backward()
                    # optimizer.step()

                    scaler.scale(reduce_all).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    print(
                        'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
                            epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
                            reduce_BCE.item(), reduce_all.item()))


                    term_all = reduce_all

                    # sum_loss += term_all
                    each_loss[now_task] += term_all
                    count_batch[now_task] += 1

                    epoch_loss_supervise.append(float(term_all))

                if task3_pool_image.num_imgs >= batch_size:
                    images = task3_pool_image.query(batch_size)
                    labels = task3_pool_mask.query(batch_size)
                    wts = task3_pool_weight.query(batch_size)
                    scales = torch.ones(batch_size).cuda()
                    for bi in range(len(scales)):
                        scales[bi] = task3_scale.pop(0)

                    now_task = 3
                    weight = edge_weight ** wts

                    'supervise_learning'
                    term_seg_Dice, term_seg_BCE, Sup_term_all = supervise_learning(images, labels, batch_size, scales,
                                                                                   model, now_task, weight,
                                                                                   loss_seg_DICE, loss_seg_CE)

                    reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
                    reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
                    reduce_all = engine.all_reduce_tensor(Sup_term_all)

                    optimizer.zero_grad()
                    # reduce_all.backward()
                    # optimizer.step()

                    scaler.scale(reduce_all).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    print(
                        'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
                            epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
                            reduce_BCE.item(), reduce_all.item()))


                    term_all = reduce_all

                    # sum_loss += term_all
                    each_loss[now_task] += term_all
                    count_batch[now_task] += 1

                    epoch_loss_supervise.append(float(term_all))

                if task4_pool_image.num_imgs >= batch_size:
                    images = task4_pool_image.query(batch_size)
                    labels = task4_pool_mask.query(batch_size)
                    wts = task4_pool_weight.query(batch_size)
                    scales = torch.ones(batch_size).cuda()
                    for bi in range(len(scales)):
                        scales[bi] = task4_scale.pop(0)

                    now_task = 4
                    weight = edge_weight ** wts

                    'supervise_learning'
                    term_seg_Dice, term_seg_BCE, Sup_term_all = supervise_learning(images, labels, batch_size, scales,
                                                                                   model, now_task, weight,
                                                                                   loss_seg_DICE, loss_seg_CE)

                    reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
                    reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
                    reduce_all = engine.all_reduce_tensor(Sup_term_all)

                    optimizer.zero_grad()
                    # reduce_all.backward()
                    # optimizer.step()

                    scaler.scale(reduce_all).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    print(
                        'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
                            epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
                            reduce_BCE.item(), reduce_all.item()))


                    term_all = reduce_all

                    # sum_loss += term_all
                    each_loss[now_task] += term_all
                    count_batch[now_task] += 1

                    epoch_loss_supervise.append(float(term_all))


                if task5_pool_image.num_imgs >= batch_size:
                    'only do supervised training'

                    images = task5_pool_image.query(batch_size)
                    labels = task5_pool_mask.query(batch_size)
                    wts = task5_pool_weight.query(batch_size)
                    scales = torch.ones(batch_size).cuda()
                    for bi in range(len(scales)):
                        scales[bi] = task5_scale.pop(0)

                    now_task = 5
                    weight = edge_weight ** wts

                    'supervise_learning'
                    term_seg_Dice, term_seg_BCE, Sup_term_all = supervise_learning(images, labels, batch_size, scales,
                                                                                   model, now_task, weight,
                                                                                   loss_seg_DICE, loss_seg_CE)

                    reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
                    reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
                    reduce_all = engine.all_reduce_tensor(Sup_term_all)

                    optimizer.zero_grad()
                    # reduce_all.backward()
                    # optimizer.step()

                    scaler.scale(reduce_all).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    print(
                        'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
                            epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
                            reduce_BCE.item(), reduce_all.item()))


                    term_all = reduce_all

                    # sum_loss += term_all
                    each_loss[now_task] += term_all
                    count_batch[now_task] += 1

                    epoch_loss_supervise.append(float(term_all))

            epoch_loss_supervise_mean = np.mean(epoch_loss_supervise)
            all_tr_loss_supervise.append(epoch_loss_supervise_mean)

            if (args.local_rank == 0):
                print('Epoch_sum {}: lr = {:.4}, loss_Sum_supervise = {:.4}'.format(epoch, optimizer.param_groups[0]['lr'],
                                                                          epoch_loss_supervise_mean.item()))
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
                writer.add_scalar('Train_loss_supervise', epoch_loss_supervise_mean.item(), epoch)
                plt.plot(all_tr_loss_supervise, label="Supervise")
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Loss')
                plt.legend()
                plt.savefig('Cellsegmentation_supervised_%s.png' % (os.path.basename(args.snapshot_dir)))
                plt.clf()

                row = len(df_loss)
                df_loss.loc[row] = [epoch, epoch_loss_supervise_mean]
                df_loss.to_csv('Cellsegmentation_supervised.csv')


            if (epoch >= 0) and (args.local_rank == 0) and (((epoch % 10 == 0) and (epoch >= 800)) or (epoch % 1 == 0)):
                print('save validation image ...')

                model.eval()
                # semi_pool_image = ImagePool(8 * 6)
                task0_pool_image = ImagePool(8)
                task0_pool_mask = ImagePool(8)
                task0_scale = []
                task1_pool_image = ImagePool(8)
                task1_pool_mask = ImagePool(8)
                task1_scale = []
                task2_pool_image = ImagePool(8)
                task2_pool_mask = ImagePool(8)
                task2_scale = []
                task3_pool_image = ImagePool(8)
                task3_pool_mask = ImagePool(8)
                task3_scale = []
                task4_pool_image = ImagePool(8)
                task4_pool_mask = ImagePool(8)
                task4_scale = []
                task5_pool_image = ImagePool(8)
                task5_pool_mask = ImagePool(8)
                task5_scale = []

                val_loss = np.zeros((6))
                val_F1 = np.zeros((6))
                val_Dice = np.zeros((6))
                val_TPR = np.zeros((6))
                val_PPV = np.zeros((6))
                cnt = np.zeros((6))

                # for iter, batch1, batch2 in enumerate(zip(valloaderloader, semi_valloaderloader)):
                with torch.no_grad():
                    for iter, batch1 in enumerate(valloader):

                        # if iter > 100:
                        #     break

                        'dataloader'
                        imgs = batch1[0].cuda()
                        lbls = batch1[1].cuda()
                        wt = batch1[2].cuda().float()
                        volumeName = batch1[3]
                        t_ids = batch1[4].cuda()
                        s_ids = batch1[5]


                        # semi_img = batch2[0]

                        for ki in range(len(imgs)):
                            now_task = t_ids[ki]

                            if now_task == 0:
                                task0_pool_image.add(imgs[ki].unsqueeze(0))
                                task0_pool_mask.add(lbls[ki].unsqueeze(0))
                                task0_scale.append((s_ids[ki]))
                            elif now_task == 1:
                                task1_pool_image.add(imgs[ki].unsqueeze(0))
                                task1_pool_mask.add(lbls[ki].unsqueeze(0))
                                task1_scale.append((s_ids[ki]))
                            elif now_task == 2:
                                task2_pool_image.add(imgs[ki].unsqueeze(0))
                                task2_pool_mask.add(lbls[ki].unsqueeze(0))
                                task2_scale.append((s_ids[ki]))
                            elif now_task == 3:
                                task3_pool_image.add(imgs[ki].unsqueeze(0))
                                task3_pool_mask.add(lbls[ki].unsqueeze(0))
                                task3_scale.append((s_ids[ki]))
                            elif now_task == 4:
                                task4_pool_image.add(imgs[ki].unsqueeze(0))
                                task4_pool_mask.add(lbls[ki].unsqueeze(0))
                                task4_scale.append((s_ids[ki]))
                            elif now_task == 5:
                                task5_pool_image.add(imgs[ki].unsqueeze(0))
                                task5_pool_mask.add(lbls[ki].unsqueeze(0))
                                task5_scale.append((s_ids[ki]))

                        output_folder = os.path.join(args.snapshot_dir.replace('snapshots_2D/','/Data3/HumanKidney/cell_segmentation/validation_'), str(epoch))
                        #output_folder = os.path.join('/Data/DoDNet/a_DynConv/validation_noscale_0829', str(epoch))
                        if not os.path.exists(output_folder):
                            os.makedirs(output_folder)
                        optimizer.zero_grad()

                        if task0_pool_image.num_imgs >= batch_size:
                            images = task0_pool_image.query(batch_size)
                            labels = task0_pool_mask.query(batch_size)
                            now_task = torch.tensor(0)
                            scales = torch.ones(batch_size).cuda()
                            for bi in range(len(scales)):
                                scales[bi] = task0_scale.pop(0)
                            preds, simscore, conscore = model(images, torch.ones(batch_size).cuda()*0, scales, labels)

                            #confident_flag, confident_e, simattention_flag, simattention_e, simloss_flag, simloss_econfident_flag, simattention_flag, simloss_flag

                            F1 = 0
                            DICE = 0
                            TPR = 0
                            PPV = 0

                            now_preds = preds[:,1,...] > preds[:,0,...]
                            now_preds_onehot = one_hot_3D(now_preds.long())

                            labels_onehot = one_hot_3D(labels.long())

                            rmin, rmax, cmin, cmax = mask_to_box(images)

                            cntt = 0
                            if confident_flag > 0:
                                if confident_e:
                                    now_F1, now_DICE, now_TPR, now_PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin,
                                                                 cmax, 1.5 ** (conscore * labels))
                                else:
                                    now_F1, now_DICE, now_TPR, now_PPV = count_score(now_preds_onehot, labels_onehot,
                                                                                     rmin, rmax, cmin,
                                                                                     cmax, conscore * labels)

                                F1 += confident_flag * now_F1
                                DICE += confident_flag * now_DICE
                                TPR += confident_flag * now_TPR
                                PPV += confident_flag * now_PPV


                                cntt += confident_flag

                            if simattention_flag > 0:
                                if simattention_e:
                                    now_F1, now_DICE, now_TPR, now_PPV = count_score(now_preds_onehot, labels_onehot,
                                                                                     rmin, rmax, cmin,
                                                                                     cmax, 1.5 ** (simscore * labels))
                                else:
                                    now_F1, now_DICE, now_TPR, now_PPV = count_score(now_preds_onehot, labels_onehot,
                                                                                     rmin, rmax, cmin,
                                                                                     cmax, simscore * labels)
                                F1 += simattention_flag * now_F1
                                DICE += simattention_flag * now_DICE
                                TPR += simattention_flag * now_TPR
                                PPV += simattention_flag * now_PPV
                                cntt += simattention_flag

                            if simloss_flag:

                                final_sim = simscore * labels

                                sim_score_onehot = torch.zeros((labels_onehot.shape)).cuda()
                                sim_score_onehot[:, 0] = 1 - final_sim
                                sim_score_onehot[:, 1] = final_sim

                                if simloss_e:
                                    now_F1, now_DICE, now_TPR, now_PPV = count_score(now_preds_onehot, sim_score_onehot, rmin, rmax, cmin, cmax, 1.5 ** final_sim)
                                else:
                                    now_F1, now_DICE, now_TPR, now_PPV = count_score(now_preds_onehot, sim_score_onehot,
                                                                                     rmin, rmax, cmin, cmax,final_sim)

                                F1 += simloss_flag * now_F1
                                DICE += simloss_flag * now_DICE
                                TPR += simloss_flag * now_TPR
                                PPV += simloss_flag * now_PPV

                                cntt += simloss_flag

                            val_F1[0] += F1 / cntt
                            val_Dice[0] += DICE / cntt
                            val_TPR[0] += TPR / cntt
                            val_PPV[0] += PPV / cntt
                            cnt[0] += 1

                            for pi in range(len(images)):
                                prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
                                num = len(glob.glob(os.path.join(output_folder, '*')))
                                out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                                img = (out_image - out_image.min())/ (out_image.max() - out_image.min())
                                plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
                                           img)
                                plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
                                           labels[pi, ...].detach().cpu().numpy())
                                plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' %(now_task.item())),
                                           prediction.detach().cpu().numpy())
                                plt.imsave(os.path.join(output_folder, str(num) + '_sim_%s.png' %(now_task.item())),
                                           simscore[pi, ...].detach().cpu().numpy())
                                plt.imsave(os.path.join(output_folder, str(num) + '_sim*label_%s.png' % (now_task.item())),
                                           simscore[pi, ...].detach().cpu().numpy() * labels[pi, ...].detach().cpu().numpy())
                                plt.imsave(os.path.join(output_folder, str(num) + '_con_%s.png' % (now_task.item())),
                                           conscore[pi, ...].detach().cpu().numpy())
                                plt.imsave(
	                                os.path.join(output_folder, str(num) + '_con*label_%s.png' % (now_task.item())),
	                                conscore[pi, ...].detach().cpu().numpy() * labels[pi, ...].detach().cpu().numpy())



                        if task1_pool_image.num_imgs >= batch_size:
                            images = task1_pool_image.query(batch_size)
                            labels = task1_pool_mask.query(batch_size)
                            scales = torch.ones(batch_size).cuda()
                            for bi in range(len(scales)):
                                scales[bi] = task1_scale.pop(0)
                            preds, simscore, conscore = model(images, torch.ones(batch_size).cuda()*1, scales, labels)

                            F1 = 0
                            DICE = 0
                            TPR = 0
                            PPV = 0

                            now_preds = preds[:, 1, ...] > preds[:, 0, ...]
                            now_preds_onehot = one_hot_3D(now_preds.long())

                            labels_onehot = one_hot_3D(labels.long())

                            rmin, rmax, cmin, cmax = mask_to_box(images)

                            cntt = 0
                            if confident_flag > 0:
                                if confident_e:
                                    now_F1, now_DICE, now_TPR, now_PPV = count_score(now_preds_onehot, labels_onehot,
                                                                                     rmin, rmax, cmin,
                                                                                     cmax, 1.5 ** (conscore * labels))
                                else:
                                    now_F1, now_DICE, now_TPR, now_PPV = count_score(now_preds_onehot, labels_onehot,
                                                                                     rmin, rmax, cmin,
                                                                                     cmax, conscore * labels)

                                F1 += confident_flag * now_F1
                                DICE += confident_flag * now_DICE
                                TPR += confident_flag * now_TPR
                                PPV += confident_flag * now_PPV

                                cntt += confident_flag

                            if simattention_flag > 0:
                                if simattention_e:
                                    now_F1, now_DICE, now_TPR, now_PPV = count_score(now_preds_onehot, labels_onehot,
                                                                                     rmin, rmax, cmin,
                                                                                     cmax, 1.5 ** (simscore * labels))
                                else:
                                    now_F1, now_DICE, now_TPR, now_PPV = count_score(now_preds_onehot, labels_onehot,
                                                                                     rmin, rmax, cmin,
                                                                                     cmax, simscore * labels)
                                F1 += simattention_flag * now_F1
                                DICE += simattention_flag * now_DICE
                                TPR += simattention_flag * now_TPR
                                PPV += simattention_flag * now_PPV
                                cntt += simattention_flag

                            if simloss_flag:

                                final_sim = simscore * labels

                                sim_score_onehot = torch.zeros((labels_onehot.shape)).cuda()
                                sim_score_onehot[:, 0] = 1 - final_sim
                                sim_score_onehot[:, 1] = final_sim

                                if simloss_e:
                                    now_F1, now_DICE, now_TPR, now_PPV = count_score(now_preds_onehot, sim_score_onehot,
                                                                                     rmin, rmax, cmin, cmax,
                                                                                     1.5 ** final_sim)
                                else:
                                    now_F1, now_DICE, now_TPR, now_PPV = count_score(now_preds_onehot, sim_score_onehot,
                                                                                     rmin, rmax, cmin, cmax, final_sim)

                                F1 += simloss_flag * now_F1
                                DICE += simloss_flag * now_DICE
                                TPR += simloss_flag * now_TPR
                                PPV += simloss_flag * now_PPV

                                cntt += simloss_flag

                            val_F1[1] += F1 / cntt
                            val_Dice[1] += DICE / cntt
                            val_TPR[1] += TPR / cntt
                            val_PPV[1] += PPV / cntt
                            cnt[1] += 1

                            for pi in range(len(images)):
                                prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
                                num = len(glob.glob(os.path.join(output_folder, '*')))
                                out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                                img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                                plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
                                           img)
                                plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
                                           labels[pi, ...].detach().cpu().numpy())
                                plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
                                           prediction.detach().cpu().numpy())
                                plt.imsave(os.path.join(output_folder, str(num) + '_sim_%s.png' % (now_task.item())),
                                           simscore[pi, ...].detach().cpu().numpy())
                                plt.imsave(
                                    os.path.join(output_folder, str(num) + '_sim*label_%s.png' % (now_task.item())),
                                    simscore[pi, ...].detach().cpu().numpy() * labels[pi, ...].detach().cpu().numpy())
                                plt.imsave(os.path.join(output_folder, str(num) + '_con_%s.png' % (now_task.item())),
                                           conscore[pi, ...].detach().cpu().numpy())
                                plt.imsave(
	                                os.path.join(output_folder, str(num) + '_con*label_%s.png' % (now_task.item())),
	                                conscore[pi, ...].detach().cpu().numpy() * labels[pi, ...].detach().cpu().numpy())


                        # if task2_pool_image.num_imgs >= batch_size:
                        #     images = task2_pool_image.query(batch_size)
                        #     labels = task2_pool_mask.query(batch_size)
                        #     scales = torch.ones(batch_size).cuda()
                        #     for bi in range(len(scales)):
                        #         scales[bi] = task2_scale.pop(0)
                        #     preds, _ = model(images, torch.ones(batch_size).cuda()*2, scales)
						#
                        #     now_task = torch.tensor(2)
						#
                        #     now_preds = preds[:, 1, ...] > preds[:, 0, ...]
                        #     now_preds_onehot = one_hot_3D(now_preds.long())
						#
                        #     labels_onehot = one_hot_3D(labels.long())
                        #     rmin, rmax, cmin, cmax = mask_to_box(images)
                        #     F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)
						#
                        #     val_F1[2] += F1
                        #     val_Dice[2] += DICE
                        #     val_TPR[2] += TPR
                        #     val_PPV[2] += PPV
                        #     cnt[2] += 1
						#
                        #     for pi in range(len(images)):
                        #         prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
                        #         num = len(glob.glob(os.path.join(output_folder, '*')))
                        #         out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                        #         img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                        #         plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
                        #                    img)
                        #         plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
                        #                    labels[pi, ...].detach().cpu().numpy())
                        #         plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
                        #                    prediction.detach().cpu().numpy())
						#
                        # if task3_pool_image.num_imgs >= batch_size:
                        #     images = task3_pool_image.query(batch_size)
                        #     labels = task3_pool_mask.query(batch_size)
                        #     scales = torch.ones(batch_size).cuda()
                        #     for bi in range(len(scales)):
                        #         scales[bi] = task3_scale.pop(0)
                        #     preds, _ = model(images, torch.ones(batch_size).cuda()*3, scales)
						#
                        #     now_task = torch.tensor(3)
						#
                        #     now_preds = preds[:, 1, ...] > preds[:, 0, ...]
                        #     now_preds_onehot = one_hot_3D(now_preds.long())
						#
                        #     labels_onehot = one_hot_3D(labels.long())
                        #     rmin, rmax, cmin, cmax = mask_to_box(images)
                        #     F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)
						#
                        #     val_F1[3] += F1
                        #     val_Dice[3] += DICE
                        #     val_TPR[3] += TPR
                        #     val_PPV[3] += PPV
                        #     cnt[3] += 1
						#
                        #     for pi in range(len(images)):
                        #         prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
                        #         num = len(glob.glob(os.path.join(output_folder, '*')))
                        #         out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                        #         img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                        #         plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
                        #                    img)
                        #         plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
                        #                    labels[pi, ...].detach().cpu().numpy())
                        #         plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
                        #                    prediction.detach().cpu().numpy())
						#
                        # if task4_pool_image.num_imgs >= batch_size:
                        #     images = task4_pool_image.query(batch_size)
                        #     labels = task4_pool_mask.query(batch_size)
                        #     scales = torch.ones(batch_size).cuda()
                        #     for bi in range(len(scales)):
                        #         scales[bi] = task4_scale.pop(0)
                        #     preds, _ = model(images, torch.ones(batch_size).cuda()*4, scales)
						#
                        #     now_task = torch.tensor(4)
						#
                        #     now_preds = preds[:, 1, ...] > preds[:, 0, ...]
                        #     now_preds_onehot = one_hot_3D(now_preds.long())
						#
                        #     labels_onehot = one_hot_3D(labels.long())
                        #     rmin, rmax, cmin, cmax = mask_to_box(images)
                        #     F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)
						#
                        #     val_F1[4] += F1
                        #     val_Dice[4] += DICE
                        #     val_TPR[4] += TPR
                        #     val_PPV[4] += PPV
                        #     cnt[4] += 1
						#
                        #     for pi in range(len(images)):
                        #         prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
                        #         num = len(glob.glob(os.path.join(output_folder, '*')))
                        #         out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                        #         img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                        #         plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
                        #                    img)
                        #         plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
                        #                    labels[pi, ...].detach().cpu().numpy())
                        #         plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
                        #                    prediction.detach().cpu().numpy())
						#
                        # if task5_pool_image.num_imgs >= batch_size:
                        #     images = task5_pool_image.query(batch_size)
                        #     labels = task5_pool_mask.query(batch_size)
                        #     scales = torch.ones(batch_size).cuda()
                        #     for bi in range(len(scales)):
                        #         scales[bi] = task5_scale.pop(0)
						#
                        #     preds, _ = model(images, torch.ones(batch_size).cuda()*5, scales)
						#
                        #     now_task = torch.tensor(5)
						#
                        #     now_preds = preds[:, 1, ...] > preds[:, 0, ...]
                        #     now_preds_onehot = one_hot_3D(now_preds.long())
						#
                        #     labels_onehot = one_hot_3D(labels.long())
                        #     rmin, rmax, cmin, cmax = mask_to_box(images)
                        #     F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)
						#
                        #     val_F1[5] += F1
                        #     val_Dice[5] += DICE
                        #     val_TPR[5] += TPR
                        #     val_PPV[5] += PPV
                        #     cnt[5] += 1
						#
                        #     for pi in range(len(images)):
                        #         prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
                        #         num = len(glob.glob(os.path.join(output_folder, '*')))
                        #         out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                        #         img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                        #         plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
                        #                    img)
                        #         plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
                        #                    labels[pi, ...].detach().cpu().numpy())
                        #         plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
                        #                    prediction.detach().cpu().numpy())


                    avg_val_F1 = val_F1 / cnt
                    avg_val_Dice = val_Dice / cnt
                    avg_val_TPR = val_TPR / cnt
                    avg_val_PPV = val_PPV / cnt

                    print('Validate \n 0dt_f1={:.4} 0dt_dsc={:.4} 0dt_tpr={:.4} 0dt_ppv={:.4}'
                          ' \n 1pt_f1={:.4} 1pt_dsc={:.4} 1pt_tpr={:.4} 1pt_ppv={:.4}\n'
                          ' \n 2cps_f1={:.4} 2cps_dsc={:.4} 2cps_tpr={:.4} 2cps_ppv={:.4}\n'
                          ' \n 3tf_f1={:.4} 3tf_dsc={:.4} 3tf_tpr={:.4} 3tf_ppv={:.4}\n'
                          ' \n 4vs_f1={:.4} 4vs_dsc={:.4} 4vs_tpr={:.4} 4vs_ppv={:.4}\n'
                          ' \n 5ptc_f1={:.4} 5ptc_dsc={:.4} 5ptc_tpr={:.4} 5ptc_ppv={:.4}\n'
                          .format(avg_val_F1[0].item(), avg_val_Dice[0].item(), avg_val_TPR[0].item(), avg_val_PPV[0].item(),
                                  avg_val_F1[1].item(), avg_val_Dice[1].item(), avg_val_TPR[1].item(), avg_val_PPV[1].item(),
                                  avg_val_F1[2].item(), avg_val_Dice[2].item(), avg_val_TPR[2].item(), avg_val_PPV[2].item(),
                                  avg_val_F1[3].item(), avg_val_Dice[3].item(), avg_val_TPR[3].item(), avg_val_PPV[3].item(),
                                  avg_val_F1[4].item(), avg_val_Dice[4].item(), avg_val_TPR[4].item(), avg_val_PPV[4].item(),
                                  avg_val_F1[5].item(), avg_val_Dice[5].item(), avg_val_TPR[5].item(), avg_val_PPV[5].item()))

                    df = pd.DataFrame(columns = ['task','F1','Dice','TPR','PPV'])
                    df.loc[0] = ['0dt', avg_val_F1[0].item(), avg_val_Dice[0].item(), avg_val_TPR[0].item(), avg_val_PPV[0].item()]
                    df.loc[1] = ['1pt', avg_val_F1[1].item(), avg_val_Dice[1].item(), avg_val_TPR[1].item(), avg_val_PPV[1].item()]
                    df.loc[2] = ['2capsule', avg_val_F1[2].item(), avg_val_Dice[2].item(), avg_val_TPR[2].item(), avg_val_PPV[2].item()]
                    df.loc[3] = ['3tuft', avg_val_F1[3].item(), avg_val_Dice[3].item(), avg_val_TPR[3].item(), avg_val_PPV[3].item()]
                    df.loc[4] = ['4vessel', avg_val_F1[4].item(), avg_val_Dice[4].item(), avg_val_TPR[4].item(), avg_val_PPV[4].item()]
                    df.loc[5] = ['5ptc', avg_val_F1[5].item(), avg_val_Dice[5].item(), avg_val_TPR[5].item(), avg_val_PPV[5].item()]
                    df.to_csv(os.path.join(output_folder,'validation_result.csv'))


                print('save model ...')
                if args.FP16:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'amp': amp.state_dict()
                    }
                    torch.save(checkpoint, osp.join(args.snapshot_dir, 'MOTS_DynConv_' + args.snapshot_dir.split('/')[-2] + '_e' + str(epoch) + '.pth'))
                else:
                    torch.save(model.state_dict(),osp.join(args.snapshot_dir, 'MOTS_DynConv_' + args.snapshot_dir.split('/')[-2] + '_e' + str(epoch) + '.pth'))

            if (epoch >= args.num_epochs - 1) and (args.local_rank == 0):
                print('save model ...')
                if args.FP16:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'amp': amp.state_dict()
                    }
                    torch.save(checkpoint, osp.join(args.snapshot_dir, 'MOTS_DynConv_' + args.snapshot_dir.split('/')[-2] + '_final_e' + str(epoch) + '.pth'))
                else:
                    torch.save(model.state_dict(),osp.join(args.snapshot_dir, 'MOTS_DynConv_' + args.snapshot_dir.split('/')[-2] + '_final_e' + str(epoch) + '.pth'))
                break

        end = timeit.default_timer()
        print(end - start, 'seconds')


if __name__ == '__main__':
    main()
