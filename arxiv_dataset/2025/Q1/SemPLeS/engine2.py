# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import math
import sys
from typing import Iterable

import torch
import torch.nn.functional as F
import utils

from sklearn.metrics import average_precision_score
import numpy as np
import cv2
import os
from pathlib import Path


import torchvision as tv
from clip_utils import clip_forward, clip_forward_global, clip_forward_vis, clip_forward_btp2text
from clip_loss import SimMaxLoss, SimMinLoss, BackgroundSuppressionLoss


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    epoch: int, loss_scaler, max_norm: float = 0,
                    set_training_mode=True, clip_model=None, writer=None, hypers=None, btp=None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10


    Loss_Max = SimMaxLoss()
    Loss_Min = SimMinLoss()

    model.eval()
    def preprocess(labels):
        new_labels = []
        for n in range(labels.size(0)):
            for idx in range(0, labels.size(1)):
                temp = torch.zeros(1, labels.size(1)).long()
                if labels[n, idx] == 1:
                    temp[0, idx] = 1
                new_labels.append(temp)
        return torch.cat(new_labels, dim=0).cuda()


    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            # outputs, patch_outputs, attn_outputs = model(samples)

            # loss = F.multilabel_soft_margin_loss(outputs, targets)
            # metric_logger.update(cls_loss=loss.item())

            # ploss = F.multilabel_soft_margin_loss(patch_outputs, targets)
            # metric_logger.update(pat_loss=ploss.item())
            # loss = loss + ploss

            # aloss = F.multilabel_soft_margin_loss(attn_outputs, targets)
            # metric_logger.update(attn_loss=aloss.item())
            # loss = loss + aloss
            with torch.no_grad():
                output, cams, patch_attn = model(samples, return_att=True, attention_type='fused')
            patch_attn = patch_attn.squeeze(0)
            fine_cam = torch.matmul(patch_attn.unsqueeze(1), cams.view(cams.shape[0],
                                                                             cams.shape[1], -1, 1)). \
                reshape(cams.shape[0], cams.shape[1], cams.shape[2], cams.shape[3])



            fg_label = preprocess(targets.cpu())

            N, _, _, _ = fine_cam.size()

            # foreground indices
            clip_input_size = 224
            fg_indices = torch.nonzero(targets.reshape(-1) == 1, as_tuple=False).squeeze()

            cam_224 = F.interpolate(fine_cam, (clip_input_size, clip_input_size), mode='bilinear', align_corners=True).reshape(N * fine_cam.shape[1], 1, clip_input_size,
                                                                                                clip_input_size)
            # img_224 = F.interpolate(img, (clip_input_size, clip_input_size), mode='bilinear', align_corners=True)
            # Is_224 = F.interpolate(Is, (clip_input_size, clip_input_size), mode='bilinear', align_corners=True)

            min_ = cam_224.min(dim=2, keepdim=True)[0]
            min_ = min_.min(dim=3, keepdim=True)[0]
            max_ = cam_224.max(dim=2, keepdim=True)[0]
            max_ = max_.max(dim=3, keepdim=True)[0]
            cam_224 = (cam_224 - min_) / (max_ - min_ + 1e-4)
            # cam_224 = (cam_224 > 0.45).float()

            fg_224_eval = []
            bg_224_eval = []
            temp_idx = torch.nonzero(targets == 1, as_tuple=False)
            for j in range(temp_idx.shape[0]):
                fg_224_eval.append(cam_224[fg_indices[j]] * samples[temp_idx[j, 0]])
                bg_224_eval.append((1 - cam_224[fg_indices[j]]) * samples[temp_idx[j, 0]])

            fg_224_eval = torch.stack(fg_224_eval, dim=0)
            bg_224_eval = torch.stack(bg_224_eval, dim=0)

            if model.num_classes == 20:
                dname = 'voc'
            else:
                dname = 'coco'

            img_similarity, label_similarity = clip_forward_global(clip_model, bg_224_eval, fg_label[fg_indices], btp.prompts, dname=dname)
            L_Max = Loss_Max(img_similarity, 1)
            label_similarity = label_similarity.clamp(0.0001, 0.9999)
            L_Min = torch.log(label_similarity).mean()
            loss = L_Max + hypers[0] * L_Min

            print(img_similarity.clamp(0.0001, 0.9999).mean().item(), label_similarity.mean().item())
            print(label_similarity.max().item())

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=btp.parameters(), create_graph=is_second_order)


        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    writer.add_image('I', tv.utils.make_grid(samples, normalize=True, scale_each=True), epoch)
    with torch.no_grad():
        clip_forward_btp2text(clip_model, btp.prompts, dname=dname)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.MultiLabelSoftMarginLoss()
    mAP = []
    patch_mAP = []
    attn_mAP = []

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        batch_size = images.shape[0]

        with torch.cuda.amp.autocast():
            output, patch_output, attn_output = model(images)

            loss = criterion(output, target)
            output = torch.sigmoid(output)

            mAP_list = compute_mAP(target, output)
            mAP = mAP + mAP_list
            metric_logger.meters['mAP'].update(np.mean(mAP_list), n=batch_size)

            ploss = criterion(patch_output, target)
            loss += ploss
            patch_output = torch.sigmoid(patch_output)

            mAP_list = compute_mAP(target, patch_output)
            patch_mAP = patch_mAP + mAP_list
            metric_logger.meters['patch_mAP'].update(np.mean(mAP_list), n=batch_size)

            aloss = criterion(attn_output, target)
            loss += aloss
            attn_output = torch.sigmoid(attn_output)

            mAP_list = compute_mAP(target, attn_output)
            attn_mAP = attn_mAP + mAP_list
            metric_logger.meters['attn_mAP'].update(np.mean(mAP_list), n=batch_size)

        metric_logger.update(loss=loss.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print(
        '* mAP {mAP.global_avg:.3f} patch_mAP {patch_mAP.global_avg:.3f} attn_mAP {attn_mAP.global_avg:.3f} loss {losses.global_avg:.3f}'
        .format(mAP=metric_logger.mAP, patch_mAP=metric_logger.mAP, attn_mAP=metric_logger.mAP,
                losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def compute_mAP(labels, outputs):
    y_true = labels.cpu().numpy()
    y_pred = outputs.cpu().numpy()
    AP = []
    for i in range(y_true.shape[0]):
        if np.sum(y_true[i]) > 0:
            ap_i = average_precision_score(y_true[i], y_pred[i])
            AP.append(ap_i)
            # print(ap_i)
    return AP


@torch.no_grad()
def generate_attention_maps_ms(data_loader, model, device, args, epoch=None):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generating attention maps:'
    if args.attention_dir is not None:
        Path(args.attention_dir).mkdir(parents=True, exist_ok=True)
    if args.cam_npy_dir is not None:
        Path(args.cam_npy_dir).mkdir(parents=True, exist_ok=True)

    # switch to evaluation mode
    model.eval()

    img_list = open(args.img_ms_list).readlines()
    index = args.rank
    for image_list, target in metric_logger.log_every(data_loader, 10, header):
        if index >= len(img_list):
            continue
        images1 = image_list[0].to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        img_name = img_list[index].strip()
        index += args.world_size

        img_temp = images1.permute(0, 2, 3, 1).detach().cpu().numpy()
        orig_images = np.zeros_like(img_temp)
        orig_images[:, :, :, 0] = (img_temp[:, :, :, 0] * 0.229 + 0.485) * 255.
        orig_images[:, :, :, 1] = (img_temp[:, :, :, 1] * 0.224 + 0.456) * 255.
        orig_images[:, :, :, 2] = (img_temp[:, :, :, 2] * 0.225 + 0.406) * 255.

        w_orig, h_orig = orig_images.shape[1], orig_images.shape[2]

        with torch.cuda.amp.autocast():
            cam_list = []
            for s in range(len(image_list)):
                images = image_list[s].to(device, non_blocking=True)
                w, h = images.shape[2] - images.shape[2] % args.patch_size, images.shape[3] - images.shape[
                    3] % args.patch_size
                w_featmap = w // args.patch_size
                h_featmap = h // args.patch_size

                output, cams, patch_attn = model(images, return_att=True, attention_type=args.attention_type)
                patch_attn = torch.sum(patch_attn, dim=0)

                if args.patch_attn_refine:
                    cams = torch.matmul(patch_attn.unsqueeze(1),
                                                  cams.view(cams.shape[0], cams.shape[1],
                                                                      -1, 1)).reshape(cams.shape[0],
                                                                                      cams.shape[1],
                                                                                      w_featmap, h_featmap)

                cams = \
                    F.interpolate(cams, size=(w_orig, h_orig), mode='bilinear', align_corners=False)[0]
                cams = cams * target.clone().view(args.nb_classes, 1, 1)

                if s % 2 == 1:
                    cams = torch.flip(cams, dims=[-1])

                cam_list.append(cams)

            sum_cam = torch.sum(torch.stack(cam_list), dim=0)
            sum_cam = sum_cam.unsqueeze(0)

            output = torch.sigmoid(output)

        if args.visualize_cls_attn:
            for b in range(images.shape[0]):
                if (target[b].sum()) > 0:
                    cam_dict = {}
                    for cls_ind in range(args.nb_classes):
                        if target[b, cls_ind] > 0:
                            cls_score = format(output[b, cls_ind].cpu().numpy(), '.3f')

                            cam = sum_cam[b, cls_ind, :]

                            cam = (cam - cam.min()) / (
                                    cam.max() - cam.min() + 1e-8)
                            cam = cam.cpu().numpy()

                            cam_dict[cls_ind] = cam

                            if args.attention_dir is not None:
                                file_name = img_name + '_' + str(cls_ind) + '_' + str(cls_score) + '.png'
                                fname = os.path.join(args.attention_dir, file_name)
                                show_cam_on_image(orig_images[0], cam, fname)

                    if args.cam_npy_dir is not None:
                        np.save(os.path.join(args.cam_npy_dir, img_name + '.npy'), cam_dict)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return


def show_cam_on_image(img, mask, save_path):
    img = np.float32(img) / 255.
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + img
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    cv2.imwrite(save_path, cam)