import torch
import torch.nn as nn
import torch.nn.functional as F

import grid_res
grid_h = grid_res.GRID_H
grid_w = grid_res.GRID_W

min_w = (512 / grid_w) / 8
min_h = (512 / grid_h) / 8

# intra-grid constraint
def intra_grid_loss(pts):
    batch_size = pts.shape[0]

    delta_x = pts[:, :, 0:grid_w, 0] - pts[:, :, 1:grid_w + 1, 0]
    delta_y = pts[:, 0:grid_h, :, 1] - pts[:, 1:grid_h + 1, :, 1]

    loss_x = F.relu(delta_x + min_w)
    loss_y = F.relu(delta_y + min_h)

    loss = torch.mean(loss_x) + torch.mean(loss_y)
    return loss


# inter-grid constraint
def inter_grid_loss(train_mesh):
    w_edges = train_mesh[:, :, 0:grid_w, :] - train_mesh[:, :, 1:grid_w + 1, :]
    cos_w = torch.sum(w_edges[:, :, 0:grid_w - 1, :] * w_edges[:, :, 1:grid_w, :], 3) / \
            (torch.sqrt(torch.sum(w_edges[:, :, 0:grid_w - 1, :] * w_edges[:, :, 0:grid_w - 1, :], 3))
             * torch.sqrt(torch.sum(w_edges[:, :, 1:grid_w, :] * w_edges[:, :, 1:grid_w, :], 3)))
    # print("cos_w.shape")
    # print(cos_w.shape)
    delta_w_angle = 1 - cos_w

    h_edges = train_mesh[:, 0:grid_h, :, :] - train_mesh[:, 1:grid_h + 1, :, :]
    cos_h = torch.sum(h_edges[:, 0:grid_h - 1, :, :] * h_edges[:, 1:grid_h, :, :], 3) / \
            (torch.sqrt(torch.sum(h_edges[:, 0:grid_h - 1, :, :] * h_edges[:, 0:grid_h - 1, :, :], 3))
             * torch.sqrt(torch.sum(h_edges[:, 1:grid_h, :, :] * h_edges[:, 1:grid_h, :, :], 3)))
    delta_h_angle = 1 - cos_h

    loss = torch.mean(delta_w_angle) + torch.mean(delta_h_angle)
    return loss
    

def l_num_loss(img1, img2, l_num=1):
    return torch.mean(torch.abs((img1 - img2)**l_num))


def cal_lp_loss(input1, output_H, output_H_ref, output_H_tgt, output_tps_ref, output_tps_tgt):
    batch_size, _, img_h, img_w = input1.size()

    # # part one:
    overlap = output_H[:,3:6,:,:]
    lp_loss_1_1 = l_num_loss(input1*overlap, output_H[:,0:3,:,:]*overlap, 1)

    overlap = output_H_ref[:,3:6,:,:] * output_H_tgt[:,3:6,:,:]
    lp_loss_1_2 = l_num_loss(output_H_ref[:,0:3,:,:]*overlap, output_H_tgt[:,0:3,:,:]*overlap, 1)
 
    lp_loss_1 = (lp_loss_1_1 + lp_loss_1_2) / 2.

    # # part two:
    overlap = output_tps_ref[:,3:6,:,:] * output_tps_tgt[:,3:6,:,:]
    lp_loss_2 = l_num_loss(output_tps_ref[:,0:3,:,:]*overlap, output_tps_tgt[:,0:3,:,:]*overlap, 1)


    lp_loss = 1. * lp_loss_1 + 4. * lp_loss_2

    return lp_loss

def cal_lp_loss2(output_tps_ref, output_tps_tgt):
    overlap = output_tps_ref[:,3:6,:,:] * output_tps_tgt[:,3:6,:,:]
    lp_loss_2 = l_num_loss(output_tps_ref[:,0:3,:,:]*overlap, output_tps_tgt[:,0:3,:,:]*overlap, 1)
    return lp_loss_2





