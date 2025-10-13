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


    lp_loss = 3. * lp_loss_1 + 1. * lp_loss_2
    return lp_loss



def cal_lp_loss2(output_tps_ref, output_tps_tgt):
    overlap = output_tps_ref[:,3:6,:,:] * output_tps_tgt[:,3:6,:,:]
    lp_loss_2 = l_num_loss(output_tps_ref[:,0:3,:,:]*overlap, output_tps_tgt[:,0:3,:,:]*overlap, 1)
    return lp_loss_2


def get_vgg19_FeatureMap(vgg_model, input_255, layer_index):
    
    vgg_mean = torch.tensor([123.6800, 116.7790, 103.9390]).reshape((1,3,1,1))
    if torch.cuda.is_available():
        vgg_mean = vgg_mean.cuda()
    vgg_input = input_255-vgg_mean
    #x = vgg_model.features[0](vgg_input)
    #FeatureMap_list.append(x)

    x_list = []

    for i in range(0,layer_index+1):
        if i == 0:
            x = vgg_model.features[0](vgg_input)
        else:
            x = vgg_model.features[i](x)
            if i == 6 or i == 13 or i ==24:
                x_list.append(x)

    return x_list



def cal_ddmperception_loss(vgg_model, input1_tesnor, input2_tesnor, DDM_ref_list, DDM_tgt_list):
    bs = input1_tesnor.size(0)
    iter_num = len(DDM_ref_list)

    # resize DDM to the resolution of semantic features
    DDM_ref1 = F.interpolate(DDM_ref_list[0], (128, 128), mode='bicubic', align_corners=False)
    DDM_tgt1 = F.interpolate(DDM_tgt_list[0], (128, 128), mode='bicubic', align_corners=False)
    # DDM_ref2 = F.interpolate(DDM_ref_list[1], (128, 128), mode='bicubic', align_corners=False)
    # DDM_tgt2 = F.interpolate(DDM_tgt_list[1], (128, 128), mode='bicubic', align_corners=False)

    # get feature maps
    H_ref_feature_list = get_vgg19_FeatureMap(vgg_model, (input1_tesnor+1)*127.5, 13)
    H_tgt_feature_list = get_vgg19_FeatureMap(vgg_model, (input2_tesnor+1)*127.5, 13)
    feature_ref = torch.mean(H_ref_feature_list[-1], 1, keepdim=True)  # bs, 1, h, w
    feature_tgt = torch.mean(H_tgt_feature_list[-1], 1, keepdim=True)  # bs, 1, h, w

    # normalized ref features
    reshaped_ref = feature_ref.view(bs, -1)
    reshaped_ref_min = reshaped_ref.min(dim=1)[0].view(bs, 1, 1, 1)
    reshaped_ref_max = reshaped_ref.max(dim=1)[0].view(bs, 1, 1, 1)
    reshaped_tgt = feature_tgt.view(bs, -1)
    reshaped_tgt_min = reshaped_tgt.min(dim=1)[0].view(bs, 1, 1, 1)
    reshaped_tgt_max = reshaped_tgt.max(dim=1)[0].view(bs, 1, 1, 1)
    reshaped_min = torch.min(reshaped_ref_min, reshaped_tgt_min)
    reshaped_max = torch.max(reshaped_ref_max, reshaped_tgt_max)
    normalized_feature_ref = (feature_ref - reshaped_min) / (reshaped_max - reshaped_min + 1e-8)  
    normalized_feature_tgt = (feature_tgt - reshaped_min) / (reshaped_max - reshaped_min + 1e-8)  


    #loss_list = []
    total_loss = 0.
    for i in range(iter_num):
        DDM_ref1 = F.interpolate(DDM_ref_list[i], (128, 128), mode='bicubic', align_corners=False)
        DDM_tgt1 = F.interpolate(DDM_tgt_list[i], (128, 128), mode='bicubic', align_corners=False)
        loss_ref1 = torch.mean((normalized_feature_ref+0.1) * DDM_ref1)
        loss_tgt1 = torch.mean((normalized_feature_tgt+0.1) * DDM_tgt1)
        loss1 = torch.max(loss_ref1, loss_tgt1)
        #loss_list.append(loss1)
        total_loss = total_loss + loss1*(0.5**(iter_num-1-i))

    return total_loss

def cal_ddmperception_loss_test(vgg_model, input1_tesnor, input2_tesnor, DDM_ref, DDM_tgt):
    bs = input1_tesnor.size(0)

    # resize DDM to the resolution of semantic features
    DDM_ref1 = F.interpolate(DDM_ref, (128, 128), mode='bicubic', align_corners=False)
    DDM_tgt1 = F.interpolate(DDM_tgt, (128, 128), mode='bicubic', align_corners=False)

    # get feature maps
    H_ref_feature_list = get_vgg19_FeatureMap(vgg_model, (input1_tesnor+1)*127.5, 13)
    H_tgt_feature_list = get_vgg19_FeatureMap(vgg_model, (input2_tesnor+1)*127.5, 13)
    feature_ref = torch.mean(H_ref_feature_list[-1], 1, keepdim=True)  # bs, 1, h, w
    feature_tgt = torch.mean(H_tgt_feature_list[-1], 1, keepdim=True)  # bs, 1, h, w

    # normalized ref features
    reshaped_ref = feature_ref.view(bs, -1)
    reshaped_ref_min = reshaped_ref.min(dim=1)[0].view(bs, 1, 1, 1)
    reshaped_ref_max = reshaped_ref.max(dim=1)[0].view(bs, 1, 1, 1)
    reshaped_tgt = feature_tgt.view(bs, -1)
    reshaped_tgt_min = reshaped_tgt.min(dim=1)[0].view(bs, 1, 1, 1)
    reshaped_tgt_max = reshaped_tgt.max(dim=1)[0].view(bs, 1, 1, 1)
    reshaped_min = torch.min(reshaped_ref_min, reshaped_tgt_min)
    reshaped_max = torch.max(reshaped_ref_max, reshaped_tgt_max)
    normalized_feature_ref = (feature_ref - reshaped_min) / (reshaped_max - reshaped_min + 1e-8)  
    normalized_feature_tgt = (feature_tgt - reshaped_min) / (reshaped_max - reshaped_min + 1e-8)  

    loss_ref1 = torch.mean((normalized_feature_ref+0.1) * DDM_ref1)
    loss_tgt1 = torch.mean((normalized_feature_tgt+0.1) * DDM_tgt1)
    loss = torch.max(loss_ref1, loss_tgt1)

    return loss
