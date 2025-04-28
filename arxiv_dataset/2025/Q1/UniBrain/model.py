import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import matplotlib
from torch.nn import Parameter
import math
import random
from torch_geometric.nn import GCNConv, BatchNorm

class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        I = y_true
        J = y_pred

        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        win = [9] * ndims if self.win is None else self.win

        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        conv_fn = getattr(F, 'conv%dd' % ndims)

        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)


class GCC(nn.Module):
    """
    global normalized cross correlation (sqrt)
    """

    def __init__(self):
        super(GCC, self).__init__()

    def forward(self, I, J):
        I2 = I.pow(2)
        J2 = J.pow(2)
        IJ = I * J

        I_ave, J_ave = I.mean(), J.mean()
        I2_ave, J2_ave = I2.mean(), J2.mean()
        IJ_ave = IJ.mean()

        cross = IJ_ave - I_ave * J_ave
        I_var = I2_ave - I_ave.pow(2)
        J_var = J2_ave - J_ave.pow(2)

        #        cc = cross*cross / (I_var*J_var + np.finfo(float).eps)#1e-5
        cc = cross / (I_var.sqrt() * J_var.sqrt() + np.finfo(float).eps)  # 1e-5

        return -1.0 * cc + 1

def histogram_2d(x, y, bins, min_val, max_val):
    """
    Compute a 2D histogram for the input arrays x and y.
    """
    hist = torch.histc(x * max_val + y, bins=bins * bins, min=0, max=max_val * max_val)
    hist = hist.view(bins, bins)
    return hist

def mutual_information(hist):
    """
    Calculate the Mutual Information based on a joint histogram.
    """
    pxy = hist / torch.sum(hist)
    px = torch.sum(pxy, dim=1) # marginal for x over y
    py = torch.sum(pxy, dim=0) # marginal for y over x
    pxy = torch.where(pxy != 0, pxy, torch.ones_like(pxy))
    px_py = px[:, None] * py[None, :]
    mi = torch.sum(pxy * torch.log2(pxy / (px_py + 1e-6)))
    return mi

class NMI_Loss(torch.nn.Module):
    def __init__(self, bins=64, min_val=0.0, max_val=1.0):
        super(NMI_Loss, self).__init__()
        self.bins = bins
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, img1, img2):
        # Flatten the images
        img1 = img1.flatten(start_dim=1)
        img2 = img2.flatten(start_dim=1)

        # Compute histograms
        hist_2d = histogram_2d(img1, img2, self.bins, self.min_val, self.max_val)

        # Calculate Mutual Information
        mi = mutual_information(hist_2d)

        # Calculate marginal histograms and entropies
        hist1 = torch.sum(hist_2d, dim=1)
        hist2 = torch.sum(hist_2d, dim=0)
        entropy1 = -torch.sum(hist1[hist1 != 0] * torch.log2(hist1[hist1 != 0]))
        entropy2 = -torch.sum(hist2[hist2 != 0] * torch.log2(hist2[hist2 != 0]))

        # Normalize Mutual Information to get NMI
        nmi = 2.0 * mi / (entropy1 + entropy2 + 1e-6)
        return -nmi  # Negative NMI for loss

    
    
class first_Grad(nn.Module):
    """
    N-D gradient loss
    """

    def __init__(self, penalty):
        super(first_Grad, self).__init__()
        self.penalty = penalty

    def forward(self, pred):

        dy = torch.abs(pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :])
        dx = torch.abs(pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :])
        dz = torch.abs(pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1])

        if self.penalty == 'l2':

            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        elif self.penalty == 'l1':

            dy = dy
            dx = dx
            dz = dz

        d = torch.mean(dy) + torch.mean(dx) + torch.mean(dz)
        grad = d / 3.0

        return grad


class conv_block(nn.Module):
    def __init__(self, inChan, outChan, stride=1):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(inChan, outChan, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.BatchNorm3d(outChan),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        return x


class Unet(nn.Module):
    def __init__(self, enc_nf=[2, 16, 32, 32, 64, 64], dec_nf=[64, 32, 32, 32, 16, 3]):
        super(Unet, self).__init__()

        self.inconv = conv_block(enc_nf[0], enc_nf[1])
        self.down1 = conv_block(enc_nf[1], enc_nf[2], 2)
        self.down2 = conv_block(enc_nf[2], enc_nf[3], 2)
        self.down3 = conv_block(enc_nf[3], enc_nf[4], 2)
        self.down4 = conv_block(enc_nf[4], enc_nf[5], 2)
        self.up1 = conv_block(enc_nf[-1], dec_nf[0])
        self.up2 = conv_block(dec_nf[0] + enc_nf[4], dec_nf[1])
        self.up3 = conv_block(dec_nf[1] + enc_nf[3], dec_nf[2])
        self.up4 = conv_block(dec_nf[2] + enc_nf[2], dec_nf[3])
        self.same_conv = conv_block(dec_nf[3] + enc_nf[1], dec_nf[4])
        self.outconv = nn.Conv3d(
            dec_nf[4], dec_nf[5], kernel_size=3, stride=1, padding=1, bias=True)
        # init last_conv
        self.outconv.weight.data.normal_(mean=0, std=1e-5)
        if self.outconv.bias is not None:
            self.outconv.bias.data.zero_()

    def forward(self, x):
        # down-sample path (encoder)
        skip1 = self.inconv(x)
        skip2 = self.down1(skip1)
        skip3 = self.down2(skip2)
        skip4 = self.down3(skip3)
        x = self.down4(skip4)
        # up-sample path (decoder)
        x = self.up1(x)
        #        x = self.upsample(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat((x, skip4), 1)
        x = self.up2(x)
        #        x = self.upsample(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat((x, skip3), 1)
        x = self.up3(x)
        #        x = self.upsample(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat((x, skip2), 1)
        x = self.up4(x)
        #        x = self.upsample(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat((x, skip1), 1)
        x = self.same_conv(x)
        x = self.outconv(x)

        return x


def get_batch_identity_theta_4_4(batch_size):
    theta = Variable(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], dtype=torch.float)).cuda()
    i_theta = theta.view(4, 4).unsqueeze(0).repeat(batch_size, 1, 1)
    return i_theta


def get_batch_identity_theta_3_4(batch_size):
    theta = Variable(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float)).cuda()
    i_theta = theta.view(3, 4).unsqueeze(0).repeat(batch_size, 1, 1)
    return i_theta


def theta_3_4_to_4_4(batch_size, theta):
    i_theta = get_batch_identity_theta_4_4(batch_size)
    i_theta[:, 0:3, :] = theta
    return i_theta


def theta_dot(batch_size, theta_cur, theta_pre):
    theta_cur_4_4 = theta_3_4_to_4_4(batch_size, theta_cur)
    theta_pre_4_4 = theta_3_4_to_4_4(batch_size, theta_pre)

    theta_cat_4_4 = theta_cur_4_4 @ theta_pre_4_4
    theta_cat_3_4 = theta_cat_4_4[:, 0:3, :]
    return theta_cat_3_4


def theta_dot_with_inv(batch_size, theta_cur, theta_pre):
    theta_cur_4_4 = theta_3_4_to_4_4(batch_size, theta_cur)
    theta_pre_4_4 = theta_3_4_to_4_4(batch_size, theta_pre)

    theta_cat_4_4 = theta_cur_4_4 @ theta_pre_4_4
    theta_cat_4_4_inv = torch.linalg.inv(theta_cat_4_4)

    theta_cat_3_4 = theta_cat_4_4[:, 0:3, :]
    theta_cat_3_4_inv = theta_cat_4_4_inv[:, 0:3, :]

    return theta_cat_3_4, theta_cat_3_4_inv


class encoder(nn.Module):
    def __init__(self, enc_nf=[2, 16, 32, 64, 128, 256, 512]):
        super(encoder, self).__init__()

        self.inconv = conv_block(enc_nf[0], enc_nf[1])
        self.down1 = conv_block(enc_nf[1], enc_nf[2], 2)
        self.down2 = conv_block(enc_nf[2], enc_nf[3], 2)
        self.down3 = conv_block(enc_nf[3], enc_nf[4], 2)
        self.down4 = conv_block(enc_nf[4], enc_nf[5], 2)
        self.down5 = conv_block(enc_nf[5], enc_nf[6], 2)

        self.fc_loc = nn.Sequential(
            nn.Linear(512 * 3 * 3 * 3, 128),
            nn.LeakyReLU(0.2, True),
            nn.Linear(128, 4 * 3))

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0,
                                                     0, 1, 0, 0,
                                                     0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        x = self.inconv(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        x = x.view(-1, )

        x = self.fc_loc(x)

        theta = x.view(-1, 3, 4)

        return theta


class multi_stage_ext(nn.Module):
    def __init__(self, img_size, stage, gamma=10,
                 enc_nf=[1, 16, 32, 32, 64, 64], dec_nf=[64, 32, 32, 32, 16, 1], ):
        super(multi_stage_ext, self).__init__()
        self.unet = Unet(enc_nf, dec_nf)
        self.stage = stage
        self.gamma = gamma

    def forward(self, mov, if_train):
        img_size = mov.shape[-1]
        batch_size = mov.shape[0]

        striped_list = []
        mask_list = []

        for i in range(self.stage):
            mask = self.unet(mov)
            mask = torch.nn.Sigmoid()(self.gamma * mask)

            if if_train == False:
                mask[mask < 0.5] = 0.0
                mask[mask >= 0.5] = 1.0

            mov = mask * mov

            striped_list.append(mov)
            mask_list.append(mask)

        return striped_list, mask_list


class multi_stage_reg(nn.Module):
    def __init__(self, img_size, stage,
                 enc_affine=[2, 16, 32, 64, 128, 256, 512]):
        super(multi_stage_reg, self).__init__()
        self.affine = encoder(enc_affine)
        self.stage = stage

    def forward(self, ref, mov):
        img_size = ref.shape[-1]
        batch_size = ref.shape[0]

        warped_list = []
        theta_list = []
        theta_list_inv = []

        theta_previous = get_batch_identity_theta_3_4(batch_size)
        cur_mov = mov

        for i in range(self.stage):
            image = torch.cat((ref, cur_mov), 1)
            theta_cur = self.affine(image)

            theta_out, theta_out_inv = theta_dot_with_inv(batch_size, theta_cur, theta_previous)

            cur_grid = F.affine_grid(theta_out, ref.size(), align_corners=True)
            cur_grid_inv = F.affine_grid(theta_out_inv, ref.size(), align_corners=True)
            cur_mov = F.grid_sample(mov, cur_grid, mode="bilinear", align_corners=True)

            theta_previous = theta_out

            warped_list.append(cur_mov)
            theta_list.append(theta_out)
            theta_list_inv.append(theta_out_inv)

        return warped_list, theta_list, theta_list_inv 

def am_1d_2_Nd_torch(am):
    am=am.squeeze()
    mask_label=torch.unique(am)
    
    multi_d=torch.zeros((len(mask_label),96,96,96))
    for i in range(len(mask_label)):
        label_idx=mask_label[i]
        multi_d[i][am==label_idx]=1.0
    return multi_d.unsqueeze(0).float().cuda()

class Roi_feature_net(nn.Module):
    def __init__(self, hidden_size, output_size, input_size=96*96*96):
        super(Roi_feature_net, self).__init__()

        self.hidden1 = nn.Linear(input_size, hidden_size)

        self.hidden2 = nn.Linear(hidden_size, hidden_size)

        self.output = nn.Linear(hidden_size, output_size)

        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        
        x = x.view(-1,96*96*96)
        
        x = self.leaky_relu(self.hidden1(x))

        x = self.leaky_relu(self.hidden2(x))
        
        x = self.output(x)
        return x

class Graph_gen_net_norm(nn.Module):

    def __init__(self):
        super(Graph_gen_net_norm, self).__init__()

    def forward(self, x):
        
        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        
        ajc_m = torch.einsum('ijk,ipk->ijp', x, x)
        
        return ajc_m  
    

class GNN_predictor(nn.Module):
    def __init__(self, hidden_dim, node_input_dim, roi_num=116):
        super(GNN_predictor, self).__init__()
        self.roi_num = roi_num
        #hidden_dim = 128

        self.conv1 = GCNConv(node_input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.classifier = nn.Linear(hidden_dim * roi_num, 2)
    
    def get_edge_info(self, m):
        bz = m.shape[0]
        all_edge_indices = []
        all_edge_weights = []

        for b in range(bz):
            row, col = torch.where(m[b] > 0)

            row += b * self.roi_num
            col += b * self.roi_num

            all_edge_indices.append(torch.stack([row, col], dim=0))
            all_edge_weights.append(m[b, m[b] > 0])

        edge_index = torch.cat(all_edge_indices, dim=1)
        edge_weight = torch.cat(all_edge_weights)

        return edge_index, edge_weight

    def forward(self, m, node_feature):
        bz = m.shape[0]

        edge_index, edge_weight = self.get_edge_info(m)

        x = F.leaky_relu(self.conv1(node_feature.squeeze(1), edge_index, edge_weight), 0.2, inplace=True)
        x = F.leaky_relu(self.conv2(x, edge_index, edge_weight), 0.2, inplace=True)

        x = x.view(bz, -1)
        x = self.classifier(x)

        return x

def feature_perturbation(node_features, perturb_rate=0.01):
    noise = torch.randn_like(node_features) * perturb_rate
    perturbed_features = node_features + noise
    return perturbed_features

def edge_perturbation(edge_index, num_nodes, perturb_rate=0.01):
    # Ensure the input is two-dimensional
    if edge_index.dim() == 3:
        edge_index = edge_index.squeeze(0)

    # Create a copy of the adjacency matrix in two dimensions to represent the graph's edges
    edges = edge_index.nonzero().t()

    # Add edges
    num_new_edges = int(edges.shape[1] * perturb_rate)
    new_edges = torch.randint(0, num_nodes, (2, num_new_edges), device=edge_index.device)

    # Remove edges
    num_edges = edges.shape[1]
    num_edges_to_remove = int(num_edges * perturb_rate)
    indices_to_remove = torch.randperm(num_edges, device=edge_index.device)[:num_edges_to_remove]
    remaining_edges = edges[:, ~indices_to_remove]

    # Merge original and new edges
    perturbed_edges = torch.cat([remaining_edges, new_edges], dim=1)

    # Create a new adjacency matrix
    new_edge_index = torch.zeros_like(edge_index)
    new_edge_index[perturbed_edges[0], perturbed_edges[1]] = 1

    # Convert back to a three-dimensional tensor
    new_edge_index = new_edge_index.unsqueeze(0)
    return new_edge_index


class UniBrain(nn.Module):
    def __init__(self, img_size, ext_stage, reg_stage,if_pred_aal,
                 enc_nf=[1, 16, 32, 32, 64, 64, 128, 128], dec_nf=[128, 64, 64, 32, 32, 32, 16, 1],
                 enc_affine=[2, 16, 32, 64, 128, 256, 512],
                 enc_seg=[1, 128, 256, 256, 512, 512], dec_seg=[512, 256, 256, 256, 128, 4],
                 enc_par=[1, 128, 256, 256, 512, 512], dec_par=[512, 256, 256, 256, 128, 117],
                 feature_net_hidden=256,feature_dim=128,
                 gnn_hidden=128):
        super(UniBrain, self).__init__()
        
        #vision
        self.ext_net = multi_stage_ext(img_size, ext_stage)
        self.reg_net = multi_stage_reg(img_size, reg_stage)
        self.seg_net = Unet(enc_seg, dec_seg)
        self.par_net = Unet(enc_par, dec_par)
        #graph
        self.if_pred_aal = if_pred_aal
        self.feature_net = Roi_feature_net(feature_net_hidden,feature_dim)
        self.graph_gen = Graph_gen_net_norm()
        self.gnn_pred = GNN_predictor(gnn_hidden,feature_dim)
        
    def forward(self, ref, mov, ref_am, ref_aal, if_train):
        
        ##vision
        striped_list, mask_list = self.ext_net(mov, if_train)
        warped_list, theta_list, theta_list_inv = self.reg_net(ref, striped_list[-1])
        am_mov_pred = self.seg_net(mov)
        aal_mov_pred = self.par_net(mov)
        
        inv_grid = F.affine_grid(theta_list_inv[-1], ref_am.size(), align_corners=True)
        am_ref_2_mov = F.grid_sample(ref_am, inv_grid, mode="nearest", align_corners=True)
        aal_ref_2_mov = F.grid_sample(ref_aal, inv_grid, mode="nearest", align_corners=True)
        
        ##graph
        if self.if_pred_aal==True:
            aal_Nd=F.softmax(aal_mov_pred, dim=1)
        else:
            aal_Nd=am_1d_2_Nd_torch(aal_ref_2_mov)
        
        roi_images = aal_Nd * mov
        
        ## 1. roi_image squeeze first dim to allow roi_feature_net to process all roi via batch, 
        ## 2. do not use roi=0 (background) to create graph
        roi_images=roi_images.squeeze(0)[1:,:,:,:]

        roi_features=self.feature_net(roi_images).unsqueeze(0)
        
        # feature perturbation
        if if_train:
            roi_features = feature_perturbation(roi_features, perturb_rate=0.05)

        ajc_m = self.graph_gen(roi_features)
        
        # edge perturbation
        if if_train:
            num_nodes = roi_features.size(1)
            ajc_m = edge_perturbation(ajc_m, num_nodes, perturb_rate=0.05)
        
        y_predicted=self.gnn_pred(ajc_m,roi_features)
        
        return striped_list, mask_list, warped_list, theta_list, theta_list_inv, am_mov_pred, am_ref_2_mov, aal_mov_pred, aal_ref_2_mov, ajc_m, y_predicted
    