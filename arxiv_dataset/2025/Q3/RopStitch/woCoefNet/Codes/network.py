import torch
import torch.nn as nn
import utils.torch_DLT as torch_DLT
import utils.torch_homo_transform as torch_homo_transform
import utils.torch_tps_transform as torch_tps_transform
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
resize_512 = T.Resize((512,512))

import grid_res
grid_h = grid_res.GRID_H
grid_w = grid_res.GRID_W



#Covert global homo into mesh
def H2Mesh(H, rigid_mesh):

    H_inv = torch.inverse(H)
    ori_pt = rigid_mesh.reshape(rigid_mesh.size()[0], -1, 2)
    ones = torch.ones(rigid_mesh.size()[0], (grid_h+1)*(grid_w+1),1)
    if torch.cuda.is_available():
        ori_pt = ori_pt.cuda()
        ones = ones.cuda()

    ori_pt = torch.cat((ori_pt, ones), 2) # bs*(grid_h+1)*(grid_w+1)*3
    tar_pt = torch.matmul(H_inv, ori_pt.permute(0,2,1)) # bs*3*(grid_h+1)*(grid_w+1)

    mesh_x = torch.unsqueeze(tar_pt[:,0,:]/tar_pt[:,2,:], 2)
    mesh_y = torch.unsqueeze(tar_pt[:,1,:]/tar_pt[:,2,:], 2)
    mesh = torch.cat((mesh_x, mesh_y), 2).reshape([rigid_mesh.size()[0], grid_h+1, grid_w+1, 2])

    return mesh


# get rigid mesh
def get_rigid_mesh(batch_size, height, width):

    ww = torch.matmul(torch.ones([grid_h+1, 1]), torch.unsqueeze(torch.linspace(0., float(width), grid_w+1), 0))
    hh = torch.matmul(torch.unsqueeze(torch.linspace(0.0, float(height), grid_h+1), 1), torch.ones([1, grid_w+1]))
    if torch.cuda.is_available():
        ww = ww.cuda()
        hh = hh.cuda()

    ori_pt = torch.cat((ww.unsqueeze(2), hh.unsqueeze(2)),2) # (grid_h+1)*(grid_w+1)*2
    ori_pt = ori_pt.unsqueeze(0).expand(batch_size, -1, -1, -1)

    return ori_pt

# normalize mesh from -1 ~ 1
def get_norm_mesh(mesh, height, width):
    batch_size = mesh.size()[0]
    mesh_w = mesh[...,0]*2./float(width) - 1.
    mesh_h = mesh[...,1]*2./float(height) - 1.
    norm_mesh = torch.stack([mesh_w, mesh_h], 3) # bs*(grid_h+1)*(grid_w+1)*2

    return norm_mesh.reshape([batch_size, -1, 2]) # bs*-1*2


# random augmentation
# it seems to do nothing to the performance
def data_aug(img1, img2):
    # Randomly shift brightness
    random_brightness = torch.randn(1).uniform_(0.7,1.3).cuda()
    img1_aug = img1 * random_brightness
    random_brightness = torch.randn(1).uniform_(0.7,1.3).cuda()
    img2_aug = img2 * random_brightness

    # Randomly shift color
    white = torch.ones([img1.size()[0], img1.size()[2], img1.size()[3]]).cuda()
    random_colors = torch.randn(3).uniform_(0.7,1.3).cuda()
    color_image = torch.stack([white * random_colors[i] for i in range(3)], axis=1)
    img1_aug  *= color_image

    random_colors = torch.randn(3).uniform_(0.7,1.3).cuda()
    color_image = torch.stack([white * random_colors[i] for i in range(3)], axis=1)
    img2_aug  *= color_image

    # clip
    img1_aug = torch.clamp(img1_aug, -1, 1)
    img2_aug = torch.clamp(img2_aug, -1, 1)

    return img1_aug, img2_aug


# for train.py / test.py
def build_model(net, input1_tensor, input2_tensor, is_training = True):
    batch_size, _, img_h, img_w = input1_tensor.size()

    # random values
    coef = torch.rand((batch_size, 4, 1)).cuda()
    alpha = torch.rand((batch_size, 2, 1, 1)).cuda()
    # network
    if is_training == True:
        aug_input1_tensor, aug_input2_tensor = data_aug(input1_tensor, input2_tensor)
        H_motion, mesh_motion_ref, mesh_motion_tgt = net(aug_input1_tensor, aug_input2_tensor, coef, alpha)
    else:
        H_motion, mesh_motion_ref, mesh_motion_tgt = net(input1_tensor, input2_tensor, coef, alpha)

    H_motion = H_motion.reshape(-1, 4, 2)
    mesh_motion_ref = mesh_motion_ref.reshape(-1, grid_h+1, grid_w+1, 2)
    mesh_motion_tgt = mesh_motion_tgt.reshape(-1, grid_h+1, grid_w+1, 2)

    # initialize the source points bs x 4 x 2
    src_p = torch.tensor([[0., 0.], [img_w, 0.], [0., img_h], [img_w, img_h]])
    if torch.cuda.is_available():
        src_p = src_p.cuda()
    src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)
    # target points
    dst_p = src_p + H_motion
    # solve homo using DLT
    H = torch_DLT.Tensor_ACA_rect(src_p, dst_p)

    M_tensor = torch.tensor([[img_w / 2.0, 0., img_w / 2.0],
                      [0., img_h / 2.0, img_h / 2.0],
                      [0., 0., 1.]]).cuda()
    M_tile = M_tensor.unsqueeze(0).expand(batch_size, -1, -1)
    M_tensor_inv = torch.inverse(M_tensor)
    M_tile_inv = M_tensor_inv.unsqueeze(0).expand(batch_size, -1, -1)
    mask = torch.ones_like(input2_tensor).cuda()

    H_mat = torch.matmul(torch.matmul(M_tile_inv, H), M_tile)
    output_H = torch_homo_transform.transformer(torch.cat((input2_tensor, mask), 1), H_mat, (img_h, img_w))

    ################# Homogrpahy Decomposition #############
    # coef = torch.rand((batch_size, 4, 1)).cuda()
    dst_p_tgt = src_p + (H_motion * coef)
    #dst_p_tgt = src_p + (H_motion * 0.5)
    H_tgt = torch_DLT.Tensor_ACA_rect(src_p, dst_p_tgt)
    H_ref = torch.matmul(torch.inverse(H), H_tgt)
    H_mat_ref = torch.matmul(torch.matmul(M_tile_inv, H_ref), M_tile)
    H_mat_tgt = torch.matmul(torch.matmul(M_tile_inv, H_tgt), M_tile)
    output_H_ref = torch_homo_transform.transformer(torch.cat((input1_tensor, mask), 1), H_mat_ref, (img_h, img_w))
    output_H_tgt = torch_homo_transform.transformer(torch.cat((input2_tensor, mask), 1), H_mat_tgt, (img_h, img_w))

    ##### stage 2 ####
    rigid_mesh = get_rigid_mesh(batch_size, img_h, img_w)
    ini_mesh_ref = H2Mesh(H_ref, rigid_mesh)
    mesh_ref = ini_mesh_ref + mesh_motion_ref
    ini_mesh_tgt = H2Mesh(H_tgt, rigid_mesh)
    mesh_tgt = ini_mesh_tgt + mesh_motion_tgt

    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
    norm_mesh_ref = get_norm_mesh(mesh_ref, img_h, img_w)
    norm_mesh_tgt = get_norm_mesh(mesh_tgt, img_h, img_w)

    output_tps_ref = torch_tps_transform.transformer(torch.cat((input1_tensor, mask), 1), norm_mesh_ref, norm_rigid_mesh, (img_h, img_w))
    output_tps_tgt = torch_tps_transform.transformer(torch.cat((input2_tensor, mask), 1), norm_mesh_tgt, norm_rigid_mesh, (img_h, img_w))
    
    out_dict = {}
    out_dict.update(output_H=output_H, output_H_ref=output_H_ref, output_H_tgt=output_H_tgt)
    out_dict.update(output_tps_ref = output_tps_ref, mesh_ref = mesh_ref)
    out_dict.update(output_tps_tgt = output_tps_tgt, mesh_tgt = mesh_tgt)

    return out_dict



# for test_output.py
def build_output_model(net, input1_tensor, input2_tensor, alpha=0.5, max_out_height=1200):
    batch_size, _, img_h, img_w = input1_tensor.size()
    # use middle plane to warp
    coef = torch.rand((batch_size, 4, 1)).cuda()
    coef = coef*0 + 0.5

    # input resize
    resized_input1 = resize_512(input1_tensor)
    resized_input2 = resize_512(input2_tensor)
    H_motion, mesh_motion_ref, mesh_motion_tgt = net(resized_input1, resized_input2, coef, alpha)

    # output mesh motion resize
    H_motion = H_motion.reshape(batch_size, 4, 2)
    mesh_motion_ref = mesh_motion_ref.reshape(batch_size, grid_h+1, grid_w+1, 2)
    mesh_motion_tgt = mesh_motion_tgt.reshape(batch_size, grid_h+1, grid_w+1, 2)

    H_motion = torch.stack([H_motion[...,0]*img_w/512, H_motion[...,1]*img_h/512], 2)
    mesh_motion_ref = torch.stack([mesh_motion_ref[...,0]*img_w/512, mesh_motion_ref[...,1]*img_h/512], 3)
    mesh_motion_tgt = torch.stack([mesh_motion_tgt[...,0]*img_w/512, mesh_motion_tgt[...,1]*img_h/512], 3)

    ######### warping img1 and img2 to the middle plane ########
    # initialize the source points bs x 4 x 2
    src_p = torch.tensor([[0., 0.], [img_w, 0.], [0., img_h], [img_w, img_h]]).cuda()
    src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)
    # target points
    dst_p = src_p + H_motion
    dst_p_tgt = src_p + H_motion*coef
    H = torch_DLT.Tensor_ACA_rect(src_p, dst_p)
    H_tgt = torch_DLT.Tensor_ACA_rect(src_p, dst_p_tgt)
    H_ref = torch.matmul(torch.inverse(H), H_tgt)


    # then, calculate the final mesh
    rigid_mesh = get_rigid_mesh(batch_size, img_h, img_w)
    ini_mesh_ref = H2Mesh(H_ref, rigid_mesh)
    mesh_ref = ini_mesh_ref + mesh_motion_ref
    ini_mesh_tgt = H2Mesh(H_tgt, rigid_mesh)
    mesh_tgt = ini_mesh_tgt + mesh_motion_tgt


    # calculate the size of stitched image
    width_max = torch.maximum(torch.max(mesh_ref[...,0]), torch.max(mesh_tgt[...,0]))
    width_min = torch.minimum(torch.min(mesh_ref[...,0]), torch.min(mesh_tgt[...,0]))
    height_max = torch.maximum(torch.max(mesh_ref[...,1]), torch.max(mesh_tgt[...,1]))
    height_min = torch.minimum(torch.min(mesh_ref[...,1]), torch.min(mesh_tgt[...,1]))

    out_width = width_max - width_min
    out_height = height_max - height_min

    # in case of the original image resoultion is so huge.
    if max(out_height,out_width) >= max_out_height:
        return None, False
    # convert the mesh from [img_h, img_w] to [out_h, out_w]
    mesh_trans_ref = torch.stack([mesh_ref[...,0]-width_min, mesh_ref[...,1]-height_min], 3)
    mesh_trans_tgt = torch.stack([mesh_tgt[...,0]-width_min, mesh_tgt[...,1]-height_min], 3)

    # normalization
    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
    norm_mesh_trans_ref = get_norm_mesh(mesh_trans_ref, out_height, out_width)
    norm_mesh_trans_tgt = get_norm_mesh(mesh_trans_tgt, out_height, out_width)

    # # transformation
    mask = torch.ones_like(input2_tensor).cuda()
    output_ref = torch_tps_transform.transformer(torch.cat((input1_tensor+1, mask), 1), norm_mesh_trans_ref, norm_rigid_mesh, (out_height.int(), out_width.int()))
    output_tgt = torch_tps_transform.transformer(torch.cat((input2_tensor+1, mask), 1), norm_mesh_trans_tgt, norm_rigid_mesh, (out_height.int(), out_width.int()))

    out_dict = {}
    out_dict.update(output_ref=output_ref, output_tgt = output_tgt)
    #out_dict.update(output_H_ref=output_H_ref, output_H_tgt = output_H_tgt)

    return out_dict, True



def get_res18_FeatureMap(resnet18_model):

    layers_list = []


    layers_list.append(resnet18_model.conv1)    #stride 2*2     H/2
    #layers_list.append(nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False))
    layers_list.append(resnet18_model.bn1)
    layers_list.append(resnet18_model.relu)
    layers_list.append(resnet18_model.maxpool)  #stride 2       H/4

    layers_list.append(resnet18_model.layer1)                  #H/4
    layers_list.append(resnet18_model.layer2)                  #H/8

    feature_extractor_stage1 = nn.Sequential(*layers_list)
    feature_extractor_stage2 = nn.Sequential(resnet18_model.layer3)



    return feature_extractor_stage1, feature_extractor_stage2


class ResNetBackBone(nn.Module):
    def __init__(self, is_frozen=False,resnet_name='resnet18'):
        super(ResNetBackBone, self).__init__()
        if resnet_name == 'resnet18':
            resnet18_model = models.resnet.resnet18(pretrained=True)
        elif  resnet_name == 'resnet34':
            resnet18_model = models.resnet.resnet34(pretrained=True)
        elif  resnet_name == 'resnet50':
            resnet18_model = models.resnet.resnet50(pretrained=True)

        if torch.cuda.is_available():
            resnet18_model = resnet18_model.cuda()
        self.feature_extractor_stage1, self.feature_extractor_stage2 = get_res18_FeatureMap(resnet18_model)

        if is_frozen:
            for i, (name, param) in enumerate(self.feature_extractor_stage1.named_parameters()):
                param.requires_grad = False

            for i, (name, param) in enumerate(self.feature_extractor_stage2.named_parameters()):
                param.requires_grad = False

    def forward(self, input1_tesnor, input2_tesnor):
        feature_1_64 = self.feature_extractor_stage1(input1_tesnor)
        feature_1_32 = self.feature_extractor_stage2(feature_1_64)
        feature_2_64 = self.feature_extractor_stage1(input2_tesnor)
        feature_2_32 = self.feature_extractor_stage2(feature_2_64)
        return feature_1_64,feature_1_32,feature_2_64,feature_2_32


# define and forward
class Network(nn.Module):

    def __init__(self, frozen_backbone='resnet18', activate_backbone='resnet18'):
        super(Network, self).__init__()

        tps_in_channel =256
        if activate_backbone == 'resnet18':
            tps_in_channel = 256
        elif activate_backbone == 'resnet34':
            tps_in_channel = 512
        elif activate_backbone == 'resnet50':
            tps_in_channel = 1024
        elif activate_backbone == 'resnet50_moco':
            tps_in_channel = 1024
        elif activate_backbone == 'resnet50_dino':
            tps_in_channel = 1024

        self.regressHomo_part1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # 12, 20

            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # 6, 10

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
            # 3, 5
        )

        self.regressHomo_part2 = nn.Sequential(
            nn.Linear(in_features=2048, out_features=512, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=512, out_features=128, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=128, out_features=8, bias=True)
        )

        self.regressTPS_part1 = nn.Sequential(
            nn.Conv2d(tps_in_channel, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # 23, 40

            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # 12, 20

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # 6, 10

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
            # 3, 5
        )

        self.regressTPS_part2 = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=1024, out_features=512, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=512, out_features=(grid_w+1)*(grid_h+1)*2, bias=True)
        )



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        #-----------------------------------------
        self.fronzen_resnet_backbone = ResNetBackBone(is_frozen=True, resnet_name=frozen_backbone)
        self.active_resnet_backbone = ResNetBackBone(is_frozen=False, resnet_name=activate_backbone)
        

    # forward
    def forward(self, input1_tesnor, input2_tesnor, coef, alpha=0.5):
        batch_size, _, img_h, img_w = input1_tesnor.size()

        # feature pyramid
        fronzen_feature_1_64,fronzen_feature_1_32,fronzen_feature_2_64,fronzen_feature_2_32 = self.fronzen_resnet_backbone(input1_tesnor, input2_tesnor)
        active_feature_1_64,active_feature_1_32,active_feature_2_64,active_feature_2_32 = self.active_resnet_backbone(input1_tesnor, input2_tesnor)

        ######### stage 1
        # random fusion 
        fronzen_correlation_32 = self.CCL(fronzen_feature_1_32, fronzen_feature_2_32)
        active_correlation_32 = self.CCL(active_feature_1_32, active_feature_2_32)
        correlation_32 = (1 - alpha) * fronzen_correlation_32 + alpha * active_correlation_32
        # homography regression 
        temp_1 = self.regressHomo_part1(correlation_32)
        temp_1 = temp_1.view(temp_1.size()[0], -1)
        offset_1 = self.regressHomo_part2(temp_1)

        # homo decomposition
        H_motion_1 = offset_1.reshape(-1, 4, 2)
        src_p = torch.tensor([[0., 0.], [img_w, 0.], [0., img_h], [img_w, img_h]])
        if torch.cuda.is_available():
            src_p = src_p.cuda()
        src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)
        dst_p = src_p + H_motion_1
        dst_p_tgt = src_p + H_motion_1 * coef
        H = torch_DLT.Tensor_ACA_rect(src_p/8, dst_p/8)
        H_tgt = torch_DLT.Tensor_ACA_rect(src_p/8, dst_p_tgt/8)
        H_ref = torch.matmul(torch.inverse(H), H_tgt)

        M_tensor = torch.tensor([[img_w/8 / 2.0, 0., img_w/8 / 2.0],
                      [0., img_h/8 / 2.0, img_h/8 / 2.0],
                      [0., 0., 1.]]).cuda()
        M_tile = M_tensor.unsqueeze(0).expand(batch_size, -1, -1)
        M_tensor_inv = torch.inverse(M_tensor)
        M_tile_inv = M_tensor_inv.unsqueeze(0).expand(batch_size, -1, -1)

        # warping by two homo
        H_mat_ref = torch.matmul(torch.matmul(M_tile_inv, H_ref), M_tile)
        warp_feature_1_64_ref = torch_homo_transform.transformer(active_feature_1_64, H_mat_ref, (int(img_h/8), int(img_w/8)))
        H_mat_tgt = torch.matmul(torch.matmul(M_tile_inv, H_tgt), M_tile)
        warp_feature_2_64_tgt = torch_homo_transform.transformer(active_feature_2_64, H_mat_tgt, (int(img_h/8), int(img_w/8)))

        ######### stage 2: share weights ########
        # for img1
        #img1_correlation_64 = self.cost_volume(warp_feature_2_64_tgt, warp_feature_1_64_ref, search_range=5, norm=False)
        img1_temp_2 = self.regressTPS_part1(torch.cat([warp_feature_1_64_ref, warp_feature_2_64_tgt], 1))
        img1_temp_2 = img1_temp_2.reshape(img1_temp_2.size()[0], -1)
        offset_2_ref = self.regressTPS_part2(img1_temp_2)
        offset_2_ref = torch.clamp(offset_2_ref, min=-img_h/8,max=img_h/8)

        # for img2
        #img2_correlation_64 = self.cost_volume(warp_feature_1_64_ref, warp_feature_2_64_tgt, search_range=5, norm=False)
        img2_temp_2 = self.regressTPS_part1(torch.cat([warp_feature_2_64_tgt, warp_feature_1_64_ref], 1))
        img2_temp_2 = img2_temp_2.reshape(img2_temp_2.size()[0], -1)
        offset_2_tgt = self.regressTPS_part2(img2_temp_2)
        offset_2_tgt = torch.clamp(offset_2_tgt, min=-img_h/8,max=img_h/8)


        return offset_1, offset_2_ref, offset_2_tgt

    @staticmethod
    def cost_volume(x1, x2, search_range, norm=True, fast=True):
        if norm:
            x1 = F.normalize(x1, p=2, dim=1)
            x2 = F.normalize(x2, p=2, dim=1)
        bs, c, h, w = x1.shape
        padded_x2 = F.pad(x2, [search_range] * 4)  # [b,c,h,w] -> [b,c,h+sr*2,w+sr*2]
        max_offset = search_range * 2 + 1

        if fast:
            # faster(*2) but cost higher(*n) GPU memory
            patches = F.unfold(padded_x2, (max_offset, max_offset)).reshape(bs, c, max_offset ** 2, h, w)
            cost_vol = (x1.unsqueeze(2) * patches).mean(dim=1, keepdim=False)
        else:
            # slower but save memory
            cost_vol = []
            for j in range(0, max_offset):
                for i in range(0, max_offset):
                    x2_slice = padded_x2[:, :, j:j + h, i:i + w]
                    cost = torch.mean(x1 * x2_slice, dim=1, keepdim=True)
                    cost_vol.append(cost)
            cost_vol = torch.cat(cost_vol, dim=1)

        cost_vol = F.leaky_relu(cost_vol, 0.1)

        return cost_vol


    def extract_patches(self, x, kernel=3, stride=1):
        if kernel != 1:
            x = nn.ZeroPad2d(1)(x)
        x = x.permute(0, 2, 3, 1)
        all_patches = x.unfold(1, kernel, stride).unfold(2, kernel, stride)
        return all_patches


    def CCL(self, feature_1, feature_2):
        bs, c, h, w = feature_1.size()

        norm_feature_1 = F.normalize(feature_1, p=2, dim=1)
        norm_feature_2 = F.normalize(feature_2, p=2, dim=1)
        #print(norm_feature_2.size())

        patches = self.extract_patches(norm_feature_2)
        if torch.cuda.is_available():
            patches = patches.cuda()

        matching_filters  = patches.reshape((patches.size()[0], -1, patches.size()[3], patches.size()[4], patches.size()[5]))

        match_vol = []
        for i in range(bs):
            single_match = F.conv2d(norm_feature_1[i].unsqueeze(0), matching_filters[i], padding=1)
            match_vol.append(single_match)

        match_vol = torch.cat(match_vol, 0)
        #print(match_vol .size())

        # scale softmax
        softmax_scale = 10
        match_vol = F.softmax(match_vol*softmax_scale,1)

        channel = match_vol.size()[1]

        h_one = torch.linspace(0, h-1, h)
        one1w = torch.ones(1, w)
        if torch.cuda.is_available():
            h_one = h_one.cuda()
            one1w = one1w.cuda()
        h_one = torch.matmul(h_one.unsqueeze(1), one1w)
        h_one = h_one.unsqueeze(0).unsqueeze(0).expand(bs, channel, -1, -1)

        w_one = torch.linspace(0, w-1, w)
        oneh1 = torch.ones(h, 1)
        if torch.cuda.is_available():
            w_one = w_one.cuda()
            oneh1 = oneh1.cuda()
        w_one = torch.matmul(oneh1, w_one.unsqueeze(0))
        w_one = w_one.unsqueeze(0).unsqueeze(0).expand(bs, channel, -1, -1)

        c_one = torch.linspace(0, channel-1, channel)
        if torch.cuda.is_available():
            c_one = c_one.cuda()
        c_one = c_one.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(bs, -1, h, w)

        flow_h = match_vol*(c_one//w - h_one)
        flow_h = torch.sum(flow_h, dim=1, keepdim=True)
        flow_w = match_vol*(c_one%w - w_one)
        flow_w = torch.sum(flow_w, dim=1, keepdim=True)

        feature_flow = torch.cat([flow_w, flow_h], 1)
        #print(flow.size())

        return feature_flow
