import torch
import torch.nn as nn
import utils.torch_DLT as torch_DLT
import utils.torch_homo_transform as torch_homo_transform
import utils.torch_tps_transform as torch_tps_transform
import ssl
import torch.nn.functional as F
import cv2
import numpy as np
import torchvision.models as models
import utils.torch_homo_ddm as torch_homo_ddm
import utils.torch_homo_flow as torch_homo_flow

import torchvision.transforms as T
resize_512 = T.Resize((512,512))

import grid_res
grid_h = grid_res.GRID_H
grid_w = grid_res.GRID_W

# fitting the translation, 2D rotation and scale to formulate a similarity transformation
def get_SimilarityMatrix(src_p, dst_p):
    print(src_p[0])
    print(dst_p[0])
    #input: bs, 4, 2
    batch_size = src_p.shape[0]
    img_w = 512
    img_h = 512

    #------------------------ get similarity transformation for dst_p_tgt -------------------#
    # | s*cos  -s*sin t_w |  translation: (t_w, t_h)     scale: s
    # | s*sin  s*cos  t_h |  rotation:  cos  -sin
    # |   0      0     1  |             sin   cos
    ## 1: translation
    translation = torch.mean(dst_p - src_p, 1) # bs, 2

    ## 2: scale
    area_dst = 0.5 * torch.abs(
    dst_p[:,0,0] * dst_p[:,1,1] + dst_p[:,1,0] * dst_p[:,3,1] + dst_p[:,3,0] * dst_p[:,2,1] + dst_p[:,2,0] * dst_p[:,0,1] -
    (dst_p[:,0,1] * dst_p[:,1,0] + dst_p[:,1,1] * dst_p[:,3,0] + dst_p[:,3,1] * dst_p[:,2,0] + dst_p[:,2,1] * dst_p[:,0,0]))
    #area_src = 0.5 * torch.abs(
    #src_p[:,0,0] * src_p[:,1,1] + src_p[:,1,0] * src_p[:,3,1] + src_p[:,3,0] * src_p[:,2,1] + src_p[:,2,0] * src_p[:,0,1] -
    #(src_p[:,0,1] * src_p[:,1,0] + src_p[:,1,1] * src_p[:,3,0] + src_p[:,3,1] * src_p[:,2,0] + src_p[:,2,1] * src_p[:,0,0]))
    scale = area_dst / (img_w * img_h)
    # scale = area_dst / area_src

    ## 3: rotation
    center_dst = torch.mean(dst_p, 1).unsqueeze(1) # bs, 1, 2
    vector_dst = dst_p - center_dst          #bs, 4, 2
    center_src = torch.mean(src_p, 1).unsqueeze(1) # bs, 1, 2
    vector_src = src_p - center_src          #bs, 4, 2

    cos_sum = torch.sum(vector_dst * vector_src, 2) / (torch.sqrt(torch.sum(vector_dst*vector_dst, 2)) * torch.sqrt(torch.sum(vector_src*vector_src, 2))+1e-7) # bs, 4
    cos_sum = torch.clip(cos_sum, -1, 1)

    seta = torch.mean(torch.acos(cos_sum), 1) # bs
    #cos_seta = torch.cos(seta)

    print(scale[0])

    S_Matrix = torch.stack([scale*torch.cos(seta), -scale*torch.sin(seta), translation[:,0],
                            scale*torch.sin(seta), scale*torch.cos(seta), translation[:,1],
                            scale*0, scale*0, scale*0+1], 1).reshape(batch_size, 3, 3)
    # -----------------------------------------------------------------------------------------

    return S_Matrix # bs, 3, 3

def get_DistortionPoint(pt):
    # input: pt [bs, 4, 2]

    # ------------------ distance score --------------------
    center = torch.mean(pt, 1).unsqueeze(1) #bs, 1, 2
    diag = torch.sqrt(torch.sum((pt - center)**2, 2)) # bs, 4
    #print(diag.shape)
    diag_min, _  = torch.min(diag, dim=1, keepdim=True)
    #print(type(diag_min))
    DistanceScore =  diag/(diag_min+1e-7)   #bs, 4
    DistanceScore = torch.clamp(DistanceScore, 1, 10)
    # print(DistanceScore[0].max())
    # print(DistanceScore[0].min())



    # ------------------ angle score --------------------
    # 0  1
    # 2  3
    vector01 = pt[:,1,:]-pt[:,0,:]  # bs, 2
    vector02 = pt[:,2,:]-pt[:,0,:]
    cos0 = torch.sum(vector01*vector02, 1)/(torch.sqrt(torch.sum(vector01**2, 1)) *torch.sqrt(torch.sum(vector02**2, 1)))
    vector10 = pt[:,0,:]-pt[:,1,:]
    vector13 = pt[:,3,:]-pt[:,1,:]
    cos1 = torch.sum(vector10*vector13, 1)/(torch.sqrt(torch.sum(vector10**2, 1)) *torch.sqrt(torch.sum(vector13**2, 1)))
    vector20 = pt[:,0,:]-pt[:,2,:]
    vector23 = pt[:,3,:]-pt[:,2,:]
    cos2 = torch.sum(vector20*vector23, 1)/(torch.sqrt(torch.sum(vector20**2, 1)) *torch.sqrt(torch.sum(vector23**2, 1)))
    vector31 = pt[:,1,:]-pt[:,3,:]
    vector32 = pt[:,2,:]-pt[:,3,:]
    cos3 = torch.sum(vector31*vector32, 1)/(torch.sqrt(torch.sum(vector31**2, 1)) *torch.sqrt(torch.sum(vector32**2, 1)))
    cos_cat = torch.stack([cos0, cos1, cos2, cos3], 1)  # bs, 4
    #print(cos_cat[0])
    #cos_max, _ = torch.max(cos_cat, dim=1, keepdim=True)
    #AngleScore = cos_max/(cos_cat+1e-7)
    AngleScore = torch.abs(cos_cat)
    #AngleScore = torch.clamp(AngleScore, 1, 10)

    # print(AngleScore[0].max())
    # print(AngleScore[0].min())

    # --------------- global score ------------
    vector03 = pt[:,3,:]-pt[:,0,:]
    vector21 = pt[:,1,:]-pt[:,2,:]
    cos_global = torch.sum(vector03*vector21, 1)/(torch.sqrt(torch.sum(vector03**2, 1)) *torch.sqrt(torch.sum(vector21**2, 1)))
    global_score = torch.abs(cos_global).unsqueeze(1)


    #DistortionScore = DistanceScore + AngleScore
    # DistortionScore = DistanceScore * (AngleScore+1) + global_score*5

    ## potential version
    DistortionScore = (DistanceScore-1) + AngleScore + global_score
    # print("-----------")
    # print(DistanceScore[0])
    # print(AngleScore[0])
    # print(global_score[0])

    return DistortionScore

# draw mesh on image
# warp: h*w*3
# f_local: grid_h*grid_w*2
def draw_mesh_on_warp(warp, f_local):

    warp = np.ascontiguousarray(warp)

    point_color = (0, 255, 0) # BGR
    thickness = 2
    lineType = 8

    num = 1
    for i in range(grid_h+1):
        for j in range(grid_w+1):

            num = num + 1
            if j == grid_w and i == grid_h:
                continue
            elif j == grid_w:
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i+1,j,0]), int(f_local[i+1,j,1])), point_color, thickness, lineType)
            elif i == grid_h:
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i,j+1,0]), int(f_local[i,j+1,1])), point_color, thickness, lineType)
            else :
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i+1,j,0]), int(f_local[i+1,j,1])), point_color, thickness, lineType)
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i,j+1,0]), int(f_local[i,j+1,1])), point_color, thickness, lineType)

    return warp


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



def H2Offset(H, ori_pt):
    #H_inv = torch.inverse(H)
    ones = torch.ones(ori_pt.size()[0], 4, 1)
    if torch.cuda.is_available():
        ori_pt = ori_pt.cuda()
        ones = ones.cuda()

    ori_pt = torch.cat((ori_pt, ones), 2) # bs*4*3
    tar_pt = torch.matmul(H, ori_pt.permute(0,2,1)) # bs*3*4

    out_x = tar_pt[:,0,:]/tar_pt[:,2,:]
    out_y = tar_pt[:,1,:]/tar_pt[:,2,:]
    out = torch.stack((out_x, out_y), 2)

    return out

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


def build_model(net, coef_net, input1_tensor, input2_tensor, is_training = True):
    batch_size, _, img_h, img_w = input1_tensor.size()
    alpha = torch.rand((batch_size, 2, 1, 1)).cuda()
    # network
    if is_training == True:
        aug_input1_tensor, aug_input2_tensor = data_aug(input1_tensor, input2_tensor)
        dst_p_ref_list, dst_p_tgt_list, final_coef_list, mesh_ref, mesh_tgt = net(aug_input1_tensor, aug_input2_tensor, coef_net, alpha)
    else:
        dst_p_ref_list, dst_p_tgt_list, final_coef_list, mesh_ref, mesh_tgt = net(input1_tensor, input2_tensor, coef_net, alpha)

    DDM_ref_list = []
    DDM_tgt_list = []
    for i in range(len(dst_p_ref_list)):
        # get ddm for ref/tgt
        DisPt_tgt = get_DistortionPoint(dst_p_tgt_list[i])    # bs, 4
        DDM_tgt = torch_homo_ddm.transformer(input2_tensor, DisPt_tgt)
        DisPt_ref = get_DistortionPoint(dst_p_ref_list[i])    # bs, 4
        DDM_ref = torch_homo_ddm.transformer(input2_tensor, DisPt_ref)
        DDM_tgt_list.append(DDM_tgt)
        DDM_ref_list.append(DDM_ref)

    #############################################################

    ##### stage 2 ####
    # Do not use the following code to save GPU memory.

    # rigid_mesh = get_rigid_mesh(batch_size, img_h, img_w)
    # norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
    # norm_mesh_ref = get_norm_mesh(mesh_ref, img_h, img_w)
    # norm_mesh_tgt = get_norm_mesh(mesh_tgt, img_h, img_w)

    # mask = torch.ones_like(input2_tensor).cuda()
    # output_tps_ref = torch_tps_transform.transformer(torch.cat((input1_tensor, mask), 1), norm_mesh_ref, norm_rigid_mesh, (img_h, img_w))
    # output_tps_tgt = torch_tps_transform.transformer(torch.cat((input2_tensor, mask), 1), norm_mesh_tgt, norm_rigid_mesh, (img_h, img_w))

    out_dict = {}
    # out_dict.update(output_tps_ref = output_tps_ref, mesh_ref = mesh_ref)
    # out_dict.update(output_tps_tgt = output_tps_tgt, mesh_tgt = mesh_tgt)
    out_dict.update(mesh_ref = mesh_ref)
    out_dict.update(mesh_tgt = mesh_tgt)
    out_dict.update(DDM_ref_list=DDM_ref_list, DDM_tgt_list=DDM_tgt_list)

    return out_dict



# for train_ft.py
def build_new_ft_model(net, coef_net, input1_tensor, input2_tensor, alpha=0.5):
    batch_size, _, img_h, img_w = input1_tensor.size()

    dst_p_ref_list, dst_p_tgt_list, final_coef_list, mesh_motion_ref, mesh_motion_tgt = net.forward_finetune(input1_tensor, input2_tensor, coef_net, alpha)

    mesh_motion_ref = mesh_motion_ref.reshape(-1, grid_h+1, grid_w+1, 2)
    mesh_motion_tgt = mesh_motion_tgt.reshape(-1, grid_h+1, grid_w+1, 2)
    #mesh_motion = torch.stack([mesh_motion[...,0]*img_w/512, mesh_motion[...,1]*img_h/512], 3)

    ##### stage 2 ####
    rigid_mesh = get_rigid_mesh(batch_size, img_h, img_w)
    mesh_ref =  mesh_motion_ref
    mesh_tgt = mesh_motion_tgt

    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
    norm_mesh_ref = get_norm_mesh(mesh_ref, img_h, img_w)
    norm_mesh_tgt = get_norm_mesh(mesh_tgt, img_h, img_w)

    mask = torch.ones_like(input2_tensor)
    if torch.cuda.is_available():
        mask = mask.cuda()
    output_tps_ref = torch_tps_transform.transformer(torch.cat((input1_tensor+1, mask), 1), norm_mesh_ref, norm_rigid_mesh, (img_h, img_w))
    output_tps_tgt = torch_tps_transform.transformer(torch.cat((input2_tensor+1, mask), 1), norm_mesh_tgt, norm_rigid_mesh, (img_h, img_w))


    out_dict = {}
    out_dict.update(rigid_mesh = rigid_mesh)
    out_dict.update(output_tps_ref = output_tps_ref, mesh_ref = mesh_ref)
    out_dict.update(output_tps_tgt = output_tps_tgt, mesh_tgt = mesh_tgt)

    return out_dict

# for train_ft.py
def get_stitched_result(input1_tensor, input2_tensor, rigid_mesh, mesh_ref, mesh_tgt, max_out_height=4000):
    batch_size, _, img_h, img_w = input1_tensor.size()

    rigid_mesh = torch.stack([rigid_mesh[...,0]*img_w/512, rigid_mesh[...,1]*img_h/512], 3)
    mesh_ref = torch.stack([mesh_ref[...,0]*img_w/512, mesh_ref[...,1]*img_h/512], 3)
    mesh_tgt = torch.stack([mesh_tgt[...,0]*img_w/512, mesh_tgt[...,1]*img_h/512], 3)

    ######################################

    # calculate the size of stitched image
    width_max = torch.maximum(torch.max(mesh_ref[...,0]), torch.max(mesh_tgt[...,0]))
    width_min = torch.minimum(torch.min(mesh_ref[...,0]), torch.min(mesh_tgt[...,0]))
    height_max = torch.maximum(torch.max(mesh_ref[...,1]), torch.max(mesh_tgt[...,1]))
    height_min = torch.minimum(torch.min(mesh_ref[...,1]), torch.min(mesh_tgt[...,1]))

    out_width = width_max - width_min
    out_height = height_max - height_min
   

    # in case of image size is so huge.
    if max(out_height,out_width) >= max_out_height:
        print(out_width)
        print(out_height)
        return None, False

    # convert the mesh from [img_h, img_w] to [out_h, out_w]
    mesh_trans_ref = torch.stack([mesh_ref[...,0]-width_min, mesh_ref[...,1]-height_min], 3)
    mesh_trans_tgt = torch.stack([mesh_tgt[...,0]-width_min, mesh_tgt[...,1]-height_min], 3)

    # normalization
    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
    norm_mesh_trans_ref = get_norm_mesh(mesh_trans_ref, out_height, out_width)
    norm_mesh_trans_tgt = get_norm_mesh(mesh_trans_tgt, out_height, out_width)

    # transformation
    mask = torch.ones_like(input2_tensor).cuda()
    output_ref = torch_tps_transform.transformer(torch.cat((input1_tensor+1, mask), 1), norm_mesh_trans_ref, norm_rigid_mesh, (out_height.int(), out_width.int()))
    output_tgt = torch_tps_transform.transformer(torch.cat((input2_tensor+1, mask), 1), norm_mesh_trans_tgt, norm_rigid_mesh, (out_height.int(), out_width.int()))

    img1 = output_ref[0,0:3,...]
    img2 = output_tgt[0,0:3,...]
    stitched =  img1*(img1/(img1+img2+1e-6)) + img2*(img2/(img1+img2+1e-6))

    out_dict = {}
    out_dict.update(output_ref=output_ref, output_tgt = output_tgt)
    out_dict.update(stitched=stitched)

    return out_dict, True


# for test_output.py
def build_output_model(net, coef_net, input1_tensor, input2_tensor, alpha=0.5, max_out_height=1200):
    batch_size, _, img_h, img_w = input1_tensor.size()

    # input resize
    resized_input1 = resize_512(input1_tensor)
    resized_input2 = resize_512(input2_tensor)
    dst_p_ref_list, dst_p_tgt_list, final_coef_list, mesh_ref, mesh_tgt = net(resized_input1, resized_input2, coef_net, alpha)
    
    # print('final_coef_list:',final_coef_list[-1])
    # get ddm for ref/tgt
    DisPt_tgt = get_DistortionPoint(dst_p_tgt_list[-1])    # bs, 4
    DDM_tgt = torch_homo_ddm.transformer(input2_tensor, DisPt_tgt)
    DisPt_ref = get_DistortionPoint(dst_p_ref_list[-1])    # bs, 4
    DDM_ref = torch_homo_ddm.transformer(input2_tensor, DisPt_ref)

    # then, calculate the final mesh
    rigid_mesh = get_rigid_mesh(batch_size, img_h, img_w)

    # calculate the size of stitched image
    mesh_ref = torch.stack([mesh_ref[...,0]*img_w/512, mesh_ref[...,1]*img_h/512], 3)
    mesh_tgt = torch.stack([mesh_tgt[...,0]*img_w/512, mesh_tgt[...,1]*img_h/512], 3)

    width_max = torch.maximum(torch.max(mesh_ref[...,0]), torch.max(mesh_tgt[...,0]))
    width_min = torch.minimum(torch.min(mesh_ref[...,0]), torch.min(mesh_tgt[...,0]))
    height_max = torch.maximum(torch.max(mesh_ref[...,1]), torch.max(mesh_tgt[...,1]))
    height_min = torch.minimum(torch.min(mesh_ref[...,1]), torch.min(mesh_tgt[...,1]))

    out_width = width_max - width_min
    out_height = height_max - height_min

    # in case of image size is so huge.
    if max(out_height,out_width) >= max_out_height:
        return None, False

    # convert the mesh from [img_h, img_w] to [out_h, out_w]
    mesh_trans_ref = torch.stack([mesh_ref[...,0]-width_min, mesh_ref[...,1]-height_min], 3)
    mesh_trans_tgt = torch.stack([mesh_tgt[...,0]-width_min, mesh_tgt[...,1]-height_min], 3)

    # normalization
    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
    norm_mesh_trans_ref = get_norm_mesh(mesh_trans_ref, out_height, out_width)
    norm_mesh_trans_tgt = get_norm_mesh(mesh_trans_tgt, out_height, out_width)

    # transformation
    mask = torch.ones_like(input2_tensor).cuda()
    output_ref = torch_tps_transform.transformer(torch.cat((input1_tensor+1, mask), 1), norm_mesh_trans_ref, norm_rigid_mesh, (out_height.int(), out_width.int()))
    output_tgt = torch_tps_transform.transformer(torch.cat((input2_tensor+1, mask), 1), norm_mesh_trans_tgt, norm_rigid_mesh, (out_height.int(), out_width.int()))


    out_dict = {}
    out_dict.update(output_ref=output_ref, output_tgt = output_tgt)
    out_dict.update(DDM_ref=DDM_ref, DDM_tgt = DDM_tgt)

    return out_dict, True




def get_res18_FeatureMap(resnet18_model):
    layers_list = []

    layers_list.append(resnet18_model.conv1)    #stride 2*2     H/2
    layers_list.append(resnet18_model.bn1)
    layers_list.append(resnet18_model.relu)
    layers_list.append(resnet18_model.maxpool)  #stride 2       H/4

    layers_list.append(resnet18_model.layer1)                  #H/4
    layers_list.append(resnet18_model.layer2)                  #H/8

    feature_extractor_stage1 = nn.Sequential(*layers_list)
    feature_extractor_stage2 = nn.Sequential(resnet18_model.layer3)

    return feature_extractor_stage1, feature_extractor_stage2


def get_feature_down8(resnet18_model):
    # feature extractor

    layers_list = []

    layers_list.append(resnet18_model.conv1) #stride 2*2     H/2
    layers_list.append(resnet18_model.bn1)
    layers_list.append(resnet18_model.relu)
    layers_list.append(resnet18_model.maxpool)  #stride 2       H/4

    layers_list.append(resnet18_model.layer1)                  #H/4
    layers_list.append(resnet18_model.layer2)                  #H/8

    coefficient_generator = nn.Sequential(*layers_list)

    return coefficient_generator

def get_motion_down8(resnet18_model):
    # feature extractor

    layers_list = []

    layers_list.append(nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)) #stride 2*2     H/2
    layers_list.append(resnet18_model.bn1)
    layers_list.append(resnet18_model.relu)
    layers_list.append(resnet18_model.maxpool)  #stride 2       H/4

    layers_list.append(resnet18_model.layer1)                  #H/4
    layers_list.append(resnet18_model.layer2)                  #H/8

    coefficient_generator = nn.Sequential(*layers_list)

    return coefficient_generator

# define and forward
class CoefNetwork(nn.Module):

    def __init__(self):
        super(CoefNetwork, self).__init__()


        self.regressCoef_part1 = nn.Sequential(
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=(2, 2), bias=False),   #32
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),     #16
            # 12, 20

            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),     #8
            # 6, 10

            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)      # 4
            # 3, 5
        )

        self.regressCoef_part2 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=128, bias=True),
            nn.Dropout(0.2 ),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=128, out_features=4, bias=True),

        )



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        resnet18_model = models.resnet.resnet18(weights=None)
        if torch.cuda.is_available():
            resnet18_model = resnet18_model.cuda()
        #print(resnet18_model)
        self.get_feature_map = get_feature_down8(resnet18_model)
        self.get_motion_map = get_motion_down8(resnet18_model)


    # forward
    def forward(self, input1_tesnor, input2_tesnor, H_motion):
        batch_size, _, img_h, img_w = input1_tesnor.size()
        coef_list = []
        #final_coef_list = []
        dst_p_ref_list = []
        dst_p_tgt_list = []
        final_coef_list = []

        feat1 = self.get_feature_map(input1_tesnor)
        feat2 = self.get_feature_map(input2_tesnor)
        H_motion_tgt = H_motion
        H_motion_ref = H_motion*0
        homo_embed1 = self.homography_embedding(H_motion_ref)
        homo_embed2 = self.homography_embedding(H_motion_tgt)


        for i in range(3):
            motion1 = self.get_motion_map(homo_embed1)
            motion2 = self.get_motion_map(homo_embed2)

            temp = self.regressCoef_part1(torch.cat([feat1, motion1, feat2, motion2], 1))
            coef = self.regressCoef_part2(temp.reshape(temp.size()[0], -1))
            coef_list.append(coef.unsqueeze(2))
            #print(coef_list[-1].shape)
            final_coef = torch.sigmoid(torch.sum(torch.stack(coef_list), dim=0))
            # print('final_coef:',final_coef)
            final_coef_list.append(final_coef)

            # update H_motion
            src_p = torch.tensor([[0., 0.], [img_w, 0.], [0., img_h], [img_w, img_h]])
            if torch.cuda.is_available():
                src_p = src_p.cuda()
            src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)
            dst_p = src_p + H_motion
            dst_p_tgt = src_p + H_motion*final_coef                              
            H = torch_DLT.Tensor_ACA_rect(src_p, dst_p)
            H_tgt = torch_DLT.Tensor_ACA_rect(src_p, dst_p_tgt)
            H_ref = torch.matmul(torch.inverse(H), H_tgt)
            dst_p_ref = H2Offset(H_ref, src_p)
            homo_embed1 = self.homography_embedding(dst_p_ref)
            homo_embed2 = self.homography_embedding(dst_p_tgt)
            dst_p_ref_list.append(dst_p_ref)
            dst_p_tgt_list.append(dst_p_tgt)

        return dst_p_ref_list, dst_p_tgt_list, final_coef_list



    @staticmethod
    def homography_embedding(H_motion):
        batch_size = H_motion.size()[0]
        img_w = 512
        img_h = 512
        # ----------- embedding homo ------------
        H_motion = H_motion.reshape(-1, 4, 2)
        # initialize the source points bs x 4 x 2
        src_p = torch.tensor([[0., 0.], [img_w, 0.], [0., img_h], [img_w, img_h]]).cuda()
        src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)
        dst_p = src_p + H_motion
        H = torch_DLT.Tensor_ACA_rect(src_p, dst_p)
        M_tensor = torch.tensor([[img_w / 2.0, 0., img_w / 2.0],
                          [0., img_h / 2.0, img_h / 2.0],
                          [0., 0., 1.]]).cuda()
        M_tile = M_tensor.unsqueeze(0).expand(batch_size, -1, -1)
        M_tensor_inv = torch.inverse(M_tensor)
        M_tile_inv = M_tensor_inv.unsqueeze(0).expand(batch_size, -1, -1)
        homo_embed = torch_homo_flow.transformer((batch_size, img_h, img_w), torch.matmul(torch.matmul(M_tile_inv, H), M_tile))

        return homo_embed


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
        #-----------------------------------------

    # forward
    def forward(self, input1_tesnor, input2_tesnor, coef_net, alpha=0.5):
        batch_size, _, img_h, img_w = input1_tesnor.size()

        with torch.no_grad():
            fronzen_feature_1_64,fronzen_feature_1_32,fronzen_feature_2_64,fronzen_feature_2_32 = self.fronzen_resnet_backbone(input1_tesnor, input2_tesnor)
            active_feature_1_64,active_feature_1_32,active_feature_2_64,active_feature_2_32 = self.active_resnet_backbone(input1_tesnor, input2_tesnor)

            ######### stage 1
            # for img2
            fronzen_correlation_32 = self.CCL(fronzen_feature_1_32, fronzen_feature_2_32)
            active_correlation_32 = self.CCL(active_feature_1_32, active_feature_2_32)
            correlation_32 = (1 - alpha) * fronzen_correlation_32 + alpha * active_correlation_32
            temp_1 = self.regressHomo_part1(correlation_32)
            temp_1 = temp_1.view(temp_1.size()[0], -1)
            offset_1 = self.regressHomo_part2(temp_1)

        # homo decomposition
        H_motion_1 = offset_1.reshape(-1, 4, 2)                                 # newly added
        dst_p_ref_list, dst_p_tgt_list,final_coef_list = coef_net(input1_tesnor, input2_tesnor, H_motion_1)        # newly added
        dst_p_ref = dst_p_ref_list[-1]
        dst_p_tgt = dst_p_tgt_list[-1]
        final_coef = final_coef_list[-1]

        # for img offset
        src_p = torch.tensor([[0., 0.], [img_w, 0.], [0., img_h], [img_w, img_h]]).cuda()
        src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)

        H_ref = torch_DLT.Tensor_ACA_rect(src_p, dst_p_ref)
        H_tgt = torch_DLT.Tensor_ACA_rect(src_p, dst_p_tgt)
        rigid_mesh = get_rigid_mesh(batch_size, img_h, img_w)
        ini_mesh_ref = H2Mesh(H_ref, rigid_mesh)
        ini_mesh_tgt = H2Mesh(H_tgt, rigid_mesh)

        # for feature offset and warp
        #H = torch_DLT.Tensor_ACA_rect(src_p/8, dst_p/8)
        H_ref_feat = torch_DLT.Tensor_ACA_rect(src_p/8, dst_p_ref/8)
        H_tgt_feat = torch_DLT.Tensor_ACA_rect(src_p/8, dst_p_tgt/8)
        # H_ref = torch.matmul(torch.inverse(H), H_tgt)

        M_tensor = torch.tensor([[img_w/8 / 2.0, 0., img_w/8 / 2.0],
                      [0., img_h/8 / 2.0, img_h/8 / 2.0],
                      [0., 0., 1.]]).cuda()
        M_tile = M_tensor.unsqueeze(0).expand(batch_size, -1, -1)
        M_tensor_inv = torch.inverse(M_tensor)
        M_tile_inv = M_tensor_inv.unsqueeze(0).expand(batch_size, -1, -1)

        # warping by two homo
        H_mat_ref = torch.matmul(torch.matmul(M_tile_inv, H_ref_feat), M_tile)
        warp_feature_1_64_ref = torch_homo_transform.transformer(active_feature_1_64, H_mat_ref, (int(img_h/8), int(img_w/8)))
        H_mat_tgt = torch.matmul(torch.matmul(M_tile_inv, H_tgt_feat), M_tile)
        warp_feature_2_64_tgt = torch_homo_transform.transformer(active_feature_2_64, H_mat_tgt, (int(img_h/8), int(img_w/8)))

        ######### stage 2
        with torch.no_grad():
            # for img1
            # img1_correlation_64 = self.cost_volume(warp_feature_2_64_tgt, warp_feature_1_64_ref, search_range=5, norm=False)
            img1_temp_2 = self.regressTPS_part1(torch.cat([warp_feature_1_64_ref, warp_feature_2_64_tgt], 1))
            img1_temp_2 = img1_temp_2.reshape(img1_temp_2.size()[0], -1)
            offset_2_ref = self.regressTPS_part2(img1_temp_2)
            # for img2
            # img2_correlation_64 = self.cost_volume(warp_feature_1_64_ref, warp_feature_2_64_tgt, search_range=5, norm=False)
            img2_temp_2 = self.regressTPS_part1(torch.cat([warp_feature_2_64_tgt, warp_feature_1_64_ref], 1))
            img2_temp_2 = img2_temp_2.reshape(img2_temp_2.size()[0], -1)
            offset_2_tgt = self.regressTPS_part2(img2_temp_2)
            # post process
            mesh_motion_ref = offset_2_ref.reshape(-1, grid_h+1, grid_w+1, 2)
            mesh_motion_tgt = offset_2_tgt.reshape(-1, grid_h+1, grid_w+1, 2)
            mesh_ref = ini_mesh_ref + mesh_motion_ref
            mesh_tgt = ini_mesh_tgt + mesh_motion_tgt


        return dst_p_ref_list, dst_p_tgt_list, final_coef_list,  mesh_ref, mesh_tgt

        # forward
    def forward_finetune(self, input1_tesnor, input2_tesnor, coef_net, alpha=0.5):
        batch_size, _, img_h, img_w = input1_tesnor.size()

        fronzen_feature_1_64,fronzen_feature_1_32,fronzen_feature_2_64,fronzen_feature_2_32 = self.fronzen_resnet_backbone(input1_tesnor, input2_tesnor)
        active_feature_1_64,active_feature_1_32,active_feature_2_64,active_feature_2_32 = self.active_resnet_backbone(input1_tesnor, input2_tesnor)

        ######### stage 1
        # for img2
        fronzen_correlation_32 = self.CCL(fronzen_feature_1_32, fronzen_feature_2_32)
        active_correlation_32 = self.CCL(active_feature_1_32, active_feature_2_32)
        correlation_32 = (1 - alpha) * fronzen_correlation_32 + alpha * active_correlation_32
        temp_1 = self.regressHomo_part1(correlation_32)
        temp_1 = temp_1.view(temp_1.size()[0], -1)
        offset_1 = self.regressHomo_part2(temp_1)

        # homo decomposition
        H_motion_1 = offset_1.reshape(-1, 4, 2)                                
        dst_p_ref_list, dst_p_tgt_list,final_coef_list = coef_net(input1_tesnor, input2_tesnor, H_motion_1)        
        dst_p_ref = dst_p_ref_list[-1]
        dst_p_tgt = dst_p_tgt_list[-1]
        final_coef = final_coef_list[-1]

        # for img offset
        src_p = torch.tensor([[0., 0.], [img_w, 0.], [0., img_h], [img_w, img_h]]).cuda()
        src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)

        H_ref = torch_DLT.Tensor_ACA_rect(src_p, dst_p_ref)
        H_tgt = torch_DLT.Tensor_ACA_rect(src_p, dst_p_tgt)
        rigid_mesh = get_rigid_mesh(batch_size, img_h, img_w)
        ini_mesh_ref = H2Mesh(H_ref, rigid_mesh)
        ini_mesh_tgt = H2Mesh(H_tgt, rigid_mesh)

        # for feature offset and warp
        #H = torch_DLT.Tensor_ACA_rect(src_p/8, dst_p/8)
        H_ref_feat = torch_DLT.Tensor_ACA_rect(src_p/8, dst_p_ref/8)
        H_tgt_feat = torch_DLT.Tensor_ACA_rect(src_p/8, dst_p_tgt/8)
        # H_ref = torch.matmul(torch.inverse(H), H_tgt)

        M_tensor = torch.tensor([[img_w/8 / 2.0, 0., img_w/8 / 2.0],
                      [0., img_h/8 / 2.0, img_h/8 / 2.0],
                      [0., 0., 1.]]).cuda()
        M_tile = M_tensor.unsqueeze(0).expand(batch_size, -1, -1)
        M_tensor_inv = torch.inverse(M_tensor)
        M_tile_inv = M_tensor_inv.unsqueeze(0).expand(batch_size, -1, -1)

        # warping by two homo
        H_mat_ref = torch.matmul(torch.matmul(M_tile_inv, H_ref_feat), M_tile)
        warp_feature_1_64_ref = torch_homo_transform.transformer(active_feature_1_64, H_mat_ref, (int(img_h/8), int(img_w/8)))
        H_mat_tgt = torch.matmul(torch.matmul(M_tile_inv, H_tgt_feat), M_tile)
        warp_feature_2_64_tgt = torch_homo_transform.transformer(active_feature_2_64, H_mat_tgt, (int(img_h/8), int(img_w/8)))

        ######### stage 2
        # for img1
        # img1_correlation_64 = self.cost_volume(warp_feature_2_64_tgt, warp_feature_1_64_ref, search_range=5, norm=False)
        img1_temp_2 = self.regressTPS_part1(torch.cat([warp_feature_1_64_ref, warp_feature_2_64_tgt], 1))
        img1_temp_2 = img1_temp_2.reshape(img1_temp_2.size()[0], -1)
        offset_2_ref = self.regressTPS_part2(img1_temp_2)
        # for img2
        # img2_correlation_64 = self.cost_volume(warp_feature_1_64_ref, warp_feature_2_64_tgt, search_range=5, norm=False)
        img2_temp_2 = self.regressTPS_part1(torch.cat([warp_feature_2_64_tgt, warp_feature_1_64_ref], 1))
        img2_temp_2 = img2_temp_2.reshape(img2_temp_2.size()[0], -1)
        offset_2_tgt = self.regressTPS_part2(img2_temp_2)
        # post process
        mesh_motion_ref = offset_2_ref.reshape(-1, grid_h+1, grid_w+1, 2)
        mesh_motion_tgt = offset_2_tgt.reshape(-1, grid_h+1, grid_w+1, 2)
        mesh_ref = ini_mesh_ref + mesh_motion_ref
        mesh_tgt = ini_mesh_tgt + mesh_motion_tgt


        return dst_p_ref_list, dst_p_tgt_list, final_coef_list, mesh_ref, mesh_tgt

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
