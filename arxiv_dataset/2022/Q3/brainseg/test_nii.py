import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import test_images, test_single_nii
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import SimpleITK as sitk
from data_reader import H5DataLoader
import h5py
import uuid



def inference(input_nii, model, output_fname=None, do_bfc=True):
    model.eval()

    # run N4 bias field correction
    inputImage = sitk.ReadImage(input_nii, sitk.sitkFloat32)
    nii_filename = str(uuid.uuid4())+'.nii.gz'

    if do_bfc:
        maskImage = sitk.OtsuThreshold(inputImage, 0, 1, 200)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrected_image = corrector.Execute(inputImage, maskImage)
        sitk.WriteImage(corrected_image, nii_filename)
    else:
        sitk.WriteImage(inputImage, nii_filename)


    test_single_nii(nii_filename, net, patch_size=[256, 256], output_fname=output_fname)

    os.remove(nii_filename)



if __name__ == "__main__":

    seed = 1234
    vit_name='R50-ViT-B_16'
    num_classes = 9
    is_pretrain = True
    n_skip = 3
    img_size = 256
    vit_patches_size = 16    
    #snapshot = '/project/ajoshi_27/code_farm/brainseg/model/T1_SkullScalp_t1256/TU_R50-ViT-B_16_skip3_30k_epo150_bs16_256/epoch_10.pth'
    #snapshot = '/project/ajoshi_27/code_farm/brainseg/model/T1T2_SkullScalp_t1t2256/TU_R50-ViT-B_16_skip3_30k_epo150_bs16_256/epoch_10.pth' #os.path.join(snapshot_path, 'best_model.pth')
    snapshot = '/scratch1/wenhuicu/brainseg/model/T1_SkullScalp_t1256/TU_R50-ViT-B_16_skip3_30k_epo150_bs16_256_BCE_beta_0.0001_warmup_2/epoch_10.pth'
    # snapshot = '/home1/ajoshi/epoch_10.pth'

    input_nii = '/project/ajoshi_27/headreco_out/m2m_sub-CC110033/T1fs_conform.nii.gz'
    output_file = 'm2m_sub-CC110033_robust.label.nii.gz'

    cudnn.benchmark = True
    cudnn.deterministic = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #torch.cuda.manual_seed(args.seed)




    config_vit = CONFIGS_ViT_seg[vit_name]
    config_vit.n_classes = num_classes
    config_vit.n_skip = n_skip
    config_vit.patches.size = (vit_patches_size, vit_patches_size)
    if vit_name.find('R50') !=-1:
        config_vit.patches.grid = (int(img_size/vit_patches_size), int(img_size/vit_patches_size))
    net = ViT_seg(config_vit, img_size=img_size, num_classes=config_vit.n_classes).cuda()


    net.load_state_dict(torch.load(snapshot,map_location=torch.device('cuda')))


    inference(input_nii, net, output_fname=output_file, do_bfc=False)



