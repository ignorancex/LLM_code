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

parser = argparse.ArgumentParser()

parser.add_argument('--beta', type=float, default=0.001,
                    help='ss')
parser.add_argument('--loss', type=str, default='BCE',
                    help='ss')
parser.add_argument('--warmup', type=int, default=0, help='warmup phase, use cross entropy first before robust loss')
parser.add_argument('--suffix', type=str,
                    default='', help='suffix name')

args = parser.parse_args()

if __name__ == "__main__":

    seed = 1234
    vit_name='R50-ViT-B_16'
    num_classes = 9
    is_pretrain = True
    n_skip = 3
    img_size = 256
    vit_patches_size = 16    
    # snapshot = '/project/ajoshi_27/code_farm/brainseg/model/T1_SkullScalp_t1256/TU_R50-ViT-B_16_skip3_30k_epo150_bs16_256/epoch_8.pth'
    snapshot = '/scratch1/wenhuicu/brainseg/model/T1_SkullScalp_t1256/TU_R50-ViT-B_16_skip3_30k_epo150_bs16_256_' + args.loss + '_beta_' + str(args.beta)
    if args.warmup > 0:
        snapshot = snapshot + '_warmup_' + str(args.warmup)
    snapshot = snapshot + args.suffix + '/epoch_10.pth'
    #snapshot = '/project/ajoshi_27/code_farm/brainseg/model/T1T2_SkullScalp_t1t2256/TU_R50-ViT-B_16_skip3_30k_epo150_bs16_256/epoch_10.pth' #os.path.join(snapshot_path, 'best_model.pth')
    test_data_file = '/project/ajoshi_27/code_farm/brainseg/test_t1.h5'
    output_file = '/scratch1/wenhuicu/brainseg/test_t1_output_BCE' + str(args.beta) + args.suffix
    
    if args.warmup > 0:
        output_file = output_file + str(args.warmup)
    output_file = output_file + '.h5'

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


    net.load_state_dict(torch.load(snapshot, map_location=torch.device('cuda')))

    db_loader = H5DataLoader(test_data_file)


    predictions = test_images(db_loader.images, net)
    
    hf = h5py.File(output_file, 'w')
    hf.create_dataset('Y', data=predictions)
    hf.close()



