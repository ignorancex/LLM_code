import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import argparse
import time
import numpy as np

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from utils.utils import calculate_accuracy, calculate_result
from train import model_dir, model_dir2

#---------------- MODEL ----------------------
from models.segnet import SegNet
rgb_channels = 3

#---------------- DATASET --------------------
from dataloaders.VARGEM import MVARGEMDataset

from torch.utils.data import DataLoader
root = "/home/deep/NunoCunha/src/"   # path to the root directory of the dataset 
batch_size = 1

def main():
    num_classes=2
    model = SegNet(num_classes, 
                n_init_features = rgb_channels,
                )
                
    cf = np.zeros((num_classes, num_classes))

    #model = eval(args.model_name)(num_classes=num_classes)

    if args.gpu >= 0: model.cuda(args.gpu)
    print('| loading model file %s... ' % final_model_file, end='')
    #map_location=torch.device('cuda:0')
    #map_location=torch.device('cpu')
    #checkpoint = torch.load(final_model_file,map_location=torch.device('cpu'))
    model.load_state_dict(torch.load(final_model_file, map_location=torch.device('cuda:0')))
    print('done!')

    test_loader = MVARGEMDataset(root=root, set='train', rgb_dir = 'RGB', mask_dir = 'Masks', nir_dir = 'NIR', red_dir = 'RED', num_classes = num_classes)

    test_dataloader = DataLoader(test_loader, batch_size=batch_size, shuffle=True)
    
    loss_avg = 0.
    acc_avg  = 0.
    model.eval()
    with torch.no_grad():
        for it, data in enumerate(test_dataloader):
            rgb,nir,red,mask,id = data
            rgb = Variable(rgb)
            #dsm = Variable(dsm).cuda(args.gpu) 
            mask = Variable(mask)

            if args.gpu >= 0:
                rgb = rgb.cuda(args.gpu)
                mask = mask.cuda(args.gpu)
                #dsm = dsm.cuda(args.gpu)

            #inputs = torch.cat([rgb, dsm], dim=1)
            output,_  = model(rgb)

            single_mask = output[0].cpu().numpy()
            single_mask = np.argmax(single_mask, axis=0)

            id = id[0].replace("(", "").replace(")", "").replace(",", "")

            mask_filename = os.path.join(model_dir2,  f'{id}.jpg')
            plt.imsave(mask_filename, single_mask, cmap='gray')

        

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test SegNet with pytorch')
    parser.add_argument('--model_name',  '-M',  type=str, default='SegNet')
    parser.add_argument('--batch_size',  '-B',  type=int, default=batch_size)
    parser.add_argument('--gpu',         '-G',  type=int, default=0)
    parser.add_argument('--num_workers', '-j',  type=int, default=8)
    args = parser.parse_args()

    model_dir        = os.path.join(model_dir, args.model_name)
    final_model_file = os.path.join(model_dir, 'final.pth')
    assert os.path.exists(final_model_file), 'model file `%s` do not exist' % (final_model_file)

    print('| testing %s on GPU #%d with pytorch' % (args.model_name, args.gpu))

    main()