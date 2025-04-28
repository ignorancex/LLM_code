import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from loss import *

def eval_net(net, loader, device, writer,e):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    n_val = len(loader)  # the number of batch
    tot_L1 = 0
    tot_grad = 0
    tot_model = 0
    count = 0
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['target']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.float32)
            
            with torch.no_grad():
                mask_pred = net(imgs)
                
            if count == int(n_val/2):
                inp = imgs.cpu().numpy()
                im = true_masks.cpu().numpy()
                im_pred = mask_pred.cpu().detach().numpy()
                fig1 = plt.figure(1)
                plt.imshow(np.concatenate((im[0,0,:,:],im_pred[0,0,:,:]),axis=1),cmap=plt.get_cmap('gray'),vmin=0, vmax=1)
                writer.add_figure('T2_test', fig1, e)

            count += 1
            tot_L1 += F.l1_loss(mask_pred, true_masks).item()
            tot_grad += grad_loss(mask_pred, true_masks,device).item()
            
            pbar.update()

    net.train()
    return tot_L1 / n_val, tot_grad / n_val