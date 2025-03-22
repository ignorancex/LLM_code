import argparse
import numpy as np
import torch
import os
import cv2
import model

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
from utils import random_masking



def store_NLL(x, recon, mu, logvar, z):
    # used for NLL test
    with torch.no_grad():
        sigma = torch.exp(0.5*logvar)
        b = x.size(0)
        if x.shape != recon.shape:
            target = Variable(x.data.view(-1) * 255).long()
        else:
            target = x
            _, target= torch.max(target, dim=4)
            target = target.contiguous().view(-1)

        recon = recon.contiguous()
        recon = recon.view(-1,256)

        cross_entropy = F.cross_entropy(recon, target, reduction='none')
        log_p_x_z = -torch.sum(cross_entropy.view(b ,-1), 1)

        log_p_z = -torch.sum(z**2/2+np.log(2*np.pi)/2,1)
        z_eps = (z - mu)/sigma
        z_eps = z_eps.view(opt.repeat,-1)
        log_q_z_x = -torch.sum(z_eps**2/2 + np.log(2*np.pi)/2 + logvar/2, 1)
        
        weights = log_p_x_z+log_p_z-log_q_z_x
        
    return weights

def compute_NLL(weights):
    #used for IWAE

    NLL_loss = -(torch.log(torch.mean(torch.exp(weights - weights.max())))+weights.max())

    return NLL_loss


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='./data', help='path to dataset')

    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--nc', type=int, default=3, help='input image channels')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--test_num', type=int, default=1000)
    parser.add_argument('--lam', type=int, default=1, help='factor for image complexity')

    parser.add_argument('--repeat', type=int, default=1, help='used for IWAE')
    parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
    parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
    
    
    parser.add_argument('--state_E', default='./saved_models/netE.pth', help='path to encoder checkpoint')
    parser.add_argument('--state_G', default='./saved_models/netG.pth', help='path to decoder checkpoint')


    opt = parser.parse_args()
    
    cudnn.benchmark = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    
    dataset_cifar_test = dset.CIFAR10(root=opt.dataroot, download=False,train = False,
                               transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.ToTensor()
                           ]))

    dataset_svhn= dset.SVHN(root = opt.dataroot, download=False,
                                       transform=transforms.Compose([
                                       transforms.Resize(opt.imageSize),
                                       transforms.ToTensor()
                           ]))


    test_loader = torch.utils.data.DataLoader(dataset_cifar_test, batch_size=1,
                                           shuffle=True, num_workers=int(opt.workers))

    
    test_loader_svhn = torch.utils.data.DataLoader(dataset_svhn, batch_size=1,
                                           shuffle=True, num_workers=int(opt.workers))

    
    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    nc = int(opt.nc)
    
    print('Building models...')


    image_size = [32, 32, 3]
    netG = model.Decoder(image_size, nz, 256)
    netG.load_state_dict(torch.load(opt.state_G, map_location = device))

    netE = model.Encoder(image_size, nz)
    netE.load_state_dict(torch.load(opt.state_E, map_location = device))


    netG.to(device)
    netG.eval()
    netE.to(device)
    netE.eval()

    
    loss_fn = nn.CrossEntropyLoss(reduction = 'none')
    
    print('Building complete...')

    difference_indist = []
    for i, (x1, _) in enumerate(test_loader):

        x1 = x1.expand(opt.repeat,-1,-1,-1).contiguous()

        E_agg  = []

        res_list = []

        x1 = x1.to(device)
        b = x1.size(0)

        x2 = F.interpolate(x1, scale_factor=0.25)
        x2 = F.interpolate(x2, scale_factor=4)


        img = (x1[0].clone()).permute(1,2,0)
        img = img.detach().cpu().numpy()
        img *= 255
        img = img.astype(np.uint8)
        img_encoded=cv2.imencode('.png',img,[int(cv2.IMWRITE_PNG_COMPRESSION),9])
        L = len(img_encoded[1])*8

        img = (x2[0].clone()).permute(1,2,0)
        img = img.detach().cpu().numpy()
        img *= 255
        img = img.astype(np.uint8)
        img_encoded=cv2.imencode('.png',img,[int(cv2.IMWRITE_PNG_COMPRESSION),9])
        Ld = len(img_encoded[1])*8

        x = Variable(x1.data.view(-1) * 255).long()
        x_p = Variable(x2[0].data.view(-1) * 255).long()
        xx = torch.zeros((x_p.shape[0], 256)).to(device)
        xx[:,x_p[:]] = 1
        xx = xx.expand(opt.repeat,-1,-1).contiguous().view(-1, 256)


        with torch.no_grad():


            [z,mu,logvar] = netE(x2.float())
            recon = netG(z)



            z = z.view(z.size(0),z.size(1))
            mu = mu.view(mu.size(0),mu.size(1))
            logvar = logvar.view(logvar.size(0), logvar.size(1))

            recon_p1 = recon.contiguous().view(-1,256)

            E = torch.sum(loss_fn(xx, x).view(b ,-1), 1) - torch.sum(loss_fn(recon_p1, x).view(b ,-1), 1) 
            

            E_agg.append(E)
        
                
            E_agg = torch.stack(E_agg).view(-1).mean().detach().cpu().numpy()

            # E is supposed to be positive for ID, no need to add L if E < 0.

            if E_agg >= 0:

                difference_indist.append(E_agg+(L-Ld)* opt.lam)
            else:
                difference_indist.append(E_agg)
            
        if i >= opt.test_num -1:
            break

    difference_indist = np.asarray(difference_indist)

    np.save('./array/ER/testindist.npy', difference_indist)


    

    difference_ood = []
    for i, (x1, _) in enumerate(test_loader_svhn):


        x1 = x1.expand(opt.repeat,-1,-1,-1).contiguous()

        E_agg  = []
        x1 = x1.to(device)
        b = x1.size(0)

        x2 = F.interpolate(x1, scale_factor=0.25)
        x2 = F.interpolate(x2, scale_factor=4)
        

        img = (x1[0].clone()).permute(1,2,0)
        img = img.detach().cpu().numpy()
        img *= 255
        img = img.astype(np.uint8)
        img_encoded=cv2.imencode('.png',img,[int(cv2.IMWRITE_PNG_COMPRESSION),9])
        L = len(img_encoded[1])*8

        img = (x2[0].clone()).permute(1,2,0)
        img = img.detach().cpu().numpy()
        img *= 255
        img = img.astype(np.uint8)
        img_encoded=cv2.imencode('.png',img,[int(cv2.IMWRITE_PNG_COMPRESSION),9])
        Ld = len(img_encoded[1])*8

        x = Variable(x1.data.view(-1) * 255).long()
        x_p = Variable(x2[0].data.view(-1) * 255).long()
        xx = torch.zeros((x_p.shape[0], 256)).to(device)
        xx[:,x_p[:]] = 1
        xx = xx.expand(opt.repeat,-1,-1).contiguous().view(-1, 256)


        with torch.no_grad():


            [z,mu,logvar] = netE(x2.float())
            recon = netG(z)
            
            z = z.view(z.size(0),z.size(1))
            mu = mu.view(mu.size(0),mu.size(1))
            logvar = logvar.view(logvar.size(0), logvar.size(1))

            recon_p1 = recon.contiguous().view(-1,256)

            E = torch.sum(loss_fn(xx, x).view(b ,-1), 1) - torch.sum(loss_fn(recon_p1, x).view(b ,-1), 1) 


            E_agg.append(E)

            E_agg = torch.stack(E_agg).view(-1).mean().detach().cpu().numpy() 

            if E_agg >= 0:

                difference_ood.append(E_agg+(L-Ld)*opt.lam)
            else:
                difference_ood.append(E_agg)
            

        if i >= opt.test_num -1:
            break

    difference_ood = np.asarray(difference_ood)


    np.save('./array/ER/testood.npy', difference_ood)
