import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import torch
from nets.net import L_net,fusionnet
from utils import *
import matplotlib.pyplot as plt
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net_fusion=fusionnet().to(device)
net_L=L_net().to(device)


path_model=r"model\ckpt.pth"

path_img1=r"test_case\under"
path_img2=r"test_case\over"
path_result=r"test_case\results"

exposure_adjustment=False #If True, adjust the exposure level according to the preset exposure value.
exposureLevel=0.6

net_fusion.load_state_dict(torch.load(path_model)['net_fusion'])
net_fusion.eval()
net_L.load_state_dict(torch.load(path_model)['net_L'])
net_L.eval()
os.makedirs(path_result,exist_ok=True)
with torch.no_grad():
    for imgname in tqdm(os.listdir(path_img1)):
        img1 = image_read(os.path.join(path_img1,imgname), 'RGB').transpose(2,0,1)[np.newaxis,...]/255.
        img2 = image_read(os.path.join(path_img2, imgname),'RGB').transpose(2,0,1)[np.newaxis,...]/255.
        
        img1, img2 = torch.FloatTensor(img1).to(device), torch.FloatTensor(img2).to(device)
        
        L1,L2=net_L(img1),net_L(img2)

        if exposure_adjustment:
            k=calculate_k(L1,L2,exposureLevel)
            L=FuseIlluminance(L1,L2,k)
        else:
            L=(L1+L2)/2
        
        Ihat,R,Rhat = net_fusion(img1, img2,L)
        imgout=np.uint8(np.squeeze((L*Rhat).detach().cpu().numpy()).transpose(1,2,0)*255)
        plt.imsave(os.path.join(path_result,imgname),imgout)


