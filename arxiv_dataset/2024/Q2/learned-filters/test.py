# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 16:01:49 2023

@author: matthias
"""

import numpy as np
import torch


from skimage.transform import radon, iradon
from matplotlib import pyplot as plt


import os
import argparse
from tqdm import tqdm
import cv2
from utils import *
import json



parser = argparse.ArgumentParser(description= 'Define parameters for testing. Of course only parameters were trained on are allowed!')



parser.add_argument('--wave', help="Define which wavelet transform should be used", type=str, default="haar")
parser.add_argument('--levels', help="Number of levels in wavelet transform. Has to be integer between 1 and 8", 
                    type=int, default=8)
parser.add_argument('--alpha', help="possible signal-to-noise ratios are: 4,8,12,16,20,24,28", 
                    type=int, default=16)
parser.add_argument('--noise', help= "Noise type", type=str, default ="gaussian")
parser.add_argument('--types', help= "What type of network should be evaluated? Posibilities are: unconstrained, proposed, linear, nonexpasive ", 
                   nargs='+', type=str, default= ["proposed"] )

args = parser.parse_args()

def psnr(x,rec):
    #rescale to original values
    x = x/2
    rec = rec/2
    x = x+(1/2)
    rec = rec+(1/2)
    x = x*255
    rec = rec*255
    #calculate mse
    mse = np.mean((x - rec) ** 2)
    # Compute PSNR
    psnr = 20 * np.log10(255 / np.sqrt(mse))
    return psnr
    

wave= args.wave
levels=args.levels #Maximal 8 possible levels
alpha=args.alpha
delta=np.sqrt(185856)*alpha
noise = args.noise


f = open("path_to_data.txt", "r")
folder=f.read()
folder=folder+ '/testset/'


angles=512
filelist=os.listdir(folder)


device = "cuda" if torch.cuda.is_available() else "cpu"

best='best'
methods = args.types
results={}
for method in methods:
    savefolder="RESULTS_FOLDER/"+ wave +"/"+ noise +"/" +'alpha_'+str(alpha)+"/" + method +"/results/"
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)
    results[method]= {}
    MSE=[]
    rel_MSE=[]
    PSNR =[]
    MAE=[]
    rel_MAE=[]
    MSE_fbp=[]
    rel_MSE_fbp= []
    MAE_fbp= []
    rel_MAE_fbp=[]
    PSNR_fbp = []
    nets=[]
    for i in range(levels):
        loadpath = 'RESULTS_FOLDER/'+wave+'/'+noise+'/'+'alpha_'+str(alpha)+'/' +method +'/weights/level_'+str(i+1)+'_'+best+'.pth'
        weights = torch.load(loadpath, map_location=torch.device(device))
        if method == "linear":
            net=learned_filter_linear(1,1).to(device)
        elif method == "nonexpansive":
            net=learned_filter_nonexpansive(1,1).to(device)
        elif method == "proposed":
            net=learned_filter_proposed(1,1).to(device)
        elif method == "unconstrained":
            net=learned_filter(1,1).to(device)
        else:
            print("Worng type spezified")

        net.load_state_dict(weights)
        net.eval()
        nets.append(net)
    
    rel_noise =[]
    count = 0
    for file in tqdm(filelist):
        x=cv2.imread(folder+file,cv2.IMREAD_GRAYSCALE)
        x=x/255
        x=x-(1/2)
        x=2*x
        x=cv2.resize(x, (256,256))
        y=radon(x,np.arange(angles)/(angles/180), circle=False)
        if noise =="gaussian":
            sigma=alpha
            z=sigma*np.random.randn(y.shape[0], angles)
        elif noise =="poisson":
            m=np.min(y)
            y=y-m
            z=np.random.poisson(y)-y
            y=y+m
            scale=delta/np.linalg.norm(z)
            z=z*np.sqrt(scale)
        elif noise =="uniform":
            a=np.sqrt(3)*alpha
            z= np.random.uniform(low=-a, high=a, size=(y.shape[0], angles))
        elif noise == "saltpepper":
            z=np.zeros_like(y)
            ma=np.max(y)
            mi=np.min(y)
            while np.linalg.norm(z)<delta:
                x_coord=np.random.randint(0, y.shape[0])
                y_coord=np.random.randint(0, angles)
                if np.random.uniform() <0.5:
                    z[x_coord,y_coord]=mi-y[x_coord,y_coord]
                else:
                    z[x_coord,y_coord]=ma-y[x_coord,y_coord]
        else: 
            print("Wrong argument for --noise!")
        y_delta=y+z
        rel_noise.append((np.mean(z**2)/np.mean(y**2)))
        x_fbp=iradon(y_delta, circle=False)
        inp=np.expand_dims(x_fbp,0)
        inp = torch.from_numpy(inp.astype("float32")).to(device)
        rec=reconstruct(nets, inp, levels, wave)
        rec=rec[0,0,...].cpu().detach().numpy()
        #print(rec.shape,x.shape, x_fbp.shape)
        
        if count <4:
            fig, axes = plt.subplots(1,3)
            axes[0].imshow(x, cmap='gray',vmin=-1.5,vmax=1.5)
            axes[0].axis('off')
            axes[0].title.set_text('Ground truth')
            axes[1].imshow(x_fbp, cmap='gray', vmin=-1.5,vmax=1.5)
            axes[1].axis('off')
            axes[1].title.set_text('FBP')
            im=axes[2].imshow(rec, cmap='gray', vmin=-1.5,vmax=1.5)
            axes[2].axis('off')
            axes[2].title.set_text('Method')
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.33, 0.05, 0.35])
            fig.colorbar(im, cax=cbar_ax)
            plt.savefig(savefolder+f"/Case_{count}_{method}.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
            plt.close()
        
        count+=1
        
        PSNR.append(psnr(x,rec))
        MSE.append(np.mean((x-rec)**2))
        rel_MSE.append(np.mean((x-rec)**2)/np.mean(x**2))
        MAE.append(np.mean(abs(x-rec)))
        rel_MAE.append(np.mean(abs(x-rec))/np.mean(abs(x)))
        
        if method == "linear":
            MSE_fbp.append(np.mean((x-x_fbp)**2))
            rel_MSE_fbp.append(np.mean((x-x_fbp)**2)/np.mean(x**2))
            MAE_fbp.append(np.mean(abs(x-x_fbp)))
            rel_MAE_fbp.append(np.mean(abs(x-x_fbp))/np.mean(abs(x)))   
            PSNR_fbp.append(psnr(x,x_fbp))
    
    if method == "linear":
        results["FBP"]={}
        results["FBP"]['Mean squared error']=np.mean(MSE_fbp)
        results["FBP"]['Mean rel squared error']=np.mean(rel_MSE_fbp)
        results["FBP"]['Mean absolute error']=np.mean(MAE_fbp)
        results["FBP"]['Mean rel absolute error']=np.mean(rel_MAE_fbp)
        results["FBP"]['Peak signal to noise ratio']=np.mean(PSNR_fbp)
        
        
    results[method]['Mean squared error']=np.mean(MSE)
    results[method]['Mean rel squared error']=np.mean(rel_MSE)
    results[method]['Mean absolute error']=np.mean(MAE)
    results[method]['Mean rel absolute error']=np.mean(rel_MAE)
    results[method]['Peak signal to noise ratio']=np.mean(PSNR)
    
    
    print(np.mean(rel_noise)*100)
    

summary_path = "RESULTS_FOLDER/"+ wave +"/"+ noise +"/" +'alpha_'+str(alpha)+"/"     
with open(summary_path+'summary.txt', 'w') as file:
    
    json.dump(results, file, indent=4, sort_keys=False)

