# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 16:23:29 2023

@author: Schwab Matthias
"""


import numpy as np
import torch


import shutil
import os
import argparse

from utils import *

#define parameters
parser = argparse.ArgumentParser(description= 'Define parameters for training')

parser.add_argument('--wave', help="Define which wavelet transform should be used", type=str, default="haar")
parser.add_argument('--levels', help="Number of levels in wavelet transform. Has to be integer between 1 and 8", 
                    type=int, default=8)
parser.add_argument('--alpha', help="possible noise levels: 32,28,24,20,16,12,8,4,0", 
                    type=int, default=4)
parser.add_argument('--N_epochs', help="Specify how many epochs should be trained", type=int, default=20)
parser.add_argument('--noise', help= "Noise type", type=str, default ="gaussian")
parser.add_argument('--type', 
                    help= "What type of network should be learned? Posibilities are: unconstrained, proposed, linear, nonexpasive ", 
                    type= str, default = "proposed")
args = parser.parse_args()


wave= args.wave
levels=args.levels #Maximal 8 possible levels
alpha=args.alpha #possible signal to noise ratios are: 2,4,8,16,32,64,128,256,512 
N_epochs=args.N_epochs
noise = args.noise

#get path to preprocessed data
f = open("path_to_data.txt", "r")
folder=f.read()
folder=folder+ '/preprocessed_'+ noise +'/'

if not os.path.exists('RESULTS_FOLDER/'):
    os.makedirs('RESULTS_FOLDER/')


savefolder = f"RESULTS_FOLDER/{wave}/{noise}/alpha_{str(alpha)}/{args.type}/" 



#delete old results
if os.path.exists(savefolder):
    shutil.rmtree(savefolder)

#get dataloaders
train_data = Dataset(folder=folder, alpha=alpha)  
test_data = Dataset(folder=folder,  alpha=alpha, train=False)    
collator = Collator()
dataloader= torch.utils.data.DataLoader(train_data, batch_size=1, collate_fn=collator,shuffle=True)
dataloader_test= torch.utils.data.DataLoader(test_data, batch_size=1, collate_fn=collator,shuffle=False)     


#Prepare networks for training
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

hist=get_histogram()
best_loss=np.inf

    
nets=[]
for i in range(levels):
    if args.type == "nonexpansive":
        nets.append(learned_filter_nonexpansive(1,1).to(device))
    elif args.type == "proposed":
        nets.append(learned_filter_proposed(1,1).to(device))
    elif args.type == "linear":
        nets.append(learned_filter_linear(1,1).to(device))
    elif args.type == "unconstrained":
        nets.append(learned_filter(1,1).to(device))
    else:
        print("Worng type spezified")

params=[]
for i in range(levels):
    d={}
    d['params']=nets[i].parameters()
    d['lr']=1e-3
    params.append(d)

lambda1 = lambda epoch: (1-epoch/N_epochs)**0.9    
optimizer = torch.optim.AdamW(params, weight_decay=1e-3,)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1 for i in range(levels)])

#train
for epoch in range(N_epochs):
        print(f"Epoch {epoch}/{N_epochs}\n-------------------------------")
        for net in nets:
            net.train()
        count=0
        l=0
        for x_fbp, x in dataloader:
            optimizer.zero_grad()
            rec=reconstruct(nets, x_fbp, levels, wave)
            loss=torch.mean((rec-x)**2)
            loss.backward()
            for i in range(levels):
                torch.nn.utils.clip_grad_norm_(params[i]['params'], 2)
            optimizer.step()
            l+=loss.item()
            count+=1
        hist['trainloss'].append(l/count)
        scheduler.step()
        if l < best_loss:
            print(f"New best loss: {l/count}, -->Saving models")
            best_loss=l
            save_checkpoint(nets, savefolder, 'best')
            plot_filter(nets, savefolder, device)
        #print(scheduler.get_last_lr())
        
        
        #validation
        for net in nets:
            net.eval()
        count=0
        l=0
        n=0
        for x_fbp, x in dataloader_test:
            with torch.no_grad():
                rec=reconstruct(nets, x_fbp, levels, wave)
                loss=torch.mean((rec-x)**2)
                l+=loss.item()
                n+=torch.mean((x_fbp-x)**2).item()
                count+=1
            
        hist['testloss'].append(l/count)
        hist['fbploss'].append(n/count)
        
        #print('Mean rec error on train data set:', hist['trainloss'][-1])
        #print('Mean rec error on test data set:', hist['testloss'][-1])
        plot_hist(hist, savefolder)

#save histogram and weights
np.save(f"{savefolder}/histogram.npy", hist)


      
        



