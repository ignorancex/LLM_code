# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 15:58:48 2023

@author: matthias
"""

import numpy as np
import torch
import os
import torch.nn as nn
import ptwt
from matplotlib import pyplot as plt
import torch.nn.utils as utils


class Dataset(torch.utils.data.Dataset):
    """
    Dataset for the depth/text data.
    This dataset loads an image and applies some transformations to it.
    """

    def __init__(self, folder= "C:/Users/matthias/Desktop/Data/CT-for-filters/preprocessed/", alpha=4, 
                 train=True, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train=train
        if self.train:
            self.folder = folder+"train/"
        else:
            self.folder = folder+"test/"
        self.examples = self.get_examples()
        self.alpha=alpha
        
        

    def __getitem__(self,idx):
        data = np.load(os.path.join(self.folder, self.examples[idx]), allow_pickle=True).item()
        
        alpha_index=np.where(data['alpha']==self.alpha)[0][0]
        inp=data['x_fbp'][alpha_index]
        inp=np.expand_dims(inp,0)
        outp=data['x']
        outp=np.expand_dims(outp,0)
                            
        inp = torch.from_numpy(inp.astype("float32")).to(self.device)
        outp = torch.from_numpy(outp.astype("float32")).to(self.device)
        return inp, outp

    def __len__(self):
        return len(self.examples)
    

    def get_examples(self):
        examples = [f for f in os.listdir(self.folder) if f.endswith('.npy')]
        return examples

class Collator:
    """
    Data collator for different types of mask-combinations.
    """


    def __call__(self, batch, *args, **kwargs):
        inputs = []
        outputs = []
        
        for inp, outp in batch:
            inputs.append(inp)
            outputs.append(outp)
        return torch.stack(inputs), torch.stack(outputs)





class ContrainedLinear(nn.Module):
    def __init__(self, in_features, out_features, non_expansive=False, eps=0):
        super(ContrainedLinear, self).__init__()
        # Initialize the weight and bias parameters
        self.weights = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.non_expansive = non_expansive
        self.eps = eps
        

    def forward(self, x):
        # Apply ReLU to the weights to enforce positivity
        weights = torch.abs(self.weights)+self.eps
        if self.non_expansive == True:
            weight_norm = torch.linalg.norm(weights, ord=2)  # Compute the spectral norm (largest singular value)
            weights = weights / weight_norm.clamp(min=1.0)  # Scale the weights if norm > 1
        return torch.matmul(x, weights)+ self.bias




    
class learned_filter_nonexpansive(torch.nn.Module):

    def __init__(self, indim, outdim):
        super().__init__()  
   
        self.lin1 = ContrainedLinear(in_features=indim, out_features=20, non_expansive= True)
        self.lin2 = ContrainedLinear(in_features=20, out_features=20, non_expansive= True)
        self.lin3 = ContrainedLinear(in_features=20, out_features=outdim, non_expansive= True)
        self.act = nn.ReLU()
        
    def custom_act(self,xx):
        m = xx.shape[1]
        xx1=xx[:,0:m//3]
        xx2=xx[:,m//3 : 2*(m//3)]
        xx3=xx[:, 2*(m//3):]
        out1 = self.act(xx1)
        out2 = -self.act(-xx2)
        out3 = self.act(xx3+1) -1 -self.act(xx3-1)
        return torch.cat([out1, out2, out3],1)    
        
            
    def phi_0(self, inp):
        xx = torch.zeros_like(inp)
        xx = self.custom_act(self.lin1(xx))
        xx = self.custom_act(self.lin2(xx))
        out = - self.lin3(xx)
        return  out


    def forward(self, inp):
        xx = self.custom_act(self.lin1(inp))
        xx = self.custom_act(self.lin2(xx))
        out = inp - self.lin3(xx)
        out = out -self.phi_0(inp)
        return  out

class learned_filter_proposed(torch.nn.Module):

    def __init__(self, indim, outdim):
        super().__init__()  
   
        self.lin1 = ContrainedLinear(in_features=indim, out_features=20, non_expansive= False, eps=1e-15)
        self.lin2 = ContrainedLinear(in_features=20, out_features=20, non_expansive= False, eps=1e-15)
        self.lin3 = ContrainedLinear(in_features=20, out_features=outdim, non_expansive= False, eps=1e-15)
        self.act = nn.ReLU()
        
    def custom_act(self,xx):
        m = xx.shape[1]
        xx1=xx[:,0:m//3]
        xx2=xx[:,m//3 : 2*(m//3)]
        xx3=xx[:, 2*(m//3):]
        out1 = self.act(xx1)
        out2 = -self.act(-xx2)
        out3 = self.act(xx3+1) -1 -self.act(xx3-1)
        return torch.cat([out1, out2, out3],1)    
        
            
    def phi_0(self, inp):
        inp = torch.zeros_like(inp)
        xx = self.custom_act(self.lin1(inp))
        xx = self.custom_act(self.lin2(xx))
        out = self.lin3(xx)
        return  out


    def forward(self, inp):
        xx = self.custom_act(self.lin1(inp))
        xx = self.custom_act(self.lin2(xx))
        out = self.lin3(xx)
        out = out -self.phi_0(inp)
        return  out


class learned_filter(torch.nn.Module):

    def __init__(self, indim, outdim):
        super().__init__()  
        self.lin1 = nn.Linear(in_features=indim, out_features=20)
        self.lin2 = nn.Linear(in_features=20, out_features=20)
        self.lin3 = nn.Linear(in_features=20, out_features=outdim)
        self.actr = nn.ReLU()
        

    def forward(self, inp):
        xx = self.actr(self.lin1(inp))
        xx = self.actr(self.lin2(xx))
        out = inp + self.lin3(xx)
        return  out

class learned_filter_linear(torch.nn.Module):

    def __init__(self, indim, outdim):
        super().__init__()  
        self.lin1 = nn.Linear(in_features=indim, out_features=20, bias = False)
        self.lin2 = nn.Linear(in_features=20, out_features=20, bias = False)
        self.lin3 = nn.Linear(in_features=20, out_features=outdim, bias = False)
        
    def forward(self, inp):
        xx = self.lin1(inp)
        xx = self.lin2(xx)
        out = self.lin3(xx)
        return  out

    
    
def get_histogram():
    hist={}
    hist['trainloss']=[]
    hist['testloss']=[]
    hist['fbploss']=[]
    return hist

def reconstruct(nets, x_fbp, levels, wave):
    coefficients= ptwt.wavedec2(x_fbp, wave, level=levels)
    denoised_coeff=[]
    denoised_coeff.append(coefficients[0])
    for i in range(levels):
        inp=torch.flatten(torch.stack([c for c in coefficients[i+1]]))
        filtered=nets[i](inp[:,None])
        filtered_reshaped=[]
        shape=coefficients[i+1][0].shape
        lenght=shape[-1]*shape[-2]
        for k in range(3):
            filtered_reshaped.append(torch.reshape(filtered[k*lenght:(k+1)*lenght,0], shape))
        denoised_coeff.append(filtered_reshaped)
        rec=ptwt.waverec2(denoised_coeff, wave)
    return rec



def save_checkpoint(nets, savepath, best):
    if not os.path.exists(savepath+'/weights/'):
        os.makedirs(savepath+'/weights/')
    for i in range(len(nets)):
        save= savepath+'/weights/level_'+str(i+1)+'_'+best+'.pth'
        torch.save(nets[i].state_dict(), save)


def plot_filter(nets, savepath, device):
    if not os.path.exists(savepath+'/plots/'):
        os.makedirs(savepath+'/plots/')
    for i in range(len(nets)):
            path=savepath+'/plots/level_'+str(i+1)+'.png'
            j=i+1 #der Scale index: -j 
            kappa=2**(-j/2) #quasi-singulär-wert 
            x=kappa*torch.linspace(-10, 10, 100, device=device)
            x=x[:,None]
            y=kappa*nets[i](x/kappa)
            plt.figure()
            plt.plot(x[:,0].cpu().detach().numpy(),y[:,0].cpu().detach().numpy())
            plt.plot(x[:,0].cpu().detach().numpy(),x[:,0].cpu().detach().numpy())
            plt.title('Level:'+str(i+1))
            plt.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.1)
            plt.close()

def plot_hist(hist,savepath):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    savepath = savepath+'/train_progress.png'
    plt.figure()
    plt.plot(hist['trainloss'])
    plt.plot(hist['testloss'])
    plt.plot(hist['fbploss'])
    plt.savefig(savepath, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    train_data = Dataset(s2n_ratio=8)    
    collator = Collator()
    dataloader= torch.utils.data.DataLoader(train_data, batch_size=1, collate_fn=collator,shuffle=True) 
    
    i=0
    for x_fbp, x in dataloader:
        i+=1
        if i%100==0:
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(x[0,0,...].cpu())
            plt.axis('off')
            plt.subplot(1,2,2)
            plt.imshow(x_fbp[0,0,...].cpu())
            plt.axis('off')
            plt.show()
            

    