import torch
from torch.utils.data import DataLoader,Dataset,Sampler
import os
import numpy as np
import random as rd
import torch.optim as optim
import torch.nn.functional as F
from models.embedding import *
from models.backbone import *
from models.predictor import *
from loss.make_loss import *
from torch.nn import Module
import sys
import scipy.special as special
from omegaconf import OmegaConf
sys.append('..')
from train_smaller_model import TransformerGeneratorModel

class MuonGenerator:
    def __init__(self,cfg:dict,depth='0m',device='cuda:0'):
        assert depth in cfg.keys()
        ckpt_path=cfg[depth]["ckpt_path"]
        self.energy_mean=cfg[depth]["energy_mean"]
        self.ckpt_path = ckpt_path
        self.device = device
        self.model = TransformerGeneratorModel()
        self.model.load_state_dict(torch.load(ckpt_path,map_location=device)['net'])
        self.model.to(device)
        self.model.eval()
        
    def gen(self):
        inputs=torch.rand(1000,100)*2-1
        inputs=inputs.to(self.device)
        with torch.no_grad():
            outputs=self.model(inputs)
        outputs=outputs.cpu().numpy()
        output=output.reshape(-1,3)
        for i in range(outputs.shape[0]):
            # 能量变换
            energy=outputs[i,0]
            energy=np.arctanh(energy)+self.energy_mean
            # boxcox变换
            outputs[i,0]=special.boxcox(energy,9)
            
        return outputs
    
    
if __name__ == '__main__':
    cfg=OmegaConf.load('config.yaml')
    muon_generator=MuonGenerator(cfg)
    muon_data=muon_generator.gen()
    np.save('muon_data_0m.npy',muon_data)
    