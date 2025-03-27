import pytorch_lightning as pl

import torch 

class GradientNoiseAdd(pl.Callback):
    def __init__(self):
        print('Gradient Noise Initialized')

    def on_before_optimizer_step(trainer, pl_module, optimizer, opt_idx):
        for param in pl_module.parameters():
            param.grad += self.lr*torch.randn(param.grad.shape()) ### or rand_like for uniform 
            
             