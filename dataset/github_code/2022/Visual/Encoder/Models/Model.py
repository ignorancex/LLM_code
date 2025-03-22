# Abstract class from which torch Neural Network modules should inherit.

import torch
import torch.nn as nn

from abc import ABC, abstractmethod

class Model(nn.Module, ABC):
    
    # Function to load a trained model on the GPU (if available) or on the 
    # cpu if not available.
    # INPUTS:
    # - loadFile: path to the file from where the model should be loaded.
    # - device: 'cpu' or 'cuda'
    def LoadTrainedModel(self, loadFile, device):
        
        self.load_state_dict(torch.load(loadFile,map_location=torch.device(device)))  
        self.to(device) # bring to CPU or GPU
        
        return self