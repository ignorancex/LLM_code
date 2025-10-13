import numpy as np
import torchvision
import torch
import os
import torch.nn as nn
from pathlib import Path
from torchvision.transforms.functional import to_pil_image
os.sys.path.append('./repos/emoca')

from repos.emoca.gdl_apps.EmotionRecognition.utils.io import load_model, test
from src.utils.crops import emoca_crop

class EmotionLoss(nn.Module):

    def __init__(self, path_to_models="./repos/emoca/assets/EmotionRecognition/image_based_networks", model_name = "ResNet50"):
        super().__init__()
        self.model = load_model(Path(path_to_models) / model_name)
        self.model.cuda()
        self.model.eval()

    def forward(self, pred, target, kpts):
        '''pred is RGB in range [0, 1]'''
        crop_list = emoca_crop(pred, kpts)
                
        batch = {'image': crop_list}
        output_pred = self.model(batch)['features']
        
        with torch.no_grad():
            batch = {'image': target}
            output_target = self.model(batch)['features']

        loss = 0
        for f_pred, f_target in zip(output_pred, output_target):
            loss += torch.nn.functional.l1_loss(f_pred, f_target)

        return loss
        
        