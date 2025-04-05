import torch
import torch.nn as nn
import torchvision.transforms.functional as F

class SquarePad:
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return F.pad(image, padding, 0, 'constant')
    
class SquarePad255:
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return F.pad(image, padding, 255, 'constant')

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, ignore=255):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        ignore_msk = (targets != ignore).clone().detach().unsqueeze(1)
      
        inputs = nn.functional.softmax(inputs,dim=1)       
        #flatten label and prediction tensors
        #inputs = inputs.view(-1)
        with torch.no_grad():
            all_classes = torch.unique(targets)
            targets_full = torch.zeros_like(inputs)
   
            for i in range(inputs.shape[1]):
                targets_full[:,i,:,:] = (targets==i)
            #targets = targets.view(-1)
        
        intersection = (inputs[:,1:,:,:] * targets_full[:,1:,:,:]*ignore_msk).sum()                            
        dice = (2.*intersection + smooth)/((inputs[:,1:,:,:]*ignore_msk).sum() + (targets_full[:,1:,:,:]*ignore_msk).sum() + smooth)  
        
        return 1 - dice
    
