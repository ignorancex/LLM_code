from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from torch import nn

def img_preprocess_dino(img_np):
    x = Image.fromarray(img_np.astype(np.uint8)).convert("RGB")
    T = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
    ])
    return np.asarray(T(x)).clip(0, 255).astype(np.uint8)



class DINOV2_fx():
    def __init__(self, name="dinov2_vitb14_reg_lc", device="cuda"):
        self.model = torch.hub.load('facebookresearch/dinov2', name).backbone
        self.model.to(device)
        self.model.eval()
        self.name = name
    
    def __call__(self, img_t):
        img_x = img_t/255.0
        T_norm = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        img_x = T_norm(img_x)
        assert torch.is_tensor(img_x)
        if len(img_x.shape)==3:
            img_x = img_x.unsqueeze(0)
        B,C,H,W = img_x.shape
        with torch.no_grad():
            z = self.model(img_x)
        return z

class DINOV2_fx_model(nn.Module):
    def __init__(self,name="dinov2_vitb14_reg_lc", device="cuda"):
        super(DINOV2_fx_model, self).__init__()
        self.layers = torch.hub.load('facebookresearch/dinov2', name).backbone
        self.layers.to(device)
        self.layers.eval()
        self.name = name

    def __call__(self, img_t):
        img_x = img_t/255.0
        T_norm = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        img_x = T_norm(img_x)
        assert torch.is_tensor(img_x)
        if len(img_x.shape)==3:
            img_x = img_x.unsqueeze(0)
        z = self.layers(img_x)
        return z