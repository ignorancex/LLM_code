import sys
sys.path.append("<your path to InsightFace repository>\\insightface-master\\recognition\\arcface_torch")
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from backbones import get_model

def img_preprocess_insightface(img_np):
    x = Image.fromarray(img_np.astype(np.uint8)).convert("RGB")
    T = transforms.Compose([
            transforms.Resize(112, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(112),
    ])
    return np.asarray(T(x)).clip(0, 255).astype(np.uint8)

weights ={
    "arcface":"<your path>\\arcface_r100_MS1MV3_backbone.pth"
}
models = {
    "arcface":"r100"
}

class Insightface_fx():
    def __init__(self, name="arcface", device="cuda"):
        self.model = get_model(models[name], fp16=False)
        self.model.load_state_dict(torch.load(weights[name]))
        self.model.to(device)
        self.model.eval()
        self.name = "insightface_"+name.lower().replace("-","_").replace("/","_")
    
    def __call__(self, img_t):
        img_x = img_t/255.0
        T_norm = transforms.Normalize(0.5, .05)
        img_x = T_norm(img_x)
        assert torch.is_tensor(img_x)
        if len(img_x.shape)==3:
            img_x = img_x.unsqueeze(0)
        B,C,H,W = img_x.shape
        with torch.no_grad():
            z = self.model(img_x)
        return z

class Insightface_fx_model(torch.nn.Module):
    def __init__(self, name="arcface", device="cuda"):
        super(Insightface_fx_model, self).__init__()
        self.layers = get_model(models[name], fp16=False)
        self.layers.load_state_dict(torch.load(weights[name]))
        self.layers.to(device)
        self.layers.eval()
        self.name = "insightface_"+name.lower().replace("-","_").replace("/","_")
    
    def __call__(self, img_t):
        img_x = img_t/255.0
        T_norm = transforms.Normalize(0.5, .05)
        img_x = T_norm(img_x)
        assert torch.is_tensor(img_x)
        if len(img_x.shape)==3:
            img_x = img_x.unsqueeze(0)
        z = self.layers(img_x)
        return z
 