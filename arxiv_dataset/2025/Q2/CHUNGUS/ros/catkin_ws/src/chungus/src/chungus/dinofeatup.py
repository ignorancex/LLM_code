import torch
import torch.nn as nn
import torch.nn.functional as F
from featup.core.dinov2 import DINOv2UpFeatBackbone
from torchvision import transforms


class BasicPredictionHead(nn.Module):
    """ Basic (non-reconstruction) prediction head """
    def __init__(self, emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(emb_dim, 128, kernel_size=1, padding=0, stride=1, bias=True)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(128, 1, kernel_size=1, padding=0, stride=1, bias=True)
        self.act2 = nn.Sigmoid()
    
    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        return {'prediction': x}


class ReconstructionBasicPredictionHead(nn.Module):
    """ Reconstruction-based prediction head """
    def __init__(self, emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(emb_dim, 128, kernel_size=1, padding=0, stride=1, bias=True)
        self.act1 = nn.ReLU()
        self.convP = nn.Conv2d(128, 1, kernel_size=1, padding=0, stride=1, bias=True)
        self.actP = nn.Sigmoid()
        self.convR = nn.Conv2d(128, emb_dim, kernel_size=1, padding=0, stride=1, bias=True)
    
    def forward(self, x):
        enc = self.act1(self.conv1(x))
        p = self.actP(self.convP(enc))
        r = torch.square(self.convR(enc) - x).mean(dim=1, keepdims=True)
        return {'prediction': p, 'reconstruction': r}


class DINOv2FeatUp(nn.Module):
    def __init__(self, output_size, reconstruction=True):
        """ DINOv2+FeatUp predictor
        
        :param output_size: core inference and output resolution (H,W)
        :param reconstruction: if True, include a reconstruction head
        """
        super().__init__()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.output_size = output_size
        self.reconstruction = reconstruction

        self.embedder = DINOv2UpFeatBackbone.load_backbone(pretrained=True, use_norm=True)
        self.predictor = self.get_new_prediction_head()

    def get_model_name(self):
        """ Return the name of the model """
        if self.reconstruction:
            return "dinov2featup_recons"
        else:
            return "dinov2featup"

    def get_new_prediction_head(self):
        """ Return a new prediction head """
        if self.reconstruction:
            return ReconstructionBasicPredictionHead(384)
        else:
            return BasicPredictionHead(384)
    
    def forward(self, x):
        """ Perform inference
        
        :param x: image tensor of shape (3,H,W), expected to not be normalized
        """
        embed = self.embedder(self.normalize(x).unsqueeze(0))
        features = F.interpolate(embed['features'], size=self.output_size, mode='bilinear')
        out = self.predictor(features)
        out = {k: v[0,0] for k,v in out.items()}

        return {**out, **{
            'features': features[0], # (384,H,W)
            'cls_token': embed['cls_token'][0] # (384)
        }}
    
    def update_model(self, state_dict):
        """ Update the model state dict """
        self.predictor.load_state_dict(state_dict)
    
    def load_model(self, filename):
        """ Load model from a file """
        self.predictor.load_state_dict(torch.load(filename))

    def save_model(self, filename):
        """ Save model to a file """
        torch.save(self.predictor.state_dict(), filename)
