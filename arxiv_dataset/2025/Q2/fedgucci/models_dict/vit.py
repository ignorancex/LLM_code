import torch
from torch import nn
import clip

class ViT(nn.Module):
    def __init__(self, num_classes, normalize=False) -> None:
        super(ViT, self).__init__()
        self.model, _ = clip.load("ViT-B/32", device='cuda')
        self.classifier = torch.nn.Linear(512, num_classes)
        self.normalize = normalize
    
    def forward(self, images, return_hidden_state = False):
        features = self.model.encode_image(images).float()
        if self.normalize:
            features = features / features.norm(dim=-1, keepdim=True)
        logits = self.classifier(features)
        if return_hidden_state:
            return None, logits, features
        else:
            return logits