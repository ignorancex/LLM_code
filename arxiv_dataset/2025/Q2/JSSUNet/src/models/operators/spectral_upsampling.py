from torch import nn
import torch


class ClustersUp(nn.Module):
    def __init__(self, ms_channels, hs_channels, classes=5, features=64, **kwargs):
        super(ClustersUp, self).__init__()
        self.classes = classes
        self.ms_channels = ms_channels
        self.hs_channels = hs_channels
        self.mlps = nn.ModuleList([nn.Sequential(*[ nn.Linear(ms_channels, features), nn.ReLU(),  nn.Linear(features, hs_channels), nn.ReLU()]) for _ in range(classes)])

    def forward(self, image, clusters):
        B, C, H, W = image.shape
        hs_image = torch.zeros(B, self.hs_channels, H, W).to(image.device)
        for label in range(self.classes):
            mask = clusters == label
            indices = mask.nonzero(as_tuple=True)
            if indices[0].numel() == 0:
                continue
            pixel_values = image[indices[0], :, indices[2], indices[3]]
            transformed = self.mlps[label](pixel_values)
            hs_image[indices[0], :, indices[2], indices[3]] = transformed
        return hs_image
