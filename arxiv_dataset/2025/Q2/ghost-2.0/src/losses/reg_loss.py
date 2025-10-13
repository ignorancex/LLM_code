import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class ReferenceRegularizationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, reference_bgr, source_gs):
        reference_rgb = reference_bgr[:, [2, 1, 0], ...]
        reference_gs = TF.rgb_to_grayscale(reference_rgb)
        
        loss = F.l1_loss(reference_gs, source_gs)
        
        return loss