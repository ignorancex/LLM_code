### ====== PRISM LOSS FUNCTIONS ======

import torch
from torch import nn
from torch.nn import functional as F

class PerceptualLoss(nn.Module):
    def __init__(self, vgg_model):
        super().__init__()
        for param in vgg_model.parameters():
            param.requires_grad = False
        self.vgg = nn.Sequential(*list(vgg_model.children())[:13]).eval()

    def forward(self, x, y):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if y.shape[1] == 1:
            y = y.repeat(1, 3, 1, 1)
        return F.l1_loss(self.vgg(x), self.vgg(y))


class PatchNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.temperature = temperature

    def forward(self, query_feature, positive_feature, negative_feature):
        B, C, N_query = query_feature.shape
        _, _, N_positive = positive_feature.shape
        _, _, N_negative = negative_feature.shape

        # Reshape query_feature to [B*N_query, C, 1]
        query_feature = query_feature.permute(0, 2, 1).reshape(B*N_query, C, 1)

        # Reshape positive_feature to [B*N_query, C, N_positive]
        positive_feature = positive_feature.unsqueeze(1).expand(-1, N_query, -1, -1)
        positive_feature = positive_feature.reshape(B*N_query, C, N_positive)

        # Compute l_positive
        l_positive = torch.bmm(query_feature.transpose(1, 2), positive_feature).squeeze(1)

        # Reshape negative_feature to [B*N_query, C, N_negative]
        negative_feature = negative_feature.unsqueeze(1).expand(-1, N_query, -1, -1)
        negative_feature = negative_feature.reshape(B*N_query, C, N_negative)

        # Compute l_negative
        l_negative = torch.bmm(query_feature.transpose(1, 2), negative_feature).squeeze(1)

        # Concatenate l_positive and l_negative
        logits = torch.cat((l_positive, l_negative), dim=1) / self.temperature

        # Create targets
        targets = torch.zeros(B * N_query, dtype=torch.long, device=query_feature.device)

        return self.ce_loss(logits, targets).mean()


class KLDivergenceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mu, logvar):
        kld_loss = -0.5 * logvar + 0.5 * (torch.exp(logvar) + torch.pow(mu, 2)) - 0.5
        return kld_loss

    
class BilateralLoss(nn.Module):
    def __init__(self, spatial_sigma=1.0, intensity_sigma=1.0):
        super(BilateralLoss, self).__init__()
        self.spatial_sigma = spatial_sigma
        self.intensity_sigma = intensity_sigma

    def forward(self, output):
        N, C, H, W = output.shape
        spatial_weight = self._compute_spatial_weight(H, W).to(output.device)

        # Compute intensity difference between neighboring pixels in the output
        intensity_diff = self._compute_intensity_diff(output)

        # Compute spatial and range kernels
        spatial_kernel = torch.exp(-spatial_weight / (2 * self.spatial_sigma ** 2))  # Spatial proximity
        range_kernel = torch.exp(-intensity_diff / (2 * self.intensity_sigma ** 2))  # Intensity similarity

        # Bilateral weight for smoothing the output
        bilateral_weight = spatial_kernel * range_kernel

        # The loss is based on smoothness of the output itself
        loss = bilateral_weight * intensity_diff
        return loss.mean()

    def _compute_spatial_weight(self, H, W):
        x = torch.arange(W).float().view(1, -1)
        y = torch.arange(H).float().view(-1, 1)
        xx = (x - x.T).pow(2)
        yy = (y - y.T).pow(2)
        spatial_weight = (xx[None, None, :, :] + yy[None, None, :, :]).float()  # (1, 1, H, W)
        return spatial_weight

    def _compute_intensity_diff(self, output):
        # Compute pixel intensity differences between neighboring pixels (for smoothing)
        diff_x = (output[:, :, 1:, :] - output[:, :, :-1, :]).pow(2)  # Size: (N, C, H-1, W)
        diff_y = (output[:, :, :, 1:] - output[:, :, :, :-1]).pow(2)  # Size: (N, C, H, W-1)

        # Pad diff_x to match the shape of diff_y
        diff_x_padded = F.pad(diff_x, (0, 0, 0, 1))  # Pad diff_x at the bottom by 1
        # Pad diff_y to match the shape of diff_x
        diff_y_padded = F.pad(diff_y, (0, 1, 0, 0))  # Pad diff_y at the right by 1

        # Now both tensors should have the shape (N, C, H, W)
        intensity_diff = diff_x_padded + diff_y_padded
        return intensity_diff

    
class TotalVariationLoss(nn.Module):
    def forward(self, output):
        diff_x = torch.mean(torch.abs(output[:, :, 1:, :] - output[:, :, :-1, :]))
        diff_y = torch.mean(torch.abs(output[:, :, :, 1:] - output[:, :, :, :-1]))
        return diff_x + diff_y