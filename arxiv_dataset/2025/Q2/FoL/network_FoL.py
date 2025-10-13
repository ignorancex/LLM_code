import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from backbone.dinov2_FoL import DINOv2
from aggregators.FoL import FoL
import math
import sys

class FoLNet(nn.Module):
    """The used networks are composed of a backbone and an aggregation layer.
    """
    def __init__(self, num_channels=1024, model_name='dinov2_vitl14', num_trainable_blocks=4, pretrained_foundation = False, foundation_model_path = None):
        super().__init__()
        self.backbone = DINOv2(model_name=model_name, num_trainable_blocks=num_trainable_blocks, return_token=True,
                               norm_layer=True)
        self.aggregator = FoL(num_channels=num_channels, num_clusters=64, cluster_dim=128, token_dim=256)
        self.upconv = torch.nn.ConvTranspose2d(in_channels=1024, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.upconv2 = torch.nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x, test=False):
        x = self.backbone(x)
        mask1 = x[2]
        mask_guide = torch.where(mask1 >= 0.01, mask1, torch.zeros_like(mask1))
        x, local_f, mask, loss_kl_1, loss_kl_2, mix_matrix, weak_supervision_info = self.aggregator(x, mask_guide, test=test)

        x0 = self.upconv(local_f)
        x0 = self.relu(x0)
        x0 = self.upconv2(x0)

        x0 = x0.permute(0, 2, 3, 1)
        local_all = torch.nn.functional.normalize(x0, p=2, dim=-1)
        local_feature_separate = x0.detach()
        local_feature_separate = torch.nn.functional.normalize(local_feature_separate, p=2, dim=-1)

        BS, H, W, C = x0.shape
        mask2 = mix_matrix.reshape(BS, mask.shape[1], mask.shape[2]).float()
        mask_interpolated = F.interpolate(mask2.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=True).squeeze(1)
        mask_separate = mask_interpolated
        mask = nn.functional.interpolate(mask.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)
        local_f_flat = x0.view(x0.size(0), -1, x0.size(-1))
        mask_flat = mask.view(mask.size(0), -1)
        indices = mask_flat.nonzero(as_tuple=True)
        selected_batch_tokens = local_f_flat[indices]
        split_sizes = mask_flat.sum(dim=1).to(torch.int).tolist()
        selected_tokens = selected_batch_tokens.split(split_sizes)

        if H > 80:
            max_len = 3600
        elif H > 70:
            max_len = 3500
        elif H > 40:
            max_len = 1000
        else:
            max_len = H * H

        selected_tokens = list(selected_tokens)
        padded_first_token = torch.cat([
            selected_tokens[0],
            torch.zeros((max_len - selected_tokens[0].size(0), selected_tokens[0].size(1)),
                        device=selected_tokens[0].device)
        ], dim=0)
        selected_tokens[0] = padded_first_token
        padded_tokens = torch.nn.utils.rnn.pad_sequence(selected_tokens, batch_first=True, padding_value=0)

        local_feature = torch.nn.functional.normalize(padded_tokens, p=2, dim=-1)
        if weak_supervision_info is not None:
            weak_supervision_info.append(local_all)
        return x, local_feature, loss_kl_1, loss_kl_2, [local_feature_separate, mask_separate], weak_supervision_info , mask_guide