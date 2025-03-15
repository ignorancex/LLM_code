import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from ..builder import LOSSES


class ContrastiveHead(nn.Module):
    """MLP head for contrastive representation learning, https://arxiv.org/abs/2003.04297
    Args:
        dim_in (int): dimension of the feature intended to be contrastively learned
        feat_dim (int): dim of the feature to calculated contrastive loss

    Return:
        feat_normalized (tensor): L-2 normalized encoded feature,
            so the cross-feature dot-product is cosine similarity (https://arxiv.org/abs/2004.11362)
    """

    def __init__(self, dim_in, feat_dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, feat_dim),
        )
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    init.zeros_(layer.bias)

    def forward(self, x):
        feat = self.head(x)
        feat_normalized = F.normalize(feat, dim=1)
        return feat_normalized


@LOSSES.register_module()
class SupConLossV2(nn.Module):
    def __init__(self, temperature=0.7, weight=1, k = 16, x = 4):
        super().__init__()
        self.temperature = temperature
        self.k = k
        self.x = x


    def calculate_similarity_between_sets(set1, set2, k, x):
        segment_length = set1.size(1) // k
        
        # Reshape the sets into segments
        segments1 = set1.view(set1.size(0), k, segment_length, -1)
        segments2 = set2.view(set2.size(0), k, segment_length, -1)
        
        # Calculate cosine similarities for all segments
        similarities = torch.sum(segments1 * segments2, dim=(2, 3)) / (torch.norm(segments1, dim=(2, 3)) * torch.norm(segments2, dim=(2, 3)))
        
        # Find the x segments with the highest similarities for each pair
        _, most_similar_indices = torch.topk(similarities, x, dim=1)
        
        return most_similar_indices, similarities

    def forward(self, features, labels, preds):
        # weight_label = self.updata_weight(labels, preds)
        # print(labels)
        # print(labels.shape)
        # labels = labels[:10]
        # features = features[:10]
        
        labels_binary = labels.clone().detach()
        # labels_binary[labels_binary != 12] = 0
        # print(labels_binary.sum() / 12)
        
        if len(labels_binary.shape) == 1:
            labels_binary = labels_binary.reshape(-1, 1)

        # mask of shape [None, None], mask_{i, j}=1 if sample i and sample j have the same label
        label_mask = torch.eq(labels_binary, labels_binary.T).float().cuda(labels.device)
        label_mask = label_mask - torch.diag(label_mask.diag())
        segment_length = features.size(1) // self.k
        feature_segments = features.view(features.size(0), self.k, segment_length, -1)
        # set2_segments = set2.view(set2.size(0), k, segment_length, -1)
        # print((set1_segments[:,None,:,:,:]* set2_segments[None,:,:,:,:]).shape)
        similarities = torch.div(torch.sum(feature_segments[None,:,:,:]* feature_segments[:,None,:,:], dim=(-1, -2)), self.temperature)
        sim_row_max, _ = torch.max(similarities, dim=-1, keepdim=True)
        similarities = similarities - sim_row_max
        exp_sim = torch.exp(similarities)
        # print(exp_sim.shape)
        value, idx = torch.topk(exp_sim, self.x, dim=-1)
        # print(value.shape)
        loss = -torch.log(value.sum(-1) / exp_sim.sum(-1)) * label_mask
        return loss.mean() 
        # print(torch.sum(tmp))
