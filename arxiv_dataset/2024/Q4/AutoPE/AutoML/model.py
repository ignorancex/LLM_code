# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

esm2_model_mapping = {
    'esm2_t48_15B_UR50D': {
        'layers': 48,
        'embedding_dim': 5120,
        'params': '15B',
        'model_url': 'https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t48_15B_UR50D.pt'
    },
    'esm2_t36_3B_UR50D': {
        'layers': 36,
        'embedding_dim': 2560,
        'params': '3B',
        'model_url': 'https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt'
    },
    'esm2_t33_650M_UR50D': {
        'layers': 33,
        'embedding_dim': 1280,
        'params': '650M',
        'model_url': 'https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt'
    },
    'esm2_t30_150M_UR50D': {
        'layers': 30,
        'embedding_dim': 640,
        'params': '150M',
        'model_url': 'https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t30_150M_UR50D.pt'
    },
    'esm2_t12_35M_UR50D': {
        'layers': 12,
        'embedding_dim': 480,
        'params': '35M',
        'model_url': 'https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t12_35M_UR50D.pt'
    },
    'esm2_t6_8M_UR50D': {
        'layers': 6,
        'embedding_dim': 320,
        'params': '8M',
        'model_url': 'https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t6_8M_UR50D.pt'
    }
}

class MaskedAveragePooling(nn.Module):
    def __init__(self):
        super(MaskedAveragePooling, self).__init__()

    def forward(self, x, mask):
        mask = mask.unsqueeze(-1) if mask.dim() == 2 else mask
        x_masked = x * mask
        sum_x = torch.sum(x_masked, dim=1)
        sum_mask = torch.sum(mask, dim=1)
        sum_mask = torch.where(sum_mask == 0, torch.ones_like(sum_mask), sum_mask)
        average = sum_x / sum_mask
        return average

class Model(nn.Module):
    def __init__(self, pretrained_model, embedding_dim, output_dim, self_attention_layers=False, repr_layers=33, dropout=0.1):
        super(Model, self).__init__()
        self.self_attention_layers = self_attention_layers
        self.pretrained_model = pretrained_model
        self.dropout = dropout
        if self.self_attention_layers:
            self.self_attention = nn.MultiheadAttention(embedding_dim, num_heads=8)
        self.masked_avg_pool = MaskedAveragePooling()
        self.fc = nn.Linear(embedding_dim, output_dim)
        self.repr_layers = repr_layers

    def forward(self, tokens, mask):
        results = self.pretrained_model(tokens, repr_layers=[self.repr_layers], return_contacts=True)
        token_representations = results["representations"][self.repr_layers]
        if self.self_attention_layers:
            x = token_representations.permute(1, 0, 2)
            x_skip, _ = self.self_attention(x, x, x)
            x = x + x_skip
            x = x.permute(1, 0, 2)
            x = F.dropout(x, p=self.dropout, training=self.training)
        else:
            x = token_representations
        avg_pooled = self.masked_avg_pool(x, mask)
        output = self.fc(avg_pooled)
        output = output.squeeze(-1)  # 确保输出是一维的
        return output, results