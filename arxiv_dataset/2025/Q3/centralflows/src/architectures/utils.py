import torch
import torch.nn as nn


class UntrainableLayerNorm(nn.Module):
    def __init__(self, features):
        super().__init__()

    def forward(self, x):
        mean = x.mean(-1).unsqueeze(-1)
        std = x.var(-1, unbiased=False).unsqueeze(-1).sqrt()
        return (x - mean) / std
    
    
class TrainableLayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(TrainableLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


def replace_layernorm(model, new_layer_norm_class):
    for name, module in model.named_children():
        if isinstance(module, nn.LayerNorm):
            setattr(
                model,
                name,
                new_layer_norm_class(module.weight.shape[0]).to(module.weight.device),
            )
        else:
            replace_layernorm(module, new_layer_norm_class)

