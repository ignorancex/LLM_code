import torch
import torch.nn.functional as F


def dropout_matmul(input: torch.Tensor, weight: torch.Tensor, p: float):
    x = F.dropout(input, p=p)
    return F.linear(x, weight)
