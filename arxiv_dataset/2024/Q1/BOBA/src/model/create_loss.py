import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset


def create_loss(name='ce'):
    """
    loss function must be differentiable
    """
    if name == 'ce':  # cross-entropy loss
        return F.cross_entropy
    elif name == 'bce':  # binary cross-entropy loss
        return lambda output, target: F.binary_cross_entropy_with_logits(output.view(-1), target.view(-1).float())
    else:
        raise NotImplementedError('Unknown loss name: %s' % name)
