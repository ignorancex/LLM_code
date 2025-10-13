import torch
import torch.hub
import torch.nn as nn
import torch.nn.functional as F
from .dinofeatup import DINOv2FeatUp


def create_model(name, **kwargs):
    if name == "dinov2featup":
        return DINOv2FeatUp(**kwargs)
    elif name == "dinov2featup_recons":
        return DINOv2FeatUp(reconstruction=True, **kwargs)
    else:
        raise ValueError("Invalid model name '{}' provided".format(name))
