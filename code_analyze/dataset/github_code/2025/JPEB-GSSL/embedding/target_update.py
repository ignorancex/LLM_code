import torch
from torch_geometric import nn


def ema_target_weights(target_encoder, context_encoder, sf=0.9):
    # Update based on EMA
    common_keys = target_encoder.state_dict().keys()

    for k in common_keys:
        target_encoder.state_dict()[k] = (target_encoder.state_dict()[
            k]*sf)+(context_encoder.state_dict()[k]*(1-sf))
