import torch.nn as nn
import torch
from sao.eco import ECO_Constructor
from sao.delta import Delta_Constructor, Linear_Constructor
import torch.nn.utils.prune

def Delta_Init(model, **kwargs):
    with torch.no_grad():
        for name_m, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                print("processing {} of shape {}".format(name_m, module.weight.shape))
                vals = Delta_Constructor(module, **kwargs)
                if isinstance(vals, tuple):
                    module.weight = nn.Parameter(vals[0] * kwargs['gain'])
                    torch.nn.utils.prune.custom_from_mask(module, "weight", torch.abs(vals[1]))
                else:
                    module.weight = nn.Parameter(vals * kwargs['gain'])
                if hasattr(module, "bias"):
                    torch.nn.init.normal_(module.bias.data, mean=0.0, std=kwargs['sigma_b'])
            elif isinstance(module, nn.Linear):
                torch.nn.init.orthogonal_(module.weight, kwargs['gain'])
                if hasattr(module, "bias"):
                    torch.nn.init.normal_(module.bias.data, mean=0.0, std=kwargs['sigma_b'])

    return model


def Delta_ECO_Init(model, **kwargs):
    for _, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.padding_mode != "circular":
            vals = Delta_Constructor(module, **kwargs)
            if isinstance(vals, tuple):
                module.weight = nn.Parameter(vals[0])
                torch.nn.utils.prune.custom_from_mask(
                    module, "weight", torch.abs(vals[1])
                )
            else:
                module.weight = nn.Parameter(vals)
        elif isinstance(module, nn.Conv2d) and module.padding_mode == "circular":
            module.weight = nn.Parameter(ECO_Constructor(module, **kwargs))
            torch.nn.utils.prune.custom_from_mask(
                module, "weight", (module.weight != 0) * 1
            )
        elif isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, 1)

    return model


def ECO_Init(model, **kwargs):
    for _, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            module.weight = nn.Parameter(ECO_Constructor(module, **kwargs))
            torch.nn.utils.prune.custom_from_mask(
                module, "weight", (module.weight != 0) * 1
            )
        elif isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, 1)

    return model


def Kaiming_Init(model, args):
    for _, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(
                module.weight, mode="fan_out", nonlinearity=args.activation
            )
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, 1)

    return model

def Linear_Init(model, **kwargs):
    with torch.no_grad():
        for name_m, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if module.weight.shape[1] != module.weight.shape[0]:
                    torch.nn.init.orthogonal_(module.weight, kwargs['gain'])
                else:
                    vals = Linear_Constructor(module, **kwargs)
                    module.weight = nn.Parameter(vals[0] * kwargs['gain'])
                    torch.nn.utils.prune.custom_from_mask(module, "weight", torch.abs(vals[1]))
                if hasattr(module, "bias"):
                    torch.nn.init.normal_(module.bias.data, mean=0.0, std=kwargs['sigma_b'])
    return model
