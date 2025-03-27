import os


import sys
sys.path.append(".")
from argparse import ArgumentParser
from typing import Dict

import torch
from omegaconf import OmegaConf

from utils.common import instantiate_from_config

hyper_encoder_weight_change_list = [
    'hyper_encoder.entropy_bottleneck._offset',
    'hyper_encoder.entropy_bottleneck._quantized_cdf',
    'hyper_encoder.entropy_bottleneck._cdf_length',
    'hyper_encoder.gaussian_conditional._offset',
    'hyper_encoder.gaussian_conditional._quantized_cdf',
    'hyper_encoder.gaussian_conditional._cdf_length',
    'hyper_encoder.gaussian_conditional.scale_table'
]


def load_weight(weight_path: str) -> Dict[str, torch.Tensor]:
    weight = torch.load(weight_path, map_location='cpu')
    if "state_dict" in weight:
        weight = weight["state_dict"]

    pure_weight = {}
    for key, val in weight.items():
        if key.startswith("module."):
            key = key[len("module."):]
        pure_weight[key] = val

    return pure_weight

parser = ArgumentParser()
parser.add_argument("--cldm_config", type=str, default='configs/model/cldm_stage2.yaml')
parser.add_argument("--sd_weight", type=str, default='/mnt/massive/wangce/yyx/SDcompression/magc_ckpts/ckpts_stage2/v50_step=91999-lpips=0.2130.ckpt')
parser.add_argument("--hyper_encoder_weight", type=str, default='/mnt/massive/wangce/yyx/SDcompression/magc_ckpts_map_in_decoder/ckpts_stage1/bpp6.ckpt')
parser.add_argument("--output", type=str, default='pretrained/stage2_ft/bbp6.ckpt')
args = parser.parse_args()

model = instantiate_from_config(OmegaConf.load(args.cldm_config))

sd_weights = load_weight(args.sd_weight)
if args.hyper_encoder_weight is not None:
    hyper_encoder_weight = load_weight(args.hyper_encoder_weight)
scratch_weights = model.state_dict()

init_weights = {}
for weight_name in scratch_weights.keys(): # 依次给现在模型的所有参数赋予初值
    # find target pretrained weights for this weight
    if weight_name.startswith("control_"): # control_model from sd_weights
        suffix = weight_name[len("control_"):]
        target_name = f"model.diffusion_{suffix}"
        target_model_weights = sd_weights
    elif weight_name.startswith("cond_encoder."):  # none
        suffix = weight_name[len("cond_encoder."):]
        target_name = F"first_stage_model.{suffix}"
        target_model_weights = sd_weights

    elif weight_name.startswith("hyper_encoder."):  # hyper_encoder from ours
        target_name = weight_name
        target_model_weights = hyper_encoder_weight
        # 熵模型用了factorize
        if weight_name in hyper_encoder_weight_change_list:
            hyper_encoder_weight[weight_name] = torch.tensor([], dtype=torch.int32)
    else: 
        target_name = weight_name
        target_model_weights = sd_weights
    
    # if target weight exist in pretrained model
    print(f"copy weights: {target_name} -> {weight_name}")
    if target_name in target_model_weights:
        # get pretrained weight
        target_weight = target_model_weights[target_name]
        target_shape = target_weight.shape
        model_shape = scratch_weights[weight_name].shape
        # if pretrained weight has the same shape with model weight, we make a copy
        if model_shape == target_shape:
            init_weights[weight_name] = target_weight.clone()
        # else we copy pretrained weight with additional channels initialized to zero
        else:
            newly_added_channels = model_shape[1] - target_shape[1]
            oc, _, h, w = target_shape
            zero_weight = torch.zeros((oc, newly_added_channels, h, w)).type_as(target_weight)
            init_weights[weight_name] = torch.cat((target_weight.clone(), zero_weight), dim=1)
            print(f"add zero weight to {target_name} in pretrained weights, newly added channels = {newly_added_channels}")
    else:
        init_weights[weight_name] = scratch_weights[weight_name].clone()
        print(f"These weights are newly added: {weight_name}")

model.load_state_dict(init_weights, strict=True)
torch.save(model.state_dict(), args.output)
print("Done.")
