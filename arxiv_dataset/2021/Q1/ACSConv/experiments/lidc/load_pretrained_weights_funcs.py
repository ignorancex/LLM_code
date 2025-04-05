import torch
import torch.nn as nn
from collections import OrderedDict

def load_video_pretrained_weights(model, video_resnet18_pretrain_path):
    state_dict = torch.load(video_resnet18_pretrain_path)['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove 'module.' of dataparallel
        new_state_dict[name]=v
    new_state_dict.pop('fc.weight')
    new_state_dict.pop('fc.bias')
    incompatible_keys = model.backbone.load_state_dict(new_state_dict, strict=False)
    print('load video pretrained weights\n', incompatible_keys)
    return model

def load_mednet_pretrained_weights(model, mednet_resnet18_pretrain_path):
    model.backbone.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(1, 1, 1), padding=(3, 3, 3), bias=False)
    state_dict = torch.load(mednet_resnet18_pretrain_path)['state_dict']    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove 'module.' of dataparallel
        new_state_dict[name]=v
    incompatible_keys = model.backbone.load_state_dict(new_state_dict, strict=False)
    print('load mednet pretrained weights\n', incompatible_keys)
    return model