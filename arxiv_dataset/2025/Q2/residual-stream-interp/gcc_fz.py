from pathlib import Path
import os
import argparse
from PIL import Image
import numpy as np
import torch
from torch import nn
from torchvision import models, transforms
from lucent.optvis import render, objectives, transform
import lucent.optvis.param as param

from group_cc import ModelWrapper


parser = argparse.ArgumentParser()
parser.add_argument('--basedir', type=str)
parser.add_argument('--gcc', type=str)
parser.add_argument('--module', type=str, default='0')
parser.add_argument('--neuron', type=int)
parser.add_argument('--chunk', type=int, default=-1)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--jitter', type=int, default=16)
parser.add_argument('--direction', action='store_true', default=False)
args = parser.parse_args()

device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
transform.device = device
param.spatial.device = device
param.color.device = device

# model = models.resnet18(True)
model = models.resnet50(weights='IMAGENET1K_V2')
gcc_states = torch.load(f'/media/andrelongon/DATA/resnet_sae_exp/ckpts/layer3.5.prerelu_out/{args.gcc}.pth')

if 'bn2' in args.module:
    # layer_dirs = torch.transpose(gcc_states['W_enc'], 0, 1)
    layer_dirs = gcc_states['W_dec']#[:, 256:]
else:
    # layer_dirs = torch.transpose(gcc_states['W_enc'], 0, 1)
    layer_dirs = gcc_states['W_dec']

layer_dirs = layer_dirs.to(device)
if args.direction:
    print(f"DIRECTION FOR LATENT {args.neuron}")
    subdir = f'direction/{args.module}'
    obj = objectives.neuron_weight(args.module.replace('.', '_'), layer_dirs[args.neuron])
else:
    print(f"LATENT {args.neuron}")
    subdir = 'latent'
    model = ModelWrapper(model, 32, device, use_gcc=True)
    states = model.state_dict()
    if args.chunk != -1:
        layer_dirs = torch.reshape(layer_dirs, (layer_dirs.shape[0], 2, 512))[:, args.chunk]
    states['map.weight'] = layer_dirs
    model.load_state_dict(states)
    model = nn.Sequential(model)
    obj = objectives.neuron(args.module, args.neuron)

savedir = os.path.join(args.basedir, args.gcc, subdir, f"unit{args.neuron}")
Path(savedir).mkdir(parents=True, exist_ok=True)
model.to(device).eval()

augs = None
if args.jitter < 4:
    augs = [
        transform.random_scale([1, 0.975, 1.025, 0.95, 1.05]),
        transform.random_rotate([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]),
    ]
else:
    augs = [
        transform.pad(args.jitter),
        transform.jitter(args.jitter),
        transform.random_scale([1, 0.975, 1.025, 0.95, 1.05]),
        transform.random_rotate([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]),
        transform.jitter(int(args.jitter/2)),
        transforms.CenterCrop(224),
    ]
param_f = lambda: param.images.image(224, decorrelate=True)

imgs = render.render_vis(model, obj, param_f=param_f, transforms=augs, thresholds=(2560,), show_image=False)

img = Image.fromarray((imgs[0][0]*255).astype(np.uint8))
img.save(os.path.join(savedir, f"0_dec_distill_center.png"))