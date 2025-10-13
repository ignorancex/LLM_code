import torch
from pathlib import Path
import os
import argparse
from thingsvision import get_extractor
from PIL import Image
import numpy as np

from torchvision import models, transforms
from lucent.optvis import render, objectives, transform
import lucent.optvis.param as param


parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str)
parser.add_argument('--basedir', type=str)
parser.add_argument('--module', type=str)
parser.add_argument('--neuron', type=int)
parser.add_argument('--type', type=str)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--jitter', type=int, default=16)
args = parser.parse_args()


print(f"BEGIN MODULE {args.module} NEURON {args.neuron}")

savedir = os.path.join(args.basedir, args.network, args.module, f"unit{args.neuron}")
Path(savedir).mkdir(parents=True, exist_ok=True)

device_str = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
transform.device = torch.device(device_str)
param.spatial.device = torch.device(device_str)
param.color.device = torch.device(device_str)

extractor = get_extractor(model_name=args.network, source='torchvision', device=device_str, pretrained=True)

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

extractor.model.eval()

obj = objectives.neuron(args.module.replace('.', '_'), args.neuron)
imgs = render.render_vis(extractor.model, obj, param_f=param_f, transforms=augs, thresholds=(2560,), show_image=False)

img = Image.fromarray((imgs[0][0]*255).astype(np.uint8))
img.save(os.path.join(savedir, f"0_distill_center.png"))