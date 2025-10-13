import argparse
import os
import numpy as np
import torch
import torchvision
from torchvision import models, transforms
from thingsvision import get_extractor

from group_cc import ModelWrapper
from stream_inspect import get_activations


parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', type=str)
parser.add_argument('--scale', type=float, default=1)
parser.add_argument('--save_dir', type=str)
parser.add_argument('--batch_size', type=int, default=2048)
args = parser.parse_args()

device = "cuda:1" if torch.cuda.is_available() else "cpu"
scale = args.scale
IMAGE_SIZE = 224
# specify ImageNet mean and standard deviation
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
norm_transform = transforms.Normalize(mean=MEAN, std=STD)

#   scale = 1 -> no scale
#   scale > 1 -> crop to scale fraction of 224, then resize to 224
#   scale < 1 -> crop to 224, then resize to fraction of 224, pad to 224
transform = None
if scale == 1:
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
    ])
elif scale > 1:
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(int(IMAGE_SIZE*(scale-1))),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])
elif scale < 1:
    resize = int(IMAGE_SIZE*scale) if int(IMAGE_SIZE*scale) % 2 == 0 else math.ceil(IMAGE_SIZE*scale)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.Resize(resize),
        transforms.Pad(int((IMAGE_SIZE - resize) / 2), padding_mode='constant', fill=fill),
        transforms.ToTensor(),
    ])

imagenet_data = torchvision.datasets.ImageFolder(args.train_dir, transform=transform)
trainloader = torch.utils.data.DataLoader(imagenet_data, batch_size=args.batch_size, shuffle=False, drop_last=False)

# model = models.resnet50(weights='IMAGENET1K_V2').to(device).eval()
# model = ModelWrapper(model, 0, device)

extractor = get_extractor(
    model_name='resnet50',
    source='torchvision',
    device="cuda" if torch.cuda.is_available() else "cpu",
    pretrained=True,
    model_parameters={'weights': 'IMAGENET1K_V2'}
)

all_acts = []
for i, data in enumerate(trainloader, 0):
    inputs, _ = data
    inputs = norm_transform(inputs.to(device))

    acts = get_activations(extractor, inputs, 'layer4.1.bn2', None, None, use_center=True)

    # acts = model(inputs).cpu().detach().numpy()
    # center_coord = acts.shape[-1] // 2
    # acts = acts[:, :, center_coord, center_coord]

    all_acts += acts.tolist()

np.save(os.path.join(args.save_dir, f'resnet50_4.1.bn2_scale{scale}.npy'), np.array(all_acts))