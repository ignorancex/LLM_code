"""Extract features from all Imagenet benchmarks."""
import collections
import os, sys
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(ROOT_DIR)
import argparse
from tqdm import tqdm

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import ResNet50_Weights, Swin_T_Weights, ViT_B_16_Weights, RegNet_Y_16GF_Weights
from torchvision import transforms as trn
from torch.hub import load_state_dict_from_url

from openood.networks import ResNet50, Swin_T, ViT_B_16, RegNet_Y_16GF
from openood.evaluation_api.datasets import DATA_INFO, data_setup, get_id_ood_dataloader
from openood.evaluation_api.preprocessor import get_default_preprocessor


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

@torch.no_grad()
def extract_features_batch(net, data, model="vit-b-16"):
    if model == 'dinov2':
        features = net.backbone(data)
    else:
        _, features = net(data, return_feature=True)
    return features


def extract_features(
        net: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        model: str = "vit-b-16",
        progress: bool = True
    ):
    feature_list, label_list = [], []
    for batch in tqdm(data_loader, disable=not progress):
        data = batch['data'].cuda()
        label = batch['label'].cuda()
        feats = extract_features_batch(net, data, model=model)

        feature_list.append(feats.cpu())
        label_list.append(label.cpu())

    # convert values into numpy array
    features = torch.cat(feature_list, 0)
    labels = torch.cat(label_list).long()
    return features, labels


parser = argparse.ArgumentParser()
parser.add_argument('--arch',
                    default='resnet50',
                    choices= clip.available_models() + [
                        'resnet50', 'swin-t', 'vit-b-16', 'regnet',
                        'ViT-B/14', 'ViT-S/14', 'ViT-L/14'])
parser.add_argument('--model-type',
                    default='',
                    choices=['clip', 'dinov2'])
parser.add_argument('--tvs-version', default=1, choices=[1, 2])
parser.add_argument('--tvs-pretrained', action='store_true')
parser.add_argument('--fsood', action='store_true')
parser.add_argument('--batch-size', default=200, type=int)
args = parser.parse_args()

# assuming the model is either
# 1) torchvision pre-trained; or
# 2) Dinov2
if args.tvs_pretrained:
    if args.arch == 'resnet50':
        net = ResNet50()
        weights = eval(f'ResNet50_Weights.IMAGENET1K_V{args.tvs_version}')
        net.load_state_dict(load_state_dict_from_url(weights.url))
        preprocessor = weights.transforms()
    elif args.arch == 'swin-t':
        net = Swin_T()
        weights = eval(f'Swin_T_Weights.IMAGENET1K_V{args.tvs_version}')
        net.load_state_dict(load_state_dict_from_url(weights.url))
        preprocessor = weights.transforms()
    elif args.arch == 'vit-b-16':
        net = ViT_B_16()
        weights = eval(f'ViT_B_16_Weights.IMAGENET1K_V{args.tvs_version}')
        net.load_state_dict(load_state_dict_from_url(weights.url))
        preprocessor = weights.transforms()
    elif args.arch == 'regnet':
        net = RegNet_Y_16GF()
        weights = eval(
            f'RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_E2E_V{args.tvs_version}')
        net.load_state_dict(load_state_dict_from_url(weights.url))
        preprocessor = weights.transforms()
    else:
        raise NotImplementedError
elif args.model_type == 'dinov2':
    model_tag = args.arch.lower().replace('/', '').replace('-', '')
    net = torch.hub.load('facebookresearch/dinov2', f'dinov2_{model_tag}_lc')

    preprocessor = trn.Compose([
        trn.Resize(256, interpolation=trn.InterpolationMode.BICUBIC),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
else:
    raise NotImplementedError

net.cuda()
net.eval()

model = args.arch if args.model_type == '' else args.model_type
# Setup data loaders
id_name = "imagenet"
data_root = os.path.join(ROOT_DIR, 'data')

if preprocessor is None:
    preprocessor = get_default_preprocessor(id_name)


data_setup(data_root, id_name)
loader_kwargs = {
    'batch_size': args.batch_size,
    'shuffle': False,
    'num_workers': 8,
}
dataloader_dict = get_id_ood_dataloader(id_name, data_root,
                                        preprocessor, **loader_kwargs)


# Loop over each dataset and extract features
print(f"Extracting features from {id_name} training set...")
feats, labels = extract_features(net, dataloader_dict['id']['train'], model=model)
fname = f"{model}-img1k-feats.pkl"
print(f"Saving train set features to {fname}")
torch.save(dict(feats=feats, labels=labels), fname)

print(f"Extracting features from {id_name} test set...")
feats, labels = extract_features(net, dataloader_dict['id']['test'], model=model)
fname = f"{model}-img1k-test-feats.pkl"
print(f"Saving test set features to {fname}")
torch.save(dict(feats=feats, labels=labels), fname)

for split,v in dataloader_dict['ood'].items():
    if split == 'val':
        continue
    print(f"Extracting features from {split} OOD datasets...")
    for dset,loader in v.items():
        print(f"Extracting features from {dset}...")
        feats, labels = extract_features(net, loader, model=model)
        fname = f"{model}-img1k-{split}-{dset}-feats.pkl"
        print(f"Saving {dset} set features to {fname}")
        torch.save(dict(feats=feats, labels=labels), fname)
