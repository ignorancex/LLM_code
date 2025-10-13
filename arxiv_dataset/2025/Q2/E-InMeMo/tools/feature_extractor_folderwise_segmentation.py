"""
Extract features for feature-map-level retriever (FMLR).
"""
import os
import sys
import glob
from PIL import Image
from tqdm import tqdm
import numpy as np

import torch
import torchvision.models as models
from torchvision import transforms as T
from torch.nn import functional as F

import timm
import sys


model_name = sys.argv[1]
feature_name = sys.argv[2]
split = sys.argv[3]
dataset_name = sys.argv[4]
gpu = str(sys.argv[5])
device = torch.device(gpu if torch.cuda.is_available() else "cpu")

"""
model names:
# vit_large_patch14_224_clip_laion2b
# eva_large_patch14_196.in22k_ft_in22k_in1k
# resnet50
# vit_large_patch16_224.augreg_in21k_ft_in1k
# resnet18
# vit_large_patch14_clip_224.laion2b_ft_in12k_in1k
# vit_base_patch16_224.dino
# vit_large_patch14_dinov2.lvd142m
# vit_large_patch14_clip_336.openai_ft_in12k_in1k
"""
model = timm.create_model(model_name, pretrained=True)
model.eval()
model = model.to(device)


# load the image transformer
t = []
t.append(T.Resize(model.pretrained_cfg['input_size'][1], interpolation=Image.BICUBIC))
t.append(T.CenterCrop(model.pretrained_cfg['input_size'][1]))
t.append(T.ToTensor())
t.append(T.Normalize(model.pretrained_cfg['mean'], model.pretrained_cfg['std']))
center_crop = T.Compose(t)


# Dataset-specific configuration
data_configs = {
    'pascal': {
        'save_dir': './pascal-5i/VOC2012',
        'meta_root': './evaluate/splits/pascal',
        'image_root': './pascal-5i/VOC2012/JPEGImages', # TODO
        'num_folders': 4
    },
    'isic': {
        'save_dir': './med_dataset/ISIC2016',
        'meta_root': './evaluate/splits/isic',
        'image_root': './med_dataset/ISIC2016/images',
        'num_folders': 1
    },
    'kvasir': {
        'save_dir': './med_dataset/Kvasir-SEG',
        'meta_root': './evaluate/splits/kvasir',
        'image_root': './med_dataset/Kvasir-SEG/images',
        'num_folders': 5
    }
}

cfg_ds = data_configs[dataset_name]

save_dir = os.path.join(cfg_ds['save_dir'], f"{feature_name}_{split}")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

meta_root = os.path.join(cfg_ds['meta_root'], f"{split}")

image_root = os.path.join(cfg_ds['image_root'])

for folder_id in tqdm(range(cfg_ds['num_folders'])):
    print(f"Processing folder {folder_id}")
    sys.stdout.flush()
    with open(os.path.join(meta_root, 'fold'+str(folder_id)+'.txt')) as f:
        examples = f.readlines()
    if len(examples) == 0:
        print(f"zeros folder{folder_id}")
        sys.stdout.flush()
        continue

    examples = [os.path.join(image_root, example.strip()[:-4]+'.jpg') for example in examples]

    imgs = []

    global_features = []

    for example in examples:
        try:
            path = os.path.join(example)
            img = Image.open(path).convert("RGB")
            img = center_crop(img)
            imgs.append(img)
        except:
            print(f"Disappear {path}")
            sys.stdout.flush()

        if len(imgs) == 32:
            imgs = torch.stack(imgs).to(device)
            with torch.no_grad():
                features = model.forward_features(imgs)
                if 'img' in feature_name:
                    print("Using the image features")
                    features = model.forward_head(features, pre_logits=True)
                    print("img_features shape: ", features.shape)
                elif 'cls' in feature_name:
                    print("Using the cls features")
                    # Extract [CLS] token and normalize
                    cls_features = features[:, 0]
                    print("cls_features shape: ", cls_features.shape)
                    features = F.normalize(cls_features, dim=1)
                features = features.cpu().numpy()
                global_features.append(features)

            imgs = []

    if len(imgs) > 0:
        imgs = torch.stack(imgs).to(device)
        with torch.no_grad():
            features = model.forward_features(imgs)
            if 'img' in feature_name:
                print("Using the image features")
                features = model.forward_head(features, pre_logits=True)
            elif 'cls' in feature_name:
                print("Using the cls features")
                # Extract [CLS] token and normalize
                cls_features = features[:, 0]
                features = F.normalize(cls_features, dim=1)
            features = features.cpu().numpy()
            global_features.append(features)

    # Concatenate all features on CPU
    global_features = np.concatenate(global_features, axis=0).astype(np.float32)

    save_file = os.path.join(save_dir, 'folder' + str(folder_id))
    np.savez(save_file, examples=examples, features=global_features)

    print('features shape: ', global_features.shape)