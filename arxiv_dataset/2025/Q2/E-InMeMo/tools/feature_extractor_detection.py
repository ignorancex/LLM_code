"""
Extract features for UnsupPR.
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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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


save_dir = f"./pascal-5i/VOC2012/{feature_name}_{split}_detection"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# else:
#     print(f"Directory exists at {save_dir}")
#     sys.exit()


meta_root = f"./evaluate_detection/object_{split}"
image_root = "D:/Pycharmprojects/ICLVP/pascal-5i/VOC2012/JPEGImages"
sys.stdout.flush()
with open(meta_root + '.txt') as f:
    examples = f.readlines()
if len(examples) == 0:
    print(f"zeros file.")
    sys.stdout.flush()

# print("examples: ", examples)
examples = [os.path.join(image_root, example.strip()+'.jpg') for example in examples]
# print("examples: ", examples)

imgs = []
global_features = []

for example in examples:
    try:
        path = os.path.join(example)
        img = Image.open(path).convert("RGB")
        img = center_crop(img)
        imgs.append(img)
        # print("length of imgs: ", len(imgs))
    except:
        print(f"Disappear {path}")
        sys.stdout.flush()

    if len(imgs) == 8:

        imgs = torch.stack(imgs).to(device)
        with torch.no_grad():
            features = model.forward_features(imgs)
            features = features.cpu().numpy()  # Move to CPU as soon as possible
            global_features.append(features)  # Append to CPU memory list

        imgs = []

imgs = torch.stack(imgs).to(device)
with torch.no_grad():
    features = model.forward_features(imgs)
    features = features.cpu().numpy()  # Move to CPU as soon as possible
    global_features.append(features)  # Append to CPU memory list

global_features = np.concatenate(global_features, axis=0).astype(np.float32)

save_file = os.path.join(save_dir, 'folder')
np.savez(save_file, examples=examples, features=global_features)

print('features shape: ', global_features.shape)
