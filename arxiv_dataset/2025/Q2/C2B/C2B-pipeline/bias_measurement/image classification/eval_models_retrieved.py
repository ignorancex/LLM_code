import argparse
import os.path

import numpy as np
import torch
from torch import softmax

from torchvision.models import (resnet50, ResNet50_Weights, resnet101, ResNet101_Weights, resnet152, ResNet152_Weights,
                                vit_b_16, ViT_B_16_Weights)
from tqdm.auto import tqdm

from retrieved_dataset import ImageNetRetrievedDataset

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser()
parser.add_argument('retrieved_dataset_path')
args = parser.parse_args()

retrieved_dataset_path = args.retrieved_dataset_path
source = retrieved_dataset_path.split('/')[-1].split(' - ')[-1]

weights = {
    'ViT_B_16_SWAG': (vit_b_16, ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1),
    'ResNet152_V2': (resnet152, ResNet152_Weights.IMAGENET1K_V2),
    'ResNet101_V2': (resnet101, ResNet101_Weights.IMAGENET1K_V2),
    'ResNet50_V2': (resnet50, ResNet50_Weights.IMAGENET1K_V2)
}

device = 0
batch_size = 256
num_workers = 4

for model_name, (model, weights) in weights.items():
    if os.path.exists(f'results_{model_name}_retrieved_{source}.npy'):
        continue

    transforms = weights.transforms()
    dataset = ImageNetRetrievedDataset(retrieved_dataset_path, transform=transforms)
    model = model(weights=weights)

    model.eval()
    model.to(device)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False
    )

    res = []

    with torch.no_grad():
        for img, target in tqdm(loader, desc=f"Evaluating {model_name} on {source}-retrieved"):
            img = img.to(device)
            target = target.cpu().numpy()
            output = softmax(model(img), dim=-1).cpu().numpy()

            pred = np.argmax(output, axis=-1)
            correct = pred == target

            class_score = output[np.arange(len(target)), target]
            class_rank = np.argwhere((np.argsort(output, axis=1)[:, ::-1] - target[..., None]) == 0)[:, 1]

            res.append(np.stack([target.astype(np.float32),
                                       correct.astype(np.float32),
                                       class_score.astype(np.float32),
                                       class_rank.astype(np.float32)], axis=1))

    np.save(f'results_{model_name}_retrieved_{source}.npy', np.concatenate(res, axis=0))
