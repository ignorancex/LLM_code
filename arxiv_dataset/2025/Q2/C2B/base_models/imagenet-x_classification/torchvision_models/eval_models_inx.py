import os.path

import numpy as np
import torch
from torch import softmax
from torchvision.datasets import ImageNet
from torchvision.models import (resnet50, ResNet50_Weights, resnet101, ResNet101_Weights, resnet152, ResNet152_Weights,
                                vit_b_16, ViT_B_16_Weights)
from tqdm.auto import tqdm

from imagenet_x import FACTORS
from imagenet_x.evaluate import ImageNetX

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Fix deprecated np.bool -> bool to work with numpy >= 1.24
def fixed_getitem(self, index):
    img, target = ImageNet.__getitem__(self, index)
    img_id = self.samples[index][0].split("/")[-1]
    img_annotations = self.annotations_.loc[img_id]
    return img, target, img_annotations[FACTORS].values.astype(bool)


ImageNetX.__getitem__ = fixed_getitem

# Declare dataset
imagenet_val_path = '../data/imagenet/'

weights = {
    'ViT_B_16_SWAG': (vit_b_16, ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1),
    'ResNet152_V2': (resnet152, ResNet152_Weights.IMAGENET1K_V2),
    'ResNet101_V2': (resnet101, ResNet101_Weights.IMAGENET1K_V2),
    'ResNet50_V2': (resnet50, ResNet50_Weights.IMAGENET1K_V2)
}

# Load the model
device = 0
batch_size = 256
num_workers = 2

for model_name, (model, weights) in weights.items():
    if os.path.exists(f'results_inx_{model_name}.npy'):
        continue

    transforms = weights.transforms()
    dataset = ImageNetX(imagenet_val_path, transform=transforms, which_factor='multi', filter_prototypes=False)
    model = model(weights=weights)

    # Evaluate model on ImageNetX using simple loop
    model.eval()
    model.to(device)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    res = []

    with torch.no_grad():
        for data, target, annotations in tqdm(loader, desc=f"Evaluating {model_name} on Imagenet-X"):
            data = data.to(device)
            target = target.cpu().numpy()
            annotations = annotations.cpu().numpy().astype(np.float32)
            output = softmax(model(data), dim=-1).cpu().numpy()

            pred = np.argmax(output, axis=-1)
            correct = pred == target

            class_score = output[np.arange(len(target)), target]
            class_rank = np.argwhere((np.argsort(output, axis=1)[:, ::-1] - target[..., None]) == 0)[:, 1]

            batch_res = np.stack([target.astype(np.float32),
                                       pred.astype(np.float32),
                                       correct.astype(np.float32),
                                       class_score.astype(np.float32),
                                       class_rank.astype(np.float32)], axis=1)

            res.append(np.concatenate([batch_res, annotations], axis=1))

    # Save results
    np.save(f'results_inx_{model_name}.npy', np.concatenate(res, axis=0))
