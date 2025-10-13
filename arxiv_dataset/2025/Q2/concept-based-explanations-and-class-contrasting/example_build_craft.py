from torch.utils.data import DataLoader, TensorDataset

import torch
import torch.nn as nn

from tqdm import tqdm

from core.core import nmf_attribution_whole_ds_decomp, collect_patches
from core.imagenet_utils import get_dataset_for_class, load_model
from core.config import DEVICE, OUTPUT_PATH

from craft.craft_torch import Craft, torch_to_numpy
import numpy as np
from math import ceil
import torchvision
import os
from pathlib import Path


model_name = "resnet50"
model = load_model(model_name).eval().to(DEVICE)
layer = model.layer4[2]
layer_name = "_layer4[2]"

model_till_latent = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3, model.layer4)
model_latent_till_out = lambda x: model.fc(torch.mean(x, dim=(2, 3)))

for class_idx in tqdm(range(1000), ascii=True):

  save_path = os.path.join(OUTPUT_PATH, "craft", model_name + layer_name, str(class_idx), "nmf")
  Path(save_path).mkdir(parents=True, exist_ok=True)

  craft = Craft(input_to_latent = model_till_latent,
                latent_to_logit = model_latent_till_out,
                number_of_concepts = 10,
                patch_size = 74,
                batch_size = 64,
                device = "cuda")

  dataset = get_dataset_for_class(model_name="resnet50", class_idx=class_idx, use_train_ds=True)
  dataloader = DataLoader(dataset, shuffle=True, batch_size=1, num_workers=0)

  torch.manual_seed(42)
  patches = collect_patches(dataloader)

  patches = patches[:500]

  crops, crops_u, w = craft.fit(patches)
  #crops = np.moveaxis(torch_to_numpy(crops), 1, -1)

  #print(crops.shape, crops_u.shape, w.shape)


  importances = craft.estimate_importance(patches, class_id=class_idx) # 330 is the rabbit class id in imagenet

  np.save(os.path.join(save_path, "importances.npy"), importances)
  np.save(os.path.join(save_path, "nmf.npy"), w)


  """
  images_u = craft.transform(patches)

  print(images_u.shape)

  most_important_concepts = np.argsort(importances)[::-1]#[:5]

  print("importances shape: {}".format(importances.shape))

  for c_id in most_important_concepts:
    print("Concept", c_id, " has an importance value of ", importances[c_id])


  nb_crops = 10
  for c_id in most_important_concepts:

    best_crops_ids = np.argsort(crops_u[:, c_id])[::-1][:nb_crops]
    best_crops = crops[best_crops_ids]

    print("Concept", c_id, " has an importance value of ", importances[c_id])

    print(best_crops.shape)
    print(best_crops.dtype)
    #best_crops = best_crops
    best_crops = torch.from_numpy(best_crops)
    best_crops = torch.permute(best_crops, (0, 3, 1, 2))
    torchvision.utils.save_image(best_crops, os.path.join(OUTPUT_PATH, "tmp", str(c_id) + ".jpg"))
  """
