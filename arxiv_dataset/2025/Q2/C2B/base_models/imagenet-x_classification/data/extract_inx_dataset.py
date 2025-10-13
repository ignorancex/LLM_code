import numpy as np
import pandas as pd
from imagenet_x.evaluate import ImageNetX, FACTORS
from torchvision.datasets import ImageNet
from tqdm.auto import tqdm


# Fix deprecated np.bool -> bool to work with numpy >= 1.24
def fixed_getitem(self, index):
    img, target = ImageNet.__getitem__(self, index)
    img_id = self.samples[index][0].split("/")[-1]
    img_annotations = self.annotations_.loc[img_id]
    return img, target, img_annotations[FACTORS].values.astype(bool)


ImageNetX.__getitem__ = fixed_getitem

imagenet_val_path = './imagenet/'
dataset = ImageNetX(imagenet_val_path, which_factor='multi', filter_prototypes=False)

res = []

for img_path, target, annotations in tqdm(dataset):
    res.append({'img_path': img_path, 'target': target, **dict(zip(FACTORS, annotations.astype(np.int32)))})

pd.DataFrame.from_records(res).to_feather('inx_dataset.feather')
