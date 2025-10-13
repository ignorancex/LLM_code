import torch
import numpy as np

from utils.metrics import extract_topk_largest_candidates
from monai.transforms import (LoadImaged, AddChanneld, Compose, CropForegroundd, Orientationd, ScaleIntensityRanged,
                              Spacingd, ScaleIntensityRangePercentilesd)
from local_data.constants import PATH_DATASETS

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_query(args):

    transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            SelectAdaptationOrgan(args),
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(args.model_cfg["space_x"], args.model_cfg["space_y"],
                                                      args.model_cfg["space_z"]), mode=("bilinear", "nearest")),
            select_intensity_scaling(args),
            CropForegroundd(keys=["image", "label"], source_key="image"),
        ]
    )

    # Train dict part
    test_samples = 0
    img, lbl, name = [], [], []
    for iPartition in ['test']:
        for line in open(args.data_txt_path[iPartition]):
            item = line.strip().split()[0].split('/')[1]

            img.append(PATH_DATASETS + line.strip().split()[0])
            lbl.append(PATH_DATASETS + line.strip().split()[1])
            name.append(line.strip().split()[1].split('.')[0])

            test_samples += 1

    data_dicts = [{'image': image, 'label': label, 'name': name}
                  for image, label, name in zip(img, lbl, name)]

    # Get sample info
    dict_data = data_dicts[args.iFold-1]

    # Load image and labels
    dict_data = transforms(dict_data)

    image = dict_data['image']
    if args.objective == "binary":
        label = (dict_data['label'] == 1).astype(int)
        label = np.expand_dims(extract_topk_largest_candidates(np.squeeze(label), 1), 0)
    else:
        label = dict_data['label'].astype(int)

    # Select image id name
    if "TotalSegmentator" in data_dicts[args.iFold-1]["name"]:
        id_test = data_dicts[args.iFold - 1]["name"].split("/")[-3]
    else:
        id_test = data_dicts[args.iFold - 1]["name"].split("/")[-1]

    args.query = {'image': image, 'label': label, "name": id_test}

class CategoricalToOneHot():
    def __init__(self, args):
        self.classes = len(args.dataset_indexes) + 1
        self.objective = args.objective

    def __call__(self, data):
        y = data['label']

        # one hot encoding
        y = torch.nn.functional.one_hot(y.to(torch.long), num_classes=self.classes).permute(
            (-1, 1, 2, 3, 0)).squeeze(-1)

        # quit background class
        if self.objective == "binary":
            y = y[1:, :, :, :]

        data['label'] = y
        return data

class SelectRelevantKeys():
    def __call__(self, data):
        #if np.array(data["label"]).sum() == 0:
        #    print(data["name"])
        d = {key: data[key] for key in ['image', 'label', 'name']}
        return d

class SelectAdaptationOrgan():

    def __init__(self, args):
        self.indexes = args.dataset_indexes

    def __call__(self, data):
        if "KiPA" in data["name"]:  # Select kidney
            data["label"] = MetaTensor(torch.tensor(np.int8(data["label"] == 2)), meta=data["label"].meta)
        if "FLARE" in data["name"]:
            # Select only target categories in mask
            mask = np.int8(np.zeros_like(data["label"]))
            for i in range(len(self.indexes)):
                mask += (np.int8(data["label"] == self.indexes[i]) * (i+1))
            # Create new label mask
            data["label"] = MetaTensor(torch.tensor(mask), meta=data["label"].meta)

        return data

def select_intensity_scaling(args):
    if "kipa" in args.data_txt_path["train"]:
        transform = ScaleIntensityRangePercentilesd(keys=["image"], lower=10, upper=90, b_min=args.model_cfg["b_min"],
                                                    b_max=args.model_cfg["b_max"], clip=True)
    else:
        transform = ScaleIntensityRanged(keys=["image"], a_min=args.model_cfg["a_min"], a_max=args.model_cfg["a_max"],
                                         b_min=args.model_cfg["b_min"], b_max=args.model_cfg["b_max"], clip=True)
    return transform