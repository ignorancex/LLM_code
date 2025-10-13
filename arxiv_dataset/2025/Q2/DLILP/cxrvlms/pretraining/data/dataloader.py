"""
Dataset and Dataloader preparation for vision-language pre-training
"""
import os.path

import numpy as np
import pandas as pd
import ast
import random
import os
import torch

from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from tqdm import tqdm

from cxrvlms.pretraining.data.dataset import Dataset
from cxrvlms.pretraining.data.transforms import LoadImage, SelectRelevantKeys, ProduceDescription


def get_loader(dataframes_path, data_root_path, datasets, batch_size=8, num_workers=0, banned_categories=None,
               caption="A radiology image of [CLS]", cache=False, save_cache=True, norm=False, size=224):

    """
    Dataloaders generation for vision-language pretraining. Read all partitions from assembly model and combines
    them into a unified dataframe. Also, a dataloader is conditioned for training.
    """

    # Assembly partitions into a combined data structure
    print("Setting assebly data...")
    data = []
    for iDataset in datasets:
        print("Processing data: " + iDataset)

        dataframe = pd.read_csv(dataframes_path + iDataset + ".csv")

        for i in range(len(dataframe)):
            data_i = dataframe.loc[i, :].to_dict()

            # Remove banned words - for evaluating on incremental categories
            banned = False
            if banned_categories is not None:
                for iCat in data_i["study_categories"]:
                    for iiCat in banned_categories:
                        if iiCat in iCat:
                            banned = True
            if banned:
                continue

            # Add sample to general data
            data_i['study_categories'] = ast.literal_eval(data_i['study_categories'].lower())
            data_i['prompts_categories'] = ast.literal_eval(data_i['prompts_categories'].lower())
            data_i['prompts'] = ast.literal_eval(data_i['prompts'].lower())
            data_i["image_name"] = data_i["image"]
            data_i["image_path"] = data_root_path + data_i["image"]
            data.append(data_i)

    print('Total assembly data samples: {}'.format(len(data)))

    # Test/Val partition
    i = list(range(len(data)))
    random.Random(42).shuffle(i)  # shuffle the list

    idx_train, idx_val = i[int(len(data)/10):], i[:int(len(data)/10)]
    data_train, data_val = list(map(data.__getitem__, idx_train)), list(map(data.__getitem__, idx_val))

    data_train, data_val = data_train[0:200], data_val[0:100]
    print('Total training samples: {}'.format(len(data_train)))
    print('Total validation samples: {}'.format(len(data_val)))

    # Prepare data transforms for loader
    transforms = Compose([
        LoadImage(size=(size, size), canvas=True, norm=norm),
        ProduceDescription(caption=caption),
        SelectRelevantKeys()
    ])

    # Set data
    train_dataset = Dataset(data=data_train, transform=transforms, cache=cache, size=size)
    val_dataset = Dataset(data=data_val, transform=transforms, cache=cache, size=size)

    # Set dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                            pin_memory=True)

    # Preload dataset if all samples are stored in cache
    if cache:
        path_cache = "./local_data/cache/" + str("__".join(datasets)) + "/"
        print("Loading samples in cache memory prior to training")
        if os.path.exists(path_cache):
            print("Directly loading the pre-processed samples...")
            train_dataset.shared_array_img[:] = torch.tensor(np.load(path_cache + "/train_data.npy"))
            val_dataset.shared_array_img[:] = torch.tensor(np.load(path_cache + "/val_data.npy"))
            train_dataset.shared_array_flags[:] = 1
            val_dataset.shared_array_flags[:] = 1
            print("Done!")
        else:
            print("Loading and pre-processing samples...")
            # Train
            iterator_train = tqdm(train_loader, dynamic_ncols=True)
            for step, batch in enumerate(iterator_train):
                iterator_train.set_description("Loading Training samples: (%d / %d Steps) " % (
                    step + 1, len(train_loader)))
            # Validation
            iterator_val = tqdm(val_loader, dynamic_ncols=True)
            for step, batch in enumerate(iterator_val):
                iterator_val.set_description("Loading validation samples: (%d / %d Steps) " % (
                    step + 1, len(val_loader)))
            print("Done!")

            if save_cache:
                print("Storage pre-processed samples")
                os.makedirs(path_cache)
                np.save(path_cache + "/train_data.npy", train_dataset.shared_array_img.numpy())
                np.save(path_cache + "/val_data.npy", val_dataset.shared_array_img.numpy())

    # Set dataloaders in dict
    datalaoders = {"train": train_loader, "val": val_loader}

    return datalaoders
