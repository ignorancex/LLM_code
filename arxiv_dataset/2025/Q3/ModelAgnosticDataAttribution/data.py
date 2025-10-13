from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import torch
import os
from PIL import Image
import pandas as pd
import numpy as np
import math

def get_subset_celeba_attr(data_file, partition_file, target_attribute='Male', hidden_attribute='Wearing_Earrings', num_validation_per_class=182):
    # load in the dataset
    attributes = pd.read_csv(data_file)

    # create a smaller subset to do our experiment, with an induced imbalance
    partitions = pd.read_csv(partition_file)
    attributes_train = attributes[partitions.partition == 0]
    attributes_train = attributes_train[attributes_train.image_id != '090516.jpg']
    attributes_train = attributes_train.reset_index()

    # train set of 6000 images
    target_pos_attr_pos = attributes_train[(attributes_train[target_attribute] == 1) & (
        attributes_train[hidden_attribute] == 1)].sample(0).index.values
    target_pos_attr_neg = attributes_train[(attributes_train[target_attribute] == 1) & (
        attributes_train[hidden_attribute] == -1)].sample(3000).index.values
    target_neg_attr_pos = attributes_train[(attributes_train[target_attribute] == -1) & (
        attributes_train[hidden_attribute] == 1)].sample(2000).index.values
    target_neg_attr_neg = attributes_train[(attributes_train[target_attribute] == -1) & (
        attributes_train[hidden_attribute] == -1)].sample(1000).index.values
    indexes = np.concatenate(
        [target_pos_attr_pos, target_pos_attr_neg, target_neg_attr_pos, target_neg_attr_neg])

    # validation set
    attributes_val = attributes[partitions.partition == 1]
    attributes_val = attributes_val[attributes_val.image_id != '090516.jpg']
    attributes_val = attributes_val.reset_index()

    # fully balanced, 400 images 100 men 100 women with and without earrings 50/50
    targ_pos_attr_pos = attributes_val[attributes_val[target_attribute] == 1].groupby(
        hidden_attribute).sample(num_validation_per_class).index.values
    targ_neg_attr_pos = attributes_val[attributes_val[target_attribute] == -1].groupby(
        hidden_attribute).sample(num_validation_per_class).index.values
    indexes_val = np.concatenate([targ_pos_attr_pos, targ_neg_attr_pos])

    return indexes, indexes_val


class CelebDataset(Dataset):
    def __init__(self, target='all', img_folder: str = 'data/celeba/img_align_celeba/',
                 attributes_csv: str = 'data/celeba/list_attr_celeba.csv', partitions_csv: str = 'data/celeba/list_eval_partition.csv',
                 partition: int = 0,  transforms: transforms.Compose = None, return_map=False) -> None:
        """_summary_

        Args:
            target (str, optional): _description_. Defaults to 'all'.
            img_folder (str, optional): _description_. Defaults to 'data/celeba/img_align_celeba/'.
            attributes_csv (str, optional): _description_. Defaults to 'data/celeba/list_attr_celeba.csv'.
            partitions_csv (str, optional): _description_. Defaults to 'data/celeba/list_eval_partition.csv'.
            partition (int, optional): _description_. Defaults to 0.
            transforms (transforms.Compose, optional): _description_. Defaults to None.
            return_map (bool, optional): _description_. Defaults to False.
        """
        super().__init__()
        self.img_folder = img_folder
        attributes = pd.read_csv(attributes_csv)
        partitions = pd.read_csv(partitions_csv)
        attributes = attributes[partitions.partition == partition]
        attributes = attributes[attributes.image_id != '090516.jpg']

        if target == 'all':
            self.targets = attributes.drop(
                'image_id', axis=1).replace(-1, 0).values
        else:
            self.label_categories = ['No', 'Yes']
            self.targets = attributes[target].replace(-1, 0).values

        self.images = attributes["image_id"].values
        self.targets = torch.from_numpy(self.targets)
        self.transform = transforms

        self.return_map = return_map

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_folder, self.images[idx])
        pil_img = Image.open(img_name)
        img = transforms.ToTensor()(pil_img).unsqueeze_(0)
        target = self.targets[idx]
        if self.transform:
            img = self.transform(img)

        if type(img) == tuple:  # with coordinates
            # return image[0], image[1], image[2], torch.tensor(label, dtype=torch.int64)
            return img[0].squeeze(), img[1], target

        if self.return_map: # return the image, the index and the target so it can be used to find the corresponding attribution map
            return img.squeeze(), idx, target.to(torch.float32)
        return img.squeeze(), target.to(torch.float32)


class RandomCropWithFixedCoordinates(object):
    def __init__(self, input_size, output_size, n_crops=16):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.total_crops = n_crops
        self.crops_per_row = math.sqrt(n_crops)
        assert self.crops_per_row.is_integer()
        self.crops_per_row = int(self.crops_per_row)
        self.stride = (input_size - output_size) / (self.crops_per_row - 1)
        assert self.stride.is_integer()
        self.stride = int(self.stride)

    def __call__(self, image):

        # generate random coordinates
        crop_id = torch.randint(0, self.total_crops, (1,)).item()
        left = (crop_id % self.crops_per_row) * self.stride
        top = (crop_id // self.crops_per_row) * self.stride

        image_crop = torchvision.transforms.functional.crop(
            image, left, top, height=self.output_size, width=self.output_size)

        return image_crop, crop_id
