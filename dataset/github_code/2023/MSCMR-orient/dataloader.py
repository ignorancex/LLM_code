import torch
from torch.utils.data import Dataset
from torchvision import transforms
from SimpleITK import GetArrayFromImage, ReadImage
import numpy as np
import cv2
import os

NUM_CLASSES = 8
NAME = 'T2'


class MSCMRsegDatasetTrain(Dataset):
    def __init__(self, name, mode):
        super(MSCMRsegDatasetTrain, self).__init__()
        self.mode = mode
        if self.mode == 'train':
            data_path = f'./data/dataset/{name}/data_train.pt'
            labels_path = f'./data/dataset/{name}/labels_train.pt'
            if os.path.exists(data_path):
                self.data = torch.load(data_path)
                self.labels = torch.load(labels_path)
            else:
                imgs = []
                labels = []
                for i in range(1, 37):
                    img_name = f'./data/data_adjusted/{name}/patient{i}_{name}.nii.gz'
                    itk_img = ReadImage(img_name)
                    img = GetArrayFromImage(itk_img).astype(np.uint8)
                    for j in range(img.shape[0]):
                        slice = img[j]
                        imgs += [slice, np.fliplr(slice), np.flipud(slice), np.rot90(slice, 2, axes=(1, 0)),
                                 np.transpose(slice),
                                 np.rot90(slice, 1, axes=(1, 0)), np.rot90(slice, 3, axes=(1, 0)),
                                 np.rot90(np.transpose(slice), 2, axes=(1, 0))]
                        labels += list(range(8))

                folder_path = f'./data/dataset/{name}'
                os.makedirs(folder_path)
                self.data = imgs
                self.labels = labels
                torch.save(imgs, f'./data/dataset/{name}/data_train.pt')
                torch.save(labels, f'./data/dataset/{name}/labels_train.pt')
        else:
            data_path = f'./data/dataset/{name}/data_val.pt'
            labels_path = f'./data/dataset/{name}/labels_val.pt'
            if os.path.exists(data_path):
                self.data = torch.load(data_path)
                self.labels = torch.load(labels_path)
            else:
                imgs = []
                labels = []
                for i in range(37, 46):
                    img_name = f'./data/data_adjusted/{name}/patient{i}_{name}.nii.gz'
                    itk_img = ReadImage(img_name)
                    img = GetArrayFromImage(itk_img).astype(np.uint8)
                    for j in range(img.shape[0]):
                        slice = img[j]
                        imgs += [slice, np.fliplr(slice), np.flipud(slice), np.rot90(slice, 2, axes=(1, 0)),
                                 np.transpose(slice),
                                 np.rot90(slice, 1, axes=(1, 0)), np.rot90(slice, 3, axes=(1, 0)),
                                 np.rot90(np.transpose(slice), 2, axes=(1, 0))]
                        labels += list(range(8))

                self.data = imgs
                self.labels = labels
                torch.save(imgs, f'./data/dataset/{name}/data_val.pt')
                torch.save(labels, f'./data/dataset/{name}/labels_val.pt')

        self.num_classes = NUM_CLASSES
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop((256, 256), scale=(0.7, 1), ratio=(0.8, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        # self.transforms = {'train': transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.RandomRotation(10),
        #     transforms.RandomResizedCrop((256, 256), scale=(0.7, 1), ratio=(0.8, 1.2)),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        # ]),
        #     'val': transforms.Compose([
        #         transforms.ToPILImage(),
        #         transforms.Resize((256, 256)),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        #     ])
        # }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        label = self.labels[index]

        img100, img80, img60 = img.copy(), img.copy(), img.copy()
        rec1, img80 = cv2.threshold(img, 0.8 * np.max(img), 0.8 * np.max(img), cv2.THRESH_TRUNC)
        rec1, img60 = cv2.threshold(img, 0.6 * np.max(img), 0.6 * np.max(img), cv2.THRESH_TRUNC)
        img100 = cv2.equalizeHist(img100)
        img80 = cv2.equalizeHist(img80)
        img60 = cv2.equalizeHist(img60)

        img = np.stack([img100, img80, img60], axis=-1)
        return self.transforms(img), label


class MSCMRsegDatasetPredict(Dataset):
    def __init__(self, name, mode):
        super(MSCMRsegDatasetPredict, self).__init__()
        self.mode = mode
        if self.mode == 'train':
            data_path = f'./data/dataset/{name}/data_train_predict.pt'
            labels_path = f'./data/dataset/{name}/labels_train_predict.pt'
            if os.path.exists(data_path):
                self.data = torch.load(data_path)
                self.labels = torch.load(labels_path)
            else:
                imgs = []
                labels = []
                for i in range(1, 37):
                    img_name = f'./data/data_adjusted/{name}/patient{i}_{name}.nii.gz'
                    itk_img = ReadImage(img_name)
                    img = GetArrayFromImage(itk_img).astype(np.uint8)
                    imgs += [img, np.flip(img, 2), np.flip(img, 1), np.rot90(img, 2, axes=(2, 1)),
                             np.transpose(img, (0, 2, 1)),
                             np.rot90(img, 1, axes=(2, 1)), np.rot90(img, 3, axes=(2, 1)),
                             np.rot90(np.transpose(img, (0, 2, 1)), 2, axes=(2, 1))]
                    labels += list(range(8))

                self.data = imgs
                self.labels = labels
                torch.save(imgs, f'./data/dataset/{name}/data_train_predict.pt')
                torch.save(labels, f'./data/dataset/{name}/labels_train_predict.pt')
        else:
            data_path = f'./data/dataset/{name}/data_val_predict.pt'
            labels_path = f'./data/dataset/{name}/labels_val_predict.pt'
            if os.path.exists(data_path):
                self.data = torch.load(data_path)
                self.labels = torch.load(labels_path)
            else:
                imgs = []
                labels = []
                for i in range(37, 46):
                    img_name = f'./data/data_adjusted/{name}/patient{i}_{name}.nii.gz'
                    itk_img = ReadImage(img_name)
                    img = GetArrayFromImage(itk_img).astype(np.uint8)
                    imgs += [img, np.flip(img, 2), np.flip(img, 1), np.rot90(img, 2, axes=(2, 1)),
                             np.transpose(img, (0, 2, 1)),
                             np.rot90(img, 1, axes=(2, 1)), np.rot90(img, 3, axes=(2, 1)),
                             np.rot90(np.transpose(img, (0, 2, 1)), 2, axes=(2, 1))]
                    labels += list(range(8))

                self.data = imgs
                self.labels = labels
                torch.save(imgs, f'./data/dataset/{name}/data_val_predict.pt')
                torch.save(labels, f'./data/dataset/{name}/labels_val_predict.pt')

        self.num_classes = NUM_CLASSES
        self.transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        label = self.labels[index]
        imgs=[]
        for j in range(img.shape[0]):
            slice=img[j]
            img100, img80, img60 = slice.copy(), slice.copy(), slice.copy()
            rec1, img80 = cv2.threshold(img80, 0.8 * np.max(slice), 0.8 * np.max(slice), cv2.THRESH_TRUNC)
            rec1, img60 = cv2.threshold(img60, 0.6 * np.max(slice), 0.6 * np.max(slice), cv2.THRESH_TRUNC)
            img100 = cv2.equalizeHist(img100)
            img80 = cv2.equalizeHist(img80)
            img60 = cv2.equalizeHist(img60)
            img_slice = np.stack([img100, img80, img60], axis=-1)
            imgs.append(self.transforms(img_slice).unsqueeze(0))

        return torch.cat(imgs),label


# dataset_train_predict=MSCMRsegDatasetPredict(NAME,'train')
# sample, label = dataset_train_predict[0]

# dataset_train = MSCMRseg_dataset(NAME, 'train')
# dataset_val = MSCMRseg_dataset(NAME, 'val')
#
# sample, label = dataset_train[0]
#
# print(0)
