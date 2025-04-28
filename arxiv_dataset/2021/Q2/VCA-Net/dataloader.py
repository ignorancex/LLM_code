import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import warnings
warnings.filterwarnings("ignore")


class ATLASDataset(Dataset):
	def __init__(self, h5_dataset, subject_length, patient_indices, transform=None):
		self.h5_dataset = h5_dataset
		self.subject_length = subject_length
		self.transform = transform
		self.patient_indices = patient_indices

	def __len__(self):
		return self.patient_indices

	def __getitem__(self, index):
		img_train = self.h5_dataset['data']
		mask_train = self.h5_dataset['label']
		indexed_imgs = img_train[index:index+1, 5:229, 2:194]
		indexed_masks = mask_train[index:index+1, 5:229, 2:194]

		if self.transform is not None:
			indexed_imgs = self.transform(indexed_imgs)
			indexed_masks = self.transform(indexed_masks)

		return {
			'image': torch.from_numpy(indexed_imgs.copy()).type(torch.FloatTensor),
			'mask': torch.from_numpy(indexed_masks.copy()).type(torch.FloatTensor)
		}

	def __repr__(self):
		str = 'Dataset ' + self.__class__.__name__ + '\n'
		str += '  Number of data: {}\n'.format(self.__len__())
		return str
