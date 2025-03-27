import os
import cv2
import functools
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from skimage.io import imread, imshow
from torch.utils.data import Dataset
import random

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

class BatchToTensor(object):
    def __call__(self, imgs):

        return [transforms.ToTensor()(img) for img in imgs]

def randomlist(list):
	int = random.randint(1, 10)
	if int < len(list):
		newlist = random.sample(list, int)
	else:
		newlist = []
		for i in range(int):
			newlist.append(random.choice(list))
	return(newlist)


def has_file_allowed_extension(filename, extensions):

	filename_lower = filename.lower()
	return any(filename_lower.endswith(ext) for ext in extensions)


def image_seq_loader(img_seq_dir):
	img_seq_dir = os.path.expanduser(img_seq_dir)

	img_seq = []
	for root, _, fnames in sorted(os.walk(img_seq_dir)):
		for fname in sorted(fnames):
			if has_file_allowed_extension(fname, IMG_EXTENSIONS):
				image_name = os.path.join(root, fname)
				image = imread(image_name)
				img_seq.append(image)

	return img_seq


def get_default_img_seq_loader():
	return functools.partial(image_seq_loader)



class ImageSeqDataset(Dataset):
	def __init__(self, csv_file,
				 Train_img_seq_dir,
				 Label_img_dir,
				 Train_transform=None,
				 Label_transform=None,
				 get_loader=get_default_img_seq_loader,
				 randomlist = True):

		self.seqs = pd.read_csv(csv_file, sep='/n', header=None)
		self.Train_root = Train_img_seq_dir
		self.Label_img_dir = Label_img_dir
		self.Train_transform = Train_transform
		self.Label_transform = Label_transform
		self.loader = get_loader()
		self.randomlist = randomlist

	def __getitem__(self, index):
		Train_seq_dir = os.path.join(self.Train_root, str(self.seqs.iloc[index, 0]))
		I = self.loader(Train_seq_dir)
		if self.randomlist == True:
			I = randomlist(I)

		I = self.Train_transform(I)

		train = torch.stack(I, 0).contiguous()


		Label_image = imread(self.Label_img_dir + str(self.seqs.iloc[index, 0]) + ".jpg")

		Label = self.Label_transform(Label_image)


		sample = {'Train': train, 'Lable': Label}
		return sample

	def __len__(self):
		return len(self.seqs)

	@staticmethod
	def _reorderBylum(seq):
		I = torch.sum(torch.sum(torch.sum(seq, 1), 1), 1)
		_, index = torch.sort(I)
		result = seq[index, :]
		return result

