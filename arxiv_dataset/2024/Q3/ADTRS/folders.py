import os
import csv
import numpy as np
import scipy.io
from PIL import Image
import torch.utils.data as data


class BaseDataset(data.Dataset):
    def __init__(self, root, index, transform, patch_num):
        self.root = root
        self.index = index
        self.transform = transform
        self.patch_num = patch_num
        self.samples = []
        self.prepare_samples()

    def prepare_samples(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        sample = pil_loader(path)
        if self.transform:
            sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return len(self.samples)


class LIVEFolder(BaseDataset):
    def prepare_samples(self):
        img_paths, labels, refnames_all, orgs = self.load_live_data()
        refname = get_file_names(os.path.join(self.root, 'refimgs'), '.bmp')
        refname.sort()

        for i in self.index:
            train_sel = (refname[i] == refnames_all) & (~orgs)
            train_sel_indices = np.where(train_sel)[1].tolist()
            for item in train_sel_indices:
                for _ in range(self.patch_num):
                    self.samples.append((img_paths[item], labels[0][item]))

    def load_live_data(self):
        img_paths = []
        for distortion, count in zip(['jp2k', 'jpeg', 'wn', 'gblur', 'fastfading'], [227, 233, 174, 174, 174]):
            img_paths += get_distortion_filenames(os.path.join(self.root, distortion), count)

        dmos = scipy.io.loadmat(os.path.join(self.root, 'dmos_realigned.mat'))
        labels = dmos['dmos_new'].astype(np.float32)
        orgs = dmos['orgs'].astype(np.bool_)
        refnames_all = scipy.io.loadmat(os.path.join(self.root, 'refnames_all.mat'))['refnames_all']

        return img_paths, labels, refnames_all, orgs


class LIVEChallengeFolder(BaseDataset):
    def prepare_samples(self):
        img_paths = scipy.io.loadmat(os.path.join(self.root, 'Data', 'AllImages_release.mat'))['AllImages_release'][7:1169]
        labels = scipy.io.loadmat(os.path.join(self.root, 'Data', 'AllMOS_release.mat'))['AllMOS_release'][0][7:1169].astype(np.float32)

        for item in self.index:
            for _ in range(self.patch_num):
                self.samples.append((os.path.join(self.root, 'Images', img_paths[item][0][0]), labels[item]))


class CSIQFolder(BaseDataset):
    def prepare_samples(self):
        refnames_all, labels, imgnames = self.load_csiq_data()
        refname = get_file_names(os.path.join(self.root, 'src_imgs'), '.png')
        refname.sort(reverse=True)

        for i in self.index:
            train_sel_indices = np.where(refname[i] == refnames_all)[0].tolist()
            for item in train_sel_indices:
                for _ in range(self.patch_num):
                    self.samples.append((os.path.join(self.root, 'dst_imgs_all', imgnames[item]), labels[item]))

    def load_csiq_data(self):
        refnames_all, labels, imgnames = [], [], []
        with open(os.path.join(self.root, 'csiq_label.txt'), 'r') as fh:
            for line in fh:
                words = line.strip().split()
                imgnames.append(words[0])
                labels.append(float(words[1]))
                refnames_all.append('.'.join(words[0].split(".")[:-1]))

        return np.array(refnames_all), np.array(labels).astype(np.float32), imgnames


class Koniq_10kFolder(BaseDataset):
    def prepare_samples(self):
        imgname, mos_all = [], []
        with open(os.path.join(self.root, 'koniq10k_scores_and_distributions.csv')) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgname.append(row['image_name'])
                mos_all.append(float(row['MOS_zscore']))

        for item in self.index:
            for _ in range(self.patch_num):
                self.samples.append((os.path.join(self.root, '1024x768', imgname[item]), mos_all[item]))


class FBLIVEFolder(BaseDataset):
    def prepare_samples(self):
        imgname, mos_all = [], []
        with open(os.path.join(self.root, 'labels_image.csv')) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgname.append(row['name'].split('/')[1])
                mos_all.append(float(row['mos']))

        for item in self.index:
            for _ in range(self.patch_num):
                self.samples.append((os.path.join(self.root, 'FLIVE', imgname[item]), mos_all[item]))


class TID2013Folder(BaseDataset):
    def prepare_samples(self):
        refnames_all, labels, imgnames = self.load_tid_data()
        refname = get_tid_filenames(os.path.join(self.root, 'reference_images'), '.bmp.BMP')
        refname.sort()

        for i in self.index:
            train_sel_indices = np.where(refname[i] == refnames_all)[0].tolist()
            for item in train_sel_indices:
                for _ in range(self.patch_num):
                    self.samples.append((os.path.join(self.root, 'distorted_images', imgnames[item]), labels[item]))

    def load_tid_data(self):
        refnames_all, labels, imgnames = [], [], []
        with open(os.path.join(self.root, 'mos_with_names.txt'), 'r') as fh:
            for line in fh:
                words = line.strip().split()
                imgnames.append(words[1])
                labels.append(float(words[0]))
                refnames_all.append(words[1].split("_")[0][1:])

        return np.array(refnames_all), np.array(labels).astype(np.float32), imgnames


class Kadid10k(BaseDataset):
    def prepare_samples(self):
        refnames_all, labels, imgnames = self.load_kadid_data()
        refname = get_tid_filenames(os.path.join(self.root, 'reference_images'), '.png.PNG')
        refname.sort()

        for i in self.index:
            train_sel_indices = np.where(refname[i] == refnames_all)[0].tolist()
            for item in train_sel_indices:
                for _ in range(self.patch_num):
                    self.samples.append((os.path.join(self.root, 'distorted_images', imgnames[item]), labels[item]))

    def load_kadid_data(self):
        refnames_all, labels, imgnames = [], [], []
        with open(os.path.join(self.root, 'dmos.csv')) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgnames.append(row['dist_img'])
                refnames_all.append(row['ref_img'][1:3])
                labels.append(float(row['dmos']))

        return np.array(refnames_all), np.array(labels).astype(np.float32), imgnames


def get_file_names(path, suffix):
    return [file for file in os.listdir(path) if file.endswith(suffix)]


def get_tid_filenames(path, suffixes):
    return [file[1:3] for file in os.listdir(path) if any(suffix in file for suffix in suffixes.split('.'))]


def get_distortion_filenames(path, num):
    return [os.path.join(path, f'img{index}.bmp') for index in range(1, num + 1)]


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
