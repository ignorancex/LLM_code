import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import VisionDataset, ImageFolder
from torch.utils.data import Dataset, DataLoader
import random
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from imutils import paths
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import torch


from transformers import ViTFeatureExtractor, ViTForImageClassification

import transformers



def get_aug(split, imsize=224, dataset='caltech101'):
    if split == 'train':
      if dataset == "cifar100" or "gtsrb":
        return [transforms.RandomCrop(imsize, pad_if_needed=True), transforms.RandomHorizontalFlip()]
      else:
        return [transforms.RandomResizedCrop(imsize, scale=(0.2, 1.0)), transforms.RandomHorizontalFlip()]
    else:
      return [transforms.Resize(round(imsize * 1.143)), transforms.CenterCrop(imsize)]


def get_transform(split, normalize=True, transform=None, imsize=224):
  if transform is None:
    if normalize is True:
        mean, stdev = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        normalize = transforms.Normalize(mean = mean, std=stdev)
        if split == "train":
            transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((imsize, imsize))] + get_aug(split, imsize=imsize)
                                   + [transforms.ToTensor(), normalize])                     
        else: 
            transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((imsize, imsize))] + [transforms.ToTensor(), normalize]) 
    else:
        if split == "train" :   
            transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((imsize, imsize))] + get_aug(split, imsize=imsize)
                                                   + [transforms.ToTensor()])   
        else:
            transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((imsize, imsize))] + [transforms.ToTensor()])
        
  return transform


def get_transform_torchvision(split, normalize=True, transform=None, imsize=224, dataset='cifar100'):
  if transform is None:
    if normalize is True:
        mean, stdev = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        normalize = transforms.Normalize(mean = mean, std=stdev)
        if split == "train":
            transform = transforms.Compose([transforms.Resize((imsize, imsize), interpolation=3)] + get_aug(split, imsize=imsize, dataset=dataset)
                                   + [transforms.ToTensor(), normalize])                     
        else: 
            transform = transforms.Compose([transforms.Resize((imsize, imsize), interpolation=3)] + [transforms.ToTensor(), normalize]) 
    else:
        if split == "train" :   
            transform = transforms.Compose([transforms.Resize((imsize, imsize), interpolation=3)] + get_aug(split, imsize=imsize, dataset=dataset)
                                                   + [transforms.ToTensor()])   
        else:
            transform = transforms.Compose([transforms.Resize((imsize, imsize), interpolation=3)] + [transforms.ToTensor()])
        
  return transform


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, normalize=True):

        super(Caltech, self).__init__(root, transform=transform)
        self.split = split


        image_paths = list(paths.list_images(root))

        data = []
        labels = []
        for img_path in image_paths:
            label = img_path.split(os.path.sep)[-2]
            if label == "BACKGROUND_Google":
                continue
            with open(img_path, 'rb') as f:
                img = Image.open(f)
                im2arr = img.convert('RGB')
                im2arr = np.array(im2arr)
            data.append(im2arr)
            labels.append(label)
            
        data = np.array(data)
        labels = np.array(labels)
        lb = LabelEncoder()
        labels = lb.fit_transform(labels)
        print(f"Total Number of Classes: {len(lb.classes_)}")
        

        (x_train, x_test, y_train, y_test) = train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)
        (x_val, x_test, y_val, y_test) = train_test_split(x_test, y_test, test_size=0.5, random_state=42, stratify=y_test)
        
        if self.split=="train":
            self.images = x_train
            self.targets = y_train
        elif self.split=="val":
            self.images = x_val
            self.targets = y_val        
        elif self.split=="test":
            self.images = x_test
            self.targets = y_test      
        else:
            print("error invalid split")
            
        self.transforms = get_transform(self.split, normalize=normalize, imsize=224)
        
    def __getitem__(self, index):
        data = self.images[index][:]
        
        if self.transforms:
            data = self.transforms(data)
            
        return data, self.targets[index]

    def __len__(self):   
        return len(self.images)

        
        
class IMBALANCECIFAR100(datasets.CIFAR100):
    cls_num = 100

    def __init__(self, imbalance_ratio, train_dataset, root = '/datasets/cifar100', imb_type='exp', normal_size=False):
        train = True 
        super(IMBALANCECIFAR100, self).__init__(root, train, transform=None, target_transform=None, download=True)
        self.data = self.data[np.array(train_dataset.indices)]
        print(self.data.shape)
        self.targets = [self.targets[i] for i in train_dataset.indices]
        print(len(self.targets))

        self.train = train
        if self.train:
            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imbalance_ratio)
            self.gen_imbalanced_data(img_num_list)
            if normal_size==False:
                self.transform = get_transform_torchvision('train', normalize=True, imsize=224, dataset='cifar100')
            else:
                self.transform = get_transform_torchvision('train', normalize=True, imsize=32, dataset='cifar100')

        else:
            self.transform = transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                            ])

    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.get_annotations()):
            cat_id = anno["category_id"]
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict


    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)

        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx])
            new_targets.extend([self.targets[i] for i in selec_idx])
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        print(self.data.shape)
        print(len(self.targets))


    def __getitem__(self, index):
        img, label = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label
    
    def __len__(self):
        return len(self.targets)

    def get_num_classes(self):
        return self.cls_num

    def get_annotations(self):
        annos = []
        for label in self.labels:
            annos.append({'category_id': int(label)})
        return annos
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list




def get_cifar100_data(split, normalize=True, split_percent=0.9, rand_seed=42, normal_size=False, imbalance=False, imbalance_ratio=1/100):

    if normal_size==False:
        transform = get_transform_torchvision(split, normalize=normalize, imsize=224, dataset='cifar100')
    else:
        transform = get_transform_torchvision(split, normalize=normalize, imsize=32, dataset='cifar100')


    if split=='test':
        dataset = datasets.CIFAR100('datasets/cifar100', train=False, transform=transform, download=True)

    else:
        train_set = datasets.CIFAR100('datasets/cifar100', train=True, transform=transform, download=True)
        train_set_size = int(len(train_set) * split_percent)
        valid_set_size = len(train_set) - train_set_size
        train_set, valid_set = torch.utils.data.random_split(train_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(rand_seed))


        if split=='train' and imbalance==True:
            dataset= IMBALANCECIFAR100(imbalance_ratio, train_set)
        elif split=='train':
            dataset = train_set
        elif split=='val':
            dataset = valid_set

    return dataset


def get_gtsrb_data(split, normalize=True, split_percent=0.9, rand_seed=42):

    transform = get_transform_torchvision(split, normalize=normalize, imsize=224, dataset='gtsrb')

    if split=='test':
        #dataset = datasets.CIFAR100('datasets/cifar100', train=(split=='train'), transform=transform, download=True, **kwargs)
        dataset = datasets.GTSRB('datasets/cifar100', split=split, transform=transform, download=True)

    else:

        train_set = datasets.GTSRB('datasets/cifar100', split='train', transform=transform, download=True)
        train_set_size = int(len(train_set) * split_percent)
        valid_set_size = len(train_set) - train_set_size
        train_set, valid_set = torch.utils.data.random_split(train_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(rand_seed))

        if split=='train':
            dataset = train_set
        elif split=='val':
            dataset = valid_set
            
    return dataset


def get_dataset(dataset_name, split, normalize=True, imbalance=False, imbalance_ratio=1/100):
    if dataset_name == "caltech":
      dataset = Caltech("./datasets/caltech_data/caltech101/101_ObjectCategories", split=split, normalize=normalize)      

    elif dataset_name == "cifar100":  
        dataset = get_cifar100_data(split, normalize=normalize, imbalance=imbalance, imbalance_ratio=imbalance_ratio)

    elif dataset_name == "gtsrb":  
        dataset = get_gtsrb_data(split, normalize=normalize)        

    return dataset 
  
  
def get_attack_dataset(dataset_name, n_ex, split="test", normal_size=False, normalize=False):
    np.random.seed(0)
    if dataset_name == "caltech":
        dataset = Caltech("./datasets/caltech_data/caltech101/101_ObjectCategories", split=split, normalize=normalize)      
        caltech_loader = DataLoader(dataset, batch_size=n_ex, shuffle=False, num_workers=0)
        x_test, y_test = next(iter(caltech_loader))
    elif dataset_name == "cifar100":  
        dataset = get_cifar100_data(split, normalize=normalize, normal_size=normal_size)
        cifar_loader = DataLoader(dataset, batch_size=n_ex, shuffle=False, num_workers=0)
        x_test, y_test = next(iter(cifar_loader))

    elif dataset_name == "gtsrb":  
        dataset = get_gtsrb_data(split, normalize=normalize)
        gtsrb_loader = DataLoader(dataset, batch_size=n_ex, shuffle=False, num_workers=0)
        x_test, y_test = next(iter(gtsrb_loader))

    return np.array(x_test, dtype=np.float32), np.array(y_test)  
  
  