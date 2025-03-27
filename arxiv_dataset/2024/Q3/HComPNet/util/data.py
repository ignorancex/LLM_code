
import numpy as np
import argparse
import torch
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Dict
from torch import Tensor
import random
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader
from torchvision.datasets.folder import ImageFolder
from collections import Counter
from torch.utils.data import Sampler, SubsetRandomSampler

import os

def unshuffle_dataloader(dataloader):
    if type(dataloader.dataset) == ImageFolder:
        dataset = dataloader.dataset
    else:
        dataset = dataloader.dataset.dataset.dataset
    new_dataloader = DataLoader(
        dataset=dataset,
        batch_size=dataloader.batch_size,
        shuffle=False,
        num_workers=dataloader.num_workers,
        pin_memory=dataloader.pin_memory,
        drop_last=dataloader.drop_last,
        timeout=dataloader.timeout,
        worker_init_fn=dataloader.worker_init_fn,
        multiprocessing_context=dataloader.multiprocessing_context,
        generator=dataloader.generator,
        prefetch_factor=dataloader.prefetch_factor,
        persistent_workers=dataloader.persistent_workers
    )
    return new_dataloader


def create_filtered_dataloader(dataloader, new_sampler):
    if type(dataloader.dataset) == ImageFolder:
        dataset = dataloader.dataset
    else:
        dataset = dataloader.dataset.dataset.dataset
    new_dataloader = DataLoader(
        dataset=dataset,
        batch_size=dataloader.batch_size,
        shuffle=False,
        sampler=new_sampler,
        num_workers=dataloader.num_workers,
        pin_memory=dataloader.pin_memory,
        drop_last=dataloader.drop_last,
        timeout=dataloader.timeout,
        worker_init_fn=dataloader.worker_init_fn,
        multiprocessing_context=dataloader.multiprocessing_context,
        generator=dataloader.generator,
        prefetch_factor=dataloader.prefetch_factor,
        persistent_workers=dataloader.persistent_workers
    )
    return new_dataloader


class SubsetSequentialSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class ModifiedLabelLoader(DataLoader):
    def __init__(self, dataloader, node, *args, **kwargs):
        # super(ModifiedLabelLoader, self).__init__(*args, **kwargs)
        self.dataloader = dataloader

        self.node = node

        # train loaders use additional wrappers on the dataset for adding augmentation
        if type(dataloader.dataset) == ImageFolder:
            name2label = dataloader.dataset.class_to_idx
            self.dataset = dataloader.dataset
        else:
            name2label = dataloader.dataset.dataset.dataset.class_to_idx
            self.dataset = dataloader.dataset.dataset.dataset

        self.label2name = {label:name for name, label in name2label.items()}

        self.modifiedlabel2name = {label: name for name, label in node.children_to_labels.items()}

        class_counts = {self.label2name[label]:count for label, count in Counter(self.dataset.targets).items()}

        self.num_samples = 0
        for classname, count in class_counts.items():
            if classname in self.node.children_to_labels.keys():
                self.num_samples += count
        
        # the order of images in this and the dataloader must be similar since shuffle=False, but not tested
        self.filtered_imgs = [(img_path, label) for img_path, label in self.dataset.imgs \
                                                    if self.label2name[label] in self.node.descendents]

    def __iter__(self):
        for batch_images, batch_labels in self.dataloader:
            batch_names = [self.label2name[y.item()] for y in batch_labels]
            children_idx = torch.tensor([name in self.node.descendents for name in batch_names])
            batch_names_coarsest = [self.node.closest_descendent_for(name).name for name in batch_names if name in self.node.descendents] # size of sum(children_idx)
            modified_labels = torch.tensor([self.node.children_to_labels[name] for name in batch_names_coarsest]).cuda() # size of sum(children_idx)

            if len(modified_labels) == 0:
                continue

            batch_images = batch_images[children_idx]
            original_labels = batch_labels[children_idx]

            yield batch_images, original_labels, modified_labels

    def __len__(self):
        return self.num_samples


def get_data(args: argparse.Namespace): 
    """
    Load the proper dataset based on the parsed arguments
    """
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.dataset =='CUB-190':
        base_path = 'data/CUB_200_2011/dataset_cub190'
        return get_birds(True, os.path.join(base_path, 'train_crop'), # train_dir
                                os.path.join(base_path, 'train'), # project_dir
                                os.path.join(base_path, 'test_crop'), # test_dir
                                args.image_size, args.seed, args.validation_size, 
                                os.path.join(base_path, 'train'), # train_dir_pretrain
                                os.path.join(base_path, 'test')) # test_dir_projection
    if args.dataset =='BUT-51-224':
        base_path = 'data/butterfly51_224'
        return get_birds(True, os.path.join(base_path, 'train'), # train_dir
                                os.path.join(base_path, 'train'), # project_dir
                                os.path.join(base_path, 'val'), # test_dir
                                args.image_size, args.seed, args.validation_size, 
                                os.path.join(base_path, 'train'), # train_dir_pretrain
                                os.path.join(base_path, 'val')) # test_dir_projection
    if args.dataset =='FISH-38-224':
        base_path = 'data/Fish38_224/'
        return get_birds(True, os.path.join(base_path, 'train'), # train_dir
                                os.path.join(base_path, 'train'), # project_dir
                                os.path.join(base_path, 'test'), # test_dir
                                args.image_size, args.seed, args.validation_size, 
                                os.path.join(base_path, 'train'), # train_dir_pretrain
                                os.path.join(base_path, 'test')) # test_dir_projection

    raise Exception(f'Could not load data set, data set "{args.dataset}" not found!')


def get_dataloaders(args: argparse.Namespace, device):
    trainset, trainset_pretraining, trainset_normal, trainset_normal_augment, projectset, testset, testset_projection, classes, num_channels, train_indices, targets = get_data(args)
    
    # Determine if GPU should be used
    cuda = not args.disable_cuda and torch.cuda.is_available()
    to_shuffle = True
    sampler = None
    
    num_workers = args.num_workers

    if ('leave_out_classes' in args) and (args.leave_out_classes != ''):
        with open(args.leave_out_classes, 'r') as file:
            leave_out_classes = [line.strip() for line in file]
        idx_of_classes_to_keep = set()
        name2label = projectset.class_to_idx # param
        label2name = {label:name for name, label in name2label.items()}
        for label in label2name:
            if label2name[label] not in leave_out_classes:
                idx_of_classes_to_keep.add(label)

    pretrain_batchsize = args.batch_size_pretrain 
    
    if ('leave_out_classes' in args) and (args.leave_out_classes != ''):
        target_indices = []
        for i in range(len(trainset)):
            *_, label = trainset[i]
            if label in idx_of_classes_to_keep:
                target_indices.append(i)
        sampler = SubsetRandomSampler(target_indices)
        to_shuffle = False
    trainloader = torch.utils.data.DataLoader(trainset,
                                            batch_size=args.batch_size,
                                            shuffle=to_shuffle,
                                            sampler=sampler,
                                            pin_memory=cuda,
                                            num_workers=num_workers,
                                            worker_init_fn=np.random.seed(args.seed),
                                            drop_last = False
                                            # drop_last=True if ((len(trainset) % args.batch_size) / args.batch_size) < 0.2 else False
                                            )

    if trainset_pretraining is not None:
        if ('leave_out_classes' in args) and (args.leave_out_classes != ''):
            target_indices = []
            for i in range(len(trainset_pretraining)):
                *_, label = trainset_pretraining[i]
                if label in idx_of_classes_to_keep:
                    target_indices.append(i)
            sampler = SubsetRandomSampler(target_indices)
            to_shuffle = False
        trainloader_pretraining = torch.utils.data.DataLoader(trainset_pretraining,
                                            batch_size=pretrain_batchsize,
                                            shuffle=to_shuffle,
                                            sampler=sampler,
                                            pin_memory=cuda,
                                            num_workers=num_workers,
                                            worker_init_fn=np.random.seed(args.seed),
                                            drop_last=False
                                            )
                                        
    else:        
        if ('leave_out_classes' in args) and (args.leave_out_classes != ''):
            target_indices = []
            for i in range(len(trainset)):
                *_, label = trainset[i]
                if label in idx_of_classes_to_keep:
                    target_indices.append(i)
            sampler = SubsetRandomSampler(target_indices)
            to_shuffle = False
        trainloader_pretraining = torch.utils.data.DataLoader(trainset,
                                            batch_size=pretrain_batchsize,
                                            shuffle=to_shuffle,
                                            sampler=sampler,
                                            pin_memory=cuda,
                                            num_workers=num_workers,
                                            worker_init_fn=np.random.seed(args.seed),
                                            drop_last=False
                                            )

    if ('leave_out_classes' in args) and (args.leave_out_classes != ''):
        target_indices = []
        for i in range(len(trainset_normal)):
            *_, label = trainset_normal[i]
            if label in idx_of_classes_to_keep:
                target_indices.append(i)
        sampler = SubsetRandomSampler(target_indices)
        to_shuffle = False
    trainloader_normal = torch.utils.data.DataLoader(trainset_normal,
                                            batch_size=args.batch_size,
                                            shuffle=to_shuffle,
                                            sampler=sampler,
                                            pin_memory=cuda,
                                            num_workers=num_workers,
                                            worker_init_fn=np.random.seed(args.seed),
                                            drop_last=False
                                            )

    if ('leave_out_classes' in args) and (args.leave_out_classes != ''):
        target_indices = []
        for i in range(len(trainset_normal_augment)):
            *_, label = trainset_normal_augment[i]
            if label in idx_of_classes_to_keep:
                target_indices.append(i)
        sampler = SubsetRandomSampler(target_indices)
        to_shuffle = False
    trainloader_normal_augment = torch.utils.data.DataLoader(trainset_normal_augment,
                                            batch_size=args.batch_size,
                                            shuffle=to_shuffle,
                                            sampler=sampler,
                                            pin_memory=cuda,
                                            num_workers=num_workers,
                                            worker_init_fn=np.random.seed(args.seed),
                                            drop_last=False
                                            )
    
    projectloader = torch.utils.data.DataLoader(projectset,
                                              batch_size = 1,
                                              shuffle=False,
                                              pin_memory=cuda,
                                              num_workers=num_workers,
                                              worker_init_fn=np.random.seed(args.seed),
                                              drop_last=False
                                              )
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=args.batch_size,
                                             shuffle=True, 
                                             pin_memory=cuda,
                                             num_workers=num_workers,
                                             worker_init_fn=np.random.seed(args.seed),
                                             drop_last=False
                                             )
    test_projectloader = torch.utils.data.DataLoader(testset_projection,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=cuda,
                                             num_workers=num_workers,
                                             worker_init_fn=np.random.seed(args.seed),
                                             drop_last=False
                                             )
    print("Num classes (k) = ", len(classes), classes[:5], "etc.", flush=True)
    return trainloader, trainloader_pretraining, trainloader_normal, trainloader_normal_augment, projectloader, testloader, test_projectloader, classes


def create_datasets(transform1, transform2, transform_no_augment, num_channels:int, train_dir:str, project_dir: str, test_dir:str, seed:int, validation_size:float, train_dir_pretrain = None, test_dir_projection = None, transform1p=None):
    
    trainvalset = torchvision.datasets.ImageFolder(train_dir)
    classes = trainvalset.classes
    targets = trainvalset.targets
    indices = list(range(len(trainvalset)))

    train_indices = indices
    
    if test_dir is None:
        if validation_size <= 0.:
            raise ValueError("There is no test set directory, so validation size should be > 0 such that training set can be split.")
        subset_targets = list(np.array(targets)[train_indices])
        train_indices, test_indices = train_test_split(train_indices,test_size=validation_size,stratify=subset_targets, random_state=seed)
        testset = torch.utils.data.Subset(torchvision.datasets.ImageFolder(train_dir, transform=transform_no_augment), indices=test_indices)
        print("Samples in trainset:", len(indices), "of which",len(train_indices),"for training and ", len(test_indices),"for testing.", flush=True)
    else:
        testset = torchvision.datasets.ImageFolder(test_dir, transform=transform_no_augment)
    
    trainset = torch.utils.data.Subset(TwoAugSupervisedDataset(trainvalset, transform1=transform1, transform2=transform2), indices=train_indices)
    trainset_normal = torch.utils.data.Subset(torchvision.datasets.ImageFolder(train_dir, transform=transform_no_augment), indices=train_indices)
    trainset_normal_augment = torch.utils.data.Subset(torchvision.datasets.ImageFolder(train_dir, transform=transforms.Compose([transform1, transform2])), indices=train_indices)
    projectset = torchvision.datasets.ImageFolder(project_dir, transform=transform_no_augment)

    if test_dir_projection is not None:
        testset_projection = torchvision.datasets.ImageFolder(test_dir_projection, transform=transform_no_augment)
    else:
        testset_projection = testset
    if train_dir_pretrain is not None:
        trainvalset_pr = torchvision.datasets.ImageFolder(train_dir_pretrain)
        targets_pr = trainvalset_pr.targets
        indices_pr = list(range(len(trainvalset_pr)))
        train_indices_pr = indices_pr
        if test_dir is None:
            subset_targets_pr = list(np.array(targets_pr)[indices_pr])
            train_indices_pr, test_indices_pr = train_test_split(indices_pr,test_size=validation_size,stratify=subset_targets_pr, random_state=seed)

        trainset_pretraining = torch.utils.data.Subset(TwoAugSupervisedDataset(trainvalset_pr, transform1=transform1p, transform2=transform2), indices=train_indices_pr)
    else:
        trainset_pretraining = None
    
    return trainset, trainset_pretraining, trainset_normal, trainset_normal_augment, projectset, testset, testset_projection, classes, num_channels, train_indices, torch.LongTensor(targets)


def get_birds(augment: bool, train_dir:str, project_dir: str, test_dir:str, img_size: int, seed:int, validation_size:float, train_dir_pretrain = None, test_dir_projection = None): 
    shape = (3, img_size, img_size)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean,std=std)
    transform_no_augment = transforms.Compose([
                            transforms.Resize(size=(img_size, img_size)),
                            transforms.ToTensor(),
                            normalize
                        ])
    transform1p = None
    if augment:
        transform1 = transforms.Compose([
            transforms.Resize(size=(img_size+8, img_size+8)), 
            TrivialAugmentWideNoColor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(img_size+4, scale=(0.95, 1.))
        ])
        transform1p = transforms.Compose([
            transforms.Resize(size=(img_size+32, img_size+32)), #for pretraining, crop can be bigger since it doesn't matter when bird is not fully visible
            TrivialAugmentWideNoColor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(img_size+4, scale=(0.95, 1.))
        ])
        transform2 = transforms.Compose([
                            TrivialAugmentWideNoShape(),
                            transforms.RandomCrop(size=(img_size, img_size)),
                            transforms.ToTensor(),
                            normalize
                            ])
    else:
        transform1 = transform_no_augment    
        transform2 = transform_no_augment           

    return create_datasets(transform1, transform2, transform_no_augment, 3, train_dir, project_dir, test_dir, seed, validation_size, train_dir_pretrain, test_dir_projection, transform1p)


class TwoAugSupervisedDataset(torch.utils.data.Dataset):
    r"""Returns two augmentation and no labels."""
    def __init__(self, dataset, transform1, transform2):
        self.dataset = dataset
        self.classes = dataset.classes
        if type(dataset) == torchvision.datasets.folder.ImageFolder:
            self.imgs = dataset.imgs
            self.targets = dataset.targets
        else:
            self.targets = dataset._labels
            self.imgs = list(zip(dataset._image_files, dataset._labels))
        self.transform1 = transform1
        self.transform2 = transform2
        

    def __getitem__(self, index):
        image, target = self.dataset[index]
        image = self.transform1(image)
        return self.transform2(image), self.transform2(image), target

    def __len__(self):
        return len(self.dataset)


class TrivialAugmentWideNoColor(transforms.TrivialAugmentWide):
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.5, num_bins), True), 
            "ShearY": (torch.linspace(0.0, 0.5, num_bins), True), 
            "TranslateX": (torch.linspace(0.0, 16.0, num_bins), True), 
            "TranslateY": (torch.linspace(0.0, 16.0, num_bins), True), 
            "Rotate": (torch.linspace(0.0, 60.0, num_bins), True), 
        }

class TrivialAugmentWideNoShapeWithColor(transforms.TrivialAugmentWide): # used in get_cars transform2
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            "Identity": (torch.tensor(0.0), False),
            "Brightness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Color": (torch.linspace(0.0, 0.5, num_bins), True), 
            "Contrast": (torch.linspace(0.0, 0.5, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }

class TrivialAugmentWideNoShape(transforms.TrivialAugmentWide): # used in get_birds transform2
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        aug_dict = {
            "Identity": (torch.tensor(0.0), False),
            "Brightness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Color": (torch.linspace(-0.2, 1, num_bins), False), 
            "Contrast": (torch.linspace(0.0, 0.5, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }

        return aug_dict
