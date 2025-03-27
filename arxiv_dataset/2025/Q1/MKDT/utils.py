# adapted from
# https://github.com/VICO-UoE/DatasetCondensation
import random
import time
import torch
import torch.nn as nn
import os
import pandas as pd
import tqdm
import copy
import pickle
from torch.utils.data import Dataset, Subset
from networks import MLP, ConvNet, LeNet, AlexNet, VGG11BN, VGG11, ResNet18, ResNet18BN_AP, ResNet18_AP, ResNet10
from PIL import Image
from more_dataset import Aircraft, Cub2011, Dogs
from torchvision import transforms
from torchvision import datasets as torchvision_datasets

class ImageNet(torch.utils.data.Dataset):
    def __init__(self, root: str, split: str, transform=None):
        self.root = root
        df = pd.read_csv(os.path.join(root, split, "labels.csv"), on_bad_lines='skip')
        self.images = df["image"]
        self.labels = df["label"]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.root, self.images[idx])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

def get_dataset(dataset, data_path, subset_size=None, epc=None, seed=None):
    if seed is not None:    
        random.seed(seed)

    if dataset == 'CIFAR10':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = torchvision_datasets.CIFAR10(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = torchvision_datasets.CIFAR10(data_path, train=False, download=True, transform=transform)


    elif dataset == 'Tiny':
        channel = 3
        im_size = (64, 64)
        num_classes = 200
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = torchvision_datasets.ImageFolder(os.path.join(data_path + "/tiny_imagenet", "train"), transform=transform) # no augmentation
        dst_test = torchvision_datasets.ImageFolder(os.path.join(data_path + "/tiny_imagenet", "val"), transform=transform)


    elif dataset == 'ImageNet':
        channel = 3
        # im_size = (128, 128)
        im_size = (64, 64)
        num_classes = 1000
        mean = [0.485, 0.456, 0.3868]
        std = [0.2309, 0.2262, 0.2237]
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=mean, std=std),
                                        transforms.Resize(im_size),
                                        transforms.CenterCrop(im_size)])
        data_path = "/home/data/ILSVRC"
        dst_train = ImageNet(data_path, split="train_full", transform=transform) # no augmentation
        dst_test = ImageNet(data_path, split="test", transform=transform)

    elif dataset.startswith('CIFAR100'):
        channel = 3
        im_size = (32, 32)
        num_classes = 100
        mean = [0.5071, 0.4865, 0.4409]
        std = [0.2009, 0.1984, 0.2023]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = torchvision_datasets.CIFAR100(data_path, train=True, download=True, transform=transform)  # no augmentation
        dst_test = torchvision_datasets.CIFAR100(data_path, train=False, download=True, transform=transform)
    
    elif dataset.startswith('aircraft'):
        channel = 3
        im_size = (32, 32)
        num_classes = 100
        mean = [0.4804, 0.5116, 0.5349]
        std = [0.2021, 0.1953, 0.2297]

        # consistent with KRRST
        resize = lambda x: x if x.size[0] == 32 and x.size[1] == 32 else x.resize((32,32), Image.Resampling.LANCZOS)
        transform = transforms.Compose([resize, transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = Aircraft(data_path, train=True, download=True, transform=transform) 
        dst_test = Aircraft(data_path, train=False, download=True, transform=transform)
       
    elif dataset.startswith('cub2011'):
        channel = 3
        im_size = (32, 32)
        num_classes = 200
        mean = [0.4857, 0.4995, 0.4324]
        std = [0.2145, 0.2098, 0.2496]

        # consistent with KRRST
        resize = lambda x: x if x.size[0] == 32 and x.size[1] == 32 else x.resize((32,32), Image.Resampling.LANCZOS)
        transform = transforms.Compose([resize, transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        
        dst_train = Cub2011(data_path, train=True, download=True, transform=transform)  
        dst_test = Cub2011(data_path, train=False, download=True, transform=transform)
    
    elif dataset.startswith('dogs'):
        channel = 3
        im_size = (32, 32)
        num_classes = 120
        mean = [0.4765, 0.4516, 0.3911]
        std = [0.2490, 0.2435, 0.2479]

        # consistent with KRRST
        resize = lambda x: x if x.size[0] == 32 and x.size[1] == 32 else x.resize((32,32), Image.Resampling.LANCZOS)
        transform = transforms.Compose([resize, transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

        dst_train = Dogs(data_path, train=True, download=True, transform=transform) 
        dst_test = Dogs(data_path, train=False, download=True, transform=transform)

    elif dataset.startswith('flowers'):
        channel = 3
        im_size = (32, 32)
        num_classes = 102
        mean = [0.4329, 0.3820, 0.2965]
        std = [0.2828, 0.2333, 0.2615]

        # consistent with KRRST
        resize = lambda x: x if x.size[0] == 32 and x.size[1] == 32 else x.resize((32,32), Image.Resampling.LANCZOS)
        transform = transforms.Compose([resize, transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

        dst_train = torchvision_datasets.Flowers102(data_path, split = 'train', download=True, transform=transform)
        dst_test = torchvision_datasets.Flowers102(data_path, split = 'test', download=True, transform=transform)
    
    if subset_size is not None and epc is not None:
        raise ValueError("Can only set one of subset size and epc")
    
    # handle the subsets
    if subset_size is not None:
        random_subset = list(random.sample(range(len(dst_train)), int(len(dst_train) * subset_size)))
        dst_train = torch.utils.data.Subset(dst_train, random_subset)

    if epc is not None:
        indices = get_indices_per_class(dst_train, epc)
        dst_train = torch.utils.data.Subset(dst_train, indices)
        

    return channel, im_size, num_classes, dst_train, dst_test

def get_indices_per_class(dataset, epc):
    class_indices = {}
    
    for idx, (_, label) in enumerate(dataset):
        if label in class_indices:
            class_indices[label].append(idx)
        else:
            class_indices[label] = [idx]
    
    sampled_indices = []
    for indices in class_indices.values():
        if len(indices) > epc:
            sampled_indices.extend(random.sample(indices, epc))
        else:
            sampled_indices.extend(indices)
    
    return sampled_indices

def build_trainset(dataset, dst_train, train_labels_path, channel, batch_train, distill_idx=None, shuffle=False):
    # Organize real datasets (images and the target representation)
    images_all = None
    
    print("BUILDING TRAINSET")
    
    if dataset != "ImageNet":
        images_all = []
        for i in tqdm.tqdm(range(len(dst_train))):
            sample = dst_train[i]
            images_all.append(torch.unsqueeze(sample[0], dim=0))

        images_all = torch.cat(images_all, dim=0).to("cpu")

    labels_all = torch.load(train_labels_path, map_location="cpu").to("cpu")
    print("train label shape", labels_all.shape)
    
    for ch in range(channel):
        print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))
        
    if dataset != "ImageNet":
        dst_train = TensorDataset(copy.deepcopy(images_all.detach()), copy.deepcopy(labels_all.detach()))
    else:
        dst_train = TensorDatasetWrapper(dst_train, copy.deepcopy(labels_all.detach()))
    
    # Subset using Distill idx if specified 
    if distill_idx is not None:
        with open(distill_idx, "rb") as f:
            distill_idx = pickle.load(f)
            dst_train = Subset(dst_train, indices=distill_idx)

    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=batch_train, shuffle=shuffle, num_workers=4, pin_memory=True)
    
    print("Dataset creation complete")

    return trainloader, labels_all

class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach().float()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]

class TensorDatasetWrapper(Dataset):
    def __init__(self, img_dataset, labels): # images: n x c x h x w tensor
        self.img_dataset = img_dataset
        self.labels = labels.detach().float()

    def __getitem__(self, index):
        return self.img_dataset[index][0], self.labels[index]

    def __len__(self):
        return len(self.img_dataset)


class SoftLabelDataset(torch.utils.data.Dataset):
    def __init__(self, img_dataset, label_tensor, subset_idx=None):
        super().__init__()
        self.img_dataset = img_dataset
        self.label_tensor = label_tensor
        assert len(self.img_dataset) == len(self.label_tensor)
        self.subset_idx = subset_idx
        if self.subset_idx is None:
            self.subset_idx = range(len(self.img_dataset))
        
    def __getitem__(self, i):
        idx = self.subset_idx[i]
        return (self.img_dataset[idx][0], self.label_tensor[idx])
    
    def __len__(self):
        return len(self.subset_idx)


def get_default_convnet_setting():
    net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
    return net_width, net_depth, net_act, net_norm, net_pooling


def get_network(model, channel, num_classes, im_size=(32, 32), dist=False, fix_net=False):
    if not fix_net:
        torch.random.manual_seed(int(time.time() * 1000) % 100000)
    net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()

    if model == 'MLP':
        net = MLP(channel=channel, num_classes=num_classes)
    elif model == 'ConvNet':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'LeNet':
        net = LeNet(channel=channel, num_classes=num_classes)
    elif model == 'AlexNet':
        net = AlexNet(channel=channel, num_classes=num_classes)
    elif model == 'VGG11':
        net = VGG11( channel=channel, num_classes=num_classes)
    elif model == 'VGG11BN':
        net = VGG11BN(channel=channel, num_classes=num_classes)
    elif model == 'ResNet10':
        net = ResNet10(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18':
        net = ResNet18(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18BN_AP':
        net = ResNet18BN_AP(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18_AP':
        net = ResNet18_AP(channel=channel, num_classes=num_classes)

    elif model == 'ConvNetD1':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=1, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD2':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=2, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD3':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=3, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD4':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=4, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD5':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=5, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD6':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=6, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD7':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=7, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD8':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=8, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)


    elif model == 'ConvNetW32':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=32, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetW64':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=64, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetW128':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=128, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetW256':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=256, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetW512':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=512, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetW1024':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=1024, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)

    elif model == "ConvNetKIP":
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=1024, net_depth=net_depth, net_act=net_act,
                      net_norm="none", net_pooling=net_pooling)

    elif model == 'ConvNetAS':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='sigmoid', net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetAR':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='relu', net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetAL':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='leakyrelu', net_norm=net_norm, net_pooling=net_pooling)

    elif model == 'ConvNetNN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='none', net_pooling=net_pooling)
    elif model == 'ConvNetBN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='batchnorm', net_pooling=net_pooling)
    elif model == 'ConvNetLN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='layernorm', net_pooling=net_pooling)
    elif model == 'ConvNetIN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='instancenorm', net_pooling=net_pooling)
    elif model == 'ConvNetGN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='groupnorm', net_pooling=net_pooling)

    elif model == 'ConvNetNP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling='none')
    elif model == 'ConvNetMP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling='maxpooling')
    elif model == 'ConvNetAP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling='avgpooling')


    else:
        net = None
        exit('Error: unknown model')

    if dist:
        gpu_num = torch.cuda.device_count()
        if gpu_num>0:
            device = 'cuda'
            if gpu_num>1:
                net = nn.DataParallel(net)
        else:
            device = 'cpu'
        net = net.to(device)

    return net

def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))


def epoch(mode, dataloader, net, optimizer, criterion, device):
    loss_avg, num_exp = 0, 0
    net = net.to(device)
    
    desc = "Training"
    if mode == 'train':
        net.train()
    else:
        desc = "Evaluating"
        net.eval()

    for _, datum in tqdm.tqdm(enumerate(dataloader), total=len(dataloader), desc=desc):
        img = datum[0].float().to(device)
        lab = datum[1].float().to(device)
        n_b = lab.shape[0]

        output = net(img)
        loss = criterion(output, lab)

        loss_avg += loss.item()*n_b
        num_exp += n_b

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_avg /= num_exp

    return loss_avg