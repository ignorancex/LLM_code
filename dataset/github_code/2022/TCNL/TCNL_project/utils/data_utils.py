import torch
import torchvision
import torchvision.transforms as transforms
import random
import os
import cv2
from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image


def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    return image


class TrainingDataset(Dataset):
    def __init__(self, dataset_txt, root_dir):
        self.data = list()
        with open(dataset_txt, 'r') as f:
            lines = f.readlines()
        for line in lines:
            # line = [object, head, torso, leg, outline, label]
            line = line.strip('\n')
            line = line.split(' ')
            self.data.append(line)
        # random.shuffle(self.data)
        self.root_dir = root_dir

    def __getitem__(self, index):
        data = self.data[index]
         # data = [object, head, torso, leg, outline, label]
        transformer_for_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        transformer_for_concept = transforms.Compose([
            transforms.ToTensor()
        ])

        # get input image for training
        image = load_image(os.path.join(self.root_dir, data[0]))
        image = transformer_for_image(image)
        # get the class label
        label = int(data[5])
        # get head concept
        head = load_image(os.path.join(self.root_dir,  data[1]))
        head = transformer_for_concept(head)
        head = ((head - torch.min(head)) / (torch.max(head) - torch.min(head))) * 2 - 1
        # get torso concept
        torso = load_image(os.path.join(self.root_dir,  data[2]))
        torso = transformer_for_concept(torso)
        if (torch.max(torso) == torch.min(torso)):
            torso = torso * 2 - 1
        else:
            torso = ((torso - torch.min(torso)) / (torch.max(torso) - torch.min(torso))) * 2 - 1
        # get leg concept
        leg = load_image(os.path.join(self.root_dir,  data[3]))
        leg = transformer_for_concept(leg)
        if (torch.max(leg) == torch.min(leg)):
            leg = leg * 2 - 1
        else:
            leg = ((leg - torch.min(leg)) / (torch.max(leg) - torch.min(leg))) * 2 - 1
        # get outline concept
        outline = load_image(os.path.join(self.root_dir,  data[4]))
        outline = transformer_for_concept(outline)
        outline = ((outline - torch.min(outline)) / (torch.max(outline) - torch.min(outline))) * 2 - 1
        dict_data = {
            'label': label,
            'image': image,
            'image_name': data[0].split('/')[-2] + '/' + data[0].split('/')[-1],
            'head': head,
            'torso': torso,
            'leg': leg,
            'outline': outline
       }
        return dict_data

    def __len__(self):
        return len(self.data)


class TrainingDataset_Scene(Dataset):
    def __init__(self, dataset_txt, root_dir):
        self.data = list()
        with open(dataset_txt, 'r') as f:
            lines = f.readlines()
        for line in lines:
            # line = [object, head, torso, leg, outline, label]
            line = line.strip('\n')
            line = line.split(' ')
            self.data.append(line)
        # random.shuffle(self.data)
        self.root_dir = root_dir

    def __getitem__(self, index):
        data = self.data[index]
         # data = [object, head, torso, leg, outline, label]
        transformer_for_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        transformer_for_concept = transforms.Compose([
            transforms.ToTensor()
        ])

        # get input image for training
        image = load_image(os.path.join(self.root_dir, data[0]))
        image = transformer_for_image(image)
        # get the class label
        label = int(data[11])
        # get bed concept
        bed = load_image(os.path.join(self.root_dir,  data[1]))
        bed = transformer_for_concept(bed)
        if (torch.max(bed) == torch.min(bed)):
            bed = bed * 2 - 1
        else:
            bed = ((bed - torch.min(bed)) / (torch.max(bed) - torch.min(bed))) * 2 - 1
        # get bedsidetable concept
        bedsidetable = load_image(os.path.join(self.root_dir,  data[2]))
        bedsidetable = transformer_for_concept(bedsidetable)
        if (torch.max(bedsidetable) == torch.min(bedsidetable)):
            bedsidetable = bedsidetable * 2 - 1
        else:
            bedsidetable = ((bedsidetable - torch.min(bedsidetable)) / (torch.max(bedsidetable) - torch.min(bedsidetable))) * 2 - 1
        # get lamp concept
        lamp = load_image(os.path.join(self.root_dir,  data[3]))
        lamp = transformer_for_concept(lamp)
        if (torch.max(lamp) == torch.min(lamp)):
            lamp = lamp * 2 - 1
        else:
            lamp = ((lamp - torch.min(lamp)) / (torch.max(lamp) - torch.min(lamp))) * 2 - 1
        # get sofa concept
        sofa = load_image(os.path.join(self.root_dir,  data[4]))
        sofa = transformer_for_concept(sofa)
        if (torch.max(sofa) == torch.min(sofa)):
            sofa = sofa * 2 - 1
        else:
            sofa = ((sofa - torch.min(sofa)) / (torch.max(sofa) - torch.min(sofa))) * 2 - 1
        # get chair concept
        chair = load_image(os.path.join(self.root_dir,  data[5]))
        chair = transformer_for_concept(chair)
        if (torch.max(chair) == torch.min(chair)):
            chair = chair * 2 - 1
        else:
            chair = ((chair - torch.min(chair)) / (torch.max(chair) - torch.min(chair))) * 2 - 1
        # get table concept
        table = load_image(os.path.join(self.root_dir,  data[6]))
        table = transformer_for_concept(table)
        if (torch.max(table) == torch.min(table)):
            table = table * 2 - 1
        else:
            table = ((table - torch.min(table)) / (torch.max(table) - torch.min(table))) * 2 - 1
        # get shelf concept
        shelf = load_image(os.path.join(self.root_dir,  data[7]))
        shelf = transformer_for_concept(shelf)
        if (torch.max(shelf) == torch.min(shelf)):
            shelf = shelf * 2 - 1
        else:
            shelf = ((shelf - torch.min(shelf)) / (torch.max(shelf) - torch.min(shelf))) * 2 - 1
        # get seat concept
        seat = load_image(os.path.join(self.root_dir,  data[8]))
        seat = transformer_for_concept(seat)
        if (torch.max(seat) == torch.min(seat)):
            seat = seat * 2 - 1
        else:
            seat = ((seat - torch.min(seat)) / (torch.max(seat) - torch.min(seat))) * 2 - 1
        # get screen concept
        screen = load_image(os.path.join(self.root_dir,  data[9]))
        screen = transformer_for_concept(screen)
        if (torch.max(screen) == torch.min(screen)):
            screen = screen * 2 - 1
        else:
            screen = ((screen - torch.min(screen)) / (torch.max(screen) - torch.min(screen))) * 2 - 1
        # get stage concept
        stage = load_image(os.path.join(self.root_dir,  data[10]))
        stage = transformer_for_concept(stage)
        if (torch.max(stage) == torch.min(stage)):
            stage = stage * 2 - 1
        else:
            stage = ((stage - torch.min(stage)) / (torch.max(stage) - torch.min(stage))) * 2 - 1
        dict_data = {
            'label': label,
            'image': image,
            'image_name': data[0].split('/')[-2] + '/' + data[0].split('/')[-1],
            'bed': bed,
            'bedsidetable': bedsidetable,
            'lamp': lamp,
            'sofa': sofa,
            'chair': chair,
            'table': table,
            'shelf': shelf,
            'seat': seat,
            'screen': screen,
            'stage': stage
       }
        return dict_data

    def __len__(self):
        return len(self.data)


class TrainingDataset_Normal(Dataset):
    def __init__(self, dataset_txt, root_dir):
        self.data = list()
        with open(dataset_txt, 'r') as f:
            lines = f.readlines()
        for line in lines:
            # line = [object, head, torso, leg, outline, label]
            line = line.strip('\n')
            line = line.split(' ')
            self.data.append(line)
        # random.shuffle(self.data)
        self.root_dir = root_dir

    def __getitem__(self, index):
        data = self.data[index]
         # data = [object, head, torso, leg, outline, label]
        transformer_for_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        transformer_for_concept = transforms.Compose([
            transforms.ToTensor()
        ])

        # get input image for training
        image = load_image(os.path.join(self.root_dir, data[0]))
        image = transformer_for_image(image)
        # get the class label
        label = int(data[1])
        dict_data = {
            'label': label,
            'image': image,
            'image_name': data[0].split('/')[-2] + '/' + data[0].split('/')[-1]
       }
        return dict_data

    def __len__(self):
        return len(self.data)



def get_train_loader(train_txt: str, train_batch_size: int):
    train_loader = torch.utils.data.DataLoader(
        dataset=TrainingDataset(train_txt, '/home/user/wangzhihao/XCSGCNN/VOCdevkit/my_exp_data/train'),
        batch_size=train_batch_size
    )
    return train_loader


def get_val_loader(val_txt: str, val_batch_size):
    val_loader = torch.utils.data.DataLoader(
        dataset=TrainingDataset(val_txt, '/home/user/wangzhihao/XCSGCNN/VOCdevkit/my_exp_data/val'),
        batch_size=val_batch_size
    )
    return val_loader


def get_scene_train_loader(train_txt: str, train_batch_size: int):
    train_loader = torch.utils.data.DataLoader(
        dataset=TrainingDataset_Scene(train_txt, '/home/user/wangzhihao/XCSGCNN/VOCdevkit/scene/train'),
        batch_size=train_batch_size
    )
    return train_loader


def get_scene_val_loader(val_txt: str, val_batch_size):
    val_loader = torch.utils.data.DataLoader(
        dataset=TrainingDataset_Scene(val_txt, '/home/user/wangzhihao/XCSGCNN/VOCdevkit/scene/val'),
        batch_size=val_batch_size
    )
    return val_loader


def get_normal_train_loader(train_txt: str, train_batch_size: int):
    train_loader = torch.utils.data.DataLoader(
        dataset=TrainingDataset_Normal(train_txt, '/home/user/wangzhihao/XCSGCNN/VOCdevkit/my_exp_data/train_aug'),
        batch_size=train_batch_size
    )
    return train_loader


def get_normal_val_loader(val_txt: str, val_batch_size):
    val_loader = torch.utils.data.DataLoader(
        dataset=TrainingDataset_Normal(val_txt, '/home/user/wangzhihao/XCSGCNN/VOCdevkit/my_exp_data/val_aug'),
        batch_size=val_batch_size
    )
    return val_loader




if __name__ == '__main__':
    txt = '/home/user/wangzhihao/XAI/XCSGCNN/VOCdevkit/scene/train_multi_concept.txt'
    # loader = get_scene_train_loader(txt, 1)
    data = list()
    with open(txt, 'r') as f:
        lines = f.readlines()
    for line in lines:
        # line = [object, head, torso, leg, outline, label]
        line = line.strip('\n')
        line = line.split(' ')
        data.append(line)
    print(data[0])

    # for i in loader:
    #     image = i['torso']
    #     image_name = i['image_name']
    #     sum = torch.sum(image)
    #     print(image_name, sum)

    # for i, (image, label, main_object) in enumerate(train_loader):
    #     print(image, label, main_object)
    #     print("第 {} 个Batch \n{} {}".format(i, image.size(), label))