# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# from utils.data_load.data1 import MyDataset as data1
# from utils.data_load.data2 import MyDataset as data2

import torch
import torchvision.transforms as transforms
from config.cfg import BaseConfig
from training.trainer_pretrain import DefaultTrainer
import os
from utils import data_load
import numpy as np
from module.Polar_transform import Polar_transform
from module.Rotate_transform import RotateTransform

def main(args):
    runseed = 1
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    torch.manual_seed(runseed)
    np.random.seed(runseed)

    dataset = getattr(data_load, args.data_name.lower())
    
    train_data = dataset(
        dataset='train',
        pretrain = args.pretrain,
        transform1=transforms.Compose([
            # transforms.Resize((256, 256)),

            transforms.Resize((320, 320)),
            transforms.RandomResizedCrop(256, scale=(0.2, 1.)),

            transforms.RandomHorizontalFlip(),
            # RotateTransform([0, 90, 180, 270]),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            Polar_transform(args.polar),
#             transforms.RandomApply([
#                 transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
#             ], p=0.8),
#             transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),

        transform2=transforms.Compose([
            # transforms.Resize((256, 256)),

            transforms.Resize((320, 320)),
            transforms.RandomResizedCrop(256, scale=(0.2, 1.)),
            
            transforms.RandomHorizontalFlip(),
            # RotateTransform([0, 90, 180, 270]),

            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            Polar_transform(args.polar),
#             transforms.RandomApply([
#                 transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
#             ], p=0.8),
#             transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        fold=args.k_fold
    )

    train_load = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                             drop_last=True)

    val_data = dataset(
        dataset='valid',
        pretrain = args.pretrain,
        transform1=transforms.Compose([
            transforms.Resize((256, 256)),
            Polar_transform(args.polar),
#             transforms.RandomApply([
#                 transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
#             ], p=0.8),
#             transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),

        transform2=transforms.Compose([
            transforms.Resize((256, 256)),
            Polar_transform(args.polar),
#             transforms.RandomApply([
#                 transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
#             ], p=0.8),
#             transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        fold=0
    )

    val_load = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=True, num_workers=4)

    trainer = DefaultTrainer(args)
    trainer.train(train_load, val_load)
    trainer.log_wrong()


if __name__ == '__main__':
    cfg = BaseConfig()
    fixed = None
    args = cfg.initialize(fixed)
    main(args)

