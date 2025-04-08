# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# from utils.data_load.data1 import MyDataset as data1
# from utils.data_load.data2 import MyDataset as data2

import torch
import torchvision.transforms as transforms
from config.cfg import BaseConfig
from training.trainer_finetune import DefaultTrainer
import os
from utils import data_load
import numpy as np
from module.Polar_transform import Polar_transform

def main(args):
    runseed = 1
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    torch.manual_seed(runseed)
    np.random.seed(runseed)

    dataset = getattr(data_load, args.data_name.lower())
    
    train_data = dataset(
        dataset='train',
        transform=transforms.Compose([
            transforms.Resize((256, 256)),
            # transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomGrayscale(p=0.2),
            # transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
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
        transform=transforms.Compose([
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

      
        fold=args.k_fold
    )

    val_load = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=True, num_workers=4)

    trainer = DefaultTrainer(args)


    #load pretrain_model
    checkpoint = torch.load(args.pretrained, map_location="cpu")

    state_dict = checkpoint['net_state_dict']
    

    for k in list(state_dict.keys()):
        if not k.startswith('fc'):
            continue
        del state_dict[k]

    msg = trainer.model.load_state_dict(state_dict, strict=False)


    # only unsqueeze fc1
    for name, param in trainer.model.named_parameters():
            if name not in ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias']:
                param.requires_grad = False

    # only unsqueeze cnn1.fc and fc1
    # for name, param in trainer.model.named_parameters():
    #         if name not in ['fc1.weight', 'fc1.bias', 'cnn1.fc.weight', 'cnn1.fc.bias']:
    #             param.requires_grad = False

    # init the fc layer
    trainer.model.fc1.weight.data.normal_(mean=0.0, std=0.01)
    trainer.model.fc1.bias.data.zero_()

    trainer.model.fc2.weight.data.normal_(mean=0.0, std=0.01)
    trainer.model.fc2.bias.data.zero_()

    # trainer.model.cnn1.fc.weight.data.normal_(mean=0.0, std=0.01)
    # trainer.model.cnn1.fc.bias.data.zero_()
   

    trainer.train(train_load, val_load)
    trainer.log_wrong()


if __name__ == '__main__':
    cfg = BaseConfig()
    fixed = None
    args = cfg.initialize(fixed)
    main(args)

