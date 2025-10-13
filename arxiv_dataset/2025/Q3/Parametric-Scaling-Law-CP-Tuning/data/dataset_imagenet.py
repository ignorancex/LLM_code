import os
import random
import torch
from torchvision import transforms, datasets



transform_imagenet_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def build_dataloaders(data_dir, cal_num=10000, batch_size=256, num_workers=8):
    validir = os.path.join(data_dir, '/mnt/sharedata/ssd3/common/datasets/imagenet/images/val')
    testset = datasets.ImageFolder(root=validir, transform=transform_imagenet_test)

    dataset_length = len(testset)
    cc_calibset, testset = torch.utils.data.random_split(testset, [cal_num, dataset_length - cal_num])

    dataset_length = len(testset)
    cp_calibset, testset = torch.utils.data.random_split(testset, [cal_num, dataset_length - cal_num])

    cc_calibloader = torch.utils.data.DataLoader(dataset=cc_calibset, batch_size=batch_size, num_workers=num_workers)
    cp_calibloader = torch.utils.data.DataLoader(dataset=cp_calibset, batch_size=batch_size, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, num_workers=num_workers)
    return cc_calibloader, cp_calibloader, testloader
