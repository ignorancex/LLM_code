import torch
import torch.nn as nn
from torchvision import models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def vgg(num_class: int):

    student = models.vgg16_bn(pretrained=False)
    teacher = models.vgg16_bn(pretrained=False)
  
    student.classifier[-1] = nn.Linear(in_features=4096,
                                       out_features=num_class)

    teacher.classifier[-1] = nn.Linear(in_features=4096,
                                       out_features=num_class)

    return student, teacher

def resnet(num_class: int):
    student = models.resnet34(pretrained=False)
    teacher = models.resnet34(pretrained=False)
  
    student.fc = nn.Linear(in_features=512,
                           out_features=num_class)

    teacher.fc = nn.Linear(in_features=512,
                           out_features=num_class)

    return student, teacher

def load_vgg_resnet(path, dataset, model):
    if dataset in ['cifar10', 'imagewoof', 'imagenette']:
        if model == 'vgg16':
            student, teacher = vgg(num_class=10)
            checkpoint = torch.load(path, map_location=device)
        elif model == 'resnet34':
            student, teacher = resnet(num_class=10)
            checkpoint = torch.load(path, map_location=device)
            checkpoint = fix_state_dict_resnet34(checkpoint)
        if 'model' in checkpoint:
            student.load_state_dict(checkpoint['model'], strict=True)
            teacher.load_state_dict(checkpoint['model'], strict=True)
        else:
            student.load_state_dict(checkpoint, strict=True)
            teacher.load_state_dict(checkpoint, strict=True)
    elif dataset == 'food101':
        if model == 'vgg16':
            student, teacher = vgg(num_class=101)
        elif model == 'resnet34':
            student, teacher = resnet(num_class=101)
        checkpoint = torch.load(path)
        if 'model' in checkpoint:
            student.load_state_dict(checkpoint['model'])
            teacher.load_state_dict(checkpoint['model'])
        else:
            student.load_state_dict(checkpoint)
            teacher.load_state_dict(checkpoint)
    elif dataset == 'imagenet':
        student = models.vgg16_bn(pretrained=True)
        teacher = models.vgg16_bn(pretrained=True)
    else:
        print("unrecognized dataset")

    return student, teacher

def load_model(path, dataset, model):
    if model == 'vgg16' or 'resnet34':
        return load_vgg_resnet(path, dataset, model)
    else:
        print('model not valid')

def fix_state_dict_resnet34(checkpoint):
    new_dict = {}
    
    for key, value in checkpoint.items():
        new_dict[key.replace("model.", "")] = value

    return new_dict
