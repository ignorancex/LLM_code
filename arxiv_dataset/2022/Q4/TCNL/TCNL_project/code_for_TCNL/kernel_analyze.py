import argparse
from ast import arguments
from model.new_resnet import *
from model.new_vgg import *
from model.new_alexnet import *
from model.resnet import resnet
import os
import sys
import numpy as np
import torch
from torchvision.transforms import transforms
from tqdm import tqdm
from PIL import Image
from skimage import io
from torch.autograd import Variable
import cv2
from utils.data_utils import get_train_loader
from matplotlib import pyplot as plt
plt.switch_backend('agg')


def get_file_path(root_path, file_list, dir_list):
    # 获取该目录下所有的文件名称和目录名称
    dir_or_files = os.listdir(root_path)
    for dir_file in dir_or_files:
        # 获取目录或者文件的路径
        dir_file_path = os.path.join(root_path, dir_file)
        # 判断该路径为文件还是路径
        if os.path.isdir(dir_file_path):
            dir_list.append(dir_file_path)
            # 递归获取所有文件和目录的路径
            get_file_path(dir_file_path, file_list, dir_list)
        else:
            file_list.append(dir_file_path)


def choose_tlayer(model_obj):
    name_to_num = {}
    sel_module = False
    name_module = None
    module_list = ['Sequential', 'Bottleneck', 'container', 'Block', 'densenet']
    while True:
        for num, module in enumerate(model_obj.named_children()):
            if any(x in torch.typename(module[1]) for x in module_list):
                print(f'[ Number: {num},  Name: {module[0]} ] -> Module: {module[1]}\n')
                name_to_num[module[0]] = num
            else:
                print(f'[ Number: {num},  Name: {module[0]} ] -> Layer: {module[1]}\n')
                name_to_num[module[0]] = num

        print('<<      You sholud not select [classifier module], [fc layer] !!      >>')
        if sel_module == False:
            a = input('Choose "Number" or "Name" of a module containing a target layer or a target layer: ')
        else:
            a = input(
                f'Choose "Number" or "Name" of a module containing a target layer or a target layer in {name_module} module: ')

        print('\n' * 3)
        m_val = list(model_obj._modules.values())
        m_key = list(model_obj._modules.keys())
        if isInt_str(a) == False:
            a = name_to_num[a]
        try:
            if any(x in torch.typename(m_val[int(a)]) for x in module_list):
                model_obj = m_val[int(a)]
                name_module = m_key[int(a)]
                sel_module = True
            else:
                t_layer = m_val[int(a)]
                return t_layer

        except IndexError:
            print('Selected index (number) is not allowed.')
            # raise NoIndexError('Selected index (number) is not allowed.')
        except KeyError:
            print('Selected name is not allowed.')


def isInt_str(v):
    v = str(v).strip()
    return v == '0' or (v if v.find('..') > -1 else v.lstrip('-+').rstrip('0').rstrip('.')).isdigit()


def load_model(args):
    # model = resnet_multi_concept_for_scene(args)
    # model = vgg_multi_concept_for_mammal(args)
    # model = vgg(num_classes=5)
    # model = alexnet_multi_concept_for_mammal(args)
    # model = alexnet(num_classes=args['num_classes'])
    # model = alexnet_multi_concept_for_scene(args)
    # model = resnet(num_classes=4)
    # 在cpu或者cuda:0上加载模型
    # ckpt = torch.load(model_name, map_location='cuda:0')
    model = new_resnet_multi_text_concept_for_mammal(args)
    ckpt = torch.load(args['ckpt_path'], map_location=args['device'])
    model.load_state_dict(ckpt['net'])
    model.to(args['device'])
    model.eval()
    return model


class kernel_analyze:
    def __init__(self, args:dict):
        self.model_path = args['ckpt_path']
        self.cuda_device = args['device']
        self.model = load_model(args)
        self.data_loader = args['data_loader']
        self.concept_list = ['head', 'torso', 'leg', 'shape']
        self.file_list = args['file_list']
        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]

        def forward_hook(module, input, output):
            self.activations['value'] = output[0]
        
        model_obj = self.model
        self.target_layer = choose_tlayer(model_obj)
        self.target_layer.register_backward_hook(backward_hook)
        self.target_layer.register_forward_hook(forward_hook)

    def __call__(self):
        activation_list = list()
        activation_list2 = list()
        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        for file in tqdm(self.file_list):
            image = Image.open(file).convert('RGB')
            image = transformer(image)
            image = torch.unsqueeze(image, dim=0).to(self.cuda_device)
            ori_image_path = file.replace('without_seat', 'image_new/theater')
            # ori_image_path = os.path.join('/home/wangzhihao/XAI/XCSGCNN/VOCdevkit/my_exp_data/train/object', file.split('/')[-2])
            ori_image = Image.open(ori_image_path).convert('RGB')
            ori_image = transformer(ori_image)
            ori_image = torch.unsqueeze(ori_image, dim=0).to(self.cuda_device)
            feature_dict, pred_class, reconstruct_result = self.model.inference(image)
            # pred = self.model(image)
            activation = self.activations['value']
            activation_list.append(activation.cpu().data.numpy())
            feature_dict, pred_class, reconstruct_result = self.model.inference(ori_image)
            # pred = self.model(ori_image)
            activation2 = self.activations['value']
            activation_list2.append(activation2.cpu().data.numpy())

        # for data in tqdm(self.data_loader):
        #     # image = data['image'].to(self.cuda_device)
        #     path = os.path.join('/home/wangzhihao/XAI/XCSGCNN/VOCdevkit/my_exp_data/train/without_leg', data['image_name'][0])
        #     image = Image.open(path).convert('RGB')
        #     image = transformer(image)
        #     image = torch.unsqueeze(image, dim=0).to(self.cuda_device)
        #     label = data['label']
        #     feature_dict, pred_class, reconstruct_result = self.model.inference(image)
        #     activation = self.activations['value']
        #     activation_list.append(activation.cpu().data.numpy())
        return activation_list, activation_list2

if __name__ == '__main__':
    data_loader = get_train_loader('/home/wangzhihao/XAI/XCSGCNN/VOCdevkit/my_exp_data/train_multi_concept.txt', 1)
    root_dir = '/home/wangzhihao/XAI/XCSGCNN/VOCdevkit/scene/train/without_seat'
    file_list = list()
    dir_list = list()
    get_file_path(root_dir, file_list, dir_list)
    args = {
        'data_loader': data_loader,
        'ckpt_path': '/home/wangzhihao/XAI/XCSGCNN/train/ckpt/resnet_multi_concept_scene/ckpt_170_acc80.46_loss2.70.pt',
        'file_list': file_list,
        'device': 'cuda:0',
        'num_classes': 4,
        'lr_resnet': 0.001,
        'lr_R': 0.001,
        'lr_D': 0.001,
    }
    kernel_analyzer = kernel_analyze(args)
    list1, list2 = kernel_analyzer()
    list1 = np.array(list1)
    list2 = np.array(list2)
    print(list1.shape, list2.shape)
    np.save('/home/wangzhihao/XAI/XCSGCNN/train/result/new_resnet_scene_activation/seat_activation_list_noseat.npy', list1)
    np.save('/home/wangzhihao/XAI/XCSGCNN/train/result/new_resnet_scene_activation/seat_activation_list_image_with_seat.npy', list2)


    
    head_activation_image_with_head = list1
    # head_activation_image_with_head = np.load('/home/wangzhihao/XAI/XCSGCNN/train/result/new_resnet_scene_activation/bed_activation_list_image_with_bed.npy')
    head_activation_image_with_head = np.average(head_activation_image_with_head, 0)
    head_activation_nohead = list2
    # head_activation_nohead = np.load('/home/wangzhihao/XAI/XCSGCNN/train/result/new_resnet_scene_activation/bed_activation_list_noshape.npy')
    head_activation_nohead = np.average(head_activation_nohead, 0)
    difference = head_activation_image_with_head - head_activation_nohead
    difference = torch.relu(torch.tensor(difference)).data.numpy()
    print(difference.shape)
    difference = np.sum(difference, 2)
    difference = np.sum(difference, 1)
    print(difference.shape)
    avg_drop = np.average(difference)
    # avg_drop = np.percentile(difference, 60)
    print(avg_drop)
    cnt = 0
    for i in difference:
        if i > avg_drop:
            cnt += 1
    print(cnt)




