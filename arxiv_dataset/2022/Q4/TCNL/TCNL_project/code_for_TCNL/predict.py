import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import shutil
import wandb
from model.XCSGCNN import XCSGCNN_new_structure
from utils.data_utils import get_train_loader, get_val_loader
from utils.progress_utils import progress_bar
from utils.vis_utils import array_to_image
import ruamel_yaml as yaml
from tqdm import tqdm
from matplotlib import pyplot as plt
plt.switch_backend('agg')
# wandb.init(project="XCSGCNN", entity="wangzhihao")


def predict(net:XCSGCNN_new_structure, val_loader, args:dict):
    loss, correct, total = 0, 0, 0
    loss_C, loss_R, loss_D = 0, 0, 0

    with torch.no_grad():
        for index, data in enumerate(val_loader):
            image = data['image'].to(args['device'])
            label = data['label'].to(args['device'])
            main_object = data['object'].to(args['device'])
            image_name = data['image_name']
            pred_class, pred_fake, pred_real, reg, reconstruct_image, main_object = net(image, label, main_object)
            target_dir = os.path.join(args['result_path'], 'reconstruct_image')
            array_to_image(image_name, reconstruct_image.cpu().data.numpy(), target_dir)
            loss_C_curr, loss_R_curr, loss_D_curr = net.validation_check(pred_class, pred_fake, pred_real, reg, reconstruct_image, main_object, label)
            loss_curr = loss_C_curr + loss_R_curr + loss_D_curr
            loss += loss_curr
            loss_C += loss_C_curr
            loss_R += loss_R_curr
            loss_D += loss_D_curr

            _, predicted = pred_class.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()

            progress_bar(index, len(val_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (loss / (index + 1), 100. * correct / total, correct, total))


if __name__ == '__main__':
    '''
    arguments
    '''
    args = {
        'val_txt_path': '/root/XCSGCNN/VOCdevkit/reformed_data/new_val.txt',
        'ckpt_path': '/root/XCSGCNN/train/ckpt/new_XCSGCNN/ckpt_5_acc26.11_loss3.93.pt',
        'result_path': '/root/XCSGCNN/train/result/new_XCSGCNN',
        'val_batch_size': 16,
        'device': 'cuda:0',
        'num_classes': 20,
        'lr_resnet': 0.001,
        'lr_R': 0.001,
        'lr_D': 0.001,
    }

    if not os.path.exists(args['result_path']):
        os.makedirs(args['result_path'])
    hyper_parameter_path = args['result_path'] + '/parameter.yaml'
    with open(hyper_parameter_path, 'w', encoding='utf-8') as f:
        yaml.dump(args, f, Dumper=yaml.RoundTripDumper)
    current_code_file_name = os.path.basename(sys.argv[0])
    shutil.copy(sys.argv[0], os.path.join(args['result_path'], current_code_file_name))
    block_file = '/root/XCSGCNN/train/model/base_block.py'
    model_file = '/root/XCSGCNN/train/model/XCSGCNN.py'
    shutil.copy(block_file, os.path.join(args['result_path'], os.path.split(block_file)[1]))
    shutil.copy(model_file, os.path.join(args['result_path'], os.path.split(model_file)[1]))

    '''
    loading data...
    '''
    print('===> loading data...')
    val_loader = get_val_loader(args['val_txt_path'], args['val_batch_size'])
    # print('train loader: %d, val loader %d' % (len(train_loader), len(val_loader)))
    # print('trainset size %d, valset size %d' % (len(train_loader)*args['train_batch_size'], len(val_loader)*args['val_batch_size']))

    '''
    loading model
    '''
    print('===> loading model...')
    net = XCSGCNN_new_structure(args)
    ckpt = torch.load(args['ckpt_path'], map_location=args['device'])
    net.to(args['device'])
    net.load_state_dict(ckpt['net'])
    net.eval()
    

    '''
    predicting
    '''
    predict(net, val_loader, args)



