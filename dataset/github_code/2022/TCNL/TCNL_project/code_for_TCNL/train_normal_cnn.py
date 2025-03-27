import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import shutil
# import wandb
# from model.new_resnet import new_resnet, resnet_multi_concept_for_mammal
from model.resnet import resnet
# from model.new_vgg import vgg
from utils.data_utils import get_train_loader, get_val_loader, get_scene_train_loader, get_scene_val_loader, get_normal_train_loader, get_normal_val_loader
from utils.progress_utils import progress_bar
from utils.vis_utils import array_to_image
import ruamel_yaml as yaml
# from ruamel.ymal import yaml
from utils.gpu_utils import MemTracker
from matplotlib import pyplot as plt
plt.switch_backend('agg')


def train(epoch: int, net:resnet, optimizer, train_loader, criterion, args: dict, train_loss: list, train_acc: list):
    print('\nEpoch:[%d/%d]' % (epoch, args['epoch']))
    net.train()
    loss, correct, total = 0, 0, 0

    for index, data in enumerate(train_loader):
        image = data['image'].to(args['device'])
        label = data['label'].to(args['device'])

        optimizer.zero_grad()
        outputs = net(image)
        loss_curr = criterion(outputs, label)
        # print(loss_curr)
        loss_curr.backward()
        optimizer.step()

        loss += loss_curr.cpu().data.numpy()
        _, predicted = outputs.max(1)
        total += label.size(0)
        correct += predicted.eq(label).sum().item()

        progress_bar(index, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (loss/(index+1), 100.*correct/total, correct, total))

    avg_loss = loss / len(train_loader)
    train_loss.append(avg_loss)
    acc = 100. * correct / total
    train_acc.append(acc)


def val(epoch: int, net: resnet, criterion, val_loader, args: dict, val_loss: list, val_acc :list):
    global best_acc
    global best_loss
    net.eval()
    loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for index, data in enumerate(val_loader):
            image = data['image'].to(args['device'])
            label = data['label'].to(args['device'])
            outputs = net(image)
            loss_curr = criterion(outputs, label)
            loss += loss_curr.cpu().data.numpy()

            _, predicted = outputs.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()

            progress_bar(index, len(val_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (loss / (index + 1), 100. * correct / total, correct, total))

    acc = 100. * correct / total
    avg_loss = loss / len(val_loader)
    val_loss.append(avg_loss)
    val_acc.append(acc)

    # if acc > best_acc or avg_loss < best_loss:
    #     print("Saving checkpoints..")
    #     state = {
    #         'net': net.state_dict(),
    #         'acc': acc,
    #         'epoch': epoch,
    #         'loss': avg_loss
    #     }
    #     if not os.path.isdir(args['ckpt_path']):
    #         os.mkdir(args['ckpt_path'])
    #     torch.save(state, args['ckpt_path'] + str('/ckpt_%d_acc%.2f_loss%.2f.pt' % (epoch, acc, avg_loss)))
    #     if acc > best_acc:
    #         best_acc = acc
    #     if avg_loss < best_loss:
    #         best_loss = avg_loss


if __name__ == '__main__':
    '''
    arguments
    '''
    args = {
        'train_txt_path': '/home/user/wangzhihao/XCSGCNN/VOCdevkit/my_exp_data/train_multi_concept.txt',
        'val_txt_path': '/home/user/wangzhihao/XCSGCNN/VOCdevkit/my_exp_data/val_multi_concept.txt',
        'ckpt_path': '/home/user/wangzhihao/XCSGCNN/train/ckpt/resnet_multi_text_concept_check',
        'train_batch_size': 16,
        'val_batch_size': 16,
        'device': 'cuda:0',
        'epoch': 500,
        'num_classes': 5,
        'lr_resnet': 0.01,
        'lr_R': 0.001,
        'lr_D': 0.001,
        'momentum': 0.9,
        'weight_decay': 0.0005
    }

    if not os.path.exists(args['ckpt_path']):
        os.makedirs(args['ckpt_path'])
    hyper_parameter_path = args['ckpt_path'] + '/parameter.yaml'
    with open(hyper_parameter_path, 'w', encoding='utf-8') as f:
        yaml.dump(args, f, Dumper=yaml.RoundTripDumper)
    current_code_file_name = os.path.basename(sys.argv[0])
    shutil.copy(sys.argv[0], os.path.join(args['ckpt_path'], current_code_file_name))

    '''
    loading data...
    '''
    print('===> loading data...')
    # train_loader = get_normal_train_loader(args['train_txt_path'], args['train_batch_size'])
    # val_loader = get_normal_val_loader(args['val_txt_path'], args['val_batch_size'])
    train_loader = get_train_loader(args['train_txt_path'], args['train_batch_size'])
    val_loader = get_val_loader(args['val_txt_path'], args['val_batch_size'])
    # train_loader = get_scene_train_loader(args['train_txt_path'], args['train_batch_size'])
    # val_loader = get_scene_val_loader(args['val_txt_path'], args['val_batch_size'])
    print('train loader: %d, val loader %d' % (len(train_loader), len(val_loader)))
    print('trainset size %d, valset size %d' % (len(train_loader)*args['train_batch_size'], len(val_loader)*args['val_batch_size']))

    '''
    model
    '''
    print('===> loading model...')
    net = resnet(num_classes=5)
    # net = resnet(num_classes=4)
    net = net.to(args['device'])

    '''
    training
    '''
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args['lr_resnet'])
    train_loss = list()
    train_acc = list()
    val_loss = list()
    val_acc = list()

    best_acc = 0
    best_loss = 1000

    for epoch in range(args['epoch']):
        train(epoch, net, optimizer, train_loader, criterion, args, train_loss, train_acc)
        val(epoch, net, criterion, val_loader, args, val_loss, val_acc)

        plt.figure()
        plt.plot(np.linspace(0, epoch, len(train_loss)), train_loss)
        plt.savefig(os.path.join(args['ckpt_path'], 'train_loss.png'))
        plt.close()

        plt.plot(np.linspace(0, epoch, len(train_acc)), train_acc)
        plt.savefig(os.path.join(args['ckpt_path'], 'train_acc.png'))
        plt.close()

        plt.plot(np.linspace(0, epoch, len(val_loss)), val_loss)
        plt.savefig(os.path.join(args['ckpt_path'], 'val_loss.png'))
        plt.close()

        plt.plot(np.linspace(0, epoch, len(val_acc)), val_acc)
        plt.savefig(os.path.join(args['ckpt_path'], 'val_acc.png'))
        plt.close()

        # plt.plot(np.linspace(0, epoch, len(lr_list)), lr_list)
        # plt.savefig(os.path.join(args['ckpt_path'], 'lr.png'))
        # plt.close()



