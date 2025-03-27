import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import shutil
import wandb
from model.new_alexnet import new_alexnet
from utils.data_utils import get_train_loader, get_val_loader
from utils.progress_utils import progress_bar
from utils.vis_utils import array_to_image
import ruamel_yaml as yaml
from matplotlib import pyplot as plt
plt.switch_backend('agg')
# wandb.init(project="XCSGCNN", entity="wangzhihao")

# 测试鉴别器的影响


def train(epoch: int, net: new_alexnet, train_loader, args: dict, train_loss: list, train_acc: list):
    print('\nEpoch:[%d/%d]' % (epoch, args['epoch']))
    net.train()
    loss, correct, total = 0, 0, 0
    loss_C, loss_R_shape, loss_D_shape, loss_R_head, loss_D_head = 0, 0, 0, 0, 0

    # optimize_flag = 'reconstruction_discrimination'
    optimize_flag = 'all'

    for index, data in enumerate(train_loader):
        image = data['image'].to(args['device'])
        label = data['label'].to(args['device'])
        object = data['object'].to(args['device'])
        head = data['head'].to(args['device'])
        image_name = data['image_name']
        target_shape_dir = '/root/XCSGCNN/train/visulization/exp_v2/new_alexnet/train/outline/' + str(epoch)
        target_head_dir = '/root/XCSGCNN/train/visulization/exp_v2/new_alexnet/train/head/' + str(epoch)
        # pred_class, pred_fake, pred_real, reg, reconstruct_image, main_object = net(image, label, main_object)
        # if index >= 150:
        pred_class, shape_pred_fake, shape_pred_real, head_pred_fake, head_pred_real, reconstruct_shape, reconstruct_head, object, head = net(image, label, object, head)
        array_to_image(image_name, reconstruct_shape.cpu().data.numpy(), target_shape_dir)
        array_to_image(image_name, reconstruct_head.cpu().data.numpy(), target_head_dir)
    
        loss_C_curr, loss_R_shape_curr, loss_D_shape_curr, loss_R_head_curr, loss_D_head_curr = net.optimize_parameters(pred_class, shape_pred_fake, shape_pred_real, head_pred_fake, head_pred_real, reconstruct_shape, reconstruct_head, object, head, label, optimize_flag)
        loss_curr = loss_C_curr + loss_R_shape_curr + loss_D_shape_curr + loss_R_head_curr + loss_D_head_curr
        loss += loss_curr

        # loss_R += loss_R_curr
        # loss_D += loss_D_curr

        _, predicted = pred_class.max(1)
        total += label.size(0)
        correct += predicted.eq(label).sum().item()

        progress_bar(index, len(train_loader), 'Loss: %.3f ｜ Acc: %.3f%% (%d/%d)' % (loss/(index+1), 100. * correct / total, correct, total))

    avg_loss = loss / len(train_loader)
    train_loss.append(avg_loss)
    # acc = 100. * correct / total
    # train_acc.append(acc)
    # wandb.log({"train_loss_C": loss_C / len(train_loader)})
    # wandb.log({"train_loss_R": loss_R / len(train_loader)})
    # wandb.log({"train_loss_D": loss_D / len(train_loader)})


def val(epoch: int, net: new_alexnet, val_loader, args: dict, val_loss: list, val_acc :list):
    global best_acc
    global best_loss
    net.eval()
    loss, correct, total = 0, 0, 0
    loss_C, loss_R, loss_D = 0, 0, 0

    with torch.no_grad():
        for index, data in enumerate(val_loader):
            image = data['image'].to(args['device'])
            label = data['label'].to(args['device'])
            object = data['object'].to(args['device'])
            head = data['head'].to(args['device'])
            image_name = data['image_name']
            pred_class, shape_pred_fake, shape_pred_real, head_pred_fake, head_pred_real, reconstruct_shape, reconstruct_head, object, head = net(image, label, object, head)
            target_shape_dir = '/root/XCSGCNN/train/visulization/exp_v2/new_alexnet/val/outline/' + str(epoch)
            target_head_dir = '/root/XCSGCNN/train/visulization/exp_v2/new_alexnet/val/head/' + str(epoch)
            array_to_image(image_name, reconstruct_shape.cpu().data.numpy(), target_shape_dir)
            array_to_image(image_name, reconstruct_head.cpu().data.numpy(), target_head_dir)
            loss_C_curr, loss_R_shape_curr, loss_D_shape_curr, loss_R_head_curr, loss_D_head_curr = net.validation_check(pred_class, shape_pred_fake, shape_pred_real, head_pred_fake, head_pred_real, reconstruct_shape, reconstruct_head, object, head, label)
            loss_curr = loss_C_curr + loss_R_shape_curr + loss_D_shape_curr + loss_R_head_curr + loss_D_head_curr
            loss += loss_curr
            # loss_C += loss_C_curr
            # loss_R += loss_R_curr
            # loss_D += loss_D_curr

            _, predicted = pred_class.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()

            progress_bar(index, len(val_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (loss / (index + 1), 100. * correct / total, correct, total))

    acc = 100. * correct / total
    avg_loss = loss / len(val_loader)
    val_loss.append(avg_loss)
    val_acc.append(acc)
    # wandb.log({"val_loss_C": loss_C / len(val_loader)})
    # wandb.log({"val_loss_R": loss_R / len(val_loader)})
    # wandb.log({"val_loss_D": loss_D / len(val_loader)})

    if acc > best_acc or avg_loss < best_loss or epoch%10 == 0:
        print("Saving checkpoints..")
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'loss': avg_loss
        }
        if not os.path.isdir(args['ckpt_path']):
            os.mkdir(args['ckpt_path'])
        torch.save(state, args['ckpt_path'] + str('/ckpt_%d_acc%.2f_loss%.2f.pt' % (epoch, acc, avg_loss)))
        if acc > best_acc:
            best_acc = acc
        if avg_loss < best_loss:
            best_loss = avg_loss
    


if __name__ == '__main__':
    '''
    arguments
    '''
    args = {
        'train_txt_path': '/root/XCSGCNN/VOCdevkit/my_exp_data/train_outline.txt',
        'val_txt_path': '/root/XCSGCNN/VOCdevkit/my_exp_data/val_outline.txt',
        'ckpt_path': '/root/XCSGCNN/train/ckpt/exp_v2/new_alexnet',
        'train_batch_size': 8,
        'val_batch_size': 8,
        'device': 'cuda:0',
        'epoch': 300,
        'num_classes': 20,
        'lr_resnet': 0.001,
        'lr_R': 0.001,
        'lr_D': 0.001,
    }

    if not os.path.exists(args['ckpt_path']):
        os.makedirs(args['ckpt_path'])
    hyper_parameter_path = args['ckpt_path'] + '/parameter.yaml'
    with open(hyper_parameter_path, 'w', encoding='utf-8') as f:
        yaml.dump(args, f, Dumper=yaml.RoundTripDumper)
    current_code_file_name = os.path.basename(sys.argv[0])
    shutil.copy(sys.argv[0], os.path.join(args['ckpt_path'], current_code_file_name))
    block_file = '/root/XCSGCNN/train/model/alexnet_backbone.py'
    model_file = '/root/XCSGCNN/train/model/new_alexnet.py'
    shutil.copy(block_file, os.path.join(args['ckpt_path'], os.path.split(block_file)[1]))
    shutil.copy(model_file, os.path.join(args['ckpt_path'], os.path.split(model_file)[1]))
    '''
    loading data...
    '''
    print('===> loading data...')
    train_loader = get_train_loader(args['train_txt_path'], args['train_batch_size'])
    val_loader = get_val_loader(args['val_txt_path'], args['val_batch_size'])
    print('train loader: %d, val loader %d' % (len(train_loader), len(val_loader)))
    print('trainset size %d, valset size %d' % (len(train_loader)*args['train_batch_size'], len(val_loader)*args['val_batch_size']))

    '''
    model
    '''
    print('===> loading model...')
    net = new_alexnet(args)
    # ckpt = torch.load('/root/XCSGCNN/train/ckpt/XCSGCNN_exp_classification/ckpt_205_acc80.00_loss4.52.pt', map_location=args['device'])
    net.to(args['device'])
    # net.load_state_dict(ckpt['net'])
    '''
    training
    '''
    train_loss = list()
    train_acc = list()
    val_loss = list()
    val_acc = list()

    best_acc = 0
    best_loss = 1000
    # wandb.config = {
    #     "learning_rate": 0.001,
    #     "epochs": 250,
    #     "batch_size": 16
    # }

    for epoch in range(args['epoch']):
        train(epoch, net, train_loader, args, train_loss, train_acc)
        val(epoch, net, val_loader, args, val_loss, val_acc)

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



