import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import shutil
import wandb
from model.new_resnet import new_resnet, resnet_multi_concept_for_mammal
from utils.data_utils import get_train_loader, get_val_loader
from utils.progress_utils import progress_bar
from utils.vis_utils import array_to_image
import ruamel_yaml as yaml
from utils.gpu_utils import MemTracker
from matplotlib import pyplot as plt
plt.switch_backend('agg')
# wandb.init(project="XCSGCNN", entity="wangzhihao")


# gpu_tracker = MemTracker()

def train(epoch: int, net: resnet_multi_concept_for_mammal, train_loader, args: dict, train_loss: list, train_acc: list):
    print('\nEpoch:[%d/%d]' % (epoch, args['epoch']))
    net.train()
    loss, correct, total = 0, 0, 0

    # optimize_flag = 'reconstruction_discrimination'
    # optimize_flag = 'all'
    # optimize_flag = 'split_train'
    optimize_flag = 'all'
    optimize_flag = 'C'
    optimize_flag = 'R_D'

    for index, data in enumerate(train_loader):
        # gpu_tracker.track()
        image = data['image'].to(args['device'])
        label = data['label'].to(args['device'])
        shape = data['outline'].to(args['device'])
        head = data['head'].to(args['device'])
        torso = data['torso'].to(args['device'])
        leg = data['leg'].to(args['device'])
        image_name = data['image_name']
        target_dir = {
            'head': args['visualization_path'] + '/train/head/' + str(epoch),
            'shape': args['visualization_path'] + '/train/shape/' + str(epoch),
            'torso': args['visualization_path'] + '/train/torso/' + str(epoch),
            'leg': args['visualization_path'] + '/train/leg/' + str(epoch),
        }
        if (epoch < 200):
            pred_class, reconstruct_result, loss_curr = net.learning(image, label, head, torso, leg, shape, flag='C')
        else:
            pred_class, reconstruct_result, loss_curr = net.learning(image, label, head, torso, leg, shape, flag='R_D')
            array_to_image(image_name, reconstruct_result['head'], target_dir['head'])
            array_to_image(image_name, reconstruct_result['shape'], target_dir['shape'])
            array_to_image(image_name, reconstruct_result['torso'], target_dir['torso'])
            array_to_image(image_name, reconstruct_result['leg'], target_dir['leg'])
        
        # gpu_tracker.track()
        loss += loss_curr
        # gpu_tracker.track()

        _, predicted = pred_class.max(1)
        total += label.size(0)
        correct += predicted.eq(label).sum().item()

        progress_bar(index, len(train_loader), 'Loss: %.3f ｜ Acc: %.3f%% (%d/%d)' % (loss/(index+1), 100. * correct / total, correct, total))

    avg_loss = loss / len(train_loader)
    train_loss.append(avg_loss)
    acc = 100. * correct / total
    train_acc.append(acc)


def val(epoch: int, net: resnet_multi_concept_for_mammal, val_loader, args: dict, val_loss: list, val_acc :list):
    global best_acc
    global best_loss
    net.eval()
    loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for index, data in enumerate(val_loader):
            image = data['image'].to(args['device'])
            label = data['label'].to(args['device'])
            shape = data['outline'].to(args['device'])
            head = data['head'].to(args['device'])
            torso = data['torso'].to(args['device'])
            leg = data['leg'].to(args['device'])
            image_name = data['image_name']
            pred_class, reconstruct_result, loss_curr = net.validating(image, label, head, torso, leg, shape)
            target_dir = {
                'head': args['visualization_path'] + '/val/head/' + str(epoch),
                'shape': args['visualization_path'] + '/val/shape/' + str(epoch),
                'torso': args['visualization_path'] + '/val/torso/' + str(epoch),
                'leg': args['visualization_path'] + '/val/leg/' + str(epoch),
            }
            array_to_image(image_name, reconstruct_result['head'], target_dir['head'])
            array_to_image(image_name, reconstruct_result['shape'], target_dir['shape'])
            array_to_image(image_name, reconstruct_result['torso'], target_dir['torso'])
            array_to_image(image_name, reconstruct_result['leg'], target_dir['leg'])
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
        'train_txt_path': '/home/user/wangzhihao/XCSGCNN/VOCdevkit/my_exp_data/train.txt',
        'val_txt_path': '/home/user/wangzhihao/XCSGCNN/VOCdevkit/my_exp_data/val.txt',
        'ckpt_path': '/home/user/wangzhihao/train/ckpt/exp_v3/tmp',
        'visualization_path': '/home/user/wangzhihao/XCSGCNN/train/visulization/tmp',
        'train_batch_size': 8,
        'val_batch_size': 8,
        'device': 'cuda:3',
        'epoch': 400,
        'num_classes': 5,
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
    block_file = '/home/user/wangzhihao/XCSGCNN/train/model/resnet_backbone.py'
    model_file = '/home/user/wangzhihao/XCSGCNN/train/model/new_resnet.py'
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
    net = resnet_multi_concept_for_mammal(args)
    device = torch.device(args['device'])
    # ckpt = torch.load('/root/XCSGCNN/train/ckpt/XCSGCNN_exp_classification/ckpt_205_acc80.00_loss4.52.pt', map_location=args['device'])
    # net.to(args['device'])
    net.to(device)
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
