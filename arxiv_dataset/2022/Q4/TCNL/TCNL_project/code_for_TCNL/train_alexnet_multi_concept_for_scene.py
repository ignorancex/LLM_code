import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel
import torch.distributed as dist
import sys
import shutil
# import wandb
# from model.new_vgg import new_vgg, vgg_multi_concept_for_scene
from model.new_alexnet import *
from utils.data_utils import get_scene_train_loader, get_scene_val_loader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler
from utils.progress_utils import progress_bar
from utils.vis_utils import array_to_image
import ruamel_yaml as yaml
from utils.gpu_utils import MemTracker
from matplotlib import pyplot as plt
plt.switch_backend('agg')
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
# wandb.init(project="XCSGCNN", entity="wangzhihao")


# gpu_tracker = MemTracker()

def train(epoch: int, net: alexnet_multi_concept_for_scene, train_loader, args: dict, train_loss: list, train_acc: list):
    print('\nEpoch:[%d/%d]' % (epoch, args['epoch']))
    # net.train()
    loss, correct, total = 0, 0, 0

    # optimize_flag = 'reconstruction_discrimination'
    optimize_flag = 'all'
    optimize_flag = 'split_train'

    for index, data in enumerate(train_loader):
        # gpu_tracker.track()
        image = data['image'].to(args['device'])
        label = data['label'].to(args['device'])
        bed = data['bed'].to(args['device'])
        bedsidetable = data['bedsidetable'].to(args['device'])
        lamp = data['lamp'].to(args['device'])
        sofa = data['sofa'].to(args['device'])
        chair = data['chair'].to(args['device'])
        table = data['table'].to(args['device'])
        shelf = data['shelf'].to(args['device'])
        seat = data['seat'].to(args['device'])
        screen = data['screen'].to(args['device'])
        stage = data['stage'].to(args['device'])
        image_name = data['image_name']
        target_dir = {
            'bed': args['visualization_path'] + '/train/bed/' + str(epoch),
            'bedsidetable': args['visualization_path'] + '/train/bedsidetable/' + str(epoch),
            'lamp': args['visualization_path'] + '/train/lamp/' + str(epoch),
            'sofa': args['visualization_path'] + '/train/sofa/' + str(epoch),
            'chair': args['visualization_path'] + '/train/chair/' + str(epoch),
            'table': args['visualization_path'] + '/train/table/' + str(epoch),
            'shelf': args['visualization_path'] + '/train/shelf/' + str(epoch),
            'seat': args['visualization_path'] + '/train/seat/' + str(epoch),
            'screen': args['visualization_path'] + '/train/screen/' + str(epoch),
            'stage': args['visualization_path'] + '/train/stage/' + str(epoch),
        }
        pred_class, reconstruct_result, loss_curr = net.learning(image, label, bed, sofa, shelf, seat)
        if epoch%30 == 0:
            array_to_image(image_name, reconstruct_result['bed'], target_dir['bed'])
            # array_to_image(image_name, reconstruct_result['bedsidetable'], target_dir['bedsidetable'])
            # array_to_image(image_name, reconstruct_result['lamp'], target_dir['lamp'])
            array_to_image(image_name, reconstruct_result['sofa'], target_dir['sofa'])
            # array_to_image(image_name, reconstruct_result['chair'], target_dir['chair'])
            # array_to_image(image_name, reconstruct_result['table'], target_dir['table'])
            array_to_image(image_name, reconstruct_result['shelf'], target_dir['shelf'])
            array_to_image(image_name, reconstruct_result['seat'], target_dir['seat'])
            # array_to_image(image_name, reconstruct_result['screen'], target_dir['screen'])
            # array_to_image(image_name, reconstruct_result['stage'], target_dir['stage'])
        
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


def val(epoch: int, net: alexnet_multi_concept_for_scene, val_loader, args: dict, val_loss: list, val_acc :list):
    global best_acc
    global best_loss
    # net.eval()
    loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for index, data in enumerate(val_loader):
            image = data['image'].to(args['device'])
            label = data['label'].to(args['device'])
            bed = data['bed'].to(args['device'])
            bedsidetable = data['bedsidetable'].to(args['device'])
            lamp = data['lamp'].to(args['device'])
            sofa = data['sofa'].to(args['device'])
            chair = data['chair'].to(args['device'])
            table = data['table'].to(args['device'])
            shelf = data['shelf'].to(args['device'])
            seat = data['seat'].to(args['device']) 
            screen = data['screen'].to(args['device'])
            stage = data['stage'].to(args['device'])
            image_name = data['image_name']
            target_dir = {
                'bed': args['visualization_path'] + '/val/bed/' + str(epoch),
                'bedsidetable': args['visualization_path'] + '/val/bedsidetable/' + str(epoch),
                'lamp': args['visualization_path'] + '/val/lamp/' + str(epoch),
                'sofa': args['visualization_path'] + '/val/sofa/' + str(epoch),
                'chair': args['visualization_path'] + '/val/chair/' + str(epoch),
                'table': args['visualization_path'] + '/val/table/' + str(epoch),
                'shelf': args['visualization_path'] + '/val/shelf/' + str(epoch),
                'seat': args['visualization_path'] + '/val/seat/' + str(epoch),
                'screen': args['visualization_path'] + '/val/screen/' + str(epoch),
                'stage': args['visualization_path'] + '/val/stage/' + str(epoch),
            }
            pred_class, reconstruct_result, loss_curr = net.validating(image, label, bed, sofa, shelf, seat)
            if epoch%30 == 0:
                array_to_image(image_name, reconstruct_result['bed'], target_dir['bed'])
                # array_to_image(image_name, reconstruct_result['bedsidetable'], target_dir['bedsidetable'])
                # array_to_image(image_name, reconstruct_result['lamp'], target_dir['lamp'])
                array_to_image(image_name, reconstruct_result['sofa'], target_dir['sofa'])
                # array_to_image(image_name, reconstruct_result['chair'], target_dir['chair'])
                # array_to_image(image_name, reconstruct_result['table'], target_dir['table'])
                array_to_image(image_name, reconstruct_result['shelf'], target_dir['shelf'])
                array_to_image(image_name, reconstruct_result['seat'], target_dir['seat'])
            # array_to_image(image_name, reconstruct_result['screen'], target_dir['screen'])
            # array_to_image(image_name, reconstruct_result['stage'], target_dir['stage'])
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
        'train_txt_path': '/home/wangzhihao/XAI/XCSGCNN/VOCdevkit/scene/train.txt',
        'val_txt_path': '/home/wangzhihao/XAI/XCSGCNN/VOCdevkit/scene/val.txt',
        'ckpt_path': '/home/wangzhihao/XAI/XCSGCNN/train/ckpt/alexnet_multi_concept_for_scene',
        'visualization_path': '/home/wangzhihao/XAI/XCSGCNN/train/visualization/alexnet_multi_concept_scene',
        'train_batch_size': 8,
        'val_batch_size': 8,
        'device': 'cuda:0',
        'epoch': 400,
        'num_classes': 4,
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
    block_file = '/home/wangzhihao/XAI/XCSGCNN/train/model/alexnet_backbone.py'
    model_file = '/home/wangzhihao/XAI/XCSGCNN/train/model/new_alexnet.py'
    shutil.copy(block_file, os.path.join(args['ckpt_path'], os.path.split(block_file)[1]))
    shutil.copy(model_file, os.path.join(args['ckpt_path'], os.path.split(model_file)[1]))
    '''
    loading data...
    '''
    print('===> loading data...')
    train_loader = get_scene_train_loader(args['train_txt_path'], args['train_batch_size'])
    val_loader = get_scene_val_loader(args['val_txt_path'], args['val_batch_size'])
    print('train loader: %d, val loader %d' % (len(train_loader), len(val_loader)))
    print('trainset size %d, valset size %d' % (len(train_loader)*args['train_batch_size'], len(val_loader)*args['val_batch_size']))

    '''
    model
    '''
    print('===> loading model...')
    # dist.init_process_group(backend='nccl')

    net = alexnet_multi_concept_for_scene(args)
    # device = torch.device(args['device'])
    # net = DataParallel(net, device_ids=[0, 1])
    net.to(args['device'])
    # net.to(device)
    # net = net.module

    # ckpt = torch.load('/root/XCSGCNN/train/ckpt/XCSGCNN_exp_classification/ckpt_205_acc80.00_loss4.52.pt', map_location=args['device'])
    # net.to(args['device'])
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
