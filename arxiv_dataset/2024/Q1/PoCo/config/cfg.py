import argparse
from sys import implementation
import torch
import os
import torch.backends.cudnn as cudnn
import numpy as np

from utils.data import arg2str, str2bool


class BaseConfig(object):

    def __init__(self):

        self.parser = argparse.ArgumentParser()

        #basic cfg
        self.parser.add_argument('--exp_name', type=str, help='Experiment name', default='test')
        self.parser.add_argument('--gpu_id', default='2', type=str, help='GPU ID Use to train/test module')
        self.parser.add_argument('--save_folder', default='/.../', help='Path to save checkpoint models')
        self.parser.add_argument('--save_log', default='/.../',help='Path to save checkpoint models')
        self.parser.add_argument('--resume', default=None, type=str, help='Path to target resume checkpoint')
        self.parser.add_argument('--pretrained', default=None, type=str, help='Path to target pretrained checkpoint')

        self.parser.add_argument('--data_name', default='DR', choices=['DR', 'Cataract_C', 'Cataract_P', 'DR_pretrain', 'amd', 'pm', 'me'], type=str, help='module')
        self.parser.add_argument('--model_name', default='ResNet', choices=['VGG', 'ResNet50', 'pvtb2', 'resnet18p', 'resnet18f', 'resnet18'], type=str, help='module')
        self.parser.add_argument('--eval_only', action='store_true', help='use eval mode')
        self.parser.add_argument('--num_classes', default=7, type=int,help='number class)')

        #train cfg

        self.parser.add_argument('--start_iter', default=0, type=int,help='Begin counting iterations starting from this value (should be used with resume)')
        self.parser.add_argument('--max_iter', default=4000, type=int, help='Number of training iterations')
        self.parser.add_argument('--warmup_steps', default=100, type=int, help='Number of warmup iterations')
        self.parser.add_argument('--loss_name', default='CELoss', choices=['CELoss', 'FocalLoss'], type=str, help='loss')
        self.parser.add_argument('--pretrain', action='store_true', help='use pretrain')

        self.parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, help='initial learning rate')
        self.parser.add_argument('--lr_adjust', default='fix', choices=['fix', 'poly'], type=str, help='Learning Rate Adjust Strategy')
        self.parser.add_argument('--stepvalues', default=[1000,2000,3000], nargs='*', type=int, help='# of iter to change lr')

        self.parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float, help='Weight decay for SGD')
        self.parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD lr')
        self.parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
        self.parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
        self.parser.add_argument('--optim', default='SGD', type=str, choices=['SGD', 'Adam'], help='Optimizer')
        self.parser.add_argument('--save_freq', default=400, type=int, help='save weights every # iterations')
        self.parser.add_argument('--display_freq', default=10, type=int,help='display training metrics every # iterations,  =train_epoch_size/2, train_epoch_size = train_total_num/batch_size')  # 20 for one step data show
        self.parser.add_argument('--val_freq', default=10, type=int,help='do validation every # iterations, =train_epoch_size-1')
        self.parser.add_argument('--k_fold', default=4, type=int, help='use k fold validation')
        self.parser.add_argument('--k_fold_start', default=0, type=int, help='start from k fold')
        self.parser.add_argument('--dropout', default=1, type=float, help='Dropout rate of Resnet')

      
        self.parser.add_argument('--polar', default=256, type=int, help='polar size (0 means no apply)')

        self.parser.add_argument('--details', default=None, type=str, help='implementation details')


    def initialize(self, fixed=None):
        # print(fixed)
        self.args = self.parse(fixed)
        print('--------------Options Log-------------')
        print(arg2str(self.args))
        print('--------------Options End-------------')

        #Create save dict
        if not os.path.exists(self.args.save_folder):
            os.makedirs(self.args.save_folder, exist_ok=True)



        return self.args


    def parse(self, fixed=None):

        if fixed is not None:
            args = self.parser.parse_known_args(fixed)[0]
        else:
            args = self.parser.parse_known_args()[0]

        return args


    def save_result(self, acc_list, loss_list, loss_refine_list):
        acc_mean = np.mean(acc_list)
        acc_std = np.std(acc_list)

        loss_mean = np.mean(loss_list)
        loss_std = np.std(loss_list)

        loss_refine_mean = np.mean(loss_refine_list)
        loss_refine_std = np.std(loss_refine_list)



        acc_save_path = os.path.join(self.args.save_folder, self.args.exp_name) + '/acc_{}_std_{}.txt'.format(acc_mean, acc_std)
        loss_save_path = os.path.join(self.args.save_folder, self.args.exp_name) + '/loss_{}_std_{}.txt'.format(loss_mean, loss_std)
        loss_refine_save_path = os.path.join(self.args.save_folder, self.args.exp_name) + '/loss2_{}_std_{}.txt'.format(loss_refine_mean, loss_refine_std)

        with open(acc_save_path, "w") as f:
            for c, ac in enumerate(acc_list):
                f.write('train_{}_acc is {}\n'.format(c, ac))
        f.close()

        with open(loss_save_path, "w") as f:
            for c, ac in enumerate(loss_list):
                f.write('train_{}_loss is {}\n'.format(c, ac))
        f.close()

        with open(loss_refine_save_path, "w") as f:
            for c, ac in enumerate(loss_refine_list):
                f.write('train_{}_loss is {}\n'.format(c, ac))
        f.close()



