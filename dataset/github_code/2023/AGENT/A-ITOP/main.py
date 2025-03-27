from __future__ import print_function

import os
import time
import argparse
import logging
import hashlib
import copy
import numpy as np
import pickle

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import sparselearning
from sparselearning.core import Masking, CosineDecay, LinearDecay
from sparselearning.models import AlexNet, VGG16, LeNet_300_100, LeNet_5_Caffe, WideResNet, MLP_CIFAR10, ResNet34, ResNet18
from sparselearning.utils import get_mnist_dataloaders, get_cifar10_dataloaders, get_cifar100_dataloaders
from sparselearning.train_helper import SGD, SGD_snap
import torchvision
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
cudnn.benchmark = True
cudnn.deterministic = True

if not os.path.exists('./models'): os.mkdir('./models')
if not os.path.exists('./logs'): os.mkdir('./logs')
logger = None

models = {}
models['MLPCIFAR10'] = (MLP_CIFAR10,[])
models['lenet5'] = (LeNet_5_Caffe,[])
models['lenet300-100'] = (LeNet_300_100,[])
models['ResNet34'] = ()
models['ResNet18'] = ()
models['alexnet-s'] = (AlexNet, ['s', 10])
models['alexnet-b'] = (AlexNet, ['b', 10])
models['vgg-c'] = (VGG16, ['C', 10])
models['vgg-d'] = (VGG16, ['D', 10])
models['vgg-like'] = (VGG16, ['like', 10])
models['wrn-28-2'] = (WideResNet, [28, 2, 10, 0.3])
models['wrn-22-8'] = (WideResNet, [22, 8, 10, 0.3])
models['wrn-16-8'] = (WideResNet, [16, 8, 10, 0.3])
models['wrn-16-10'] = (WideResNet, [16, 10, 10, 0.3])

def setup_logger(args):
    global logger
    if logger == None:
        logger = logging.getLogger()
    else:  # wish there was a logger.close()
        for handler in logger.handlers[:]:  # make a copy of the list
            logger.removeHandler(handler)

    args_copy = copy.deepcopy(args)
    # copy to get a clean hash
    # use the same log file hash if iterations or verbose are different
    # these flags do not change the results
    args_copy.iters = 1
    args_copy.verbose = False
    args_copy.log_interval = 1
    args_copy.seed = 0

    log_path = './logs/{0}_{1}_{2}.log'.format(args.model, args.density, hashlib.md5(str(args_copy).encode('utf-8')).hexdigest()[:8])

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%H:%M:%S')

    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def print_and_log(msg):
    global logger
    print(msg)
    logger.info(msg)

def train(args, model, model_snap, device, train_loader, optimizer, optimizer_snap, epoch, mask=None, tau=0.1, alpha = 0.3, gamma = 0.1):
    model.train()
    model_snap.train()
    train_loss = 0
    correct = 0
    n = 0

    tr_acc_seq = []; tr_loss_seq = []
    tau_seq = []; tau_curr_seq = []

    loss_curr = []; loss_last = []

    optimizer_snap.zero_grad() 
    mul = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        mul += 1
        data, target = data.to(device), target.to(device)
        #optimizer.zero_grad()
        output = model(data)
        output_snap = model_snap(data)
        loss = F.nll_loss(output, target)
        loss_snap = F.nll_loss(output_snap, target)
        loss.backward()

        loss_curr.append(loss.detach().cpu().numpy())
        loss_last.append(loss_snap.detach().cpu().numpy())
    
    tau_curr = np.cov(loss_curr, loss_last, ddof=1)[0][1] / np.var(loss_last, ddof=1)
    tau = (1 - alpha) * tau + alpha * tau_curr
    tau_seq.append(tau); tau_curr_seq.append(tau_curr)
    print("curr tau: {}; tau used: {}".format(tau_curr, tau))

    u = optimizer.get_param_groups()
    mask.optimizer.set_u(u)
    optimizer_snap.set_param_groups(optimizer.get_param_groups())

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        if args.fp16: data = data.half()

        mask.optimizer.zero_grad()
        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output, target)

        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        n += target.shape[0]

        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()

        optimizer_snap.zero_grad()
        output_snap = model_snap(data)
        loss_snap = F.nll_loss(output_snap, target)
        loss_snap.backward()

        if mask is not None: mask.step(optimizer_snap.get_param_groups(), tau * gamma, mul)
        else: optimizer.step(optimizer_snap.get_param_groups(), tau * gamma, mul)

        tr_acc_seq.append(correct / float(n)); tr_loss_seq.append(loss.item())

        if batch_idx % args.log_interval == 0:
            print_and_log('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Accuracy: {}/{} ({:.3f}% '.format(
                epoch, batch_idx * len(data), len(train_loader)*args.batch_size,
                100. * batch_idx / len(train_loader), loss.item(), correct, n, 100. * correct / float(n)))


    # training summary
    print_and_log('\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        'Training summary' ,
        train_loss/batch_idx, correct, n, 100. * correct / float(n)))

    return tr_acc_seq, tr_loss_seq, tau_seq, tau_curr_seq, tau

def evaluate(args, model, device, test_loader, is_test_set=False):
    model.eval()
    test_loss = 0
    correct = 0
    n = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if args.fp16: data = data.half()
            model.t = target
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            n += target.shape[0]

    test_loss /= float(n)

    print_and_log('\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        'Test evaluation' if is_test_set else 'Evaluation',
        test_loss, correct, n, 100. * correct / float(n)))
    return correct / float(n)

def evaluate_more(args, model, device, test_loader, is_test_set=False):
    model.eval()
    test_loss = 0
    correct = 0
    n = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if args.fp16: data = data.half()
            model.t = target
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            n += target.shape[0]

    test_loss /= float(n)

    print_and_log('\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        'Test evaluation' if is_test_set else 'Evaluation',
        test_loss, correct, n, 100. * correct / float(n)))
    return correct / float(n), test_loss


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--multiplier', type=int, default=1, metavar='N',
                        help='extend training time by multiplier times')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=17, metavar='S', help='random seed (default: 17)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--optimizer', type=str, default='sgd', help='The optimizer to use. Default: sgd. Options: sgd, adam.')
    randomhash = ''.join(str(time.time()).split('.'))
    parser.add_argument('--save', type=str, default=randomhash + '.pt',
                        help='path to save the final model')
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--decay_frequency', type=int, default=25000)
    parser.add_argument('--l1', type=float, default=0.0)
    parser.add_argument('--fp16', action='store_true', help='Run in fp16 mode.')
    parser.add_argument('--valid_split', type=float, default=0.1)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--start-epoch', type=int, default=1)
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--l2', type=float, default=5.0e-4)
    parser.add_argument('--iters', type=int, default=1, help='How many times the model should be run after each other. Default=1')
    parser.add_argument('--save-features', action='store_true', help='Resumes a saved model and saves its feature data to disk for plotting.')
    parser.add_argument('--bench', action='store_true', help='Enables the benchmarking of layers and estimates sparse speedups')
    parser.add_argument('--max-threads', type=int, default=10, help='How many threads to use for data loading.')
    # ITOP settings
    sparselearning.core.add_sparse_args(parser)

    args = parser.parse_args()
    setup_logger(args)
    print_and_log(args)

    if args.fp16:
        try:
            from apex.fp16_utils import FP16_Optimizer
        except:
            print('WARNING: apex not installed, ignoring --fp16 option')
            args.fp16 = False

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print_and_log('\n\n')
    print_and_log('='*80)
    torch.manual_seed(args.seed)
    for i in range(args.iters):
        print_and_log("\nIteration start: {0}/{1}\n".format(i+1, args.iters))

        if args.data == 'mnist':
            train_loader, valid_loader, test_loader = get_mnist_dataloaders(args, validation_split=args.valid_split)
        elif args.data == 'cifar10':
            train_loader, valid_loader, test_loader = get_cifar10_dataloaders(args, args.valid_split, max_threads=args.max_threads)
        elif args.data == 'cifar100':
            train_loader, valid_loader, test_loader = get_cifar100_dataloaders(args, args.valid_split, max_threads=args.max_threads)
        if args.model not in models:
            print('You need to select an existing model via the --model argument. Available models include: ')
            for key in models:
                print('\t{0}'.format(key))
            raise Exception('You need to select a model')
        elif args.model == 'ResNet18':
            model = ResNet18(c=100).to(device)
        elif args.model == 'ResNet34':
            model = ResNet34(c=100).to(device)
        else:
            cls, cls_args = models[args.model]
            model = cls(*(cls_args + [args.save_features, args.bench])).to(device)
        print_and_log(model)
        print_and_log('=' * 60)
        print_and_log(args.model)
        print_and_log('=' * 60)

        print_and_log('=' * 60)
        print_and_log('Prune mode: {0}'.format(args.death))
        print_and_log('Growth mode: {0}'.format(args.growth))
        print_and_log('Redistribution mode: {0}'.format(args.redistribution))
        print_and_log('=' * 60)

        model_snap = copy.deepcopy(model)

        optimizer = None
        if args.optimizer == 'sgd':
            optimizer = SGD(model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.l2, nesterov=True)
            #optimizer = optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.l2, nesterov=True)
            optimizer_snap = SGD_snap(model_snap.parameters())
        elif args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.l2)
        else:
            print('Unknown optimizer: {0}'.format(args.optimizer))
            raise Exception('Unknown optimizer.')

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs / 2) * args.multiplier, int(args.epochs * 3 / 4) * args.multiplier], last_epoch=-1)


        if args.resume:
            if os.path.isfile(args.resume):
                print_and_log("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                args.start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print_and_log("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
                print_and_log('Testing...')
                evaluate(args, model, device, test_loader)
                model.feats = []
                model.densities = []
                plot_class_feature_histograms(args, model, device, train_loader, optimizer)
            else:
                print_and_log("=> no checkpoint found at '{}'".format(args.resume))


        if args.fp16:
            print('FP16')
            optimizer = FP16_Optimizer(optimizer,
                                       static_loss_scale = None,
                                       dynamic_loss_scale = True,
                                       dynamic_loss_args = {'init_scale': 2 ** 16})
            model = model.half()

        mask = None
        if args.sparse:
            decay = CosineDecay(args.death_rate, len(train_loader)*(args.epochs*args.multiplier))
            mask = Masking(optimizer, death_rate=args.death_rate, death_mode=args.death, death_rate_decay=decay, growth_mode=args.growth,
                           redistribution_mode=args.redistribution, args=args)
            mask.add_module(model, sparse_init=args.sparse_init, density=args.density)

        best_acc = 0.0

        sp = str(1 - args.density)
        if len(sp) > 6:
            sp = sp[:6]
        sp = sp.split('.')
        if len(sp) == 2:
            sp = sp[0] + '_' + sp[1]
        else:
            sp = sp[0]
        f_name = str(args.data) + "_" + str(args.model) + '_' + sp + '_' + str(args.growth) + '_' + str(args.death) + '.pickle'

        tr_acc_seq_total = []; tr_loss_seq_total = []
        vr_acc_seq_total = []; vr_loss_seq_total = []
        tau_seq_total = []; tau_curr_seq_total = []
        tau = 0.1

        for epoch in range(1, args.epochs*args.multiplier + 1):
            t0 = time.time()
            tr_acc_seq, tr_loss_seq, tau_seq, tau_curr_seq, tau = train(args, model, model_snap, device, train_loader, optimizer, optimizer_snap, epoch, mask, tau=tau)
            lr_scheduler.step()
            if args.valid_split > 0.0:
                val_acc, val_loss = evaluate_more(args, model, device, valid_loader)
                vr_acc_seq_total.append(val_acc); vr_loss_seq_total.append(val_loss)

            tr_acc_seq_total.append(tr_acc_seq); tr_loss_seq_total.append(tr_loss_seq)
            tau_seq_total.append(tau_seq); tau_curr_seq_total.append(tau_curr_seq)

            if val_acc > best_acc:
                print('Saving model')
                best_acc = val_acc
                torch.save(model.state_dict(), args.save)

            with open('./stats/' + f_name, 'wb') as f_save:
                pickle.dump([tr_acc_seq_total, tr_loss_seq_total, vr_acc_seq_total, vr_loss_seq_total, tau_seq_total, tau_curr_seq_total], f_save)

            print_and_log('Current learning rate: {0}. Time taken for epoch: {1:.2f} seconds.\n'.format(optimizer.param_groups[0]['lr'], time.time() - t0))
        print('Testing model')
        model.load_state_dict(torch.load(args.save))
        evaluate(args, model, device, test_loader, is_test_set=True)
        print_and_log("\nIteration end: {0}/{1}\n".format(i+1, args.iters))

        layer_fired_weights, total_fired_weights = mask.fired_masks_update()
        for name in layer_fired_weights:
            print('The final percentage of fired weights in the layer', name, 'is:', layer_fired_weights[name])
        print('The final percentage of the total fired weights is:', total_fired_weights)

        with open('./stats/' + f_name, 'wb') as f_save:
            pickle.dump([tr_acc_seq_total, tr_loss_seq_total, vr_acc_seq_total, vr_loss_seq_total, tau_seq_total, tau_curr_seq_total], f_save)

if __name__ == '__main__':
   main()
