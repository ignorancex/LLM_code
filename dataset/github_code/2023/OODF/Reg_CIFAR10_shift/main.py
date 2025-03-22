#!/usr/bin/env python

import sys, argparse
import numpy as np
import torch
import utils
import datetime
# Arguments
parser=argparse.ArgumentParser()
# parser.add_argument('--seed', type=int, default= 0, help='(default=%(default)d)')
# parser.add_argument('--experiment',default='cifar-10', type=str,required=False, help='(default=%(default)s)')
parser.add_argument('--approach', default='OWM', type=str, choices=['OWM', 'AOP'], required=False,help='(default=%(default)s)')
parser.add_argument('--nepochs', default=25, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--lr', default=0.02, type=float, required=False, help='(default=%(default)f)')
# parser.add_argument('--parameter', type=str, default='', help='(default=%(default)s)')
parser.add_argument('--shift', action = 'store_true', default = False)
parser.add_argument('--perc', default=0.7, type=float, required=False, help='(default=%(default)f)')
parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100'], required=False,help='(default=%(default)s)')
args = parser.parse_args()
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ignore warning
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use gpu0,1

print('='*100)
print('Arguments =')
for arg in vars(args):
    print('\t'+arg+':', getattr(args,arg))
print('='*100)
# print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'GPU  '+os.environ["CUDA_VISIBLE_DEVICES"])
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print('='*100)
########################################################################################################################

# Seed
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(args.seed)
# else:
#     print('[CUDA unavailable]')
#     sys.exit()

import cifar as dataloader
if args.approach == 'OWM':
    from owm import Appr
    import cnn_owm as network
else:
    from aop import Appr
    import cnn_aop as network
########################################################################################################################
if_CL = True
# Load
print('Load data...')
t_num = 10
if args.dataset == 'cifar10':
    data, taskcla, inputsize = dataloader.get_cifar10(t_num=t_num, if_atk=args.shift, perc=args.perc)
else:
    data, taskcla, inputsize = dataloader.get_cifar100(t_num=t_num, if_atk=args.shift, perc=args.perc)
print('Input size =', inputsize, '\nTask info =', taskcla)

# Inits
print('Inits...')
net = network.Net(inputsize).cuda() if args.dataset == 'cifar10' else network.Net100(inputsize).cuda()
utils.print_model_report(net)

# appr = approach.Appr(net, nepochs=args.nepochs, lr=args.lr, args=args)
appr = Appr(net, nepochs=args.nepochs, lr=args.lr)
utils.print_optimizer_config(appr.optimizer)
print('-'*100)

# Loop tasks
acc = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
lss = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
for t, ncla in taskcla:
    print('*'*100)
    print('Task {:2d} ({:s})'.format(t+1, data[t]['name']))
    print('*'*100)

    xtrain = data[t]['train']['x'].cuda()
    ytrain = data[t]['train']['y'].cuda()
    xvalid = data[t]['test']['x'].cuda()
    yvalid = data[t]['test']['y'].cuda()

    # Train
    appr.train(t, xtrain, ytrain, xvalid, yvalid, data, t_num) if args.approach == 'AOP' else appr.train(t, xtrain, ytrain, xvalid, yvalid, data, if_CL, t_num)
    print('-'*100)

    # Test
    for u in range(t+1):
        xtest = data[u]['test']['x'].cuda()
        ytest = data[u]['test']['y'].cuda()
        test_loss, test_acc = appr.eval(xtest, ytest)
        print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.2f}% <<<'.format(u, data[u]['name'], test_loss, 100*test_acc))
        acc[t, u] = test_acc
        lss[t, u] = test_loss


# Done
print('*'*100)
print('Accuracies =')
for i in range(acc.shape[0]):
    print('\t',end='')
    for j in range(acc.shape[1]):
        print('{:5.2f}% '.format(100*acc[i, j]),end='')
    print()
print('*'*100)
print('Done!')
# print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'GPU  '+os.environ["CUDA_VISIBLE_DEVICES"])
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print('='*100)

# torch.save(net.state_dict(), "net_learn_"+str(t_num)+".pt")


