import argparse
import numpy as np
import os
import sys
import tabulate
import time
import torch
import torch.nn.functional as F

import data
import models
import utils

parser = argparse.ArgumentParser(description='PFGE training')

parser.add_argument('--dir', type=str, default=None, metavar='DIR',
                    help='training directory (default: /tmp/pfge)')

parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                    help='dataset name (default: CIFAR10)')
parser.add_argument('--use_test', action='store_true', default=True,
                    help='switches between validation and test set (default: validation)')
parser.add_argument('--data_path', type=str, default=None, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size (default: 128)')
parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                    help='number of workers (default: 4)')
parser.add_argument("--split_classes", type=int, default=None)
parser.add_argument('--model', type=str, default='VGG16', metavar='MODEL',
                    help='model name (default: None)')

parser.add_argument('--ckpt', type=str, default=None, metavar='CKPT',
                    help='checkpoint to eval (default: None)')

parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--cycle', type=int, default=4, metavar='N',
                    help='number of epochs to train (default: 4)')
parser.add_argument('--P', type=int, default=10, help='model recording period (default: 10)')
parser.add_argument('--lr_max', type=float, default=0.05, metavar='LR1',
                    help='initial learning rate (default: 0.05)')
parser.add_argument('--lr_min', type=float, default=0.0005, metavar='LR2',
                    help='initial learning rate (default: 0.0001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=5e-4, metavar='WD',
                    help='weight decay (default: 1e-4)')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

args = parser.parse_args()

use_cuda = torch.cuda.is_available()

if use_cuda:
    args.device = torch.device("cuda")
else:
    args.device = torch.device("cpu")

assert args.cycle % 2 == 0, 'Cycle length should be even'

os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, 'fge.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print("Using model %s" % args.model)
model_cfg = getattr(models, args.model)

print("Loading dataset %s from %s" % (args.dataset, args.data_path))
loaders, num_classes = data.loaders(
        args.dataset,
        args.data_path,
        args.batch_size,
        args.num_workers,
        model_cfg.transform_train,
        model_cfg.transform_test,
        use_validation=not args.use_test,
        split_classes=args.split_classes,
    )

print("Preparing model")
print(*model_cfg.args)
model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
model.to(args.device)
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.wd)
num_model = args.epochs // args.P
model_list = []
model_list.append(model)
swa_n = np.zeros(int(num_model))
optimizer_list = []
optimizer_list.append(optimizer)
for i in range(int(num_model)):
    model1 = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
    model1.to(args.device)
    optimizers = torch.optim.SGD(model1.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.wd)
    model_list.append(model1)
    optimizer_list.append(optimizers)



criterion = utils.cross_entropy

checkpoint = torch.load(args.ckpt)
start_epoch = 0
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])

ensemble_size = 0
pfge_predictions_sum = np.zeros((len(loaders['test'].dataset), num_classes))

columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_loss', 'te_acc', 'pfge_ens_acc', 'time']

for epoch in range(args.epochs):
    time_ep = time.time()
    lr_schedule = utils.cyclic_learning_rate(epoch, args.cycle, args.lr_max, args.lr_min)
    i = epoch // args.P
    if i == 0:
        train_res = utils.train_epochs(loaders['train'], model, criterion, optimizer, lr_schedule=lr_schedule,
                                       cuda=use_cuda)
        test_res = utils.eval(loaders['test'], model, criterion, cuda=use_cuda)
        time_ep = time.time() - time_ep
        pfge_ens_acc = None
        if (epoch % args.cycle + 1) == args.cycle // 2:
            utils.moving_average(model_list[i+1], model, 1.0 / (swa_n[i] + 1))
            swa_n[i] += 1
        if (epoch + 1) % args.P == 0:
            ensemble_size += 1
            utils.bn_update(loaders["train"], model_list[i+1])
            pfge_res = utils.predict(loaders["test"], model_list[i+1])
            pfge_predictions = pfge_res["predictions"]
            targets = pfge_res["targets"]
            pfge_predictions_sum += pfge_predictions
            pfge_ens_acc = 100.0 * np.mean(np.argmax(pfge_predictions_sum, axis=1) == targets)
            utils.save_checkpoint(args.dir, epoch + 1, name="pfge", state_dict=model_list[i+1].state_dict())
    else:
        train_res = utils.train_epochs(loaders['train'], model_list[i], criterion, optimizer_list[i], lr_schedule=lr_schedule, cuda=use_cuda)
        test_res = utils.eval(loaders['test'], model_list[i], criterion, cuda=use_cuda)
        time_ep = time.time() - time_ep
        if (epoch % args.cycle + 1) == args.cycle // 2:
            utils.moving_average(model_list[i+1], model_list[i], 1.0/(swa_n[i] + 1))
            swa_n[i] += 1
        if (epoch + 1) % args.P == 0:
            ensemble_size += 1
            utils.bn_update(loaders["train"], model_list[i + 1])
            pfge_res = utils.predict(loaders["test"], model_list[i + 1])
            pfge_predictions = pfge_res["predictions"]
            targets = pfge_res["targets"]
            pfge_predictions_sum += pfge_predictions
            pfge_ens_acc = 100.0 * np.mean(np.argmax(pfge_predictions_sum, axis=1) == targets)
            utils.save_checkpoint(args.dir, epoch + 1, name="pfge", state_dict=model_list[i + 1].state_dict())

    values = [epoch + 1, lr_schedule(1.0), train_res['loss'], train_res['accuracy'], test_res['loss'],
              test_res['accuracy'], pfge_ens_acc, time_ep]
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='9.4f')
    if epoch % 40 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)
