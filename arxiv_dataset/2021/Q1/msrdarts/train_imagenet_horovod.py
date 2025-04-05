import os
import sys
import numpy as np
import time
import torch
import utils
import glob
import random
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

import horovod.torch as hvd

from torch.autograd import Variable
from model import NetworkImageNet as Network

parser = argparse.ArgumentParser("imagenet")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=290, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=48, help='num of init channels')
parser.add_argument('--layers', type=int, default=14, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='MSRDARTS_layer14_imagenet_searched_cifar', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
parser.add_argument('--parallel', action='store_true', default=False, help='data parallelism')
parser.add_argument('--warmup_epochs', type=int, default=10, help='number of warmup epochs')
parser.add_argument('--resume', default=False, action='store_true')
parser.add_argument('--resume_dir', default=None, type=str)
parser.add_argument('--num_workers', default=12, type=int)
parser.add_argument('--scheduler', default="cosine", type=str, choices=['step', "cosine"])
args = parser.parse_args()

# ***************** horovod *******************
hvd.init()
torch.cuda.set_device(hvd.local_rank())
torch.cuda.manual_seed(args.seed)
#hvd.broadcast()
# ***************** horovod *******************

if hvd.rank() == 0:
  if not args.resume:
    args.save = 'eval-imagenet-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
  else:
    args.save = args.resume_dir

  log_format = '%(asctime)s %(message)s'
  logging.basicConfig(stream=sys.stdout, level=logging.INFO,
      format=log_format, datefmt='%m/%d %I:%M:%S %p')
  fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
  fh.setFormatter(logging.Formatter(log_format))
  logging.getLogger().addHandler(fh)

CLASSES = 1000
device = torch.device('cuda')

class CrossEntropyLabelSmooth(nn.Module):

  def __init__(self, num_classes, epsilon):
    super(CrossEntropyLabelSmooth, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.logsoftmax = nn.LogSoftmax(dim=1)

  def forward(self, inputs, targets):
    log_probs = self.logsoftmax(inputs)
    targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
    loss = (-targets * log_probs).mean(0).sum()
    return loss


def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  print(torch.cuda.device_count())
  np.random.seed(args.seed)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)

  if hvd.rank() == 0:
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

  best_acc_top1 = 0
  start_epoch = 0
  if args.resume:
    checkpoint = torch.load(os.path.join(args.save, 'checkpoint.pth.tar'))
    best_checkpoint = torch.load(os.path.join(args.save, 'model_best.pth.tar'))
    start_epoch = checkpoint['epoch']
    best_acc_top1 = best_checkpoint['best_acc_top1']
    start_epoch = hvd.broadcast(torch.tensor(start_epoch), root_rank=0, name='start_epoch').item()
    best_acc_top1 = hvd.broadcast(torch.tensor(best_acc_top1), root_rank=0, name='best_acc_top1').item()

  genotype = eval("genotypes.%s" % args.arch)
  model = Network(args.init_channels, CLASSES, args.layers, args.auxiliary, genotype)

  if args.parallel:
    model = nn.DataParallel(model).cuda()
  else:
    model = model.cuda()

  if hvd.rank() == 0:
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
  criterion_smooth = criterion_smooth.cuda()

  optimizer = torch.optim.SGD(
    model.parameters(),
    args.learning_rate * hvd.size(),
    momentum=args.momentum,
    weight_decay=args.weight_decay
    )

  # ***************** horovod *******************
  optimizer = hvd.DistributedOptimizer(
      optimizer, named_parameters=model.named_parameters())
  # ***************** horovod *******************

  traindir = os.path.join(args.data, 'train')
  validdir = os.path.join(args.data, 'val')
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  train_data = dset.ImageFolder(
    traindir,
    transforms.Compose([
      transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.2),
      transforms.ToTensor(),
      normalize,
    ]))
  valid_data = dset.ImageFolder(
    validdir,
    transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      normalize,
    ]))

  train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_data, num_replicas=hvd.size(), rank=hvd.rank())
  train_queue = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers,sampler=train_sampler)

  valid_queue = torch.utils.data.DataLoader(
    valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)

  if start_epoch > 0 and hvd.rank() == 0:
    checkpoint = torch.load(os.path.join(args.save, 'checkpoint.pth.tar'))
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("checkpoint {}, model, optimizer was loaded".format(start_epoch))

  hvd.broadcast_parameters(model.state_dict(), root_rank=0)
  hvd.broadcast_optimizer_state(optimizer, root_rank=0)

  if not args.resume:
    set_lr(0, 0, len(train_queue), optimizer,args.scheduler)

  for epoch in range(start_epoch, args.epochs+args.warmup_epochs):
    if hvd.rank() == 0:
      lr =  optimizer.param_groups[0]['lr']
      logging.info('epoch %d lr %e', epoch, lr)
      with open(os.path.join(args.save, 'learning_rate.txt'), mode='a') as f:
        f.write(str(lr)+'\n')

    if args.parallel:
      model.module.drop_path_prob = args.drop_path_prob * epoch / args.epochs
    else:
      model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    train_acc, train_obj = train(train_queue, train_sampler, model, criterion_smooth, optimizer, epoch)
    if hvd.rank() == 0:
      logging.info('train_acc %f', train_acc)
      with open(os.path.join(args.save, "train_acc.txt"), mode='a') as f:
        f.write(str(train_acc)+'\n')
      with open(os.path.join(args.save, "train_loss.txt"), mode='a') as f:
        f.write(str(train_obj)+'\n')

    valid_acc_top1, valid_acc_top5, valid_obj = infer(valid_queue, model, criterion)
    if hvd.rank() == 0:
      logging.info('valid_acc_top1 %f', valid_acc_top1)
      logging.info('valid_acc_top5 %f', valid_acc_top5)
      with open(os.path.join(args.save, "test_acc_1.txt"), mode='a') as f:
        f.write(str(valid_acc_top1)+'\n')
      with open(os.path.join(args.save, "test_acc_5.txt"), mode='a') as f:
        f.write(str(valid_acc_top5)+'\n')
      with open(os.path.join(args.save, "test_loss.txt"), mode='a') as f:
        f.write(str(valid_obj)+'\n')

    is_best = False
    if valid_acc_top1 > best_acc_top1:
      best_acc_top1 = valid_acc_top1
      is_best = True

    if hvd.rank() == 0:
      utils.save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_acc_top1': best_acc_top1,
        'optimizer' : optimizer.state_dict(),
        }, is_best, args.save)

def metric_ave(value):
  return hvd.allreduce(torch.tensor(value)).item()

def metric_sum(value):
  return hvd.allreduce(torch.tensor(value), op=hvd.Sum).item()

def train(train_queue, train_sampler, model, criterion, optimizer, epoch):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.train()
  train_sampler.set_epoch(epoch)

  for step, (input, target) in enumerate(train_queue):
    set_lr(epoch, step, len(train_queue), optimizer, args.scheduler)
    target = target.cuda()
    input = input.cuda()

    optimizer.zero_grad()
    logits, logits_aux = model(input)
    loss = criterion(logits, target)
    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight*loss_aux

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0 and hvd.rank() == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  count = metric_sum(top1.cnt)
  test_loss = metric_sum(objs.sum)
  top1_acc = metric_sum(top1.sum)
  top5_acc = metric_sum(top5.sum)
  #return top1.avg, objs.avg, top1.sum
  return top1_acc/count, test_loss/count

def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  with torch.no_grad():
    for step, (input, target) in enumerate(valid_queue):
      input = Variable(input).cuda()
      target = Variable(target).cuda(non_blocking=True)

      logits, _ = model(input)
      loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      objs.update(loss.item(), n)
      top1.update(prec1.item(), n)
      top5.update(prec5.item(), n)

      if step % args.report_freq == 0 and hvd.rank() == 0:
        logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, top5.avg, objs.avg

def set_lr(epoch, step, trainset_len, optimizer, scheduler_type):
  if epoch < args.warmup_epochs:
    epoch += float(step+1) / trainset_len
    adj = 1. / hvd.size() * (epoch *(hvd.size() -1) / args.warmup_epochs + 1)
  else:
    if scheduler_type == "step":
      adj = 1. * args.gamma**((epoch-args.warmup_epochs+1)//args.decay_period)
    elif scheduler_type == "cosine":
      adj = 1. * (1 + np.cos(np.pi * (epoch-args.warmup_epochs+1) / (args.epochs-args.warmup_epochs))) / 2
  for param_group in optimizer.param_groups:
    param_group['lr'] = args.learning_rate * hvd.size() * adj


if __name__ == '__main__':
  main()
