import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model_search import NetworkCIFAR as Network
import dill
import pickle

from spectralnorm.spectral_norm_utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--train_portion', type=float, default=0.8, help='portion of training data')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--parallel', action='store_true', default=False, help='data parallelism')
parser.add_argument('--resume', action='store_true', default=False, help='load checkpoint and restart training')
parser.add_argument('--singular_values', action='store_true', default=False, help='check singular values')
parser.add_argument('--resume_dir', default=None, type=str, help='directory for resume')
parser.add_argument('--separate_trainset', default=True, action="store_false")
args = parser.parse_args()

if not args.resume and not args.singular_values:
  args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
  utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

if args.resume:
  args.save = args.resume_dir
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


CIFAR_CLASSES = 10


def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled = True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
  if args.parallel:
    model = nn.DataParallel(model).cuda()
  else:
    model = model.cuda()

  if args.singular_values:
    with open(os.path.join(args.resume_dir, 'network.pickle'), 'rb') as f:
      model = pickle.load(f)
    model.cuda()
    conv_list = [(n,m) for n,m in model.named_modules() if isinstance(m, nn.Conv2d) and '_ops' in n]
    for n, m in conv_list:
      fn = list(m._forward_pre_hooks.values())[0]
      shape = fn.input_dim
      sv_list = calculate_stable_rank(m, shape[2:], full_list=True)
      print('name:{}, # of sv:{}, max sv:{:.4f}, mean sv{:.4f}'.format(n, len(sv_list), sv_list[0], sv_list.mean()))
    return

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
    model.parameters(),
    args.learning_rate,
    momentum=args.momentum,
    weight_decay=args.weight_decay)

  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
  test_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))

  if args.separate_trainset:
    train_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=2)
    valid_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=2)
  else:
    train_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
      shuffle=True, pin_memory=True, num_workers=2)
    valid_queue = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
      shuffle=False, pin_memory=True, num_workers=2)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  start_epoch = 0
  if args.resume:
    with open(os.path.join(args.resume_dir, 'network.pickle'), 'rb') as f:
      model = pickle.load(f)
    model.cuda()
    checkpoint = torch.load(os.path.join(args.resume_dir, 'checkpoint.pth.tar'))
    start_epoch = checkpoint['epoch']
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print('model was loaded. Start training from epoch {}.'.format(start_epoch))

  if not args.resume:
    import_str = "from collections import namedtuple\nGenotype_normal = namedtuple('Genotype_normal','normal normal_concat')\nGenotype_reduce = namedtuple('Genotype_reduce','reduce reduce_concat')\n"
    import_str_mean = "from collections import namedtuple\nGenotype = namedtuple('Genotype','normal normal_concat reduce reduce_concat')\n"
    with open(os.path.join(args.save, 'arch_min.py'), 'w') as f:
      f.write(import_str)
    with open(os.path.join(args.save, 'arch_max.py'), 'w') as f:
      f.write(import_str)
    with open(os.path.join(args.save, 'arch_mean_min.py'), 'w') as f:
      f.write(import_str_mean)

  for epoch in range(start_epoch,args.epochs):
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)
    write_arch(model, epoch)

    # training
    train_acc, train_obj = train(train_queue, model, criterion, optimizer, lr)
    logging.info('train_acc %f', train_acc)
    with open(os.path.join(args.save, "train_acc.txt"), mode='a') as f:
      f.write(str(train_acc)+'\n')
    with open(os.path.join(args.save, "train_loss.txt"), mode='a') as f:
      f.write(str(train_obj)+'\n')
    scheduler.step()

    # validation
    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    with open(os.path.join(args.save, "test_acc.txt"), mode='a') as f:
      f.write(str(valid_acc)+'\n')
    with open(os.path.join(args.save, "test_loss.txt"), mode='a') as f:
      f.write(str(valid_obj)+'\n')

    logging.info('valid_acc %f', valid_acc)

    utils.save(model, os.path.join(args.save, 'weights.pt'))

    if args.parallel:
      utils.save_checkpoint({'epoch':epoch+1,
                             'model_state_dict':model.module.state_dict(),
                             'optimizer_state_dict':optimizer.state_dict(),
                             'scheduler_state_dict':scheduler.state_dict()},
                            False, save=args.save)
    else:
      utils.save_checkpoint({'epoch':epoch+1,
                             'model_state_dict':model.state_dict(),
                             'optimizer_state_dict':optimizer.state_dict(),
                             'scheduler_state_dict':scheduler.state_dict()},
                            False, save=args.save)

    with open(os.path.join(args.save,'network.pickle'),'wb') as f:
      pickle.dump(model, f)
  write_arch(model, args.epochs)

def train(train_queue, model, criterion, optimizer, lr):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  # adjust weight of each operation in mixed edge before epoch begins
  model.update_stable_ranks()
  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)

    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda(non_blocking=True)

    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target)

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  with torch.no_grad():
    for step, (input, target) in enumerate(valid_queue):
      input = Variable(input).cuda()
      target = Variable(target).cuda(non_blocking=True)

      logits = model(input)
      loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      objs.update(loss.item(), n)
      top1.update(prec1.item(), n)
      top5.update(prec5.item(), n)

      if step % args.report_freq == 0:
        logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg

def write_arch(model, epoch):
  if args.parallel:
    normal_list_min, reduce_list_min, genotype_mean = model.module.genotype(sr_max=False, print_srs=True)
    #normal_list_max, reduce_list_max, genotype_mean = model.module.genotype(sr_max=True, print_srs=False)
  else:
    normal_list_min, reduce_list_min, genotype_mean = model.genotype(sr_max=False, print_srs=True)
    #normal_list_max, reduce_list_max, genotype_mean = model.genotype(sr_max=True, print_srs=False)

  print(genotype_mean)
  with open(os.path.join(args.save, "arch_min.py"), 'a') as f:
    f.write("epoch_{} = {{ \n".format(epoch))
    f.write("\t\"normal\" : [\n")
    for i in range(len(normal_list_min)):
      f.write("\t\t"+str(normal_list_min[i])+", \n")
    f.write("\t], \n")
    f.write("\t\"reduce\" : [\n")
    for i in range(len(reduce_list_min)):
      f.write("\t\t"+str(reduce_list_min[i])+", \n")
    f.write("\t]} \n")

  """
  with open(os.path.join(args.save, "arch_max.py"), 'a') as f:
    f.write("epoch_{} = {{ \n".format(epoch))
    f.write("\t\"normal\" : [\n")
    for i in range(len(normal_list_max)):
      f.write("\t\t"+str(normal_list_max[i])+", \n")
    f.write("\t], \n")
    f.write("\t\"reduce\" : [\n")
    for i in range(len(reduce_list_max)):
      f.write("\t\t"+str(reduce_list_max[i])+", \n")
    f.write("\t]} \n")
  """

  with open(os.path.join(args.save, "arch_mean_min.py"), 'a') as f:
    f.write("epoch_{} =  ".format(epoch))
    f.write(str(genotype_mean))
    f.write("\n")


if __name__ == '__main__':
    main()
