import os
import numpy as np
import torch
import shutil
import glob
import torchvision.transforms as transforms
from torch.autograd import Variable


class AvgrageMeter(object):
  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0 / batch_size))
  return res

class Cutout(object):
  def __init__(self, length):
    self.length = length

  def __call__(self, img):
    h, w = img.size(1), img.size(2)
    mask = np.ones((h, w), np.float32)
    y = np.random.randint(h)
    x = np.random.randint(w)

    y1 = np.clip(y - self.length // 2, 0, h)
    y2 = np.clip(y + self.length // 2, 0, h)
    x1 = np.clip(x - self.length // 2, 0, w)
    x2 = np.clip(x + self.length // 2, 0, w)

    mask[y1: y2, x1: x2] = 0.
    mask = torch.from_numpy(mask)
    mask = mask.expand_as(img)
    img *= mask
    return img


def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  return train_transform, valid_transform


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1. - drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

def create_imagenet_subset(args, original_dir='/work/machida/dataset/ImageNet/ILSVRC2012_imagenet/train', target_dir='./split_data', train_portion=0.1, valid_portion=0.025):
  if os.path.isfile(os.path.join(args.save, "split_train_subset")):
    print('load subset information from {}'.format(os.path.join(args.save, split_subset)))
  else:
    if not os.path.exists(target_dir):
      os.makedirs(target_dir)
    dirs = glob.glob(os.path.join(original_dir, '*'))
    target_train_dir = os.path.join(target_dir, 'train')
    target_valid_dir = os.path.join(target_dir, 'val')
    for dir in dirs:
      class_name = dir.split('/')[-1]
      target_train_woking_dir = os.path.join(target_train_dir, class_name)
      target_valid_woking_dir = os.path.join(target_valid_dir, class_name)
      image_num = len(os.listdir(dir))
      train_num = int(image_num*train_portion)
      valid_num = int(image_num*valid_portion)
      perm_list = np.random.permutation(np.array([i for i in range(image_num)]))
      train_list_idx = perm_list[:train_num]
      valid_list_idx = perm_list[train_num:train_num+valid_num]
      image_list = np.array(sorted(glob.glob(os.path.join(dir,'*'))))

      if not os.path.exists(target_train_woking_dir):
        os.makedirs(target_train_woking_dir)
      if not os.path.exists(target_valid_woking_dir):
        os.makedirs(target_valid_woking_dir)

      print(target_train_woking_dir)
      for file in image_list[train_list_idx]:
        shutil.copy(file, target_train_woking_dir)
      for file in image_list[valid_list_idx]:
        shutil.copy(file, target_valid_woking_dir)

      with open(os.path.join(args.save, 'split_train_subset'), 'a') as f:
        f.write(str(target_train_woking_dir)+':')
        for i in train_list_idx:
          f.write(str(i)+' ')
        f.write('\n')
      with open(os.path.join(args.save, 'split_valid_subset'), 'a') as f:
        f.write(str(target_train_woking_dir)+':')
        for i in valid_list_idx:
          f.write(str(i)+' ')
        f.write('\n')

    shutil.copy(os.path.join(args.save, 'split_train_subset'), target_dir)
    shutil.copy(os.path.join(args.save, 'split_valid_subset'), target_dir)
