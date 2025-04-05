import os
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import pandas as pd
import os
import time
import random
import torch.nn.functional as F
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import tarfile
import zipfile
plt.switch_backend('agg')

USE_GPU = True
# USE_GPU = False
import collections.abc
container_abcs = collections.abc
from itertools import repeat

def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse
    
_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)

if USE_GPU and torch.cuda.is_available():
    def to_var(x, requires_grad=False, gpu=None):
        x = x.cuda(gpu)
        return x.requires_grad_(requires_grad)
else:
    def to_var(x, requires_grad=False, gpu=None):
        return x.requires_grad_(requires_grad)

if USE_GPU and torch.cuda.is_available():
    def to_device(x, gpu=None):
        x = x.cuda(gpu)
        return x
else:
    def to_device(x, gpu=None):
        return x

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class MultiAverageMeter(object):
    def __init__(self):
        self.meters = []
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0        

    def update(self,val,n=1):
        if len(self.meters) < len(val):
            for i in range(len(val)-len(self.meters)):
                self.meters.append(AverageMeter())
        for i, meter in enumerate(self.meters):
            meter.update(val[i],n)
        self.val = [meter.val for meter in self.meters]
        self.avg = [meter.avg for meter in self.meters]
        self.sum = [meter.sum for meter in self.meters]
        self.count = [meter.count for meter in self.meters]



def log_results(save, epoch, log_dict, writer):
    with open(os.path.join(save, 'logs.csv'), 'a') as f:
        f.write('%03d,'%((epoch + 1),))
        for value in log_dict.values():
            f.write('%0.6f,' % (value,))
        f.write('\n')
    for key, value in log_dict.items():
        writer.add_scalar(key, value, epoch)


def one_hot_to_categorical(x, dim):
    return x.argmax(dim=dim)

def categorical_to_one_hot(x, dim=1, expand_dim=False, n_classes=None):
    '''Sequence and label.
    when dim = -1:
    b x 1 => b x n_classes
    when dim = 1:
    b x 1 x h x w => b x n_classes x h x w'''
    # assert (x - x.long().to(x.dtype)).max().item() < 1e-6
    if type(x)==np.ndarray:
        x = torch.Tensor(x)
    assert torch.allclose(x, x.long().to(x.dtype))
    x = x.long()
    if n_classes is None:
        n_classes = int(torch.max(x)) + 1
    if expand_dim:
        x = x.unsqueeze(dim)
    else:
        assert x.shape[dim] == 1
    shape = list(x.shape)
    shape[dim] = n_classes
    x_one_hot = torch.zeros(shape).to(x.device).scatter_(dim=dim, index=x, value=1.)
    return x_one_hot.long()  




def plot_multi_voxels(*multi_voxels):
    multi_voxels = [np.array(voxels.cpu()) if isinstance(voxels, torch.Tensor) else np.array(voxels) for voxels in multi_voxels]
    multi_voxels = [np.expand_dims(voxels, 0) if voxels.ndim==3 else voxels for voxels in multi_voxels]

    rows = len(multi_voxels[0])
    columns = len(multi_voxels)
    fig = plt.figure(figsize=[10*columns,8*rows])
    for row in range(rows):
        for column in range(columns):
            if row<len(multi_voxels[column]):
                ax = fig.add_subplot(rows,columns,row*columns+column+1, projection='3d')
                ax.voxels(multi_voxels[column][row], edgecolor='k')

def plot_multi_shapes(*multi_shapes):
    multi_shapes = [np.array(shapes.cpu()) if isinstance(shapes, torch.Tensor) else np.array(shapes) for shapes in multi_shapes]
    multi_shapes = [np.expand_dims(shapes, 0) if shapes.ndim==2 else shapes for shapes in multi_shapes]

    rows = len(multi_shapes[0])
    columns = len(multi_shapes)
    fig = plt.figure(figsize=[10*columns,8*rows])
    for row in range(rows):
        for column in range(columns):
            if row<len(multi_shapes[column]):
                ax = fig.add_subplot(rows,columns,row*columns+column+1)
                ax.imshow(multi_shapes[column][row])

def save_model(model, save, valid_error, best_error, save_all):
    if save_all or valid_error < best_error:
        torch.save(model.state_dict(), os.path.join(save, 'model.dat'))
    if valid_error < best_error:
        best_error = valid_error
        print('New best error: %.4f' % best_error)

    return best_error


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def initialize(modules):
    for m in modules:
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in')
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in')
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in')
            m.bias.data.zero_()

import sys, time
class Logger(object):
    def __init__(self, filename='terminal log.txt', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')
        self.log.write(''.join([time.strftime("%y-%m-%d %H:%M:%S",  time.localtime(time.time())), '\n\n']))

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

    def __del__(self):
        self.log.write(''.join(['\n', time.strftime("%y-%m-%d %H:%M:%S",  time.localtime(time.time()))]))
        self.log.close()

def redirect_stdout(save_path):
    sys.stdout = Logger(os.path.join(save_path, 'stdout.txt'), sys.stdout)
    sys.stderr = Logger(os.path.join(save_path, 'stderr.txt'), sys.stderr)		# redirect std err, if necessary

def copy_file_backup(save):
    import shutil, sys, getpass, socket
    backup_dir = os.path.join(save, 'backup_code')
    os.makedirs(backup_dir)
    with open(os.path.join(backup_dir, 'CLI argument.txt'), 'w') as f:
        res = ''.join(['hostName: ', socket.gethostname(), '\n',
                    'account: ', getpass.getuser(), '\n',
                    'save_path: ', os.path.realpath(save), '\n', 
                    'CUDA_VISIBLE_DEVICES: ', str(os.environ.get('CUDA_VISIBLE_DEVICES')), '\n'])
        f.write(res)

        for i, _ in enumerate(sys.argv):
            f.write(sys.argv[i] + '\n')
        
    script_file = sys.argv[0]
    shutil.copy(script_file, backup_dir)
    shutil.copytree(os.path.join(sys.path[0], '../', 'mylib'), os.path.join(backup_dir, 'mylib'))
    shutil.copytree(os.path.join(sys.path[0], '../../', 'acsconv'), os.path.join(backup_dir, 'acsconv'))
    os.makedirs(os.path.join(backup_dir, 'current_experiment'))
    for file_path in os.listdir(sys.path[0]):
        if file_path not in ['tmp', 'data', '__pycache__']:
            shutil.copy(os.path.join(sys.path[0], file_path), os.path.join(backup_dir, 'current_experiment'))


from .sync_batchnorm import SynchronizedBatchNorm3d, SynchronizedBatchNorm2d
def model_to_syncbn(model):
    preserve_state_dict = model.state_dict()
    _convert_module_from_bn_to_syncbn(model)
    model.load_state_dict(preserve_state_dict)
    return model
def _convert_module_from_bn_to_syncbn(module):
    for child_name, child in module.named_children(): 
        if hasattr(nn, child.__class__.__name__) and \
            'batchnorm' in child.__class__.__name__.lower():
            TargetClass = globals()['Synchronized'+child.__class__.__name__]
            arguments = TargetClass.__init__.__code__.co_varnames[1:]
            kwargs = {k: getattr(child, k) for k in arguments}
            setattr(module, child_name, TargetClass(**kwargs))
        else:
            _convert_module_from_bn_to_syncbn(child)