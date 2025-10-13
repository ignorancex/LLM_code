import pdb,time
import torch,torchvision,os,time,pickle
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models             
from mqbench.prepare_by_platform import prepare_by_platform   # add quant nodes for specific Backend
from mqbench.prepare_by_platform import BackendType           # contain various Backend, like TensorRT, NNIE, etc.
from mqbench.utils.state import enable_calibration            # turn on calibration algorithm, determine scale, zero_point, etc.
from mqbench.utils.state import enable_quantization           # turn on actually quantization, like FP32 -> INT8
from mqbench.utils.state import disable_all           
from copy import deepcopy
from mqbench.advanced_ptq import ptq_reconstruction
from functools import partial
import torchvision as tv
import torch.nn.functional as F
import argparse
import pyhessian

class arg():
    def __init__(self):
        self.bs = 64
        self.nthreads = 8
        self.cuda = 0
        self.start_batch = 0 
        self.end_batch = 719
args = arg() 

torch.manual_seed(0)
np.random.seed(0)

class I1K():
    def __init__(
            self,
            data_dir='~/data/imagenet',
            cuda=False,
            num_workers=8,
            train_batch_size=64,
            test_batch_size=500,
            shuffle=False
        ):
        gpu_conf = {
            'num_workers': num_workers,
            'pin_memory': True,
            'persistent_workers': True
        } if cuda else {}
        normalize = tv.transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225),
        )
        self.train = torch.utils.data.DataLoader(
            tv.datasets.ImageFolder(
                data_dir + '/train',
                tv.transforms.Compose([
                    tv.transforms.Resize(256),
                    tv.transforms.CenterCrop(224),
                    tv.transforms.ToTensor(),
                    normalize,
                ])),
            batch_size=train_batch_size,
            shuffle=shuffle,
            **gpu_conf)
        self.val = torch.utils.data.DataLoader(
            tv.datasets.ImageFolder(
                data_dir + '/val',
                tv.transforms.Compose([
                    tv.transforms.Resize(256),
                    tv.transforms.CenterCrop(224),
                    tv.transforms.ToTensor(),
                    normalize,
                ])),
            batch_size=test_batch_size,
            shuffle=False,
            **gpu_conf)


modelname = 'resnet50'
adv_ptq = False
dataset = 'I1K'
mn = dataset.lower()+ '_' + modelname
#ds = I1K(data_dir=os.path.join('/tools/d-matrix/ml/data', "imagenet"),
#         train_batch_size=args.bs,test_batch_size=args.bs,cuda=True,shuffle=True)
#ds = I1K(data_dir=os.path.join('/home/usr1/zd2922/data', "imagenet"),
#         train_batch_size=args.bs,test_batch_size=args.bs,cuda=True,shuffle=True)

ds = I1K(data_dir=os.path.join('/work2/07149/zdeng/frontera/data', "imagenet"),
          train_batch_size=args.bs,test_batch_size=args.bs,cuda=True,shuffle=True)
model = eval("tv.models." + modelname)(pretrained=True).cuda(args.cuda)
ds.train.num_workers = args.nthreads
ds.val.num_workers = args.nthreads

train = ds.train
test = ds.val

train.num_workers = args.nthreads
test.num_workers = args.nthreads
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(val_loader, model,
             criterion = torch.nn.CrossEntropyLoss().cuda(args.cuda),device=f'cuda:{args.cuda}'):
    s_time = time.time()
    # switch to evaluate mode
    model.eval()
    count,top1,top5,losses = 0,0,0,0
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images, target = images.to(device), target.to(device)
            # compute output
            output = model(images)
            loss = criterion(output, target)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses = losses * count/(count+images.size(0)) + loss * images.size(0)/(count+images.size(0))
            top1 = top1 * count/(count+images.size(0)) + acc1 * images.size(0)/(count+images.size(0))
            top5 = top5 * count/(count+images.size(0)) + acc5 * images.size(0)/(count+images.size(0))
            count += images.size(0)
    test_time = time.time() - s_time
    
    return {'top1':top1,'top5':top5,'loss':losses,'time':test_time}

test.num_workers = 12
train.num_workers = 12
# calibration data used to calibrate PTQ and MPQ
calib_data = []
stacked_tensor = []
calib_fp_output = []
i = 0
with torch.no_grad():
    for img,label in train:
        
        # stacked_tensor is to calibrate the model
        # calib_data is the data to calculate ltilde
        if i< 16: # always use 1024 samples to calibrate
            stacked_tensor.append(img)
            
        if i>= args.start_batch and i<= args.end_batch:
            calib_data.append((img,label))
            #calib_fp_output.append(model(img.cuda(args.cuda)))
        if i>= 16 and i>= args.end_batch:
            break
        
        i += 1
from pyhessian import hessian # Hessian computation
def getModuleByName(model,moduleName):
    '''
        replace module with name modelName.moduleName with newModule
    '''
    tokens = moduleName.split('.')
    m = model
    for tok in tokens:
        m = getattr(m,tok)
    return m
criterion = torch.nn.CrossEntropyLoss()
def estimate_trace(eval_data,layer_name):
    torch.manual_seed(0)
    np.random.seed(0)
    print(f'compute trace for layer {layer_name}')
    # first disable gradients of all modules
    for n,m in model.named_parameters():
        m.requires_grad = False
    # enable gradients for only that layer
    getModuleByName(model,layer_name).weight.requires_grad = True
    for m in model.parameters():
        if m.requires_grad:
            print('find m',m.shape)
    hessian_comp = hessian(model, criterion, data=eval_data, cuda=True)
    print('start trace estimation')
    s_time = time.time()
    res = hessian_comp.trace()
    print(f'{len(res)} samples:{time.time()-s_time:.2f} seconds')
    return res

layers_to_check = []
for n,m in model.named_modules():
    if isinstance(m,(torch.nn.Conv2d,torch.nn.Linear)):
        layers_to_check.append(n)
if not os.path.exists('./HAWQ_DELTAL'):
    os.mkdir('HAWQ_DELTAL')
for i in range(len(calib_data)):
    print(f'batch {i+args.start_batch}')
    res = {}
    for n in layers_to_check:
        res[n] = estimate_trace(eval_data=calib_data[i],layer_name=n)
    with open('HAWQ_DELTAL/TraceEST_batch'+str(args.start_batch+i)+'.pkl','wb') as f:
        pickle.dump(res,f)
#pdb.set_trace()
