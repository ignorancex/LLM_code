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

parser = argparse.ArgumentParser()
parser.add_argument("--bs", help="batchsize",type=int,default=64)
parser.add_argument("--nthreads", help="data loader threads",type=int,default=16) # v100: 12 t4: 8
parser.add_argument("--cuda", help="which cuda device to run",type=int,default=0) # v100: 12 t4: 8
parser.add_argument("--start-batch",help="starting batch index to estimate sensitivity",type=int)
parser.add_argument("--end-batch",help="ending batch index to estimate sensitivity",type=int)
parser.add_argument("--modelname",type=str,default='resnet34')
parser.add_argument("--datapath",type=str) # path to ImageNet data folder,example: '/home/usr1/zd2922/data/imagenet'
args = parser.parse_args()

modelname = args.modelname

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

ds = I1K(data_dir=args.datapth,
        train_batch_size=args.bs,test_batch_size=args.bs,cuda=True,shuffle=True)

ds.train.num_workers = args.nthreads
ds.val.num_workers = args.nthreads

train = ds.train
test = ds.val

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

# CALIBRATION AND SENSITIVITY SETS
calib_data = []
stacked_tensor = []
calib_fp_output = []
i = 0
with torch.no_grad():
    for img,label in train:

        # stacked_tensor is to calibrate the model (1024 samples)
        # calib_data is the sensitivity set (number of samples specified by user)
        if i< 1024/args.bs: # always use 1024 samples to calibrate
            stacked_tensor.append(img)

        if i>= args.start_batch and i<= args.end_batch:
            calib_data.append((img,label))
            #calib_fp_output.append(model(img.cuda(args.cuda)))
        if i>= 1024/args.bs and i>= args.end_batch:
            break

        i += 1


dataset = 'I1K'
mn = dataset.lower()+ '_' + modelname
model = eval("tv.models." + modelname)(pretrained=True).cuda(args.cuda)

MPQ_scheme = (2,4,8)
aw_scheme = [(8,2),(8,4),(8,8)]
model.eval()
# configuration

for b in MPQ_scheme:
    mqb_fp_model = deepcopy(model)

    # MSE calibration on model parameters
    backend = BackendType.Academic
    extra_config = {
        'extra_qconfig_dict': {
            'w_observer': 'MSEObserver',
            'a_observer': 'EMAMSEObserver',
            'w_fakequantize': 'FixedFakeQuantize',
            'a_fakequantize': 'FixedFakeQuantize',
            'w_qscheme': {
                'bit': b,
                'symmetry': True,
                'per_channel': False,
                'pot_scale': False,
            },
            'a_qscheme': {
                'bit': 8,
                'symmetry': False,
                'per_channel': False,
                'pot_scale': False,
            }
        }
    }
    print(f'Prepare {b}bits model using MQBench')

    exec(f'mqb_{b}bits_model=prepare_by_platform(mqb_fp_model, backend,extra_config).cuda(args.cuda)')

    # calibration loop
    enable_calibration(eval(f'mqb_{b}bits_model'))
    for img in stacked_tensor:
        eval(f'mqb_{b}bits_model')(img.cuda(args.cuda))

print(f'evaluate FP model')
res_fp = evaluate(test,model)
print(res_fp)

for b in MPQ_scheme:
    disable_all(eval(f'mqb_{b}bits_model'))
    # evaluation loop
    enable_quantization(eval(f'mqb_{b}bits_model'))
    eval(f'mqb_{b}bits_model').eval()
    print(f'evaluate mqb {b}bits model')
    res_quant = evaluate(test,eval(f'mqb_{b}bits_model'))
    print(f'{b} bits UPQ model:',res_quant)


mqb_fp_model = deepcopy(mqb_8bits_model)
disable_all(mqb_fp_model)
mqb_mix_model = deepcopy(mqb_fp_model)

def getModuleByName(model,moduleName):
    '''
        replace module with name modelName.moduleName with newModule
    '''
    tokens = moduleName.split('.')
    m = model
    for tok in tokens:
        m = getattr(m,tok)
    return m

def perturb(perturb_scheme):
    # perturb_scheme: {layer_name:(act_bits,weight_bits)}
    for layer_name in perturb_scheme:
        a_bits,w_bits = perturb_scheme[layer_name]

        if w_bits is not None:
            mix_module = getModuleByName(mqb_mix_model,layer_name)
            tar_module = getModuleByName(eval(f'mqb_{w_bits}bits_model'),layer_name)
            # replace weight quant to use a_bits quantization
            w_cmd = f'mix_module.weight_fake_quant=tar_module.weight_fake_quant'
            exec(w_cmd)

        if a_bits is not None:

            # replace act quant to use w_bits quantization
            a_cmd = f'mqb_mix_model.{layer_input_map[layer_name]}=mqb_{a_bits}bits_model.{layer_input_map[layer_name]}'
            exec(a_cmd)

# 1. record all modules we want to quantize
types_to_quant = (torch.nn.Conv2d,)

layer_input_map = {}

first,last = None,None
for node in mqb_8bits_model.graph.nodes:
    try:
        node_target = getModuleByName(mqb_mix_model,node.target)
        if isinstance(node_target,types_to_quant):
            node_args = node.args[0]
            print('input of ',node.target,' is ',node_args)
            layer_input_map[node.target] = str(node_args.target)
            if first is None:
                first = node.target
            last = node.target
    except:
        continue
print(f'first conv layer and last conv layer are',first,last)

aw_scheme = [(8,2),(8,4),(8,8)]

del layer_input_map[first] # usually the first conv layer and the last fc layers (already excluded) are kept 8 bits

layer_index = {}
cnt = 0
for layer in layer_input_map:
    for s in aw_scheme:
        layer_index[layer+f'{s}bits'] = cnt
        cnt += 1
L = cnt
index2layerscheme = [None for i in range(L)]
for name in layer_index:
    index = layer_index[name]
    layer_name = name[:-10]
    scheme = name[-10:]
    index2layerscheme[index] = (layer_name,scheme)

ref_layer_index = layer_index
ref_index2layerscheme = index2layerscheme

class layer_hook(object):

    def __init__(self):
        super(layer_hook, self).__init__()
        self.in_shape = None
        self.out_shape = None

    def hook(self, module, inp, outp):
        self.in_shape = inp[0].size()
        self.out_shape = outp.size()


hooks = {}

for layer in ref_layer_index:
    m = getModuleByName(model,layer[:-10])
    hook = layer_hook()
    hooks[layer[:-10]] = (hook,m.register_forward_hook(hook.hook))

with torch.no_grad():
    for img,label in calib_data:
        model(img.cuda(args.cuda))
        break
model(img.cuda(args.cuda))
def get_layer_bitops(layer_name,a_bits,w_bits):
    m = getModuleByName(model,layer_name)
    if isinstance(m,torch.nn.Conv2d):
        _,cin,_,_ = hooks[layer_name][0].in_shape
        _,cout,hout,wout = hooks[layer_name][0].out_shape
        n_muls = cin * m.weight.size()[2] * m.weight.size()[3] * cout * hout * wout
        n_accs = (cin * m.weight.size()[2] * m.weight.size()[3] - 1) * cout * hout * wout
        bitops_per_mul = 5*a_bits*w_bits - 5*a_bits-3*w_bits+3
        bitops_per_acc = 3*a_bits + 3*w_bits + 29
    return n_muls * bitops_per_mul + n_accs * bitops_per_acc

layer_size = np.array([0 for i in range(len(ref_layer_index))])
layer_bitops = np.array([0 for i in range(len(ref_layer_index))])


for l in ref_layer_index:
    index = ref_layer_index[l]
    layer_name, scheme = ref_index2layerscheme[index]
    a_bits,w_bits = eval(scheme[:-4])
    layer_size[index] = torch.numel(getModuleByName(model,layer_name).weight) * int(w_bits)
    layer_bitops[index] = get_layer_bitops(layer_name,a_bits,w_bits)

print('-'*20)
print(f"{len(layer_size)} layers (conv x number of bitchoices) in total")
print(f"8-bit model size: {layer_size.sum()*8/(8+4+2)/8/1024/1024:.4f} MB")

with open(f'layer_cost_{modelname}.pkl','wb') as f:
    pickle.dump({'layer_index':ref_layer_index,'layer_size':layer_size,'layer_bitops':layer_bitops},f)


def estimate_deltaL(eval_data,wbit_choices=[2,4,8]):

    tot_batches = len(eval_data)

    processed_batches = 0

    print(f'MPQCO Ltilde: {processed_batches}/{tot_batches} batch of data processed')

    for batch_img,batch_label in eval_data:

        s_time = time.time()
        # init deltaL dictionary
        deltaL = {}
        for layer in layer_input_map:
            if layer in ('conv1','fc'):
                continue
            deltaL[layer] = {}
            for wbit in wbit_choices:
                deltaL[layer][wbit] = 0

        for i in range(batch_img.size(0)):

            img,label = batch_img[i].unsqueeze(0),batch_label[i]
            model.zero_grad()
            logits = model(img.cuda(args.cuda))
            logits[0][label].backward()

            with torch.no_grad():
                for layer_name in layer_input_map:
                    if layer_name in ('conv1','fc'):
                        continue
                    for w_bits in wbit_choices:
                        tar_module = getModuleByName(eval(f'mqb_{w_bits}bits_model'),layer_name)
                        dw = tar_module.weight_fake_quant(tar_module.weight) - tar_module.weight
                        dl = (dw * getModuleByName(model,layer_name).weight.grad).sum()
                        dl /= logits[0][label]
                        dl = dl ** 2
                        deltaL[layer_name][w_bits] += dl.cpu().numpy()

        for layer in layer_input_map:
            if layer in ('conv1','fc'):
                continue
            for wbit in wbit_choices:
                deltaL[layer][wbit] /= 2 * batch_img.size(0)

        deltaL['n_samples'] = batch_img.size(0)

        if not os.path.exists(f'DELTAL_{modelname}'):
            os.mkdir(f'DELTAL_{modelname}')

        with open(f'DELTAL_{modelname}/MPQCO_DELTAL_{modelname}_batch{args.start_batch+processed_batches}(size{args.bs}).pkl','wb') as f:
            pickle.dump(deltaL,f)

        processed_batches += 1

        print(f'MPPCO Ltilde: {processed_batches+args.start_batch} {processed_batches}/{tot_batches} batch of data processed')
        print(f'batch cost:{time.time()-s_time:.2f} seconds')

kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
def kldiv(quant_logit,fp_logit):
    inp = F.log_softmax(quant_logit,dim=-1)
    tar = F.softmax(fp_logit,dim=-1)
    return kl_loss(inp,tar)

def perturb_loss(perturb_scheme,ref_metric,
                 eval_data=calib_data,printInfo=False,KL=False):

    global mqb_mix_model
    mqb_mix_model.eval()

    with torch.no_grad():
        # perturb layers
        perturb(perturb_scheme)

        # do evaluation
        if not KL:
            res = evaluate(eval_data,mqb_mix_model)
            perturbed_loss = res[ref_metric[0]] - ref_metric[1]
        else:
            perturbed_loss = []

            for (data,fp_out) in zip(calib_data,calib_fp_output):
                img,label = data
                quant_out = mqb_mix_model(img.cuda(args.cuda))
                perturbed_loss.append(kldiv(quant_out,fp_out))
            #print(perturbed_loss)
            perturbed_loss = torch.tensor(perturbed_loss).mean()

        if printInfo:
            print(f'use kl {KL} perturbed loss {perturbed_loss}')

        # recover layers
        mqb_mix_model = deepcopy(mqb_fp_model)

    return perturbed_loss

import time
import matplotlib.pyplot as plt
s_time = time.time()
cached = {}

# MPQCO sensitivities
estimate_deltaL(calib_data,wbit_choices=[2,4,8])

# CLADO sensitivities
KL=False
for clado_batch in range(0,len(calib_data)):
    print(f'clado batch {clado_batch+args.start_batch} processed {clado_batch+1}/{len(calib_data)}')
    ref_metric = ('loss',
                  evaluate([calib_data[clado_batch],],mqb_fp_model)['loss'])

    s_time = time.time()
    cached = {}
    for n in layer_input_map:
        for m in layer_input_map:
            for naw in aw_scheme:
                for maw in aw_scheme:
                    if (n,m,naw,maw) not in cached:
                        if n == m:
                            if naw == maw:
                                p = perturb_loss({n:naw},ref_metric,
                                                 [calib_data[clado_batch],],KL=KL)
                            else:
                                p = 0 # emprically negligible influence on results use 0 or the exact value
                        else:
                            p = perturb_loss({n:naw,m:maw},ref_metric,
                                             [calib_data[clado_batch],],KL=KL)
                        cached[(n,m,naw,maw)] = cached[(m,n,maw,naw)] = p

    print(f'{time.time()-s_time:.2f} seconds elapsed')

    # layer index and index2layerscheme map
    layer_index = {}
    cnt = 0
    for layer in layer_input_map:
        for s in aw_scheme:
            layer_index[layer+f'{s}bits'] = cnt
            cnt += 1
    L = cnt

    import numpy as np
    hm = np.zeros(shape=(L,L))
    for n in layer_input_map:
        for m in layer_input_map:
            for naw in aw_scheme:
                for maw in aw_scheme:
                    hm[layer_index[n+f'{naw}bits'],layer_index[m+f'{maw}bits']] = cached[(n,m,naw,maw)]

    index2layerscheme = [None for i in range(hm.shape[0])]

    for name in layer_index:
        index = layer_index[name]
        layer_name = name[:-10]
        scheme = name[-10:]

        index2layerscheme[index] = (layer_name,scheme)

    saveas = f'Ltilde_{modelname}/Ltilde_batch{args.start_batch+clado_batch}(size{args.bs})_'
    saveas += str(aw_scheme)
    saveas += mn
    saveas += '.pkl'

    if not os.path.exists(f'Ltilde_{modelname}'):
        os.mkdir(f'Ltilde_{modelname}')
    with open(saveas,'wb') as f:
        pickle.dump({'Ltilde':hm,'layer_index':layer_index,'index2layerscheme':index2layerscheme},f)
