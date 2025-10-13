# CLADO Gradient Cache code
import torch,torchvision,os,time,pickle
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from utils.util import get_loader,evaluate
from utils.layer import qConv2d,qLinear
from utils.train import QAVAT_train
import matplotlib.pyplot as plt
import torchvision.models as models             

from mqbench.prepare_by_platform import prepare_by_platform   # add quant nodes for specific Backend
from mqbench.prepare_by_platform import BackendType           # contain various Backend, like TensorRT, NNIE, etc.
from mqbench.utils.state import enable_calibration            # turn on calibration algorithm, determine scale, zero_point, etc.
from mqbench.utils.state import enable_quantization           # turn on actually quantization, like FP32 -> INT8
from mqbench.utils.state import disable_all           
from copy import deepcopy
from mqbench.advanced_ptq import ptq_reconstruction

# script parameter

from mltools.data import I1K
import torchvision as tv

torch.manual_seed(0)
np.random.seed(0)

MPQ_scheme = (2,4,8)
modelname = 'resnet50'

adv_ptq = False
    
dataset = 'I1K'
mn = dataset.lower()+ '_' + modelname
ds = I1K(data_dir=os.path.join('/tools/d-matrix/ml/data', "imagenet"),
         train_batch_size=64,test_batch_size=64,cuda=True)
model = eval("tv.models." + modelname)(pretrained=True).cuda()
ds.train.num_workers = 8
ds.val.num_workers = 8

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
             criterion = torch.nn.CrossEntropyLoss().cuda(),device='cuda'):

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

    return {'top1':top1,'top5':top5,'loss':losses}



print('fp32 model:',evaluate(test,model))

# calibration data used to calibrate PTQ and MPQ
calib_data = []
stacked_tensor = []
calib_fp_output = []
i = 0
with torch.no_grad():
    for img,label in train:
        i += 1
        stacked_tensor.append(img)
        calib_data.append((img,label))
        calib_fp_output.append(model(img.cuda()))
        if i == 16:
            break
# PTQ
# configuration
ptq_reconstruction_config_init = {
    'pattern': 'block',                   #? 'layer' for Adaround or 'block' for BRECQ and QDROP
    'scale_lr': 4.0e-5,                   #? learning rate for learning step size of activation
    'warm_up': 0.2,                       #? 0.2 * max_count iters without regularization to floor or ceil
    'weight': 0.01,                       #? loss weight for regularization item
#     'max_count': 20000,                   #? optimization iteration
    'max_count': 1,                   #? optimization iteration
    'b_range': [20,2],                    #? beta decaying range
    'keep_gpu': True,                     #? calibration data restore in gpu or cpu
    'round_mode': 'learned_hard_sigmoid', #? ways to reconstruct the weight, currently only support learned_hard_sigmoid
    'prob': 0.5,                          #? dropping probability of QDROP, 1.0 for Adaround and BRECQ
}

ptq_reconstruction_config = {
    'pattern': 'block',                   #? 'layer' for Adaround or 'block' for BRECQ and QDROP
    'scale_lr': 4.0e-5,                   #? learning rate for learning step size of activation
    'warm_up': 0.2,                       #? 0.2 * max_count iters without regularization to floor or ceil
    'weight': 0.01,                       #? loss weight for regularization item
    'max_count': 20000,                   #? optimization iteration
    'b_range': [20,2],                    #? beta decaying range
    'keep_gpu': True,                     #? calibration data restore in gpu or cpu
    'round_mode': 'learned_hard_sigmoid', #? ways to reconstruct the weight, currently only support learned_hard_sigmoid
    'prob': 0.5,                          #? dropping probability of QDROP, 1.0 for Adaround and BRECQ
}

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
ptq_reconstruction_config_init = dotdict(ptq_reconstruction_config_init)
ptq_reconstruction_config = dotdict(ptq_reconstruction_config)

def getModuleByName(model,moduleName):
    '''
        replace module with name modelName.moduleName with newModule
    '''
    tokens = moduleName.split('.')
    m = model
    for tok in tokens:
        m = getattr(m,tok)
    return m

for b in MPQ_scheme:
    mqb_fp_model = deepcopy(model)

    # MSE calibration on model parameters
    backend = BackendType.Academic
    extra_config = {
        'extra_qconfig_dict': {
            'w_observer': 'MSEObserver',                              # custom weight observer
            'a_observer': 'EMAMSEObserver',                              # custom activation observer
            'w_fakequantize': 'AdaRoundFakeQuantize' if adv_ptq else 'FixedFakeQuantize',
            'a_fakequantize': 'QDropFakeQuantize' if adv_ptq else 'FixedFakeQuantize',
            'w_qscheme': {
                'bit': b,                                             # custom bitwidth for weight,
                'symmetry': True,                                    # custom whether quant is symmetric for weight,
                'per_channel': False,                                  # custom whether quant is per-channel or per-tensor for weight,
                'pot_scale': False,                                   # custom whether scale is power of two for weight.
            },
            'a_qscheme': {
                'bit': 8,                                             # custom bitwidth for activation,
                'symmetry': False,                                    # custom whether quant is symmetric for activation,
                'per_channel': False,                                  # custom whether quant is per-channel or per-tensor for activation,
                'pot_scale': False,                                   # custom whether scale is power of two for activation.
            }
        }                                                         # custom tracer behavior, checkout https://github.com/pytorch/pytorch/blob/efcbbb177eacdacda80b94ad4ce34b9ed6cf687a/torch/fx/_symbolic_trace.py#L836
    }
    print(f'Prepare {b}bits model using MQBench')

    exec(f'mqb_{b}bits_model=prepare_by_platform(mqb_fp_model, backend,extra_config).cuda()')

    # calibration loop
    enable_calibration(eval(f'mqb_{b}bits_model'))
    for img,label in calib_data:
        eval(f'mqb_{b}bits_model')(img.cuda())  

    if adv_ptq:
        if os.path.exists(f'QDROP_{b}bits_{mn}.pt'):
            exec(f'mqb_{b}bits_model=ptq_reconstruction(mqb_{b}bits_model, stacked_tensor, ptq_reconstruction_config_init).cuda()')
            print(f'QDROP model already saved, now loading QDROP_{b}bits_{mn}.pt')
            load_from = f'QDROP_{b}bits_{mn}.pt'
            exec(f'mqb_{b}bits_model.load_state_dict(torch.load(load_from))')
        else:
            exec(f'mqb_{b}bits_model=ptq_reconstruction(mqb_{b}bits_model, stacked_tensor, ptq_reconstruction_config).cuda()')
            print(f'saving QDROP tuned model: QDROP_{b}bits_{mn}.pt...')
            torch.save(eval(f'mqb_{b}bits_model').state_dict(),f'QDROP_{b}bits_{mn}.pt')

for b in MPQ_scheme: 
    disable_all(eval(f'mqb_{b}bits_model'))
    # evaluation loop
    enable_quantization(eval(f'mqb_{b}bits_model'))
    print('evaluate mqb quantized model')
    print(evaluate(test,eval(f'mqb_{b}bits_model')))
mqb_fp_model = deepcopy(mqb_8bits_model)
disable_all(mqb_fp_model)
mqb_mix_model = deepcopy(mqb_fp_model)


# 1. record all modules we want to consider
types_to_quant = (torch.nn.Conv2d,torch.nn.Linear)

layer_input_map = {}

for node in mqb_8bits_model.graph.nodes:
    try:
        node_target = getModuleByName(mqb_mix_model,node.target)
        if isinstance(node_target,types_to_quant):
            node_args = node.args[0]
            print('input of ',node.target,' is ',node_args)
            layer_input_map[node.target] = str(node_args.target)
    except:
        continue

# ref_metric = ('loss',evaluate(calib_data,mqb_fp_model)['loss'])

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
                quant_out = mqb_mix_model(img.cuda())
                perturbed_loss.append(kldiv(quant_out,fp_out))
            #print(perturbed_loss)
            perturbed_loss = torch.tensor(perturbed_loss).mean()    

        if printInfo:
            print(f'use kl {KL} perturbed loss {perturbed_loss}')

        # recover layers
        mqb_mix_model = deepcopy(mqb_fp_model)

    return perturbed_loss

del layer_input_map['conv1']
del layer_input_map['fc']

import time
import matplotlib.pyplot as plt
s_time = time.time()
cached = {}
aw_scheme = []
for a_bits in MPQ_scheme:
    for w_bits in MPQ_scheme:
        aw_scheme.append((a_bits,w_bits))

aw_scheme = [(8,2),(8,4),(8,8)]
KL=False
for n in layer_input_map:
    for m in layer_input_map:
        for naw in aw_scheme:
            for maw in aw_scheme:
                if (n,m,naw,maw) not in cached:
                    if n == m:
                        if naw == maw:
                            p=0
                            #p = perturb_loss({n:naw},ref_metric,calib_data,KL=KL)
                            #print(f'perturb layer {n} to A{naw[0]}W{naw[1]} p={p}')
                        else:
                            p = 0

                    else:
                        p=0
                        #p = perturb_loss({n:naw,m:maw},ref_metric,calib_data,KL=KL)
                        #print(f'perturb layer {n} to A{naw[0]}W{naw[1]} and layer {m} to A{maw[0]}W{maw[1]} p={p}')
                    
                    cached[(n,m,naw,maw)] = cached[(m,n,maw,naw)] = p
                    
print(f'{time.time()-s_time:.2f} seconds elapsed')

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
index2layerscheme

def estimate_ltilde(index1,index2,clado_set=calib_data,KL=False):
    layer_name1,scheme1 = index2layerscheme[index1]
    layer_name2,scheme2 = index2layerscheme[index2]
    scheme1 = eval(scheme1[:-4])
    scheme2 = eval(scheme2[:-4])
    print(layer_name1,scheme1)
    print(layer_name2,scheme2)
    p = 0
    
    if layer_name1 == layer_name2:
        if scheme1 == scheme2:
            p = perturb_loss({layer_name1:scheme1},('loss',0),clado_set,KL=KL)
    else:
        p = perturb_loss({layer_name1:scheme1,layer_name2:scheme2},('loss',0),clado_set,KL=KL)
    
    fp32_ref = perturb_loss({},('loss',0),clado_set,KL=KL)
    
    return p - fp32_ref

def check_ltilde(index1,index2,numbatch=1000):
    est = []
    avg = []
    n_batch = 0
    all_est = 0
    for batch in train:
        batch_est = estimate_ltilde(index1,index2,clado_set=[batch,]).cpu().numpy()
        est.append(batch_est)
        all_est = (all_est * n_batch + batch_est)/(n_batch+1)
        avg.append(all_est)
        n_batch += 1
        print(f'batch {n_batch:3d} batch_est {batch_est:.4f} avg {all_est:.4f}')
        if n_batch >= numbatch:
            break
    est = np.array(est)
    avg = np.array(avg)
    return est

n_entries = 40
sampled = 0
res = {}

while sampled < n_entries:
    plt.clf()
    id1 = np.random.randint(len(index2layerscheme))
    id2 = np.random.randint(len(index2layerscheme))
    layer_name1,scheme1 = index2layerscheme[id1]
    layer_name2,scheme2 = index2layerscheme[id2]
    if layer_name1 == layer_name2 and not scheme1 == scheme2:
        continue
    if (id1,id2) in res:
        continue
    
    
    est = check_ltilde(id1,id2)
    res[(id1,id2)] = est
    
    plt.rcParams['figure.figsize'] = (12,10)

    for i in range(100):
        np.random.shuffle(est)
        avg = np.cumsum(est)
        avg /= np.cumsum(np.ones(len(avg)))
        plt.plot((np.arange(len(avg))+1)*64,avg,marker='o',alpha=0.5,markersize=1)
    plt.xlabel('number of samples')
    plt.ylabel('estimated Ltilde')
    plt.title(f'{layer_name1,scheme1} and {layer_name2,scheme2}')
    plt.savefig(f'Ltilde/{id1}_{id2}.pdf')
    sampled += 1
    
with open('saved_ltilde.pkl','wb') as f:
    pickle.dump({'res':res,'index2layerscheme':index2layerscheme},f)