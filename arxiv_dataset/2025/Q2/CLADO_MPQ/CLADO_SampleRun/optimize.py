'''
This notebook
(1) Do MIP optimization after ltildes,deltals are saved by check_model.py
(2) visualize the results for different models (specified by the argument)
'''
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
parser.add_argument("--start-batch",help="starting batch index to estimate sensitivity use under mode 1",type=int)
parser.add_argument("--end-batch",help="ending batch index to estimate sensitivity use under mode 1",type=int)
parser.add_argument("--modelname",type=str)
parser.add_argument("--datapath",type=str) # path to ImageNet data folder,example: '/home/usr1/zd2922/data/imagenet'
args = parser.parse_args()

modelname = args.modelname

dataset = 'I1K'
mn = dataset.lower()+ '_' + modelname
MPQ_scheme = (2,4,8)
aw_scheme = [(8,2),(8,4),(8,8)]

with open(f'Ltilde_{modelname}/Ltilde_batch{args.start_batch}(size{args.bs})_{str(aw_scheme)}i1k_{modelname}.pkl','rb') as f:
    hm = pickle.load(f)
ref_layer_index = hm['layer_index']
ref_index2layerscheme = hm['index2layerscheme']

with open(f'layer_cost_{modelname}.pkl','rb') as f:
    layer_cost = pickle.load(f)
    assert hm['layer_index'] == ref_layer_index

layer_size = layer_cost['layer_size']
layer_bitops = layer_cost['layer_bitops']

batch_Ltildes_clado = []

for batch_id in range(args.start_batch,args.end_batch+1):
    with open(f'Ltilde_{modelname}/Ltilde_batch{batch_id}(size{args.bs})_[(8, 2), (8, 4), (8, 8)]i1k_{modelname}.pkl','rb') as f:
        hm = pickle.load(f)

    assert hm['layer_index'] == ref_layer_index
    batch_Ltildes_clado.append(hm['Ltilde'])
batch_Ltildes_clado = np.array(batch_Ltildes_clado)
ref_Ltilde_clado = batch_Ltildes_clado.mean(axis=0)

batch_Ltildes_mpqco = []

for batch_id in range(args.start_batch,args.end_batch+1):
    with open(f'DELTAL_{modelname}/MPQCO_DELTAL_{modelname}_batch{batch_id}(size64).pkl','rb') as f:
        hm = pickle.load(f)

    deltal = np.zeros(ref_Ltilde_clado.shape)

    for layer_id in range(len(ref_index2layerscheme)):
        layer_name,scheme = ref_index2layerscheme[layer_id]
        wbit = eval(scheme[:-4])[1]
        deltal[layer_id,layer_id] = hm[layer_name][wbit]

    batch_Ltildes_mpqco.append(deltal)
batch_Ltildes_mpqco = np.array(batch_Ltildes_mpqco)
ref_Ltilde_mpqco = batch_Ltildes_mpqco.mean(axis=0)




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

ds = I1K(data_dir=args.datapath,
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

# calibration data used to calibrate PTQ and MPQ
calib_data = []
stacked_tensor = []
calib_fp_output = []
i = 0
with torch.no_grad():
    for img,label in train:

        # stacked_tensor is used to calibrate the model (1024 samples)
        # calib_data is the sensitivity set (number of samples specified by user)
        if i< 1024/args.bs: # always use 1024 samples to calibrate
            stacked_tensor.append(img)

        if i>= 1024/args.bs and i>= args.end_batch:
            break

        i += 1

model = eval("tv.models." + modelname)(pretrained=True).cuda(args.cuda)
model.eval()

for b in MPQ_scheme:
    mqb_fp_model = deepcopy(model)

    # MSE calibration on model parameters
    backend = BackendType.Academic
    extra_config = {
        'extra_qconfig_dict': {
            'w_observer': 'MSEObserver',                              # custom weight observer
            'a_observer': 'EMAMSEObserver',                              # custom activation observer
            'w_fakequantize': 'FixedFakeQuantize',
            'a_fakequantize': 'FixedFakeQuantize',
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

    exec(f'mqb_{b}bits_model=prepare_by_platform(mqb_fp_model, backend,extra_config).cuda(args.cuda)')

    # calibration loop
    enable_calibration(eval(f'mqb_{b}bits_model'))
    for img in stacked_tensor:
        eval(f'mqb_{b}bits_model')(img.cuda(args.cuda))


model.eval()


for b in MPQ_scheme:
    disable_all(eval(f'mqb_{b}bits_model'))
    # evaluation loop
    enable_quantization(eval(f'mqb_{b}bits_model'))
    eval(f'mqb_{b}bits_model').eval()


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

# 1. record all modules we want to consider
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
import matplotlib.pyplot as plt
import cvxpy as cp
def MIQCP_optimize(cached_grad,layer_bitops,layer_size,
                   schemes_per_layer=len(aw_scheme),
                   bitops_bound=np.inf,size_bound=np.inf,
                   naive=False,PSD=False):

    if cached_grad.__class__ == torch.Tensor:
        cached_grad = cached_grad.cpu().numpy()

    x = cp.Variable(cached_grad.shape[0], boolean=True)
    schemes_per_layer = schemes_per_layer
    assert cached_grad.shape[0]%schemes_per_layer == 0, 'cached_gradient shape[0] does not divde schemes per layer'
    num_layers = cached_grad.shape[0]//schemes_per_layer

    if PSD:
        es,us = np.linalg.eig(cached_grad)
        es[es<0] = 0
        cached_grad = us@np.diag(es)@us.T
        cached_grad = (cached_grad+cached_grad.T)/2
    if not naive:
        cached_grad = cp.atoms.affine.wraps.psd_wrap(cached_grad)
        objective = cp.Minimize(cp.quad_form(x,cached_grad))
    else:
        objective = cp.Minimize(np.diagonal(cached_grad)@x)

    equality_constraint_matrix = []
    for i in range(num_layers):
        col = np.zeros(cached_grad.shape[0])
        col[i*schemes_per_layer:(i+1)*schemes_per_layer] = 1
        equality_constraint_matrix.append(col)

    equality_constraint_matrix = np.array(equality_constraint_matrix)

    constraints = [equality_constraint_matrix@x == np.ones((num_layers,)),
                   layer_bitops@x/10**9<=bitops_bound,
                   layer_size@x/8/1024/1024<=size_bound]

    prob = cp.Problem(objective,constraints)
    prob.solve(solver='GUROBI',verbose=False,TimeLimit=120)

    # Print result.
    print("Solution status", prob.status)
    print("A solution x is")
    print(x.value)
    print(f"bitops: {x.value@layer_bitops}")
    return x

def Ltilde2CachedGrad(Ltilde):

    cached_grad = np.zeros_like(Ltilde)

    for i in range(cached_grad.shape[0]):
        for j in range(cached_grad.shape[0]):
            layer_i,scheme_i = ref_index2layerscheme[i]
            layer_j,scheme_j = ref_index2layerscheme[j]
            if layer_i == layer_j:
                if scheme_i == scheme_j:
                    cached_grad[i,j] = cached_grad[j,i] = 2 * Ltilde[i,j]
                else:
                    cached_grad[i,j] = cached_grad[j,i] = 0
            else:
                cached_grad[i,j] = cached_grad[j,i] = Ltilde[i,j] - Ltilde[i,i] - Ltilde[j,j]
    return cached_grad

L = ref_Ltilde_clado.shape[0]

evaluated = {}

if os.path.exists(f'evaluated_decisions_{modelname}.pkl'):
    with open(f'evaluated_decisions_{modelname}.pkl','rb') as f:
        evaluated = pickle.load(f)
        print(f'{len(evaluated)} decisions records for {mn} loaded')

def hash_decision(v):
    hashed = ''
    for i in range(v.shape[0]):
        hashed += '0' if np.abs(v[i]) < 1e-1 else '1'
    return hashed

def evaluate_decision(v,printInfo=False,test=test):

    global mqb_mix_model,evaluated

    hashed = hash_decision(v)
    if hashed in evaluated:
       	print('cache hit')
        return evaluated[hashed]


    # alpha = torch.nn.Softmax(dim=1)(v.reshape(-1,len(MPQ_scheme)))
    offset = torch.ones(int(L/len(aw_scheme)),dtype=int) * len(aw_scheme)
    offset = offset.cumsum(dim=-1) - len(aw_scheme)
    select = torch.Tensor(v).reshape(-1,len(aw_scheme)).argmax(dim=1) + offset

    modelsize = (layer_size[select]).sum()/8/1024/1024
    bitops = (layer_bitops[select]).sum()/10**9

    decisions = {}
    for scheme_id in select.numpy():
        layer,scheme = ref_index2layerscheme[scheme_id]
        decisions[layer] = eval(scheme[:-4])

    print("evaluate MPQ decision\n",decisions)
    print('evaluate hashed decisions\n',hashed)


    with torch.no_grad():

        # perturb layers
        perturb(decisions)

        # do evaluation
        res = evaluate(test,mqb_mix_model)

        # recover layers
        mqb_mix_model = deepcopy(mqb_fp_model)

    evaluated[hashed] = (res,modelsize,bitops)

    with open(f'evaluated_decisions_{modelname}.pkl','wb') as f:
        pickle.dump(evaluated,f)

    return res,modelsize,bitops

### post process trace estimation to get hawq deltal
batch_Ltildes_hawq = []

for batch_id in range(args.start_batch,args.end_batch+1):
    with open(f'HAWQ_DELTAL_{modelname}/TraceEST_batch{batch_id}.pkl','rb') as f:
        trace = pickle.load(f)

    deltal = np.zeros(ref_Ltilde_clado.shape)

    for layer_id in range(len(ref_index2layerscheme)):
        layer_name,scheme = ref_index2layerscheme[layer_id]
        wbit = eval(scheme[:-4])[1]
        tar_module = getModuleByName(eval(f'mqb_{wbit}bits_model'),layer_name)
        dw = tar_module.weight_fake_quant(tar_module.weight) - tar_module.weight

        # mean of trace times squared error
        deltal[layer_id,layer_id] = np.array(trace[layer_name]).mean()/dw.nelement() * (dw**2).sum().item()

    with open(f'HAWQ_DELTAL_{modelname}/DELTAL_batch{batch_id}.pkl','wb') as f:
        pickle.dump({'Ltilde':deltal,'layer_index':ref_layer_index},f)
    batch_Ltildes_hawq.append(deltal)

batch_Ltildes_hawq = np.array(batch_Ltildes_hawq)
ref_Ltilde_hawq = batch_Ltildes_hawq.mean(axis=0)


size_8bit = layer_size.sum()/(8+4+2)/1024/1024

clado_perf,naive_perf,mpqco_perf,hawq_perf = [], [], [], []
clado_size,naive_size,mpqco_size,hawq_size = [], [], [], []

size_bound = size_8bit/2
print(f'size_bound {size_bound:.2f}MB')

# CLADO Way
cached_grad = Ltilde2CachedGrad(ref_Ltilde_clado)
v_clado1 = MIQCP_optimize(cached_grad=cached_grad,
                   layer_bitops=layer_bitops,
                   layer_size=layer_size,
                   schemes_per_layer=len(aw_scheme),
                   bitops_bound=np.inf,size_bound=size_bound,
                   naive=False,PSD=True)
perf,size,bitops = evaluate_decision(v_clado1.value)
print(f'clado:',perf,size,'MB')
clado_perf.append(perf['top1'])
clado_size.append(size)

# naive Way
cached_grad = Ltilde2CachedGrad(ref_Ltilde_clado)
v_naive1 = MIQCP_optimize(cached_grad=cached_grad,
                   layer_bitops=layer_bitops,
                   layer_size=layer_size,
                   schemes_per_layer=len(aw_scheme),
                   bitops_bound=np.inf,size_bound=size_bound,
                   naive=True,PSD=False)
perf,size,bitops = evaluate_decision(v_naive1.value)
print(f'naive:',perf,size,'MB')
naive_perf.append(perf['top1'])
naive_size.append(size)

# MPQCO Way
v_mpqco1 = MIQCP_optimize(cached_grad=ref_Ltilde_mpqco,
                   layer_bitops=layer_bitops,
                   layer_size=layer_size,
                   schemes_per_layer=len(aw_scheme),
                   bitops_bound=np.inf,size_bound=size_bound,
                   naive=True,PSD=False)
perf,size,bitops = evaluate_decision(v_mpqco1.value)
print(f'mpqco:',perf,size,'MB')
mpqco_perf.append(perf['top1'])
mpqco_size.append(size)


# HAWQ Way
v_hawq1 = MIQCP_optimize(cached_grad=ref_Ltilde_hawq,
                   layer_bitops=layer_bitops,
                   layer_size=layer_size,
                   schemes_per_layer=len(aw_scheme),
                   bitops_bound=np.inf,size_bound=size_bound,
                   naive=True,PSD=False)
perf,size,bitops = evaluate_decision(v_hawq1.value)
print(f'hawq:',perf,size,'MB')
hawq_perf.append(perf['top1'])
hawq_size.append(size)
