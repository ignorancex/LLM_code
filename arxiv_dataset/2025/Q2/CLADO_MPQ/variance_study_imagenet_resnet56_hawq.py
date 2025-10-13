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
import argparse,time,pickle
import matplotlib.pyplot as plt
import cvxpy as cp
import pdb
################################ Arg Parser ####################################
parser = argparse.ArgumentParser()

# parser.add_argument("--mode", help="1: Estimate and Store Ltilde and DELTAL 2: CLADO and MPQCO Optimization",type=int)
# common args
parser.add_argument("--bs", help="batchsize",type=int,default=64)
parser.add_argument("--nthreads", help="data loader threads",type=int,default=12) # v100: 12 t4: 8

# under mode 1: store Ltildes
parser.add_argument("--head",help="starting batch index to estimate sensitivity use under mode 1",type=int)
parser.add_argument("--tail",help="ending batch index to estimate sensitivity use under mode 1",type=int)
parser.add_argument("--seed-start",help="seed to generate binary search",type=int)
parser.add_argument("--seed-end",help="seed to generate binary search",type=int)
parser.add_argument("--cuda",help="gpu index to run experiment",default=0,type=int)

# under mode 2: compare algorithms
# parser.add_argument("--HWcost",help="Bitops or Size use under mode 2",type=str) # constraint type
# parser.add_argument("--start-cost",help="min hardware constraint",type=float) # PF generation
# parser.add_argument("--end-cost",help="max hardware constraint",type=float) # PF generation
# parser.add_argument("--n-intervals",help="number of hw constraints between min and max constraint",type=int) # PF generation
# parser.add_argument("--n-sets",help="number of different sensitivity sets used for each constraint",type=int) # variance study
args = parser.parse_args() 

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
# ds = I1K(data_dir=os.path.join('/home/usr1/zd2922/data', "imagenet"),
#          train_batch_size=64,test_batch_size=args.bs,cuda=True,shuffle=True)
ds = I1K(data_dir=os.path.join('/work2/07149/zdeng/frontera/data', "imagenet"),
         train_batch_size=args.bs,test_batch_size=args.bs,cuda=True,shuffle=True)
model = eval("tv.models." + modelname)(pretrained=True).cuda(args.cuda)
torch.save(model,mn+'.pth')
torch.save(model.state_dict(),mn+'_state_dict.pth')
with open(f'{mn}.pkl','wb') as f:
    pickle.dump(model,f)
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
        
        # stacked_tensor is to calibrate the model
        # calib_data is the data to calculate ltilde
        if i< 16: # always use 1024 samples to calibrate
            stacked_tensor.append(img)
        if i>= 16:
            break
        
        i += 1
for i in range(args.head,args.tail+1):
    with open(f'helpZihao/HAWQ_DELTAL/TraceEST_batch{i}.pkl','rb') as f:
        TraceEst = pickle.load(f)
        break
          
MPQ_scheme = (2,4,8)
model.eval()
# configuration
ptq_reconstruction_config_init = {
    'pattern': 'block',                   #? 'layer' for Adaround or 'block' for BRECQ and QDROP
    'scale_lr': 4.0e-5,                   #? learning rate for learning step size of activation
    'warm_up': 0.2,                       #? 0.2 * max_count iters without regularization to floor or ceil
    'weight': 0.01,                       #? loss weight for regularization item
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
ptq_reconstruction_config = dotdict(ptq_reconstruction_config)
ptq_reconstruction_config_init = dotdict(ptq_reconstruction_config_init)

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

    exec(f'mqb_{b}bits_model=prepare_by_platform(mqb_fp_model, backend,extra_config).cuda(args.cuda)')
    
    # calibration loop
    enable_calibration(eval(f'mqb_{b}bits_model'))
    for img in stacked_tensor:
        eval(f'mqb_{b}bits_model')(img.cuda(args.cuda))
    
    #torch.save(eval(f'mqb_{b}bits_model'),f'{mn}_mqb_{b}bits_model.pth')
    if adv_ptq:
        if os.path.exists(f'QDROP_{b}bits_{mn}.pt'):
            exec(f'mqb_{b}bits_model=ptq_reconstruction(mqb_{b}bits_model, stacked_tensor, ptq_reconstruction_config_init).cuda(args.cuda)')
            print(f'QDROP model already saved, now loading QDROP_{b}bits_{mn}.pt')
            load_from = f'QDROP_{b}bits_{mn}.pt'
            exec(f'mqb_{b}bits_model.load_state_dict(torch.load(load_from))')
        else:
            
            exec(f'mqb_{b}bits_model=ptq_reconstruction(mqb_{b}bits_model, stacked_tensor, ptq_reconstruction_config).cuda(args.cuda)')
            print(f'saving QDROP tuned model: QDROP_{b}bits_{mn}.pt...')
            torch.save(eval(f'mqb_{b}bits_model').state_dict(),f'QDROP_{b}bits_{mn}.pt')
            
for b in MPQ_scheme: 
    disable_all(eval(f'mqb_{b}bits_model'))
    # evaluation loop
    enable_quantization(eval(f'mqb_{b}bits_model'))
    eval(f'mqb_{b}bits_model').eval()
    print(f'evaluate mqb {b}bits model')
    #print(evaluate(test,eval(f'mqb_{b}bits_model')))
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

del layer_input_map['conv1']
del layer_input_map['fc']


aw_scheme = [(8,2),(8,4),(8,8)]

############################################ following code from variance_study.py ########################################

head,tail = args.head,args.tail

with open(f'helpZihao/Ltilde_resnet50/Ltilde_batch{head}(size64)_[(8, 2), (8, 4), (8, 8)]i1k_resnet50.pkl','rb') as f:
    hm = pickle.load(f)
ref_layer_index = hm['layer_index']
ref_index2layerscheme = hm['index2layerscheme']

with open('imagenet/layer_cost_resnet50.pkl','rb') as f:
    layer_cost = pickle.load(f)
    assert hm['layer_index'] == ref_layer_index
    
layer_size = layer_cost['layer_size']
layer_bitops = layer_cost['layer_bitops']

batch_Ltildes_clado = []

for batch_id in range(head,tail+1):
    with open(f'helpZihao/Ltilde_resnet50/Ltilde_batch{batch_id}(size64)_[(8, 2), (8, 4), (8, 8)]i1k_resnet50.pkl','rb') as f:
        hm = pickle.load(f)
    
    assert hm['layer_index'] == ref_layer_index
    batch_Ltildes_clado.append(hm['Ltilde'])

batch_Ltildes_clado = np.array(batch_Ltildes_clado)
ref_Ltilde_clado = batch_Ltildes_clado.mean(axis=0)

batch_Ltildes_mpqco = []

for batch_id in range(head,tail+1):
    with open(f'helpZihao/DELTAL_resnet50/MPQCO_DELTAL_resnet50_batch{batch_id}(size64).pkl','rb') as f:
        hm = pickle.load(f)
        
    deltal = np.zeros(ref_Ltilde_clado.shape)
    
    for layer_id in range(len(ref_index2layerscheme)):
        layer_name,scheme = ref_index2layerscheme[layer_id]
        wbit = eval(scheme[:-4])[1]
        deltal[layer_id,layer_id] = hm[layer_name][wbit]
    
    batch_Ltildes_mpqco.append(deltal)
batch_Ltildes_mpqco = np.array(batch_Ltildes_mpqco)
ref_Ltilde_mpqco = batch_Ltildes_mpqco.mean(axis=0)

### post process trace estimation to get hawq deltal
batch_Ltildes_hawq = []

for batch_id in range(head,tail+1):
    with open(f'helpZihao/HAWQ_DELTAL/TraceEST_batch{batch_id}.pkl','rb') as f:
        trace = pickle.load(f)
    
    deltal = np.zeros(ref_Ltilde_clado.shape)
    
    for layer_id in range(len(ref_index2layerscheme)):
        layer_name,scheme = ref_index2layerscheme[layer_id]
        wbit = eval(scheme[:-4])[1]
        tar_module = getModuleByName(eval(f'mqb_{wbit}bits_model'),layer_name)
        dw = tar_module.weight_fake_quant(tar_module.weight) - tar_module.weight
        
        # mean of trace times squared error
        deltal[layer_id,layer_id] = np.array(trace[layer_name]).mean()/dw.nelement() * (dw**2).sum().item()
    
    with open(f'helpZihao/HAWQ_DELTAL/DELTAL_batch{batch_id}.pkl','wb') as f:
        pickle.dump({'Ltilde':deltal,'layer_index':ref_layer_index},f)
    batch_Ltildes_hawq.append(deltal)
    
batch_Ltildes_hawq = np.array(batch_Ltildes_hawq)
ref_Ltilde_hawq = batch_Ltildes_hawq.mean(axis=0)

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
    #print("Solution status", prob.status)
    #print("A solution x is")
    #print(x.value)
    #print(f"bitops: {x.value@layer_bitops}")
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

def avg_diff(decisions):
    # decisions is in shape n_runs x decision dimension
    # do majority voting first for each dimension
    mean_decision = decisions.mean(axis=0)
    mean_decision[mean_decision>0.5] = 1
    mean_decision[mean_decision<0.5] = 0
    #assert mean_decision[np.abs(mean_decision-0.5)<1e-6].shape[0] == 0, str(np.abs(mean_decision-0.5)<1e-6)
    
    diff = np.abs(decisions - mean_decision)
    return diff.sum()/decisions.shape[0]
    

diff_cladoss,diff_naivess,diff_mpqcoss = [],[],[]

L = ref_Ltilde_clado.shape[0]

evaluated = {}

if os.path.exists('evaluated_decisions_combined'):
    with open('evaluated_decisions_combined.pkl','rb') as f:
        evaluated = pickle.load(f)
        print(f'{len(evaluated)} decisions records loaded')

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
    
    print("evaluate_decision\n",decisions)
    print('hashed\n',hashed)
    

    with torch.no_grad():
        
        # perturb layers
        perturb(decisions)
            
        # do evaluation
        res = evaluate(test,mqb_mix_model)
        
        # recover layers
        mqb_mix_model = deepcopy(mqb_fp_model)
        
    evaluated[hashed] = (res,modelsize,bitops)
    
    with open(f'evaluated_decisions_{mn}_{aw_scheme}_{args.tail-args.head+1}batches_seed{args.seed_start}-{args.seed_end}_hawq_cuda{args.cuda}.pkl','wb') as f:
        pickle.dump(evaluated,f)
    
    return res,modelsize,bitops


for seed in range(args.seed_start,args.seed_end+1,1):
    print(f'seed {seed}')
    np.random.seed(seed)
    shuffle = np.random.choice(tail-head+1,tail-head+1,replace=False)
    n_batch = 1
    
    diff_clados,diff_naives,diff_mpqcos = [],[],[]

    while 2*n_batch <= shuffle.shape[0]:

        '''
        clado_ltilde1 = batch_Ltildes_clado[shuffle[:n_batch]].mean(axis=0)
        mpqco_ltilde1 = batch_Ltildes_mpqco[shuffle[:n_batch]].mean(axis=0)
        clado_ltilde2 = batch_Ltildes_clado[shuffle[n_batch:2*n_batch]].mean(axis=0)
        mpqco_ltilde2 = batch_Ltildes_mpqco[shuffle[n_batch:2*n_batch]].mean(axis=0)
        '''
        
        hawq_ltilde1 = batch_Ltildes_hawq[shuffle[:n_batch]].mean(axis=0)
        hawq_ltilde2 = batch_Ltildes_hawq[shuffle[n_batch:2*n_batch]].mean(axis=0)
        

        constraint_index = 0
        
        
        for size_bound in np.linspace(10,20,11):
            print(f'seed {seed} n_batch {n_batch} size_bound {size_bound:.2f}MB')
            '''
            # CLADO Way
            cached_grad = Ltilde2CachedGrad(clado_ltilde1)
            v_clado1 = MIQCP_optimize(cached_grad=cached_grad,
                               layer_bitops=layer_bitops,
                               layer_size=layer_size,
                               schemes_per_layer=len(aw_scheme),
                               bitops_bound=np.inf,size_bound=size_bound,
                               naive=False,PSD=True)
            evaluate_decision(v_clado1.value)
            
            
            cached_grad = Ltilde2CachedGrad(clado_ltilde2)
            v_clado2 = MIQCP_optimize(cached_grad=cached_grad,
                               layer_bitops=layer_bitops,
                               layer_size=layer_size,
                               schemes_per_layer=len(aw_scheme),
                               bitops_bound=np.inf,size_bound=size_bound,
                               naive=False,PSD=True)
            evaluate_decision(v_clado2.value)

            # naive Way
            cached_grad = Ltilde2CachedGrad(clado_ltilde1)
            v_naive1 = MIQCP_optimize(cached_grad=cached_grad,
                               layer_bitops=layer_bitops,
                               layer_size=layer_size,
                               schemes_per_layer=len(aw_scheme),
                               bitops_bound=np.inf,size_bound=size_bound,
                               naive=True,PSD=False)
            evaluate_decision(v_naive1.value)
            
            cached_grad = Ltilde2CachedGrad(clado_ltilde2)
            v_naive2 = MIQCP_optimize(cached_grad=cached_grad,
                               layer_bitops=layer_bitops,
                               layer_size=layer_size,
                               schemes_per_layer=len(aw_scheme),
                               bitops_bound=np.inf,size_bound=size_bound,
                               naive=True,PSD=False)
            evaluate_decision(v_naive2.value)

            # MPQCO Way
            v_mpqco1 = MIQCP_optimize(cached_grad=mpqco_ltilde1,
                               layer_bitops=layer_bitops,
                               layer_size=layer_size,
                               schemes_per_layer=len(aw_scheme),
                               bitops_bound=np.inf,size_bound=size_bound,
                               naive=True,PSD=False)
            evaluate_decision(v_mpqco1.value)
            
            v_mpqco2 = MIQCP_optimize(cached_grad=mpqco_ltilde2,
                               layer_bitops=layer_bitops,
                               layer_size=layer_size,
                               schemes_per_layer=len(aw_scheme),
                               bitops_bound=np.inf,size_bound=size_bound,
                               naive=True,PSD=False)
            evaluate_decision(v_mpqco2.value)
            '''
            # HAWQ Way
            v_hawq1 = MIQCP_optimize(cached_grad=hawq_ltilde1,
                               layer_bitops=layer_bitops,
                               layer_size=layer_size,
                               schemes_per_layer=len(aw_scheme),
                               bitops_bound=np.inf,size_bound=size_bound,
                               naive=True,PSD=False)
            evaluate_decision(v_hawq1.value)
            
            v_hawq2 = MIQCP_optimize(cached_grad=hawq_ltilde2,
                               layer_bitops=layer_bitops,
                               layer_size=layer_size,
                               schemes_per_layer=len(aw_scheme),
                               bitops_bound=np.inf,size_bound=size_bound,
                               naive=True,PSD=False)
            print(evaluate_decision(v_hawq2.value))
            
            constraint_index += 1
        n_batch *= 2
        

