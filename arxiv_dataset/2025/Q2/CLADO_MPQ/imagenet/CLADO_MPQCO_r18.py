import torch,torchvision,os,time
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
torch.manual_seed(0)
np.random.seed(0)

from mltools.data import I1K
import torchvision as tv

modelname = 'resnet18'
adv_ptq = False
dataset = 'I1K'
mn = dataset.lower()+ '_' + modelname
ds = I1K(data_dir=os.path.join('/tools/d-matrix/ml/data', "imagenet"),
         train_batch_size=64,test_batch_size=64,cuda=True)
model = eval("tv.models." + modelname)(pretrained=True).cuda()
ds.train.num_workers = 12
ds.val.num_workers = 12

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
        i += 1
        # stacked_tensor is to calibrate the model
        # calib_data (part of it, as defined later) is the data to calculate ltilde
        if i<= 16:
            stacked_tensor.append(img)
        
        calib_data.append((img,label))
        #calib_fp_output.append(model(img.cuda()))
        print(i)
        if i>=300:
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

    exec(f'mqb_{b}bits_model=prepare_by_platform(mqb_fp_model, backend,extra_config).cuda()')
    
    # calibration loop
    enable_calibration(eval(f'mqb_{b}bits_model'))
    for img in stacked_tensor:
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
    eval(f'mqb_{b}bits_model').eval()
    print(f'evaluate mqb {b}bits model')
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
        
        #print(layer_name)
        #print(a_cmd)
        #print(w_cmd)
import pickle
def estimate_deltaL(eval_data,wbit_choices=[2,4,8]):
    
    tot_batches = len(eval_data)
    
    processed_batches = 0
    
    print(f'MPPCO Ltilde: {processed_batches}/{tot_batches} batch of data processed')
    
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
            logits = model(img.cuda())       
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
        
        #print(deltaL)
        with open(f'MPQCO_DELTAL_resnet18_batch{processed_batches}(size64).pkl','wb') as f:
            pickle.dump(deltaL,f)
            
        processed_batches += 1
        
        print(f'MPPCO Ltilde: {processed_batches}/{tot_batches} batch of data processed')
        print(f'batch cost:{time.time()-s_time:.2f} seconds')
    
    
import torch.nn.functional as F
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

# this part is done
# saved Ltilde are stored under Ltilde_resnet18 folder
# KL=False
# estimate_deltaL(calib_data,wbit_choices=[2,4,8])
# for clado_batch in range(len(calib_data)):
#     print(f'clado batch {clado_batch+1}/{len(calib_data)}')
#     ref_metric = ('loss',
#                   evaluate([calib_data[clado_batch],],mqb_fp_model)['loss'])
    
#     s_time = time.time()
#     cached = {}
#     for n in layer_input_map:
#         for m in layer_input_map:
#             for naw in aw_scheme:
#                 for maw in aw_scheme:
#                     if (n,m,naw,maw) not in cached:
#                         if n == m:
#                             if naw == maw:
#                                 p = perturb_loss({n:naw},ref_metric,
#                                                  [calib_data[clado_batch],],KL=KL)
#                                 #print(f'perturb layer {n} to A{naw[0]}W{naw[1]} p={p}')
#                             else:
#                                 p = 0

#                         else:
#                             p = perturb_loss({n:naw,m:maw},ref_metric,
#                                              [calib_data[clado_batch],],KL=KL)
#                             #print(f'perturb layer {n} to A{naw[0]}W{naw[1]} and layer {m} to A{maw[0]}W{maw[1]} p={p}')

#                         cached[(n,m,naw,maw)] = cached[(m,n,maw,naw)] = p

#     print(f'{time.time()-s_time:.2f} seconds elapsed')
    
#     # layer index and index2layerscheme map
#     layer_index = {}
#     cnt = 0
#     for layer in layer_input_map:
#         for s in aw_scheme:
#             layer_index[layer+f'{s}bits'] = cnt
#             cnt += 1
#     L = cnt

#     import numpy as np
#     hm = np.zeros(shape=(L,L))
#     for n in layer_input_map:
#         for m in layer_input_map:
#             for naw in aw_scheme:
#                 for maw in aw_scheme:
#                     hm[layer_index[n+f'{naw}bits'],layer_index[m+f'{maw}bits']] = cached[(n,m,naw,maw)]

#     index2layerscheme = [None for i in range(hm.shape[0])]

#     for name in layer_index:
#         index = layer_index[name]
#         layer_name = name[:-10]
#         scheme = name[-10:]

#         index2layerscheme[index] = (layer_name,scheme)
    
#     import pickle

#     saveas = f'Ltilde_resnet18/Ltilde_batch{clado_batch}(size64)_'
#     saveas += 'QDROP' if adv_ptq else ''
#     saveas += str(aw_scheme)
#     saveas += mn
#     saveas += 'KL' if KL else ''
#     saveas += '.pkl'

#     with open(saveas,'wb') as f:
#         pickle.dump({'Ltilde':hm,'layer_index':layer_index,'index2layerscheme':index2layerscheme},f)
with open('Ltilde_resnet18/Ltilde_batch0(size64)_[(8, 2), (8, 4), (8, 8)]i1k_resnet18.pkl','rb') as f:
    hm = pickle.load(f)
ref_layer_index = hm['layer_index']
ref_index2layerscheme = hm['index2layerscheme']
batch_Ltildes_clado = []
for batch_id in range(300):
    with open(f'Ltilde_resnet18/Ltilde_batch{batch_id}(size64)_[(8, 2), (8, 4), (8, 8)]i1k_resnet18.pkl','rb') as f:
        hm = pickle.load(f)
    
    assert hm['layer_index'] == ref_layer_index
    batch_Ltildes_clado.append(hm['Ltilde'])
batch_Ltildes_clado = np.array(batch_Ltildes_clado)
ref_Ltilde_clado = batch_Ltildes_clado.mean(axis=0)
batch_Ltildes_mpqco = []
for batch_id in range(300):
    with open(f'Ltilde_resnet18/MPQCO_DELTAL_resnet18_batch{batch_id}(size64).pkl','rb') as f:
        hm = pickle.load(f)
        
    deltal = np.zeros(ref_Ltilde_clado.shape)
    
    for layer_id in range(len(ref_index2layerscheme)):
        layer_name,scheme = ref_index2layerscheme[layer_id]
        wbit = eval(scheme[:-4])[1]
        deltal[layer_id,layer_id] = hm[layer_name][wbit]
    
    batch_Ltildes_mpqco.append(deltal)
batch_Ltildes_mpqco = np.array(batch_Ltildes_mpqco)
ref_Ltilde_mpqco = batch_Ltildes_mpqco.mean(axis=0)

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
        model(img.cuda())
        break
        
def get_layer_bitops(layer_name,a_bits,w_bits):
    m = getModuleByName(model,layer_name)
    if isinstance(m,torch.nn.Conv2d):
        _,cin,_,_ = hooks[layer_name][0].in_shape 
        _,cout,hout,wout = hooks[layer_name][0].out_shape
        n_muls = cin * m.weight.size()[2] * m.weight.size()[3] * cout * hout * wout 
        n_accs = (cin * m.weight.size()[2] * m.weight.size()[3] - 1) * cout * hout * wout
        #bitops_per_mul = 2 * a_bits * w_bits
        #bitops_per_acc = (a_bits + w_bits) + np.ceil(np.log2(cin * m.weight.size()[2] * m.weight.size()[
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
L = ref_Ltilde_clado.shape[0]
def evaluate_decision(v,printInfo=False,test=test):
    global mqb_mix_model
    v = v.detach()
    # alpha = torch.nn.Softmax(dim=1)(v.reshape(-1,len(MPQ_scheme)))
    offset = torch.ones(int(L/len(aw_scheme)),dtype=int) * len(aw_scheme)
    offset = offset.cumsum(dim=-1) - len(aw_scheme)
    select = v.reshape(-1,len(aw_scheme)).argmax(dim=1) + offset
    
    modelsize = (layer_size[select]).sum()/8/1024/1024
    bitops = (layer_bitops[select]).sum()/10**9
    
    decisions = {}
    for scheme_id in select.numpy():
        layer,scheme = ref_index2layerscheme[scheme_id]
        decisions[layer] = eval(scheme[:-4])
    
    print("evaluate_decision\n",decisions)
    

    with torch.no_grad():
        
        # perturb layers
        perturb(decisions)
            
        # do evaluation
        res = evaluate(test,mqb_mix_model)
        
        # recover layers
        mqb_mix_model = deepcopy(mqb_fp_model)
    return res,modelsize,bitops
import cvxpy as cp

def MIQCP_optimize(cached_grad,layer_bitops,layer_size,
                   schemes_per_layer=len(aw_scheme),
                   bitops_bound=np.inf,size_bound=np.inf,
                   naive=False,PSD=True):
    
    if cached_grad.__class__ == torch.Tensor:
        cached_grad = cached_grad.cpu().numpy()
    
    x = cp.Variable(cached_grad.shape[0], boolean=True)
    schemes_per_layer = schemes_per_layer
    assert cached_grad.shape[0]%schemes_per_layer == 0, 'cached_gradient shape[0] does not divde schemes per layer'
    num_layers = cached_grad.shape[0]//schemes_per_layer
    
    if not naive:
        # convexation of cached_grad
        es,us = np.linalg.eig(cached_grad)
        if PSD:
            es[es<0] = 0
        C = us@np.diag(es)@us.T
        C = (C+C.T)/2
        C = cp.atoms.affine.wraps.psd_wrap(C)
        objective = cp.Minimize(cp.quad_form(x,C))
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
    prob.solve(verbose=False,TimeLimit=60)
    
    # Print result.
    print("Solution status", prob.status)
    print("A solution x is")
    print(x.value)
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

for n_batch in (1,4,16,64,256):
    for sid in range(5):
        
        print(f'{n_batch} batches sid {sid}')
        
        clado_perf,clado_size,clado_bitops = [],[],[]
        mpqco_perf,mpqco_size,mpqco_bitops = [],[],[]
        naive_perf,naive_size,naive_bitops = [],[],[]
        
        
        shuffle = np.random.choice(300,n_batch,replace=False)
        
        clado_ltilde = batch_Ltildes_clado[shuffle].mean(axis=0)
        mpqco_ltilde = batch_Ltildes_mpqco[shuffle].mean(axis=0)
        
        with open(f'Ltilde_resnet18/res_{n_batch}batches(size64)_sid{sid}.pkl','wb') as f:
            pickle.dump({'shuffle':shuffle,
                         'clado_perf':clado_perf,'clado_size':clado_size,'clado_bitops':clado_bitops,
                         'naive_perf':naive_perf,'naive_size':naive_size,'naive_bitops':naive_bitops,
                         'mpqco_perf':mpqco_perf,'mpqco_size':mpqco_size,'mpqco_bitops':mpqco_bitops},f)
            
        for size_bound in np.linspace(5,10,11):
            # CLADO Way
            print(f'clado with size bound {size_bound}')
            cached_grad = Ltilde2CachedGrad(clado_ltilde)
            v = MIQCP_optimize(cached_grad=cached_grad,
                               layer_bitops=layer_bitops,
                               layer_size=layer_size,
                               schemes_per_layer=len(aw_scheme),
                               bitops_bound=np.inf,size_bound=size_bound,
                               naive=False,PSD=True)
            v = torch.Tensor(v.value)
            perf,size,bitops = evaluate_decision(v)
            clado_perf.append(perf)
            clado_size.append(size)
            clado_bitops.append(bitops)
            
            # naive Way
            print(f'naive with size bound {size_bound}')
            cached_grad = Ltilde2CachedGrad(clado_ltilde)
            v = MIQCP_optimize(cached_grad=cached_grad,
                               layer_bitops=layer_bitops,
                               layer_size=layer_size,
                               schemes_per_layer=len(aw_scheme),
                               bitops_bound=np.inf,size_bound=size_bound,
                               naive=True)
            v = torch.Tensor(v.value)
            perf,size,bitops = evaluate_decision(v)
            naive_perf.append(perf)
            naive_size.append(size)
            naive_bitops.append(bitops)
            
            # MPQCO Way
            print(f'MPQCO with size bound {size_bound}')
            v = MIQCP_optimize(cached_grad=mpqco_ltilde,
                               layer_bitops=layer_bitops,
                               layer_size=layer_size,
                               schemes_per_layer=len(aw_scheme),
                               bitops_bound=np.inf,size_bound=size_bound,
                               naive=True)
            v = torch.Tensor(v.value)
            perf,size,bitops = evaluate_decision(v)
            mpqco_perf.append(perf)
            mpqco_size.append(size)
            mpqco_bitops.append(bitops)
        
        with open(f'Ltilde_resnet18/res_{n_batch}batches(size64)_sid{sid}.pkl','wb') as f:
            pickle.dump({'shuffle':shuffle,
                         'clado_perf':clado_perf,'clado_size':clado_size,'clado_bitops':clado_bitops,
                         'naive_perf':naive_perf,'naive_size':naive_size,'naive_bitops':naive_bitops,
                         'mpqco_perf':mpqco_perf,'mpqco_size':mpqco_size,'mpqco_bitops':mpqco_bitops},f)
        
