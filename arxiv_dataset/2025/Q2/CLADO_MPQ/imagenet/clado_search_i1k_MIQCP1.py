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
torch.manual_seed(0)
np.random.seed(0)

from mltools.data import I1K
import torchvision as tv


modelname = 'resnet50'
adv_ptq = False
dataset = 'I1K'
mn = dataset.lower()+ '_' + modelname
cached_pkl_name = 'CachedGrad1kx512(2)_((8, 2), (8, 4), (8, 8))i1k_resnet50.pkl'
ds = I1K(data_dir=os.path.join('/tools/d-matrix/ml/data', "imagenet"),
         train_batch_size=64,test_batch_size=64,cuda=True)
model = eval("tv.models." + modelname)(pretrained=True).cuda()
ds.train.num_workers = 8
ds.val.num_workers = 8

train = ds.train
test = ds.val

with open(cached_pkl_name,'rb') as f:
    hm = pickle.load(f)
    
    
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
        stacked_tensor.append(img)
        calib_data.append((img,label))
        calib_fp_output.append(model(img.cuda()))
        if i == 16: # remember to modify this to get 1024 images
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
    eval(f'mqb_{b}bits_model').eval()
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
ref_metric = ('loss',evaluate(calib_data,mqb_fp_model)['loss'])

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
import torch.nn.functional as F
kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
def kldiv(quant_logit,fp_logit):
    inp = F.log_softmax(quant_logit,dim=-1)
    tar = F.softmax(fp_logit,dim=-1)
    return kl_loss(inp,tar)

def perturb_loss(perturb_scheme,ref_metric=ref_metric,
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

index2layerscheme = [None for i in range(hm['Ltilde'].shape[0])]

for name in hm['layer_index']:
    index = hm['layer_index'][name]
    layer_name = name[:-10]
    scheme = name[-10:]
    a = hm['Ltilde']
    print(f'index {index} layer {layer_name} scheme {scheme} Ltilde {a[index,index].item():.6f}')
    
    index2layerscheme[index] = (layer_name,scheme)
    
L = hm['Ltilde'].shape[0]
cached_grad = np.zeros_like(hm['Ltilde'])
for i in range(L):
    for j in range(L):
        layer_i,scheme_i = index2layerscheme[i]
        layer_j,scheme_j = index2layerscheme[j]
        if layer_i == layer_j:
            if scheme_i == scheme_j:
                cached_grad[i,j] = cached_grad[j,i] = 2 * hm['Ltilde'][i,j]
            else:
                #cached_grad[i,j] = cached_grad[j,i] = 4 * hm['Ltilde'][i,j] - hm['Ltilde'][i,i] - hm['Ltilde'][j,j]
                cached_grad[i,j] = cached_grad[j,i] = 0
        else:
            cached_grad[i,j] = cached_grad[j,i] = hm['Ltilde'][i,j] - hm['Ltilde'][i,i] - hm['Ltilde'][j,j]

class layer_hook(object):

    def __init__(self):
        super(layer_hook, self).__init__()
        self.in_shape = None
        self.out_shape = None

    def hook(self, module, inp, outp):
        self.in_shape = inp[0].size()
        self.out_shape = outp.size()
    

hooks = {}

for layer in hm['layer_index']:
    m = getModuleByName(model,layer[:-10])
    hook = layer_hook()
    hooks[layer[:-10]] = (hook,m.register_forward_hook(hook.hook))
with torch.no_grad():
    for img,label in train:
        model(img.cuda())
        break

def get_layer_bitops(layer_name,a_bits,w_bits):
    
    m = getModuleByName(model,layer_name)
    
    if isinstance(m,torch.nn.Conv2d):
        _,cin,_,_ = hooks[layer_name][0].in_shape
        _,cout,hout,wout = hooks[layer_name][0].out_shape
        
        n_muls = cin * m.weight.size()[2] * m.weight.size()[3] * cout * hout * wout
        n_accs = (cin * m.weight.size()[2] * m.weight.size()[3] - 1) * cout * hout * wout
        
        bitops_per_mul = 2 * a_bits * w_bits
        bitops_per_acc = (a_bits + w_bits) + np.ceil(np.log2(cin * m.weight.size()[2] * m.weight.size()[3]))
        
        return n_muls * bitops_per_mul + n_accs * bitops_per_acc
    
layer_size = np.array([0 for i in range(L)])
for l in hm['layer_index']:
    index = hm['layer_index'][l]
    layer_name, scheme = index2layerscheme[index]
    scheme = eval(scheme[:-4])
    layer_size[index] = torch.numel(getModuleByName(model,layer_name).weight) * int(scheme[1])

layer_bitops = []

layer_size = np.array([0 for i in range(L)])
layer_bitops = np.array([0 for i in range(L)])
for l in hm['layer_index']:
    index = hm['layer_index'][l]
    layer_name, scheme = index2layerscheme[index]
    a_bits,w_bits = eval(scheme[:-4])
    #print(layer_name,a_bits,w_bits)
    layer_size[index] = torch.numel(getModuleByName(model,layer_name).weight) * int(w_bits)
    layer_bitops[index] = get_layer_bitops(layer_name,a_bits,w_bits)

cached_grad_full = deepcopy(cached_grad)
index2layerscheme_full = deepcopy(index2layerscheme)
layer_size_full = deepcopy(layer_size)
layer_bitops_full = deepcopy(layer_bitops)

MPQ_scheme = (4,8)
aw_scheme = []
for a_bits in MPQ_scheme:
    for w_bits in MPQ_scheme:
        aw_scheme.append((a_bits,w_bits))
aw_scheme = [(8,2),(8,4),(8,8)]

del_index = []
for index in range(len(index2layerscheme_full)):
    if eval(index2layerscheme_full[index][1][:-4]) not in aw_scheme:
        del_index.append(index)
        print(f'delete {index}: {index2layerscheme_full[index][1]}')

cached_grad = np.delete(cached_grad_full,[del_index],axis=0)
cached_grad = np.delete(cached_grad,[del_index],axis=1)
index2layerscheme = np.delete(index2layerscheme_full,[del_index],axis=0)
layer_size = np.delete(layer_size_full,[del_index],axis=0)
layer_bitops = np.delete(layer_bitops_full,[del_index],axis=0)

# initialize random variable v
# use recitfied sigmoid h(v) to represent alpha
# freg is 1-(1-2h(v))**beta, annealing beta to 

if not isinstance(cached_grad,torch.Tensor):
    cached_grad = torch.Tensor(cached_grad)

layer_size_tensor = torch.Tensor(layer_size)
layer_bitops_tensor = torch.Tensor(layer_bitops)

def lossfunc(v,beta,lambda1,lambda2,printInfo=False,naive=False,b=None):
    alpha = torch.nn.Softmax(dim=1)(v.reshape(-1,len(aw_scheme))).reshape(-1,)
    
    if not naive:
        outer_alpha = torch.outer(alpha,alpha)
        netloss = torch.sum(outer_alpha * cached_grad)
    else:
        netloss = torch.sum(torch.diagonal(cached_grad) * alpha)
        
    model_size = torch.sum(layer_size_tensor * alpha)/8/1024/1024 # model size in MB
    model_bitops = torch.sum(layer_bitops_tensor * alpha)/10**9
            
    regloss = torch.sum(1-(torch.abs(1-2*alpha))**beta)
    regloss *= lambda1

    if b is None:
        closs = lambda2 * model_bitops
    else:
        closs = lambda2 * torch.clamp(model_bitops-b,0)
    
    totloss = netloss + regloss + closs
    
    if printInfo:
        print(f'netloss {netloss.item():.4f} regloss {regloss.item():.4f}(beta={beta:.4f}) closs{closs.item():.4f}(bitops: {model_bitops.item():.2f}G constraint:{b})')
        print(f'model size: {model_size.item():.4f}MB')
        print('alpha:\n',alpha)
        
    return totloss    
    
L = cached_grad.size()[0]
def optimize(n_iteration,lr,beta,lambda1,lambda2,b=None,naive=False,hardInit=False):
    
    
    #v = torch.nn.Parameter(torch.randn(L))
    v = torch.nn.Parameter(torch.zeros(L))
    if hardInit:
        v = torch.nn.Parameter(torch.zeros(L))
        with torch.no_grad():
            for i in range(L):
                if i % len(aw_scheme) == len(aw_scheme)-1:
                    v[i].data += 1.
                    
    optim = torch.optim.Adam([v,],lr=lr)
    bs = np.linspace(beta[0],beta[1],n_iteration)
    
    for i in range(n_iteration):
        if i==0 or (i+1) % 1000 == 0:
            printInfo = True
            print(f'Iter {i+1}')
        else:
            printInfo = False
            
        optim.zero_grad()
        loss = lossfunc(v,bs[i],lambda1,lambda2,printInfo=printInfo,b=b,naive=naive)
        loss.backward()
        optim.step()
    
    return v

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
        layer,scheme = index2layerscheme[scheme_id]
        decisions[layer] = eval(scheme[:-4])
    
    print("evaluate_decision\n",decisions)
    

    with torch.no_grad():
        
        # perturb layers
        perturb(decisions)
            
        # do evaluation
        res = evaluate(test,mqb_mix_model)
        
        # recover layers
        mqb_mix_model = deepcopy(mqb_fp_model)
    return res,modelsize,bitops,decisions

'''
The following code use old optimization that requires too much 
hyperparameter tuning

Now we replace it with Mixed Integer Quadratic Constraint Programming (MIQCP) Solver from CVXPY
'''
# n_iters = (5000,)
# lambda1s = (1,1e-1,0)
# lambda2s = np.logspace(-3,0,50) # for 8 x (8,4)
# sample_size = 1
# results = {}
# for n_iter in n_iters:
#     for lambda1 in lambda1s:
#         for lambda2 in lambda2s:
#             feint_loss,feint_size,feint_bitops = [],[],[]
#             trial_name = f'{aw_scheme}bits_CLADO_lambda1{lambda1}_lambda2{lambda2}_{n_iter}iters'
#             print(trial_name)
#             for repeat in range(sample_size):
#                 v = optimize(n_iteration=n_iter,lr=2e-3,beta=[20,2],lambda1=lambda1,lambda2=lambda2,naive=False)
#                 perf,size,bitops = evaluate_decision(v)
#                 feint_loss.append(perf)
#                 feint_size.append(size)
#                 feint_bitops.append(bitops)
#             results[trial_name] = {'size':feint_size,'perf':feint_loss,'bitops':feint_bitops}

# n_iters = (5000,)
# lambda1s = (1,1e-1,0)
# lambda2s = np.logspace(-3,-1,50) 
# sample_size = 1

# for n_iter in n_iters:
#     for lambda1 in lambda1s:
#         for lambda2 in lambda2s:
#             naive_loss,naive_size,naive_bitops = [],[],[]
#             print('lambda2:',lambda2)
#             trial_name = f'{aw_scheme}bits_NAIVE_lambda1{lambda1}_lambda2{lambda2}_{n_iter}iters'
#             for repeat in range(sample_size):
#                 v = optimize(n_iteration=n_iter,lr=2e-3,beta=[20,2],lambda1=lambda1,lambda2=lambda2,naive=True)
#                 perf,size,bitops = evaluate_decision(v)
#                 naive_loss.append(perf)
#                 naive_size.append(size)
#                 naive_bitops.append(bitops)
#             results[trial_name] = {'size':naive_size,'perf':naive_loss,'bitops':naive_bitops}

# with open(f'general{str(aw_scheme)}i1kresnet50results_more_more.pkl','wb') as f:
#     pickle.dump(results,f)

import cvxpy as cp

def MIQCP_optimize(cached_grad,layer_bitops,layer_size,
                   schemes_per_layer=3,
                   bitops_bound=np.inf,size_bound=np.inf,
                   naive=False):
    
    if cached_grad.__class__ == torch.Tensor:
        cached_grad = cached_grad.cpu().numpy()
    
    x = cp.Variable(cached_grad.shape[0], boolean=True)
    schemes_per_layer = schemes_per_layer
    assert cached_grad.shape[0]%schemes_per_layer == 0, 'cached_gradient shape[0] does not divde schemes per layer'
    num_layers = cached_grad.shape[0]//schemes_per_layer
    
    if not naive:
        # convexation of cached_grad
        es,us = np.linalg.eig(cached_grad)
        es[es<0] = 0
        C = us@np.diag(es)@us.T
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
    prob.solve(solver='GUROBI')
    
    # Print result.
    print("\nThe optimal value is", prob.value)
    print("A solution x is")
    print(x.value)
    #print(f"bitops: {x.value@layer_bitops}")
    return x

clado_res,naive_res = [],[]

for size_bound in np.linspace(10,20,200):
#for bitops_bound in np.linspace(400,600,200):
    #print(f'Set size bound to {size_bound} MB')
    print(f'Set size bound to {size_bound} Gops')
    
    v1 = MIQCP_optimize(cached_grad=cached_grad,
                   layer_bitops=layer_bitops,
                   layer_size=layer_size,
                   schemes_per_layer=len(aw_scheme),
                   bitops_bound=np.inf,size_bound=size_bound,
                   naive=False)
    v1 = torch.Tensor(v1.value)
    
    perf_test,size,bitops,MPQ_decision = evaluate_decision(v1)
    
    perf_calib,_,_,_ = evaluate_decision(v1,test=calib_data)
    
    clado_res.append((perf_test,perf_calib,size,bitops,MPQ_decision))
    
    v2 = MIQCP_optimize(cached_grad=cached_grad,
                   layer_bitops=layer_bitops,
                   layer_size=layer_size,
                   schemes_per_layer=len(aw_scheme),
                   bitops_bound=np.inf,size_bound=size_bound,
                   naive=True)
    v2 = torch.Tensor(v2.value)
    
    perf_test,size,bitops,MPQ_decision = evaluate_decision(v2)
    
    perf_calib,_,_,_ = evaluate_decision(v2,test=calib_data)
    
    naive_res.append((perf_test,perf_calib,size,bitops,MPQ_decision))
    
with open(f'{mn}_{cached_pkl_name[:-4]}.pkl','wb') as f:
    pickle.dump({'clado_res':clado_res,'naive_res':naive_res},f)
    
    
