import torch,torchvision,os,time,pickle
import torchvision.transforms as transforms
import numpy as np
from utils.util import get_loader,evaluate
from utils.layer import qConv2d,qLinear
from utils.train import QAVAT_train
import matplotlib.pyplot as plt
import torchvision.models as models                           # for example model
from mqbench.prepare_by_platform import prepare_by_platform   # add quant nodes for specific Backend
from mqbench.prepare_by_platform import BackendType           # contain various Backend, like TensorRT, NNIE, etc.
from mqbench.utils.state import enable_calibration            # turn on calibration algorithm, determine scale, zero_point, etc.
from mqbench.utils.state import enable_quantization           # turn on actually quantization, like FP32 -> INT8
from mqbench.utils.state import disable_all           # turn on actually quantization, like FP32 -> INT8
from mqbench.advanced_ptq import ptq_reconstruction
from copy import deepcopy
import argparse
from collections import OrderedDict

def parse_args():
    parser = argparse.ArgumentParser(description='FeintLady Mixed Precision Quantization Search Algorithm')
    parser.add_argument("-ds", "--dataset", type = str, default='cifar10', help = "dataset")
    parser.add_argument("-m", "--model", type = str, default='resnet20', help = "model")
    parser.add_argument("--use_acc", action = 'store_true', default=False, 
                        help = "use acc as objective in FeintLady, default: false (use loss as objective)")
    parser.add_argument("--ptq_mode", type = int, default='0', help = "0: adaround, 1: adaround, 2: brecq")
    parser.add_argument("--l1",type=float,default=1e-3,help = "binary regularization coefficient")
    parser.add_argument("--l2",type=float,default=1e-5,help = "(search from this value to 0)harware cost regularization coefficient")
    parser.add_argument("--nl1",type=int,default=3,help = "number of points to interpolate l1 search")
    parser.add_argument("--nl2",type=int,default=50,help = "number of points to interpolate l2 search")
    
    return parser.parse_args()

args = parse_args()

def getModuleByName(modelName,moduleName):
    '''
        replace module with name modelName.moduleName with newModule
    '''
    tokens = moduleName.split('.')
    eval_str = modelName
    for token in tokens:
        try:
            eval_str += f'[{int(token)}]'
        except:
            eval_str += f'.{token}'
            
    return eval(eval_str)

# model and dataset prep
model = torch.hub.load("chenyaofo/pytorch-cifar-models", args.dataset.lower()+'_'+args.model, pretrained=True).cuda()
train,test = get_loader(args.dataset.upper(),batch_size=128,test_batch_size=128)
train.num_workers = test.num_workers = 2
train.pin_in_memory = test.pin_in_memory = True

# use 1024 calibration data to calibrate PTQ and MPQ
calib_data = []
i = 0
for img,label in train:
    i += 1
    calib_data.append((img,label))
    if i == 8:
        break

model.eval()
mqb_model = deepcopy(model)
torch_fp_model = deepcopy(model)
torch_quantized_model = deepcopy(model)
torch_perturb_model = deepcopy(model)
model = None

# MSE calibration on model parameters
backend = BackendType.Academic
extra_config = {
    'extra_qconfig_dict': {
        'w_observer': 'MSEObserver',                          
        'a_observer': 'EMAMSEObserver',                              
        'w_fakequantize': 'FixedFakeQuantize' if args.ptq_mode == 0 else 'AdaRoundFakeQuantize',                   
        'a_fakequantize': 'FixedFakeQuantize' if args.ptq_mode == 1 else 'QDropFakeQuantize',                    
        'w_qscheme': {
            'bit': 4,                                             
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
mqb_model = prepare_by_platform(mqb_model, backend,extra_config).cuda()

# calibration loop
enable_calibration(mqb_model)
for img,label in calib_data:
    mqb_model(img.cuda()) 
    
if not args.ptq_mode == 0:
    ptq_rec_config = {'pattern':'layer' if args.ptq_mode == 1 else 'block', 
                      'scale_lr': 4.0e-5, 'warm_up':0.2, 'weight':0.01,
                      'max_count':20000,'b_range':[20,2],'keep_gpu': True,
                      'round_mode':'learned_hard_sigmoid','prob':1}
    class dotdict(dict):
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__
    ptq_rec_config = dotdict(ptq_rec_config)
    
    mqb_model = ptq_reconstruction(mqb_model,calib_data,ptq_rec_config)

print('fully quantized model:')
enable_quantization(mqb_model)
print(evaluate(test,mqb_model))
print('full precision model:')
print(evaluate(test,torch_fp_model))


# pass quantized weight to torch_quantized_model
for n,m in mqb_model.named_modules():
    if isinstance(m,torch.nn.Linear) or isinstance(m,torch.nn.Conv2d):
        # print('loading quantized weight for layer',n)
        torch_module = getModuleByName('torch_quantized_model',n)
        torch_module.weight.data = m.weight_fake_quant(m.weight).data
mqb_model = None


# 1. record all modules we want to consider
layers_to_quant = OrderedDict() # layer_name:[torch_fp_module,torch_q_module,torch_p_module]
types_to_quant = (torch.nn.Conv2d,torch.nn.Linear)

for n,m in torch_fp_model.named_modules():
    if isinstance(m,types_to_quant):
        layers_to_quant[n] = [m,]
        
for n,m in torch_quantized_model.named_modules():
    if isinstance(m,types_to_quant):
        layers_to_quant[n].append(m)

for n,m in torch_perturb_model.named_modules():
    if isinstance(m,types_to_quant):
        layers_to_quant[n].append(m)

res = evaluate(calib_data,torch_fp_model)
ref_metric = ('mean_loss' if not args.use_acc else 'mean_acc',
              res['mean_loss' if not args.use_acc else 'mean_acc'])

def perturb_loss(perturb_names,ref_metric,eval_data,printInfo=False):
    with torch.no_grad():
        # perturb layers
        for n in perturb_names:
            layers_to_quant[n][2].weight.data = layers_to_quant[n][1].weight.data
        # do evaluation
        res = evaluate(eval_data,torch_perturb_model)
        perturbed_loss = res[ref_metric[0]] - ref_metric[1]
        
        if printInfo:
            print(res)
        # recover layers
        for n in perturb_names:
            layers_to_quant[n][2].weight.data = layers_to_quant[n][0].weight.data
    return perturbed_loss

if not os.path.exists(f'feintlady_{args.dataset}_{args.model}_mode{args.ptq_mode}_useacc{args.use_acc}.pkl'):
    print('gradient caching')
    s_time = time.time()
    cached = {}
    for n in layers_to_quant:
        for m in layers_to_quant:
            if n == m:
                print('perturb layer',n)
                p = perturb_loss([n,],ref_metric,calib_data)
                cached[(n,n)] = p
            if (n,m) not in cached:
                print('perturb layer',n,m)
                p = perturb_loss([n,m],ref_metric,calib_data)
                cached[(n,m)] = p
                cached[(m,n)] = p   
    print(f'{time.time()-s_time:.2f} seconds elapsed')

    layer_index = {}
    cnt = 0
    for layer in layers_to_quant:
        layer_index[layer] = cnt
        cnt += 1
    L = cnt


    hm = np.zeros(shape=(L,L))
    for n in layers_to_quant:
        for m in layers_to_quant:
            hm[layer_index[n],layer_index[m]] = cached[(n,m)]


    with open(f'feintlady_{args.dataset}_{args.model}_mode{args.ptq_mode}_useacc{args.use_acc}.pkl','wb') as f:
        pickle.dump({'Ltilde':hm,'layer_index':layer_index},f)

else:
    print('use cached gradient from')
    print(f'feintlady_{args.dataset}_{args.model}_mode{args.ptq_mode}_useacc{args.use_acc}.pkl')
    
with open(f'feintlady_{args.dataset}_{args.model}_mode{args.ptq_mode}_useacc{args.use_acc}.pkl','rb') as f:
    hm = pickle.load(f)

L = hm['Ltilde'].shape[0]
cached_grad = np.zeros_like(hm['Ltilde'])
for i in range(L):
    for j in range(L):
        if i == j:
            cached_grad[i,j] = 0.5 * hm['Ltilde'][i,j]
        else:
            cached_grad[i,j] = 0.25 * (hm['Ltilde'][i,j]-hm['Ltilde'][i,i]-hm['Ltilde'][j,j])
# model size as cost constraint
layer_size = np.array([0 for i in range(L)])
for l in hm['layer_index']:
    layer_size[hm['layer_index'][l]] = torch.numel(layers_to_quant[l][0].weight)
    
if not isinstance(cached_grad,torch.Tensor):
    cached_grad = torch.Tensor(cached_grad)

layer_size_tensor = torch.Tensor(layer_size)

def lossfunc(v,beta,lambda1,lambda2,printInfo=False,naive=False):
    alpha = (torch.sigmoid(v) * 1.2 - 0.1).clamp(0,1)
    if not naive:
        
        outer_alpha = torch.outer(alpha,alpha)
        netloss = torch.sum(outer_alpha * cached_grad)
    else:
        netloss = torch.sum(torch.diagonal(cached_grad) * alpha)
            
    regloss = torch.sum(1-(torch.abs(1-2*alpha))**beta)
    regloss *= lambda1
    
    
    closs = torch.sum(layer_size_tensor * alpha)
    closs *= -lambda2
    
    totloss = netloss + regloss + closs
    
    if printInfo:
        print(f'netloss {netloss.item():.4f} regloss {regloss.item():.4f}(beta={beta:.4f}) closs{closs.item():.4f}')
        print('alpha:\n',alpha)
        
    return totloss    

def optimize(n_iteration,lr,beta,lambda1,lambda2,naive=False):
    
    v = torch.nn.Parameter(torch.randn(L))
    optim = torch.optim.Adam([v,],lr=lr)
    bs = np.linspace(beta[0],beta[1],n_iteration)
    
    for i in range(n_iteration):
        if i==0 or (i+1) % 1000 == 0:
            printInfo = True
            print(f'Iter {i+1}')
        else:
            printInfo = False
            
        optim.zero_grad()
        loss = lossfunc(v,bs[i],lambda1,lambda2,printInfo=printInfo,naive=naive)
        loss.backward()
        optim.step()
    
    return v

def evaluate_decision(v,printInfo=False):
    v = v.detach()
    v[v>0] = 1
    v[v<0] = 0
    
    decision = []
    for layer in hm['layer_index']:
        if v[hm['layer_index'][layer]] > 0:
            decision.append(layer)
    
    
    #ploss = perturb_loss(decision,ref_metric,eval_data=test,printInfo=printInfo)
    with torch.no_grad():
        # perturb layers
        for n in decision:
            layers_to_quant[n][2].weight.data = layers_to_quant[n][1].weight.data
        # do evaluation
        res = evaluate(test,torch_perturb_model)
        # recover layers
        for n in decision:
            layers_to_quant[n][2].weight.data = layers_to_quant[n][0].weight.data
    
    orig_size = 0
    reduced_size = 0
    for i in range(L):
        orig_size += layer_size[i] * 8
        reduced_size += layer_size[i] * 8 if v[i] == 0 else layer_size[i] * 4
    if printInfo:
        print('result of quantizing the following layers')
        print(decision)
        print(res)
        print(f'8-bit model size {orig_size/8/1024/1024:.2f} MB')
        print(f'MP model size {reduced_size/8/1024/1024:.2f} MB') 
    
    return res,reduced_size/8/1024/1024

v = optimize(n_iteration=5000,lr=1e-3,beta=[20,2],lambda1=1e-3,lambda2=0,naive=False)
evaluate_decision(v,printInfo=True)
# fix lambda1 to 1e-3
# search lambda2 from 1e-5 to 1e-10
# each HP (lambda2) try 5 runs

lambda1s = np.logspace(np.log10(args.l1),-1,args.nl1) #lambda1=1e-3,n=5000,lr=1e-3,beta=[20,2] for resnet20 on cifar10
lambda2s = np.logspace(np.log10(args.l2),-10,args.nl2) #lambda1=1e-1,n=5000,lr=1e-3,beta=[20,2] for resnet20 on cifar100
sample_size = 5
naive_loss,naive_size = [],[]
feint_loss,feint_size = [],[]
config = []
for lambda1 in lambda1s:
    for lambda2 in lambda2s:
        print(f'lambda1 {lambda1} lambda2 {lambda2}')
        for repeat in range(sample_size):
            v = optimize(n_iteration=5000,lr=1e-3,beta=[20,2],lambda1=lambda1,lambda2=lambda2,naive=True)
            perf,size = evaluate_decision(v)
            naive_loss.append(perf)
            naive_size.append(size)

            v = optimize(n_iteration=5000,lr=1e-3,beta=[20,2],lambda1=lambda1,lambda2=lambda2,naive=False)
            perf,size = evaluate_decision(v)
            feint_loss.append(perf)
            feint_size.append(size)
            config.append((lambda1,lambda2))

naive_size = np.array(naive_size)
feint_size = np.array(feint_size)
naive_acc = [naive_loss[i]['mean_acc'] for i in range(len(naive_loss))]
feint_acc = [feint_loss[i]['mean_acc'] for i in range(len(feint_loss))]

with open(f'result_{args.dataset}_{args.model}_mode{args.ptq_mode}_useacc{args.use_acc}.pkl','wb') as f:
    pickle.dump({'naive_size':naive_size,'naive_acc':naive_acc,
                 'feint_size':feint_size,'feint_acc':feint_acc,'config':config},f)
    
plt.scatter(naive_size,naive_acc,color='red',alpha=0.5,label='Inter-Layer Depedency Unaware Optimization')
plt.scatter(feint_size,feint_acc,color='blue',alpha=0.5,label='FeintLady Optimization')
plt.xlabel('Hardware cost')
plt.ylabel('Performance')
plt.legend()
plt.savefig(f'{args.dataset}_{args.model}_mode{args.ptq_mode}_useacc{args.use_acc}.pdf',
            transparent=True, bbox_inches='tight', pad_inches=0)
