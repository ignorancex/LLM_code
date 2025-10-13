import os, time
import numpy as np
np.random.seed(0)
import pickle
from copy import deepcopy

import torch
torch.manual_seed(0)
import torchvision as tv
import torch.nn.functional as F

import matplotlib.pyplot as plt
import torchvision.models as models             

from mqbench.prepare_by_platform import prepare_by_platform   # add quant nodes for specific Backend
from mqbench.prepare_by_platform import BackendType           # contain various Backend, like TensorRT, NNIE, etc.
from mqbench.utils.state import enable_calibration            # turn on calibration algorithm, determine scale, zero_point, etc.
from mqbench.utils.state import enable_quantization           # turn on actually quantization, like FP32 -> INT8
from mqbench.utils.state import disable_all           
from mqbench.advanced_ptq import ptq_reconstruction

import datetime
import time
import argparse

import presets
import torch.utils.data
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn
import utils
from coco_utils import get_coco, get_coco_api_from_dataset
from coco_eval import CocoEvaluator
from group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from engine import evaluate


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=add_help)

    parser.add_argument("--data-path", default="../data/coco/images/", type=str, help="dataset path")
    parser.add_argument("--model", default="retinanet_resnet50_fpn", type=str, help="model name")
    parser.add_argument("-b", "--batch-size", default=1, type=int, help="images per gpu")
    parser.add_argument("--num_samples", default=None, type=int, help="number of val samples")
    parser.add_argument("--resolution", default=10, type=int, help="step size on x axis for Pareto plots")
    parser.add_argument("--num_iter", default=50, type=int, help="number of update steps in optimization algo")
    parser.add_argument("-j", "--workers", default=4, type=int, help="number of data loading workers (default: 4)")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--aspect-ratio-group-factor", default=3, type=int)
    parser.add_argument("--calib_size", default=1, type=int)
    parser.add_argument("--evaluate", dest="evaluate", help="run FP32 evaluation as baseline", action="store_true")
    parser.add_argument("--kl", dest="kl", help="use KL distance for sensitivity computation", action="store_true")
    parser.add_argument("--quantize_heads", dest="quantize_heads", help="quantize conv layers in heads", action="store_true")
    parser.add_argument("--quantize_fpn", dest="quantize_fpn", help="quantize conv layers in FPN", action="store_true")
    parser.add_argument("--tag", default="", type=str, help="add this tag string to any saved filename")
    parser.add_argument("--load_hm", default=None, type=str,  help="path to precomputed heatmap file")
    parser.add_argument("--naive_results", default=None, type=str,  help="path to precomputed naive algo results file")
    parser.add_argument("--clado_results", default=None, type=str,  help="path to precomputed clado algo results file")
    
    return parser

def np_array(a):
    if isinstance(a, torch.Tensor):
        return np.array(a.detach().cpu())
    return np.array(a)


@torch.inference_mode()
def evaluate(model, data_loader, device, calib_size=None, train=False, kl=False):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    #torch.set_num_threads(1)  # TODO ???
    torch.set_num_threads(n_threads)
    cpu_device = torch.device("cpu")
    count = 0
    total_loss = 0
    if train:
        model.train()
    else:
        model.eval()

    if calib_size is not None:
        calib_data = []
        stacked_tensor = []
        calib_fp_output = []
    elif not train:
        coco = get_coco_api_from_dataset(data_loader.dataset)
        coco_evaluator = CocoEvaluator(coco, ["bbox"])
        
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # training outputs are losses, testing outputs are detections (dict)
            outputs = model(images, targets)
            
            if calib_size is not None:
                stacked_tensor.append(images)
                calib_data.append((images, targets))
                if kl:
                    # cannot use prediction scores, because a quantized model can have a different number of detected objects
                    # therefore use outputs from the last pooling layer of FPN:
                    #quant_out = mqb_mix_model(img, target)[0]['scores']
                    transformed_images, _ = model.transform(images, targets)
                    calib_values = model.backbone(transformed_images.tensors)['p7']
                    calib_fp_output.append(calib_values)
                if count == calib_size:
                    break
            else:
                if train:
                    loss = sum(loss for loss in outputs.values())
                    total_loss += loss
                else:
                    outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
                    res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
                    coco_evaluator.update(res)
 
            if count % 100 == 0:
                print(f'iteration {count}/{len(data_loader)}')
            count += 1

    if calib_size is not None:
        return stacked_tensor, calib_data, calib_fp_output
    elif train:
        return total_loss
    else:
        coco_evaluator.synchronize_between_processes()
        # accumulate predictions from all images
        coco_evaluator.accumulate()
        map_scores = coco_evaluator.summarize()
        torch.set_num_threads(n_threads)
        return 100*map_scores[0]


args = get_args_parser().parse_args()

if args.output_dir:
    utils.mkdir(args.output_dir)
    
"""
1. quantize R50 only (except conv1)
2. quantize R50 + FPN (except conv1)
3. quantize R50 + FPN + heads (except conv1 and logits in heads)
"""

print(args)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
adv_ptq = False
dataset = 'coco'
modelname = 'retinanet'
mn = dataset.lower()+ '_' + modelname
if args.kl:
    kl_str = 'kl'
else:
    kl_str = 'no_kl' 
    
if args.quantize_heads:
    head_str = 'w_heads'
else:
    head_str = 'no_heads'
    
if args.num_samples is None:
    num_samples_str = 'full'
else:
    num_samples_str = f'{args.num_samples}'
    assert args.num_samples >= args.calib_size

# dataset, num_classes = get_dataset(args.dataset, "train", get_transform(True, args), args.data_path)
# dataset_test, _ = get_dataset(args.dataset, "val", get_transform(False, args), args.data_path)
# train_sampler = torch.utils.data.RandomSampler(dataset)

# if args.aspect_ratio_group_factor >= 0:
#     group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
#     train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
# else:
#     train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)

# data_loader = torch.utils.data.DataLoader(
#     dataset, batch_sampler=train_batch_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
# )
    
print("\nLoading data")
dataset_test = get_coco(
    args.data_path, 
    image_set="val", 
    transforms=presets.DetectionPresetEval(), 
    num_samples=args.num_samples
    )

dataset_train = get_coco(
    args.data_path, 
    image_set="train", 
    transforms=presets.DetectionPresetEval(), 
    num_samples=args.num_samples
    )

print("\nCreating data loaders")
test_sampler = torch.utils.data.SequentialSampler(dataset_test)
train_sampler = torch.utils.data.SequentialSampler(dataset_train)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, 
    batch_size=args.batch_size, 
    sampler=test_sampler, 
    num_workers=args.workers, 
    collate_fn=utils.collate_fn
)

data_loader_train = torch.utils.data.DataLoader(
    dataset_train, 
    batch_size=args.batch_size, 
    sampler=train_sampler, 
    num_workers=args.workers, 
    collate_fn=utils.collate_fn
)

print("\nLoading RetinaNet model")    
model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True).to(device)

train = data_loader_train
test = data_loader_test

if args.evaluate:
    map_score_baseline = evaluate(model, test, device)
    print(f'\n\nModel accuracy (mAP score): {map_score_baseline:.2f}\n\n')

print(f'\n\n\nRunning Calibration for the first {args.calib_size} images\n\n\n')
# calibration data used to calibrate PTQ and MPQ
# to use calib_data with KL distance metric, pass train dataloader in eval mode and specify calib_size
stacked_tensor, calib_data, calib_fp_output = evaluate(
    model, 
    train, 
    device, 
    calib_size=args.calib_size, 
    kl=args.kl
    )

# configuration
ptq_reconstruction_config_init = {
    'pattern': 'block',                   #? 'layer' for Adaround or 'block' for BRECQ and QDROP
    'scale_lr': 4.0e-5,                   #? learning rate for learning step size of activation
    'warm_up': 0.2,                       #? 0.2 * max_count iters without regularization to floor or ceil
    'weight': 0.01,                       #? loss weight for regularization item
    'max_count': 1,                   #? optimization iteration
    'b_range': [20, 2],                    #? beta decaying range
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
    'b_range': [20, 2],                    #? beta decaying range
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

print(f'\nSetting up MQBench\n')

map_scores = []
MPQ_scheme = (8, 6, 4)

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
        },            
        'extra_quantizer_dict': {
            # 'additional_function_type': [operator.add,],              # additional function type, a list, use function full name, like operator.add.
            # 'additional_module_type': (torch.nn.Upsample),            # additional module type, a tuple, use class full name, like torch.nn.Upsample.
            'additional_node_name': ['body_conv1'] ,                # addition node name, a list, use full node name, like layer1_1_conv1.
            #'exclude_module_name': [layer2.1.relu,],                  # skip specific module, a list, use module qualify name, like layer2.1.relu.
            #'exclude_function_type': [operator.mul,] ,                # skip specific module, a list, use function full name, like operator.mul
            #'exclude_node_name': [layer1_1_conv1],                    # skip specific module, a list, use full node name, like layer1_1_conv1.
    },                                             
        
        # custom tracer behavior, checkout https://github.com/pytorch/pytorch/blob/efcbbb177eacdacda80b94ad4ce34b9ed6cf687a/torch/fx/_symbolic_trace.py#L836
    }
    print(f'\n\n\nPreparing {b} bits model using MQBench\n\n\n')
    # quantize R50+FPN (backbone) and classification/regression heads of RetinaNet:
    print(f'\n\nPreparing backbone: Resnet-50 and FPN module\n\n')
    mqb_fp_model.backbone = prepare_by_platform(mqb_fp_model.backbone, backend, extra_config).cuda()
    
    # FPN is part of the backbone, so it gets quantized automatically, here we revert it unless quantize_fpn arg is passed
    # assuming 10 bits of precision is equivalent to FP32
    if not args.quantize_fpn:
        fpn_bits = 10
        print(f'\n\n\n**** NOT quantizing FPN layers (reverting) ****\n\n\n')
        for m in mqb_fp_model.backbone.named_modules():
            if 'fpn' in m[0] and isinstance(m[1], torch.nn.qat.modules.conv.Conv2d):
                print(f'\n\nQuantizing layer {m[0]} weights to 8 bits\n\n\n')
                print(m[1].weight_fake_quant)
                m[1].weight_fake_quant.quant_min = -2 ** (fpn_bits - 1)
                m[1].weight_fake_quant.quant_max = 2 ** (fpn_bits - 1) - 1
                print('\n\n', m[1].weight_fake_quant, '\n')
    
    if args.quantize_heads:
        print(f'\n\n\n**** Quantizing Heads *****\n\n\n')
        print(f'\n\nPreparing Classification Head\n\n')
        mqb_fp_model.head.classification_head.conv = prepare_by_platform(mqb_fp_model.head.classification_head.conv, backend, extra_config).cuda()
        print(f'\n\nPreparing Regression Head\n\n')
        mqb_fp_model.head.regression_head.conv = prepare_by_platform(mqb_fp_model.head.regression_head.conv, backend, extra_config).cuda()
        #mqb_fp_model.head.regression_head.bbox_reg = prepare_by_platform(mqb_fp_model.head.regression_head.bbox_reg, backend, extra_config).cuda()
        #mqb_fp_model.head.classification_head.cls_logits = prepare_by_platform(mqb_fp_model.head.classification_head.cls_logits, backend, extra_config).cuda()
        
        # MQBench academic quantizer will NOT quantize first and last layers
        # we override it here by quantizing the first layer in both heads
        for m in mqb_fp_model.head.classification_head.conv.named_modules():
            if isinstance(m[1], torch.nn.qat.modules.conv.Conv2d):    # and 'p' not in m[0]:  # do not quantize pooling layers (which are conv layers with stride=2)
                print(f'\n\nQuantizing layer {m[0]} weights to {b} bits\n\n\n')
                print(m[1].weight_fake_quant)
                m[1].weight_fake_quant.quant_min = -2 ** (b - 1)
                m[1].weight_fake_quant.quant_max = 2 ** (b - 1) - 1
                print('\n\n', m[1].weight_fake_quant, '\n')
                
        for m in mqb_fp_model.head.regression_head.conv.named_modules():
            if isinstance(m[1], torch.nn.qat.modules.conv.Conv2d):  # and 'p' not in m[0]:  # do not quantize pooling layers (which are conv layers with stride=2)
                print(f'\n\nQuantizing layer {m[0]} weights to {b} bits\n\n\n')
                print(m[1].weight_fake_quant)
                m[1].weight_fake_quant.quant_min = -2 ** (b - 1)
                m[1].weight_fake_quant.quant_max = 2 ** (b - 1) - 1
                print('\n\n', m[1].weight_fake_quant, '\n')
    
    # TODO examine every layer for correct quantization attributes
    exec(f'mqb_{b}bits_model = deepcopy(mqb_fp_model)')
    
    # calibration loop
    print(f'\n\nRunning MPQ_Bench calibration loop\n')
    enable_calibration(eval(f'mqb_{b}bits_model'))
    for img, label in calib_data:
        eval(f'mqb_{b}bits_model')(img)

    if adv_ptq:
        if os.path.exists(f'QDROP_{b}bits_{mn}.pt'):
            exec(f'mqb_{b}bits_model=ptq_reconstruction(mqb_{b}bits_model, stacked_tensor, ptq_reconstruction_config_init).cuda()')
            print(f'QDROP model already saved, now loading QDROP_{b}bits_{mn}.pt')
            load_from = f'QDROP_{b}bits_{mn}.pt'
            exec(f'mqb_{b}bits_model.load_state_dict(torch.load(load_from))')
        else:
            exec(f'mqb_{b}bits_model=ptq_reconstruction(mqb_{b}bits_model, stacked_tensor, ptq_reconstruction_config).cuda()')
            print(f'saving QDROP tuned model: QDROP_{b}bits_{mn}.pt...')
            torch.save(eval(f'mqb_{b}bits_model').state_dict(), f'QDROP_{b}bits_{mn}.pt')
            
      
print(f'\n\n\nQuantizing Model\n\n\n')    
map_scores = []  
for b in MPQ_scheme: 
    disable_all(eval(f'mqb_{b}bits_model'))
    enable_quantization(eval(f'mqb_{b}bits_model'))
    print('\n\nevaluate mqb quantized model')
    map_score = evaluate(eval(f'mqb_{b}bits_model'), test, device)
    print(f'\n\nTest mAP for {b} bit model is {map_score:.2f}\n\n')
    map_scores.append(map_score)
    
if args.evaluate:
    print(f'\nFP32 baseline: {map_score_baseline:.2f}')
    
print(f'\nQuantized model results:\nquantized FPN: {args.quantize_fpn}\nquantized heads: {args.quantize_heads}\ncalib_size: {args.calib_size}\n')
for s, b in zip(map_scores, MPQ_scheme):
    print(f'{b} bits: {s:.2f}')

   
raise(SystemExit)

 
mqb_fp_model = deepcopy(mqb_8bits_model)
disable_all(mqb_fp_model)
mqb_mix_model = deepcopy(mqb_fp_model)

# 1. record all modules we want to consider
print(f'\n\n\nRecording Modules to quantize (building layer_input_map)\n\n\n')
types_to_quant = (torch.nn.Conv2d, torch.nn.Linear)

layer_input_map = {}

def getModuleByName(model, moduleName, cl=False, rg=False):
    if 'body' in moduleName or 'fpn' in moduleName:
        m = model.backbone
    elif 'cl_' in moduleName or cl:
        m = model.head.classification_head.conv
    elif 'rg_' in moduleName or rg:
        m = model.head.regression_head.conv
    else:
        m = model

    if 'cl_' in moduleName or 'rg_' in moduleName:
        moduleName = moduleName[3:]
        
    tokens = moduleName.split('.')
    for tok in tokens:
        m = getattr(m, tok)
    return m

print('\n\nR50+FPN backbone conv layers\n\n') 
for node in mqb_8bits_model.backbone.graph.nodes:
    try:
        node_target = getModuleByName(mqb_mix_model, node.target)
        if isinstance(node_target, types_to_quant):
            node_args = node.args[0]
            print('input of ', node.target, ' is ', node_args)
            layer_input_map[node.target] = str(node_args.target)
    except:
        continue
    
if args.quantize_heads:
    print('\n\nClassification head conv layers\n\n') 
    for node in mqb_8bits_model.head.classification_head.conv.graph.nodes:
        try:
            node_target = getModuleByName(mqb_mix_model, node.target, cl=True)
            if isinstance(node_target, types_to_quant):
                node_args = node.args[0]
                print('input of ', node.target, ' is ', node_args)
                # both heads have the same layer names, so we prepend str identifier, will need to remove it when using the list
                layer_input_map['cl_'+node.target] = str(node_args.target)
        except:
            continue
        
    print('\n\nRegression head conv layers\n\n') 
    for node in mqb_8bits_model.head.regression_head.conv.graph.nodes:
        try:
            node_target = getModuleByName(mqb_mix_model, node.target, rg=True)
            if isinstance(node_target, types_to_quant):
                node_args = node.args[0]
                print('input of ', node.target, ' is ', node_args)
                layer_input_map['rg_'+node.target] = str(node_args.target)
        except:
            continue
      
    # for node in mqb_8bits_model.head.regression_head.bbox_reg.graph.nodes:
    #     try:
    #         node_target = getModuleByName(mqb_mix_model.head.regression_head.bbox_reg, node.target)
    #         if isinstance(node_target, types_to_quant):
    #             node_args = node.args[0]
    #             print('input of ', node.target, ' is ', node_args)
    #             layer_input_map[node.target] = str(node_args.target)
    #     except:
    #         continue  

    ## TODO this won't work because passing a single conv layer to Tracer will break down conv op into low level ops
    # for node in mqb_8bits_model.head.classification_head.cls_logits.graph.nodes:
    #     try:
    #         node_target = getModuleByName(mqb_mix_model.head.classification_head.cls_logits, node.target)
    #         if isinstance(node_target, types_to_quant):
    #             node_args = node.args[0]
    #             print('input of ', node.target, ' is ', node_args)
    #             layer_input_map[node.target] = str(node_args.target)
    #     except:
    #         continue

print(f'\n\nlayer_input_map\n{layer_input_map}\n\n')

ref_loss = evaluate(mqb_fp_model, calib_data, device, train=True)
print(f'\n\nRef loss: {ref_loss}\n\n')

def perturb(perturb_scheme):
    # perturb_scheme: {layer_name: (act_bits, weight_bits)}
    for layer_name in perturb_scheme:
        a_bits, w_bits = perturb_scheme[layer_name]

        if w_bits is not None:
            mix_module = getModuleByName(mqb_mix_model, layer_name)
            tar_module = getModuleByName(eval(f'mqb_{w_bits}bits_model'), layer_name)
            exec(f'mix_module.weight_fake_quant = tar_module.weight_fake_quant')
            
        if a_bits is not None:
            # replace act quant to use w_bits quantization
            if 'body' in layer_name or 'fpn' in layer_name:
                exec(f'mqb_mix_model.backbone.{layer_input_map[layer_name]} = mqb_{a_bits}bits_model.backbone.{layer_input_map[layer_name]}')
            elif 'cl_' in layer_name:
                exec(f'mqb_mix_model.head.classification_head.conv.{layer_input_map[layer_name]} = mqb_{a_bits}bits_model.head.classification_head.conv.{layer_input_map[layer_name]}')
            elif 'rg_' in layer_name:
                exec(f'mqb_mix_model.head.regression_head.conv.{layer_input_map[layer_name]} = mqb_{a_bits}bits_model.head.regression_head.conv.{layer_input_map[layer_name]}')
        
        #print(layer_name)
        #print(a_cmd)
        #print(w_cmd)
        
# perturb functionality test
print(f'\n\nRun perturb functionality test')
perturb_scheme = {}
for layer_name in layer_input_map:
    perturb_scheme[layer_name] = (8, 8)
perturb(perturb_scheme)
print(f'\n\nperturbed model test mAP: {evaluate(mqb_mix_model, test, device):.2f}')

mqb_mix_model = deepcopy(mqb_8bits_model)
disable_all(mqb_mix_model)
print(f'\n\nreverted model test mAP: {evaluate(mqb_mix_model, test, device):.2f}')

def kldiv(quant_logit, fp_logit):
    kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
    inp = F.log_softmax(quant_logit, dim=-1)
    tar = F.softmax(fp_logit, dim=-1)
    return kl_loss(inp, tar)

def perturb_loss(
    perturb_scheme, 
    ref_loss=None, 
    eval_data=None, 
    printInfo=False, 
    KL=False, 
    ):
    global mqb_mix_model
    
    with torch.no_grad():
        # perturb layers
        perturb(perturb_scheme)
            
        # do evaluation
        if not KL:
            loss = evaluate(mqb_mix_model, eval_data, device, train=True)
            perturbed_loss = loss - ref_loss
        else:
            perturbed_loss = []
            mqb_mix_model.eval()
            for (data, fp_out) in zip(eval_data, calib_fp_output):
                img, target = data
                # cannot use prediction scores, because a quantized model can have a different number of detected objects
                # therefore use the last pooling layer of FPN:
                #quant_out = mqb_mix_model(img, target)[0]['scores']
                img, _ = model.transform(img, target)
                quant_out = mqb_mix_model.backbone(img.tensors)['p7']
                perturbed_loss.append(kldiv(quant_out, fp_out))
                
            perturbed_loss = torch.tensor(perturbed_loss).mean()    
        
        if printInfo:
            print(f'use kl {KL} perturbed loss {perturbed_loss}')
        
        # recover layers
        mqb_mix_model = deepcopy(mqb_fp_model)
            
    return perturbed_loss


# Build Cached Grad if not done before
print(f'\n\n\nRunning CLADO algorithm\n\n\nBuilding cached gradients\n\n\n')
del layer_input_map['body.conv1'] 
# del layer_input_map['fc']    #TODO what's the equvalent for RetinaNet (dont' quantize last layer)?

#layer_input_map = {'body.conv1': layer_input_map['body.conv1']}


if args.load_hm is None:
    s_time = time.time()
    cached = {}
    aw_scheme = []
    for a_bits in MPQ_scheme:
        for w_bits in MPQ_scheme:
            aw_scheme.append((a_bits, w_bits))

    aw_scheme = [(8, 2), (8, 4), (8, 8)]

    for n in layer_input_map:
        print(n)
        for m in layer_input_map:
            print('\t', m)
            for naw in aw_scheme:
                print('\t\t', naw)
                for maw in aw_scheme:
                    print('\t\t\t', maw)
                    if (n, m, naw, maw) not in cached:
                        if n == m:
                            if naw == maw:
                                p = perturb_loss({n:naw}, ref_loss, calib_data, KL=args.kl)
                                print(f'perturb layer {n} to A{naw[0]}W{naw[1]} p={p}')
                            else:
                                p = 0
                        else:
                            p = perturb_loss({n:naw, m:maw}, ref_loss, calib_data, KL=args.kl)
                            print(f'perturb layer {n} to A{naw[0]}W{naw[1]} and layer {m} to A{maw[0]}W{maw[1]} p={p}')
                        
                        cached[(n, m, naw, maw)] = cached[(m, n, maw, naw)] = p
                        
    print(f'{time.time()-s_time:.2f} seconds elapsed')

    layer_index = {}
    cnt = 0
    for layer in layer_input_map:
        for s in aw_scheme:
            layer_index[layer+f'{s}bits'] = cnt
            cnt += 1
    L = cnt
    print(f'\n\n\nBuilding Heat Map\n\n\n')
    hm = np.zeros(shape=(L, L))
    for n in layer_input_map:
        for m in layer_input_map:
            for naw in aw_scheme:
                for maw in aw_scheme:
                    hm[layer_index[n+f'{naw}bits'], layer_index[m+f'{maw}bits']] = cached[(n, m, naw, maw)]

    fname = f'general_a48w48_rn_coco_heatmaps_calib_{args.calib_size}_{head_str}_{kl_str}_{args.tag}.pkl'
    print(f'\n\n\nSaving Heat Maps to {fname}\n\n\n')
    hm = {'Ltilde':hm, 'layer_index':layer_index}
    with open(fname, 'wb') as f:
        pickle.dump(hm, f)
else:
    print(f'\n\n\nLoading saved Heat Maps from {args.load_hm}\n\n\n')
    with open(args.load_hm, 'rb') as f:
        hm = pickle.load(f)
    
print(hm.keys())
print(hm['layer_index'])
print(hm['Ltilde'].shape)

#perturb_loss(['conv1', ], ref_metric, eval_data=calib_data)

index2layerscheme = [None for i in range(hm['Ltilde'].shape[0])]

for name in hm['layer_index']:
    index = hm['layer_index'][name]
    layer_name = name[:-10]
    scheme = name[-10:]
    a = hm['Ltilde']
    print(f'index {index} layer {layer_name} scheme {scheme} Ltilde {a[index, index].item():.6f}')
    
    index2layerscheme[index] = (layer_name, scheme)
    
    
print(f'\n\n\nPlotting Heat Maps\n\n\n')
plt.imshow(hm['Ltilde'], cmap='hot')
if args.load_hm is not None:
    fname = args.load_hm
plt.savefig(fname+'.png')

print(f'\n\nHeat Maps:\n\n{hm}\n\n')

print(f'\n\n\nBuilding Cached Grads\n\n\n')
L = hm['Ltilde'].shape[0]
cached_grad = np.zeros_like(hm['Ltilde'])
for i in range(L):
    for j in range(L):
        layer_i, scheme_i = index2layerscheme[i]
        layer_j, scheme_j = index2layerscheme[j]
        if layer_i == layer_j:
            if scheme_i == scheme_j:
                cached_grad[i, j] = cached_grad[j, i] = 2 * hm['Ltilde'][i, j]
            else:
                #cached_grad[i, j] = cached_grad[j, i] = 4 * hm['Ltilde'][i, j] - hm['Ltilde'][i, i] - hm['Ltilde'][j, j]
                cached_grad[i, j] = cached_grad[j, i] = 0
        else:
            cached_grad[i, j] = cached_grad[j, i] = hm['Ltilde'][i, j] - hm['Ltilde'][i, i] - hm['Ltilde'][j, j]
        '''
        print(index2layerscheme[i])
        print(index2layerscheme[j])
        '''
        '''
        if i == j:
            cached_grad[i, j] = 0.5 * hm['Ltilde'][i, j]
        else:
            cached_grad[i, j] = 0.25 * (hm['Ltilde'][i, j]-hm['Ltilde'][i, i]-hm['Ltilde'][j, j])
        '''
        
# cached_grad[cached_grad<0]=0
print(f'\n\n\nPlotting Cached Grads\n\n\n')
plt.imshow(cached_grad)
print(f'\n\nCached gradients:\n\n{cached_grad}\n\n')


print(f'\n\n\nRegister Forward Hooks\n\n\n')
### Hook to record input and output shapes of layers
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
    m = getModuleByName(model, layer[:-10])
    hook = layer_hook()
    hooks[layer[:-10]] = (hook, m.register_forward_hook(hook.hook))

with torch.no_grad():
    for img, label in calib_data:
        model(img)
        break
    
def get_layer_bitops(layer_name, a_bits, w_bits):
    
    m = getModuleByName(model, layer_name)
    
    if isinstance(m, torch.nn.Conv2d):
        _, cin, _, _ = hooks[layer_name][0].in_shape
        _, cout, hout, wout = hooks[layer_name][0].out_shape
        
#         print('in', hooks[layer_name][0].in_shape)
#         print('out', hooks[layer_name][0].out_shape)
        
        n_muls = cin * m.weight.size()[2] * m.weight.size()[3] * cout * hout * wout
        n_accs = (cin * m.weight.size()[2] * m.weight.size()[3] - 1) * cout * hout * wout
        
        bitops_per_mul = 2 * a_bits * w_bits
        bitops_per_acc = (a_bits + w_bits) + np.ceil(np.log2(cin * m.weight.size()[2] * m.weight.size()[3]))
        
#         print(f'n_muls {n_muls} ops_per_mul {bitops_per_mul} totl {n_muls*bitops_per_mul}')
#         print(f'n_accs {n_accs} ops_per_acc {bitops_per_acc} totl {n_accs*bitops_per_acc}')
#         print()
        
        return n_muls * bitops_per_mul + n_accs * bitops_per_acc
    
    
layer_size = np_array([0 for i in range(L)])
for l in hm['layer_index']:
    index = hm['layer_index'][l]
    layer_name, scheme = index2layerscheme[index]
    scheme = eval(scheme[:-4])
    layer_size[index] = torch.numel(getModuleByName(model, layer_name).weight) * int(scheme[1])

layer_bitops = []

print(f'\n\n\nCompute layer bitops\n\n\n')
### Calculate sizes and numbers of bitoperations for layers under different quantization options
layer_size = np_array([0 for i in range(L)])
layer_bitops = np_array([0 for i in range(L)])
for l in hm['layer_index']:
    index = hm['layer_index'][l]
    layer_name, scheme = index2layerscheme[index]
    a_bits, w_bits = eval(scheme[:-4])
    #print(layer_name, a_bits, w_bits)
    layer_size[index] = torch.numel(getModuleByName(model, layer_name).weight) * int(w_bits)
    layer_bitops[index] = get_layer_bitops(layer_name, a_bits, w_bits)
    
    
## Modify the quantization options for layers
#modify the aw_scheme to list of (a_bits, w_bits) so that CLADO will only consider to choose from these options.
#the corresponding cached_grad, layer_size, layer_bitops will also exclude quantization options that are not considered

cached_grad_full = deepcopy(cached_grad)
index2layerscheme_full = deepcopy(index2layerscheme)
layer_size_full = deepcopy(layer_size)
layer_bitops_full = deepcopy(layer_bitops)

MPQ_scheme = (2, 4, 8)
aw_scheme = []
for a_bits in MPQ_scheme:
    for w_bits in MPQ_scheme:
        aw_scheme.append((a_bits, w_bits))

aw_scheme = [(4, 8), (8, 8), (8, 4), (4, 4)]
aw_scheme = [(8, 2), (8, 4), (8, 8)]

del_index = []
for index in range(len(index2layerscheme_full)):
    if eval(index2layerscheme_full[index][1][:-4]) not in aw_scheme:
        del_index.append(index)
        print(f'delete {index}: {index2layerscheme_full[index][1]}')

cached_grad = np.delete(cached_grad_full, [del_index], axis=0)
cached_grad = np.delete(cached_grad, [del_index], axis=1)
index2layerscheme = np.delete(index2layerscheme_full, [del_index], axis=0)
layer_size = np.delete(layer_size_full, [del_index], axis=0)
layer_bitops = np.delete(layer_bitops_full, [del_index], axis=0)

print(f'\n\n\nStart CLADO algorithm\n\n\n')
# initialize random variable v
# use recitfied sigmoid h(v) to represent alpha
# freg is 1-(1-2h(v))**beta, annealing beta to 

if not isinstance(cached_grad, torch.Tensor):
    cached_grad = torch.Tensor(cached_grad)

layer_size_tensor = torch.Tensor(layer_size)
layer_bitops_tensor = torch.Tensor(layer_bitops)

def lossfunc(v, beta, lambda1, lambda2, printInfo=False, naive=False, b=None):
    alpha = torch.nn.Softmax(dim=1)(v.reshape(-1, len(aw_scheme))).reshape(-1, )
    
    if not naive:
        outer_alpha = torch.outer(alpha, alpha)
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
        closs = lambda2 * torch.clamp(model_bitops-b, 0)
    
    totloss = netloss + regloss + closs
    
    if printInfo:
        print(f'\n\nnetloss {netloss.item():.4f} regloss {regloss.item():.4f}(beta={beta:.4f}) closs{closs.item():.4f}(bitops: {model_bitops.item():.2f}G constraint:{b})')
        print(f'model size: {model_size.item():.4f}MB')
        print('alpha:\n', alpha)
        
    return totloss 


L = cached_grad.size()[0]
def optimize(n_iteration, lr, beta, lambda1, lambda2, b=None, naive=False, hardInit=False):
    #v = torch.nn.Parameter(torch.randn(L))
    v = torch.nn.Parameter(torch.zeros(L))
    if hardInit:
        v = torch.nn.Parameter(torch.zeros(L))
        with torch.no_grad():
            for i in range(L):
                if i % len(aw_scheme) == len(aw_scheme)-1:
                    v[i].data += 1.
                    
    optim = torch.optim.Adam([v, ], lr=lr)
    bs = np.linspace(beta[0], beta[1], n_iteration)
    
    for i in range(n_iteration):
        #print('\nIteration:', i, '\n')
        if i==0 or (i+1) % 100 == 0:
            printInfo = True
            print(f'Iter {i+1}')
        else:
            printInfo = False
            
        optim.zero_grad()
        loss = lossfunc(v, bs[i], lambda1, lambda2, printInfo=printInfo, b=b, naive=naive)
        loss.backward()
        optim.step()
    
    return v

def evaluate_decision(v, printInfo=False, test=test):
    global mqb_mix_model
    v = v.detach()
    # alpha = torch.nn.Softmax(dim=1)(v.reshape(-1, len(MPQ_scheme)))
    offset = torch.ones(int(L/len(aw_scheme)), dtype=int) * len(aw_scheme)
    offset = offset.cumsum(dim=-1) - len(aw_scheme)
    select = v.reshape(-1, len(aw_scheme)).argmax(dim=1) + offset
    
    modelsize = (layer_size[select]).sum()/8/1024/1024
    bitops = (layer_bitops[select]).sum()/10**9
    
    decisions = {}
    for scheme_id in select.numpy():
        layer, scheme = index2layerscheme[scheme_id]
        decisions[layer] = eval(scheme[:-4])
    
    print("\n\nevaluate_decision\n", decisions, '\n')
    
    with torch.no_grad():
        # perturb layers
        perturb(decisions)
            
        # do evaluation
        res = evaluate(mqb_mix_model, test, device)
        
        # recover layers
        mqb_mix_model = deepcopy(mqb_fp_model)
    return res, modelsize, bitops

## Sanity Check: no constraint optimization
#Without constraint, optimization should return (ideally) an 8-bit model, or performance close to 8-bit model

# true: 1e-2 --> 1e-4
# False 1e-1 --> 1e-3

print(f'\n\n\nRunning optimization to find alphas ({args.num_iter} steps)\n\n\n')
v = optimize(n_iteration=args.num_iter, lr=2e-3, beta=[20, 2], lambda1=0, lambda2=1e-2, naive=False, hardInit=True)

evaluate_decision(v)


## Random MPQ (old code to generate random MPQ schemes)
# random_size = []
# random_acc = []
# for i in range(500):
#     v = torch.randn(L)
#     res, size = evaluate_decision(v)
#     random_size.append(size)
#     random_acc.append(res['mean_acc'])

# random_size, random_acc

# with open('resnet56_random_baseline.pkl', 'wb') as f:
#     pickle.dump({'size':random_size, 'acc':random_acc}, f)
    

# plt.hist(random_size)

print(f'\n\n\nGenerating Pareto Frontiers\n\n\n')
        
if args.clado_results is None:
    print('\n\n\nGenerating CLADO Plots\n\n')
    ## Pareto-Frontier of CLADO vs Inter-Layer Dependency Unaware Optimization (Naive)
    n_iters = (args.num_iter, )
    #lambda1s = np.logspace(-5, -3, 3)
    lambda1s = (0, 1e-1, 1)
    lambda2s = np.logspace(-3, 0, args.resolution) # for 8 x (8, 4)
    sample_size = 1
    results = {}
    for n_iter in n_iters:
        for lambda1 in lambda1s:
            print('\n\nlambda1:', lambda1)
            for lambda2 in lambda2s:
                print('\n\nlambda2:', lambda2)
                feint_map, feint_size, feint_bitops = [], [], []
                trial_name = f'{aw_scheme}bits_CLADO_lambda1{lambda1}_lambda2{lambda2}_{n_iter}iters'
                print(f'\n\ntrial_name  {trial_name}\n\n')
                for repeat in range(sample_size):
                    v = optimize(n_iteration=n_iter, lr=2e-3, beta=[20, 2], lambda1=lambda1, lambda2=lambda2, naive=False)
                    perf, size, bitops = evaluate_decision(v)
                    feint_map.append(perf)
                    feint_size.append(size)
                    feint_bitops.append(bitops)
                results[trial_name] = {'size':feint_size, 'perf':feint_map, 'bitops':feint_bitops}    
        
    fname = f'general_a48w48_rn_coco_clado_results_{args.num_iter}iters_calib_{args.calib_size}_num_samples_{num_samples_str}_{head_str}_{kl_str}_{args.tag}.pkl'   
    with open(fname, 'wb') as f:
        pickle.dump(results, f)
else:
    with open(args.clado_results, 'rb') as f:
        c248_clado = pickle.load(f)

if args.naive_results is None:
    print('\n\n\nGenerating Naive Plots\n\n')
    lambda1s = np.logspace(-5, -3, 3)
    lambda1s = (0, 1e-1, 1)
    lambda2s = np.logspace(-3, 0, args.resolution) 
    sample_size = 1

    for n_iter in n_iters:
        for lambda1 in lambda1s:
            print('\n\nlambda1:', lambda1)
            for lambda2 in lambda2s:
                naive_map, naive_size, naive_bitops = [], [], []
                print('\n\nlambda2:', lambda2)
                trial_name = f'{aw_scheme}bits_NAIVE_lambda1{lambda1}_lambda2{lambda2}_{n_iter}iters'
                print(f'\n\ntrial_name  {trial_name}\n\n')
                for repeat in range(sample_size):
                    v = optimize(n_iteration=n_iter, lr=2e-3, beta=[20, 2], lambda1=lambda1, lambda2=lambda2, naive=True)
                    perf, size, bitops = evaluate_decision(v)
                    naive_map.append(perf)
                    naive_size.append(size)
                    naive_bitops.append(bitops)
                results[trial_name] = {'size':naive_size, 'perf':naive_map, 'bitops':naive_bitops}
        
    fname = f'general_a48w48_rn_coco_naive_results_{args.num_iter}iters_calib_{args.calib_size}_num_samples_{num_samples_str}_{head_str}_{kl_str}_{args.tag}.pkl'
    with open(fname, 'wb') as f:
        pickle.dump(results, f)
else:
    with open(args.naive_results, 'rb') as f:
        c248_naive = pickle.load(f)


def getPF(xs, ys):
    xs = np_array(xs)
    ys = np_array(ys)
    
    order = np.argsort(xs)
    
    xs = xs[order]
    ys = ys[order]
    
    cur_max = -1
    for i in range(ys.shape[0]):
        if ys[i] > cur_max:
            cur_max = ys[i]
        ys[i] = cur_max
    
    return xs, ys
plt.rcParams['figure.figsize'] = (12, 8)


# clado_size, clado_acc = [], []
# naive_size, naive_acc = [], []
# for trial in c48:
#     size = c48[trial]['size']
#     perf = c48[trial]['perf']
#     perf = [x['mean_acc'] for x in perf]
#     if 'NAIVE' in trial:
#         naive_size, naive_acc = naive_size+size, naive_acc+perf
#     if 'CLADO' in trial:
#         clado_size, clado_acc = clado_size+size, clado_acc+perf 
#     #size = np_array(size)
#     #perf = np_array(perf)
#     #size, perf = getPF(size, perf)
#     #plt.plot(size, perf, label=trial)
# c48_naive_pf = getPF(np_array(naive_size), np_array(naive_acc))
# c48_clado_pf = getPF(np_array(clado_size), np_array(clado_acc))
# plt.plot(c48_naive_pf[0], c48_naive_pf[1], label='(4, 8)bits naive MPQ')
# plt.plot(c48_clado_pf[0], c48_clado_pf[1], label='(4, 8)bits clado MPQ')

clado_size, clado_acc = [], []
naive_size, naive_acc = [], []
for trial in c248:
    size = c248[trial]['size']
    perf = c248[trial]['perf']
    perf = [x['top1'] for x in perf]
    if 'NAIVE' in trial:
        naive_size, naive_acc = naive_size+size, naive_acc+perf
    if 'CLADO' in trial:
        clado_size, clado_acc = clado_size+size, clado_acc+perf 
    #size = np_array(size)
    #perf = np_array(perf)
    #size, perf = getPF(size, perf)
    #plt.plot(size, perf, label=trial)
c248_naive_pf = getPF(np_array(naive_size), np_array(naive_acc))
c248_clado_pf = getPF(np_array(clado_size), np_array(clado_acc))
plt.plot(c248_naive_pf[0], c248_naive_pf[1], label='(2, 4, 8)bits naive MPQ')
plt.plot(c248_clado_pf[0], c248_clado_pf[1], label='(2, 4, 8)bits clado MPQ')
plt.legend()

plt.xlim([0.4, 0.7])
plt.ylim([0.88, 0.95])
plt.xlabel('Hardware Cost (Model Size in MB)', fontsize=20)
plt.ylabel('Performance (Accuracy)', fontsize=20)
plt.legend()
#TODO fname
plt.savefig('rn_coco_w248.pdf', transparent=True, bbox_inches='tight', pad_inches=0)


clado_bitops, clado_acc = [], []
naive_bitops, naive_acc = [], []
for trial in c248:
    bitops = c248[trial]['bitops']
    perf = c248[trial]['perf']
    perf = [x['top1'] for x in perf]
    if 'NAIVE' in trial:
        naive_bitops, naive_acc = naive_bitops+bitops, naive_acc+perf
    if 'CLADO' in trial:
        clado_bitops, clado_acc = clado_bitops+bitops, clado_acc+perf 
    #size = np_array(size)
    #perf = np_array(perf)
    #size, perf = getPF(size, perf)
    #plt.plot(size, perf, label=trial)
c248_naive_pf = getPF(np_array(naive_bitops), np_array(naive_acc))
c248_clado_pf = getPF(np_array(clado_bitops), np_array(clado_acc))
plt.plot(c248_naive_pf[0], c248_naive_pf[1], label='naive MPQ')
plt.plot(c248_clado_pf[0], c248_clado_pf[1], label='CLADO MPQ')
plt.legend()

# plt.xlim([0.4, 0.7])
# plt.ylim([0.88, 0.95])
plt.xlabel('Hardware Cost (Model bitops in Gops)', fontsize=20)
plt.ylabel('Performance (Accuracy)', fontsize=20)
plt.legend()
# TODO fname
plt.savefig('rn_coco_a48w48.pdf', transparent=True, bbox_inches='tight', pad_inches=0)


c48_naive_size = np_array(c48['naive_size'])
c48_naive_map = c48['naive_map']
c48_feint_size = np_array(c48['feint_size'])
c48_feint_map = c48['feint_map']

c48_naive_acc = []
for i in range(len(c48_naive_map)):
    c48_naive_acc.append(c48_naive_map[i]['top1'])

c48_feint_acc = []
for i in range(len(c48_feint_map)):
    c48_feint_acc.append(c48_feint_map[i]['top1'])

c248_naive_size = np_array(c248['naive_size'])
c248_naive_map = c248['naive_map']
c248_feint_size = np_array(c248['feint_size'])
c248_feint_map = c248['feint_map']

c248_naive_acc = []
for i in range(len(c248_naive_map)):
    c248_naive_acc.append(c248_naive_map[i]['top1'])

c248_feint_acc = []
for i in range(len(c248_feint_map)):
    c248_feint_acc.append(c248_feint_map[i]['top1'])
    
    
def getPF(xs, ys):
    xs = np_array(xs)
    ys = np_array(ys)
    
    order = np.argsort(xs)
    
    xs = xs[order]
    ys = ys[order]
    
    cur_max = -1
    for i in range(ys.shape[0]):
        if ys[i] > cur_max:
            cur_max = ys[i]
        ys[i] = cur_max
    
    return xs, ys

plt.rcParams["figure.figsize"] = (12, 10)

c48_feint_size, c48_feint_acc = getPF(c48_feint_size, c48_feint_acc)

c48_naive_size, c48_naive_acc = getPF(c48_naive_size, c48_naive_acc)

c248_feint_size, c248_feint_acc = getPF(c248_feint_size, c248_feint_acc)

c248_naive_size, c248_naive_acc = getPF(c248_naive_size, c248_naive_acc)

plt.scatter(c48_naive_size, c48_naive_acc, color='lightcoral', alpha=0.5, label='c48 Inter-Layer Depedency Unaware Optimization')
plt.scatter(c48_feint_size, c48_feint_acc, color='lightblue', alpha=0.5, label='c48 FeintLady Optimization')
# plt.scatter(c248_naive_size, c248_naive_acc, color='red', alpha=0.5, label='c248 CLADO Used')
# plt.scatter(c248_feint_size, c248_feint_acc, color='blue', alpha=0.5, label='c248 CLADO Not Used')

plt.xlabel('Hardware cost')
plt.ylabel('Performance')
plt.legend()
plt.savefig('rn_coco_FeintEffecacy.pdf', transparent=True, bbox_inches='tight', pad_inches=0)


plt.scatter(naive_size, naive_acc, color='red', alpha=0.5, label='naive')
# plt.scatter(naive_size, naive_acc, color='blue', alpha=0.5, label='feint')
plt.xlabel('hardware cost')
plt.ylabel('performance')
plt.legend()
plt.show()

# Visualization
import pickle
import numpy as np
import matplotlib.pyplot as plt

fname = 'result_cifar100_shufflenetv2_x2_0_mode0_useaccFalse.pkl'
with open(fname, 'rb') as f:
    res_sfn20 = pickle.load(f)

fname = 'result_cifar100_shufflenetv2_x1_5_mode0_useaccFalse.pkl'
with open(fname, 'rb') as f:
    res_sfn15 = pickle.load(f)
    
fname = 'result_cifar100_mobilenetv2_x1_4_mode0_useaccFalse.pkl'
with open(fname, 'rb') as f:
    res_mbn14 = pickle.load(f)

fname = 'result_cifar100_mobilenetv2_x0_75_mode0_useaccFalse.pkl'
with open(fname, 'rb') as f:
    res_mbn075 = pickle.load(f)
    

fname = 'result_cifar100_resnet56_mode0_useaccFalse.pkl'
with open(fname, 'rb') as f:
    res_rsn56 = pickle.load(f)
    
    
for k in res_rsn56: print(k)

def getPF_(xs, ys, mode='max', roundtoprecision=1):
    pf = {}
    for x, y in zip(xs, ys):
        new_x = round(x, roundtoprecision)
        if new_x in pf:
            pf[new_x] = eval(mode)(pf[new_x], y)
        else:
            pf[new_x] = y
    
    pf_x, pf_y = [], []
    
    for x in pf:
        pf_x.append(x)
        pf_y.append(pf[x])
    
    pf_x, pf_y = np_array(pf_x), np_array(pf_y)
    
    return pf_x, pf_y

def getPF(xs, ys):
    xs = np_array(xs)
    ys = np_array(ys)
    
    order = np.argsort(xs)
    
    xs = xs[order]
    ys = ys[order]
    
    cur_max = -1
    for i in range(ys.shape[0]):
        if ys[i] > cur_max:
            cur_max = ys[i]
        ys[i] = cur_max
    
    return xs, ys



x_1_mbn075, y_1_mbn075 = getPF(res_mbn075['naive_size'], res_mbn075['naive_acc'])
x_2_mbn075, y_2_mbn075 = getPF(res_mbn075['feint_size'], res_mbn075['feint_acc'])

x_1_mbn14, y_1_mbn14 = getPF(res_mbn14['naive_size'], res_mbn14['naive_acc'])
x_2_mbn14, y_2_mbn14 = getPF(res_mbn14['feint_size'], res_mbn14['feint_acc'])

x_1_sfn20, y_1_sfn20 = getPF(res_sfn20['naive_size'], res_sfn20['naive_acc'])
x_2_sfn20, y_2_sfn20 = getPF(res_sfn20['feint_size'], res_sfn20['feint_acc'])

x_1_sfn15, y_1_sfn15 = getPF(res_sfn15['naive_size'], res_sfn15['naive_acc'])
x_2_sfn15, y_2_sfn15 = getPF(res_sfn15['feint_size'], res_sfn15['feint_acc'])

x_1_rsn56, y_1_rsn56 = getPF(res_rsn56['naive_size'], res_rsn56['naive_acc'])
x_2_rsn56, y_2_rsn56 = getPF(res_rsn56['feint_size'], res_rsn56['feint_acc'])

#x_random, y_random = getPF(random_size, random_acc)


# random baseline vs use/not use gradient on resnet56
# plt.rcParams['figure.figsize'] = (12, 8)
fname = 'result_cifar10_resnet56_mode0_useaccFalse.pkl'
with open(fname, 'rb') as f:
    res_rsn = pickle.load(f)
fname = 'resnet56_random_baseline.pkl'
with open(fname, 'rb') as f:
    rand_rsn = pickle.load(f)
    
    
plt.scatter(res_rsn['feint_size'][:], res_rsn['feint_acc'][:], color='blue',
            marker='o', s=20, alpha=0.5, label='Cross-layer Gradients Used')

plt.scatter(res_rsn['naive_size'][:], res_rsn['naive_acc'][:], color='red',
            marker='o', s=20, alpha=0.5, label='Cross-layer Gradients Ignored')

plt.scatter(rand_rsn['size'], rand_rsn['acc'], color='black', marker='o', s=20, alpha=0.5,
            label='Random Guess')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Hardware Cost (Model Size in MB)', fontsize=20)
plt.ylabel('Performance (Accuracy)', fontsize=20)
plt.legend()
plt.savefig('rn_coco.pdf', transparent=True, bbox_inches='tight', pad_inches=0)


plt.rcParams['figure.figsize'] = (12, 8)

plt.plot(x_1_mbn14, y_1_mbn14, color='red',
         #marker='^', markersize=3, alpha=0.5,
         linewidth=1, label='mobilenetv2_x1_4(N)')
plt.plot(x_2_mbn14, y_2_mbn14, color='blue',
         #marker='v', markersize=3, alpha=0.5,
         linewidth=1, label='mobilenetv2_x1_4(A)')

plt.plot(x_1_sfn20, y_1_sfn20, color='lightcoral',
         #marker='^', markersize=3, alpha=0.5,
         linewidth=1, label='shufflenetv2_x2_0(N)')
plt.plot(x_2_sfn20, y_2_sfn20, color='lightblue',
         #marker='v', markersize=3, alpha=0.5,
         linewidth=1, label='sufflenetv2_x2_0(A)')

plt.plot(x_1_sfn15, y_1_sfn15, color='orangered',
         #marker='^', markersize=3, alpha=0.5,
         linewidth=1, label='shufflenetv2_x1_5(N)')
plt.plot(x_2_sfn15, y_2_sfn15, color='cyan',
         #marker='v', markersize=3, alpha=0.5,
         linewidth=1, label='sufflenetv2_x1_5(A)')

plt.plot(x_1_rsn56, y_1_rsn56, color='darkred',
         #marker='^', markersize=3, alpha=0.5,
         linewidth=1, label='resnet56(N)')
plt.plot(x_2_rsn56, y_2_rsn56, color='darkblue',
         #marker='v', markersize=3, alpha=0.5,
         linewidth=1, label='resnet56(A)')


# plt.scatter(x_1, y_1, color='red', marker='^', s=10, alpha=0.5)
# plt.scatter(x_2, y_2, color='blue', marker='v', s=10, alpha=0.5)

plt.ylim([0.68, 0.76])
plt.xlim([0., 4])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Hardware Cost (Model Size in MB)', fontsize=20)
plt.ylabel('Performance (Accuracy)', fontsize=20)
plt.legend()

# plt.ylim([0.7, 0.755])
# plt.xlim([2.7, 4.0])
plt.savefig('rn_coco_pareto_3nets.pdf', transparent=True, bbox_inches='tight', pad_inches=0)