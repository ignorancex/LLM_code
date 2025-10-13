#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
# Copyright 2021 NVIDIA Corporation. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for question answering.
"""
# You can also adapt this script on your own question answering task. Pointers for this are left as comments.

from copy import deepcopy
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
from datasets import load_dataset, load_metric

import torch
import quant_trainer
import transformers
from trainer_quant_qa import QuestionAnsweringTrainer
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    QDQBertConfig,
    QDQBertForQuestionAnswering,
    TrainingArguments,
    default_data_collator,
    set_seed,
)


from transformers.trainer_utils import SchedulerType, get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from utils_qa import postprocess_qa_predictions
from functools import reduce
import cvxpy as cp

import pytorch_quantization.nn as quant_nn
#from pytorch_quantization.tensor_quant import QuantDescriptor
import pickle
import numpy as np
import random

from pytorch_quantization.tensor_quant import fake_tensor_quant


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.9.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/question-answering/requirements.txt")

logger = logging.getLogger(__name__)

model_path = "../clado_mpqco_results/"
model_name = "bert-base"

def get_calib_loss(trainer, model, calib_subset):
    if len(calib_data_set) == 0:
        model.eval()
        calib_dataloader = trainer.get_calib_dataloader(trainer.train_dataset)
        total_loss = 0
        for batch_num, inputs in enumerate(calib_dataloader):
            # Prediction step
            loss, logits, labels = trainer.prediction_step(model, inputs, prediction_loss_only=True)
            total_loss += loss
            calib_data_set.append(inputs)
            #selecting claib dataset is deterministic 
            if batch_num * calib_dataloader.batch_size >= calib_subset:
                mean_loss = total_loss / batch_num
                break
    else:
        total_loss = 0
        model.eval()
        for inputs in calib_data_set:
            # Prediction step
            loss, logits, labels = trainer.prediction_step(model, inputs, prediction_loss_only=True)
            total_loss += loss
        mean_loss = total_loss / len(calib_data_set)

    return mean_loss


def get_module_by_name(module, access_string):
    names = access_string.split(sep='.')
    return reduce(getattr, names, module)


def perturb(perturb_scheme):
    # perturb_scheme: {layer_name:(act_bits,weight_bits)}
    for layer_name in perturb_scheme:
        _, w_bits = perturb_scheme[layer_name]
        with torch.no_grad():
            # perturb layers
            tar_module = get_module_by_name(torch_mix_model, layer_name)
            src_module = get_module_by_name(eval(f'torch_q_model_{w_bits}'), layer_name)
            tar_module._weight_quantizer = src_module._weight_quantizer


def recover_perturb(perturb_scheme):
    # perturb_scheme: {layer_name:(act_bits,weight_bits)}
    for layer_name in perturb_scheme:
        with torch.no_grad():
            tar_module = get_module_by_name(torch_mix_model, layer_name)
            src_module = get_module_by_name(torch_model_fp, layer_name)
            tar_module._weight_quantizer = src_module._weight_quantizer


def perturb_loss(perturb_scheme, ref_metric, calib_subset, trainer, batch_id, printInfo=False):
    with torch.no_grad():
        # perturb layers
        perturb(perturb_scheme)

        # do evaluation
        mean_loss = get_calib_loss(trainer, torch_mix_model, calib_subset)
        perturbed_loss = mean_loss - ref_metric
        
        if printInfo:
            print("batch_id: ", batch_id)
            print("ref_metric: ", ref_metric)
            print("mean_loss: ", mean_loss)

        # recover layers
        recover_perturb(perturb_scheme)
                
        return perturbed_loss


def compute_cache_gradients(ref_metric, aw_scheme, calib_num, calib_subset, trainer, batch_num):
    cached = {}
    for n in layers_to_quant:
        for m in layers_to_quant:
            for naw in aw_scheme:
                for maw in aw_scheme:
                    if (n,m,naw,maw) not in cached:
                        if n == m:
                            if naw == maw:
                                p = perturb_loss({n: naw}, ref_metric, calib_subset, trainer, batch_num)
                                #print(f'perturb layer {n} to A{naw[0]}W{naw[1]} p={p}')
                            else:
                                p = 0

                        else:
                            p = perturb_loss({n: naw, m: maw}, ref_metric, calib_subset, trainer, batch_num)
                            #print(f'perturb layer {n} to A{naw[0]}W{naw[1]} and layer {m} to A{maw[0]}W{maw[1]} p={p}')

                        cached[(n,m,naw,maw)] = cached[(m,n,maw,naw)] = p

    layer_index = {}
    cnt = 0
    for layer in layers_to_quant:
        for s in aw_scheme:
            layer_index[layer+f'{s}bits'] = cnt
            cnt += 1
    L = cnt

    hm = np.zeros(shape=(L,L))
    for n in layers_to_quant:
        for m in layers_to_quant:
            for naw in aw_scheme:
                for maw in aw_scheme:
                    hm[layer_index[n+f'{naw}bits'],layer_index[m+f'{maw}bits']] = cached[(n,m,naw,maw)]

    file_name = model_path + model_name + '/Ltilde_' + model_name +f'/Ltilde_calib{calib_num}_batch{batch_num}(size8).pkl'
    with open(file_name, 'wb') as f:
        pickle.dump({'Ltilde':hm,'layer_index':layer_index},f)


def load_gradients(file_name):
    with open(file_name,'rb') as f:
        hm = pickle.load(f)

    L = hm['Ltilde'].shape[0]
    index2layerscheme = [None for i in range(L)]

    for name in hm['layer_index']:
        index = hm['layer_index'][name]
        if model_name == "bert-tiny":
            layer_name = name[:-10]
            scheme = name[-10:]
        elif model_name == "bert-base":
            layer_name = name[:-11]
            scheme = name[-11:]
        else:
            assert "Model choices are: bert-base and bert-tiny."     
        #print(f'index {index} layer {layer_name} scheme {scheme} Ltilde {a[index,index].item():.6f}')
        index2layerscheme[index] = (layer_name,scheme)
    
    cached_grad = np.zeros_like(hm['Ltilde'])
    for i in range(L):
        for j in range(L):
            layer_i,scheme_i = index2layerscheme[i]
            layer_j,scheme_j = index2layerscheme[j]
            if layer_i == layer_j:
                if scheme_i == scheme_j:
                    cached_grad[i,j] = cached_grad[j,i] = 2 * hm['Ltilde'][i,j]
                else:
                    cached_grad[i,j] = cached_grad[j,i] = 0
            else:
                cached_grad[i,j] = cached_grad[j,i] = hm['Ltilde'][i,j] - hm['Ltilde'][i,i] - hm['Ltilde'][j,j] 

    return hm, cached_grad, index2layerscheme


def get_layer_bitops(layer_name, a_bits, w_bits):
    
    n_muls = layers_to_quant[layer_name][0].in_features * layers_to_quant[layer_name][0].out_features
    n_accs = n_muls - 1

    bitops_per_mul = 5 * a_bits * w_bits - 5 * a_bits - 3 * w_bits + 3 
    bitops_per_acc = 3 * a_bits + 3 * w_bits + 29

    layer_bitops = n_muls * bitops_per_mul + n_accs * bitops_per_acc

    return layer_bitops


def get_layer_size_bitops(hm, index2layerscheme):
    L = hm['Ltilde'].shape[0]
    layer_size = np.array([0 for i in range(L)])
    layer_bitops = np.array([0 for i in range(L)])
    for l in hm['layer_index']:
        index = hm['layer_index'][l]
        layer_name, scheme = index2layerscheme[index]
        a_bits, w_bits = eval(scheme[:-4])
        layer_size[index] = torch.numel(layers_to_quant[layer_name][0].weight) * int(w_bits)
        layer_bitops[index] = get_layer_bitops(layer_name, a_bits, w_bits)

    return layer_size, layer_bitops

def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception


global evaluated_decision
evaluated_decision = {}

def evaluate_decision(v, L, aw_scheme, layer_size, layer_bitops, index2layerscheme, calib_subset, trainer, printInfo=False):
    v = v.detach()

    offset = torch.ones(int(L/len(aw_scheme)),dtype=int) * len(aw_scheme)
    offset = offset.cumsum(dim=-1) - len(aw_scheme)
    select = v.reshape(-1,len(aw_scheme)).argmax(dim=1) + offset
    modelsize = (layer_size[select]).sum()/8/1024/1024
    bitops = (layer_bitops[select]).sum()/10**9
    
    decisions = {}
    for scheme_id in select.numpy():
        layer, scheme = index2layerscheme[scheme_id]
        decisions[layer] = eval(scheme[:-4])
    
    if printInfo:
        print("evaluate_decision\n",decisions)

    str_dec = str(decisions)
    if str_dec in evaluated_decision:
        metrics, modelsize, bitops = evaluated_decision[str_dec]
        return metrics, modelsize, bitops, decisions

    with torch.no_grad():    
        # perturb layers
        perturb(decisions)
        
        # do evaluation
        trainer.model = deepcopy(torch_mix_model)
        metrics = trainer.evaluate()

        trainer.model = deepcopy(torch_model_fp)
    
   
    # add the decision to the dictionary for later use
    evaluated_decision[str_dec] = (metrics, modelsize, bitops)

    return metrics, modelsize, bitops, decisions


def MIQCP_optimize(cached_grad,layer_bitops,layer_size,
                   schemes_per_layer=3,
                   bitops_bound=np.inf,size_bound=np.inf,
                   naive=False):
    
    if cached_grad.__class__ == torch.Tensor:
        cached_grad = cached_grad.cpu().numpy()

    x = cp.Variable(cached_grad.shape[0], boolean=True)
    assert cached_grad.shape[0]%schemes_per_layer == 0, 'cached_gradient shape[0] does not divde schemes per layer'
    num_layers = cached_grad.shape[0]//schemes_per_layer
    
    if not naive:
        # convexation of cached_grad
        es,us = np.linalg.eig(cached_grad)
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
    #solver='GUROBI'
    #['ECOS', 'ECOS_BB', 'GUROBI', 'OSQP', 'SCIP', 'SCIPY', 'SCS']
    prob.solve(solver='GUROBI', verbose=False)
    
    # Print result.
    print("\nThe optimal value is", prob.value)
    print("A solution x is")
    print(x.value)
    return x


def estimate_deltaL(eval_data, model, start_batch, wbit_choices=[8, 4, 2]):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tot_batches = len(eval_data)
    
    processed_batches = 0
    
    print(f'MPPCO Ltilde: {processed_batches}/{tot_batches} batch of data processed')
    
    for input_batch in eval_data:
        
        deltaL = {}
        s_deltaL = {}
        e_deltaL = {}
        for layer in layers_to_quant:
            deltaL[layer] = {}
            s_deltaL[layer] = {}
            e_deltaL[layer] = {}
            for wbit in wbit_choices:
                deltaL[layer][wbit] = 0
                s_deltaL[layer][wbit] = 0
                e_deltaL[layer][wbit] = 0
        
        input_batch['input_ids'] = torch.split(input_batch['input_ids'], 1)
        input_batch['token_type_ids'] = torch.split(input_batch['token_type_ids'], 1)
        input_batch['attention_mask'] = torch.split(input_batch['attention_mask'], 1)
        input_batch['start_positions'] = torch.split(input_batch['start_positions'], 1)
        input_batch['end_positions'] = torch.split(input_batch['end_positions'], 1)

        batch_size = len(input_batch['input_ids'])
        input_data = {}
        input_dataset = [ input_data.copy() for _ in range(batch_size) ]
        for i in range(batch_size):
            input_dataset[i]['input_ids'] = input_batch['input_ids'][i].to(device)
            input_dataset[i]['token_type_ids'] = input_batch['token_type_ids'][i].to(device)
            input_dataset[i]['attention_mask'] = input_batch['attention_mask'][i].to(device)
            input_dataset[i]['start_positions'] = input_batch['start_positions'][i].to(device)
            input_dataset[i]['end_positions'] = input_batch['end_positions'][i].to(device)

        #for each batch
        for i in range(batch_size):

            model.zero_grad()
            output = model(**input_dataset[i])
            s_logits = output['start_logits']
            s_label = input_dataset[i]['start_positions']
            
            s_logits[0][s_label].backward()
            with torch.no_grad():
                for layer_name in layers_to_quant:
                    for w_bits in wbit_choices:
                        tar_module = get_module_by_name(eval(f'torch_q_model_{w_bits}'), layer_name)
                        dw = fake_tensor_quant(tar_module.weight.data, tar_module._weight_quantizer.amax, tar_module._weight_quantizer.num_bits) - tar_module.weight.data
                        dl = (dw * get_module_by_name(model, layer_name).weight.grad).sum()
                        dl /= s_logits[0][s_label][0]
                        dl = dl ** 2
                        s_deltaL[layer_name][w_bits] += dl.cpu().numpy()

            model.zero_grad()
            output = model(**input_dataset[i])
            e_logits = output['end_logits']
            e_label = input_dataset[i]['end_positions']

            e_logits[0][e_label].backward()
            with torch.no_grad():
                for layer_name in layers_to_quant:
                    for w_bits in wbit_choices:
                        tar_module = get_module_by_name(eval(f'torch_q_model_{w_bits}'), layer_name)
                        dw = fake_tensor_quant(tar_module.weight.data, tar_module._weight_quantizer.amax, tar_module._weight_quantizer.num_bits) - tar_module.weight.data
                        dl = (dw * get_module_by_name(model, layer_name).weight.grad).sum()
                        dl /= e_logits[0][e_label][0]
                        dl = dl ** 2
                        e_deltaL[layer_name][w_bits] += dl.cpu().numpy()            
        
        for layer in layers_to_quant:
            for wbit in wbit_choices:
                s_deltaL[layer][wbit] /= 2 * batch_size
                e_deltaL[layer][wbit] /= 2 * batch_size
                deltaL[layer][wbit] = 0.5 * (s_deltaL[layer][wbit] + e_deltaL[layer][wbit])
            print(f'layer name: {layer}, deltaL-8-4-2: {deltaL[layer]}')    
        
        deltaL['n_samples'] = batch_size

        batch_index = processed_batches + start_batch
        file_name = model_path + model_name + '/DeltaL_MPQCO_' + model_name + f'/MPQCO_DELTAL_batch{batch_index}(size{batch_size}).pkl'
        with open(file_name,'wb') as f:
            pickle.dump(deltaL,f)
            
        processed_batches += 1
        print(f'MPPCO Ltilde: {processed_batches}/{tot_batches} batch of data processed')
    

def feintLady(fp_model, quant_model_8, quant_model_4, quant_model_2, calib_num, calib_subset, trainer, start_batch, end_batch):
    global torch_q_model_8
    torch_q_model_8 = deepcopy(quant_model_8)
    global torch_q_model_4
    torch_q_model_4 = deepcopy(quant_model_4)
    global torch_q_model_2
    torch_q_model_2 = deepcopy(quant_model_2)

    for name, module in fp_model.named_modules():
        if name.endswith("_weight_quantizer"):
            if module._calibrator is not None:
                module.disable_quant()

    global torch_model_fp
    torch_model_fp = deepcopy(fp_model)
    global torch_mix_model
    torch_mix_model = deepcopy(fp_model)

    from collections import OrderedDict
    # 1. record all modules we want to consider
    global layers_to_quant 
    layers_to_quant = OrderedDict() # layer_name:[torch_fp_module,torch_q_module,torch_p_module]
    types_to_quant = (quant_nn.QuantLinear)

    for name, layer in torch_q_model_8.named_modules():
        if isinstance(layer, types_to_quant):
            layers_to_quant[name] = [layer, ] 

    '''
    # mean training Loss for Ltilde study
    batch_list = np.arange(1, 100, 2).tolist()
    batch_list.extend(np.arange(100, 6000, 40).tolist())
    batch_list.extend(np.arange(6000, 12500, 1000).tolist())
    mean_losses = []
    for i in batch_list:
        logger.info(f'batch_num: {i}')
        mean_losses.append(get_calib_loss(trainer, torch_model_fp, i*8))
    file_name = model_path + model_name + '_mean_loss_all-train.pkl'
    with open(file_anem,'wb') as f:
        pickle.dump({'batch_nums': batch_list, 'mean_losses': mean_losses}, f)'''

    if model_name == "bert-base":
        aw_scheme = [(16,8),(16,4),(16,2)]
    elif model_name == "bert-tiny": 
        aw_scheme = [(8,8),(8,4),(8,2)]
    
    global calib_data_set
    # DeltaL for MPQCO per batch, and Ltilde for Clado
    
    calib_data_set = []
    
    
    for i in range(start_batch, end_batch, 1):
        file_name = model_path + f'Ltilde_MPQCO_dataset/batch{i}(size8).pkl'
        with open(file_name,'rb') as f:
            calib_data_set.append(pickle.load(f))
    
    estimate_deltaL(calib_data_set, torch_model_fp, start_batch, wbit_choices=[8, 4, 2])

    for i in range(start_batch, end_batch, 1):
        file_name = model_path + f'Ltilde_MPQCO_dataset/batch{i}(size8).pkl'
        with open(file_name,'rb') as f:
            calib_data_set = [ pickle.load(f) ]
        ref_metric = get_calib_loss(trainer, torch_model_fp, calib_subset)
        compute_cache_gradients(ref_metric, aw_scheme, calib_num, calib_subset, trainer, i)
    
    
    file_name = model_path + model_name + '/Ltilde_' + model_name + f'/Ltilde_calib{calib_num}_batch0(size8).pkl'
    with open(file_name,'rb') as f:
        hm = pickle.load(f)
    ref_layer_index = hm['layer_index']

    num_batches = [2, 4, 8, 16, 32, 64, 128, 256]
    for repeat in range(0,5,1):
        for n_batch in num_batches:
            if model_name == "bert-base" and (repeat >= 3 and n_batch == 256):
                continue
            else:    
                batch_Ltildes_clado = []
                # sample size: n_batch * 8
                for batch_id in range(n_batch):
                    b = batch_id + n_batch * repeat
                    file_name = model_path + model_name + '/Ltilde_' + model_name + f'/Ltilde_calib{calib_num}_batch{b}(size8).pkl'
                    with open(file_name,'rb') as f:
                        hm = pickle.load(f)
                    assert hm['layer_index'] == ref_layer_index
                    batch_Ltildes_clado.append(hm['Ltilde'])

                assert len(batch_Ltildes_clado) == n_batch
                batch_Ltildes_clado_np = np.array(batch_Ltildes_clado)
                ref_Ltilde_clado = batch_Ltildes_clado_np.mean(axis=0)

                s_batch = n_batch * repeat
                e_batch = n_batch * repeat + n_batch - 1
                n_smaples = n_batch * 8
                file_name = model_path + model_name + '/Ltilde_' + model_name + f'/multiple_batches/sample_size{n_smaples}/Ltilde_calib{calib_num}_batches_{s_batch}-{e_batch}_bs8.pkl'
                if not os.path.exists(file_name):
                    with open(file_name, 'wb') as f:
                        pickle.dump({'Ltilde': ref_Ltilde_clado,'layer_index': ref_layer_index}, f)
                
                hm, cached_grad, index2layerscheme = load_gradients(file_name)

                batch_deltaLs_mpqco = []
                for batch_id in range(n_batch):
                    b = batch_id + n_batch * repeat
                    file_name = model_path + model_name + '/DeltaL_MPQCO_' + model_name + f'/MPQCO_DELTAL_batch{b}(size8).pkl'
                    with open(file_name,'rb') as f:
                        hm_mpqco = pickle.load(f)

                    deltal = np.zeros(ref_Ltilde_clado.shape)

                    for layer_id in range(len(index2layerscheme)):
                        layer_name, scheme = index2layerscheme[layer_id]
                        wbit = eval(scheme[:-4])[1]
                        deltal[layer_id,layer_id] = hm_mpqco[layer_name][wbit]
                    batch_deltaLs_mpqco.append(deltal)
                
                batch_deltaLs_mpqco_np = np.array(batch_deltaLs_mpqco)
                ref_deltaLs_mpqco = batch_deltaLs_mpqco_np.mean(axis=0)

                file_name = model_path + model_name + '/DeltaL_MPQCO_' + model_name + f'/multiple_batches/sample_size{n_smaples}/DeltaL_calib{calib_num}_batches_{s_batch}-{e_batch}_bs8.pkl'
                if not os.path.exists(file_name):
                    with open(file_name, 'wb') as f:
                        pickle.dump({'DeltaL': ref_deltaLs_mpqco,'layer_index': ref_layer_index}, f)

                L = hm['Ltilde'].shape[0]

                if not os.path.exists('bert-base_layer-size_layer-bitops_A16-W8-4-2.pkl'):
                    layer_size, layer_bitops = get_layer_size_bitops(hm, index2layerscheme)
                    with open('bert-base_layer-size_layer-bitops_A16-W8-4-2.pkl', 'wb') as f:
                        pickle.dump({'layer_size': layer_size, 'layer_bitops': layer_bitops}, f)     
                layer_size, layer_bitops = get_layer_size_bitops(hm, index2layerscheme)

                # random quantization of weights
                #rand_quant_res = []
                #rand_dim = L // len(aw_scheme)
                #for _ in range(200):
                #    rand_decision = np.eye(len(aw_scheme))[np.random.choice(len(aw_scheme), rand_dim)].flatten()
                #    v1 = torch.Tensor(rand_decision)
                #    perf_test, size, bitops, MPQ_decision = evaluate_decision(v1, L, aw_scheme, layer_size, layer_bitops, index2layerscheme, calib_subset, trainer) 
                #    rand_quant_res.append((perf_test['eval_f1'], perf_test['eval_exact_match'], perf_test['eval_loss'], size, bitops, MPQ_decision))
                #with open(f'rand_quant_a{acc_prec}_w8-4-2_calib{calib_num}.pkl','wb') as f:
                #    pickle.dump({'rand_quant_res': rand_quant_res}, f)
                '''
                clado_res, naive_res, mpqco_res = [], [], []
                # bert tiny: np.linspace(0.0939,0.40,30)
                # bert base: np.linspace(20.2505, 81.05, 30)
                size_bounds = np.linspace(20.2505, 81.05, 30) if model_name == "bert-base" else np.linspace(0.0939,0.40,30)
                if model_name == "bert-base":
                    file_path = model_path + model_name + f'/Clado_Naive_MPQCO_optimal_decisions/sample_size{n_smaples}/'
                    file_name = file_path + f'clado_naive_mpqco_a16_w8-4-2_calib128_batches_{s_batch}-{e_batch}_bs8_optimization.pkl'
                    with open(file_name,'rb') as f:
                        decesions = pickle.load(f)
                    clado_dec = decesions['clado_res']
                    naive_dec = decesions['naive_res']
                    mpqco_dec = decesions['mpqco_res']    
                    assert len(size_bounds) == len(clado_dec)

                for i, size_bound in enumerate(size_bounds):
                    print(f'Set size bound to {size_bound} MB')
                    #clado
                    if model_name == "bert-tiny":
                        v1 = MIQCP_optimize(cached_grad=cached_grad,
                                    layer_bitops=layer_bitops,
                                    layer_size=layer_size,
                                    schemes_per_layer=3,
                                    bitops_bound=np.inf,size_bound=size_bound,
                                    naive=False)
                        v1 = torch.Tensor(v1.value)            
                    else:  # read optimization decision from file
                        v1 = torch.Tensor(clado_dec[i]._value) 

                    perf_test, size, bitops, MPQ_decision = evaluate_decision(v1, L, aw_scheme, layer_size, layer_bitops, index2layerscheme, calib_subset, trainer)
                    clado_res.append((perf_test['eval_f1'], perf_test['eval_exact_match'], perf_test['eval_loss'], size, bitops, MPQ_decision))    
                    
                    #naive
                    if model_name == "bert-tiny":
                        v2 = MIQCP_optimize(cached_grad=cached_grad,
                                    layer_bitops=layer_bitops,
                                    layer_size=layer_size,
                                    schemes_per_layer=3,
                                    bitops_bound=np.inf,size_bound=size_bound,
                                    naive=True)
                        v2 = torch.Tensor(v2.value)            
                    else:  # read optimization decision from file      
                        v2 = torch.Tensor(naive_dec[i]._value)
                    
                    perf_test, size, bitops, MPQ_decision = evaluate_decision(v2, L, aw_scheme, layer_size, layer_bitops, index2layerscheme, calib_subset, trainer)
                    naive_res.append((perf_test['eval_f1'], perf_test['eval_exact_match'], perf_test['eval_loss'], size, bitops, MPQ_decision))

                    #mpqco
                    if model_name == "bert-tiny":
                        v3 = MIQCP_optimize(cached_grad=ref_deltaLs_mpqco,
                                    layer_bitops=layer_bitops,
                                    layer_size=layer_size,
                                    schemes_per_layer=3,
                                    bitops_bound=np.inf,size_bound=size_bound,
                                    naive=True)
                        v3 = torch.Tensor(v3.value)            
                    else:  # read optimization decision from file
                        v3 = torch.Tensor(mpqco_dec[i]._value)
                    
                    perf_test, size, bitops, MPQ_decision = evaluate_decision(v3, L, aw_scheme, layer_size, layer_bitops, index2layerscheme, calib_subset, trainer)
                    mpqco_res.append((perf_test['eval_f1'], perf_test['eval_exact_match'], perf_test['eval_loss'], size, bitops, MPQ_decision))

                acc_prec = 8 if model_name == "bert-tiny" else 16
                file_name = model_path + model_name + f'/Clado_Naive_MPQCO_res/sample_size{n_smaples}/clado_naive_mpqco_a{acc_prec}_w8-4-2_calib{calib_num}_batches_{s_batch}-{e_batch}_bs8.pkl'
                with open(file_name,'wb') as f:
                    pickle.dump({'clado_res': clado_res, 'naive_res': naive_res, 'mpqco_res': mpqco_res}, f)'''
            

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to directory to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    do_calib: bool = field(default=False, metadata={"help": "Whether to run calibration of quantization ranges."})
    do_clado: bool = field(default=False, metadata={"help": "Whether to run the caldo quantization method."})
    calib_num: int = field(
        default=128,
        metadata={"help": "Number of samples used for calibration."},
    )
    calib_subset: int = field(
        default=16, #calib_num /8 (batch_size)
        metadata={"help": "Number of samples from the calibration dataset used for clado."},
    )
    num_calib_batch: int = field(
        default=4,
        metadata={"help": "Number of batches for calibration. 0 will disable calibration "},
    )
    start_batch: int = field(
        default=0,
        metadata={"help": "index of the start batch"},
    )
    end_batch: int = field(
        default=256,
        metadata={"help": "index of the end batch"},
    )
    save_onnx: bool = field(default=False, metadata={"help": "Whether to save model to onnx."})


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. If False, will pad the samples dynamically when"
                " batching to the maximum length in the batch (which can be faster on GPU but will be slower on TPU)."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    version_2_with_negative: bool = field(
        default=False, metadata={"help": "If true, some of the examples do not have an answer."}
    )
    null_score_diff_threshold: float = field(
        default=0.0,
        metadata={
            "help": (
                "The threshold used to select the null answer: if the best answer has a score that is less than "
                "the score of the null answer minus this threshold, the null answer is selected for this example. "
                "Only useful when `version_2_with_negative=True`."
            )
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
    )
    n_best_size: int = field(
        default=20,
        metadata={"help": "The total number of n-best predictions to generate when looking for an answer."},
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": (
                "The maximum length of an answer that can be generated. This is needed because the start "
                "and end predictions are not conditioned on one another."
            )
        },
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
            and self.test_file is None
        ):
            raise ValueError("Need either a dataset name or a training/validation file/test_file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    # Set seed 
    set_seed(1023)
    torch.manual_seed(1023)
    np.random.seed(1023)
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    # quant_trainer arguments
    quant_trainer.add_arguments(parser)

    # if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    #     # If we pass only one argument to the script and it's the path to a json file,
    #     # let's parse it to get our arguments.
    #     model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    # else:

    model_args, data_args, training_args, quant_trainer_args = parser.parse_args_into_dataclasses()

    global calib_data_set
    if not os.path.exists(f'calib_dataset_list_{model_args.calib_num}.pkl'):
        #assert ("Calib dataset does not exist!") 
        calib_data_set = []
    else:
        with open(f'calib_dataset_list_{model_args.calib_num}.pkl','rb') as f:
            d_list = pickle.load(f)
        calib_data_set = d_list['calib_dataset_list']

        #calib_data_set = d_list
        #calib_data_set = []
        '''if model_args.calib_num != model_args.calib_subset:
            calib_subset = random.sample(calib_data_set, model_args.calib_subset)
            calib_data_set = calib_subset'''
        
    # setup QAT training args for scheduler (default to use cosine annealing learning rate schedule)
    training_args.lr_scheduler_type = SchedulerType.COSINE

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and \
        (training_args.do_train or model_args.do_clado) \
            and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]

        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files, field="data", cache_dir=model_args.cache_dir)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # set default quantization parameters before building model
    #quant_trainer.set_default_quantizers(quant_trainer_args)

    
    if not model_args.do_clado:
        # Load pretrained model and tokenizer
        config = QDQBertConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model = QDQBertForQuestionAnswering.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        quant_trainer.set_default_quantizers(quant_trainer_args) 
    
    elif model_args.do_clado:
        if model_name == "bert-base":
            model_path_checkpoit_fp32 = "models/fp32/bert-base-uncased"
        elif model_name == "bert-tiny": #extra 2 epochs fine-tuning for bert-tiny
            model_path_checkpoit_fp32 = "models/finetuned/prajjwal1/bert-tiny/fp32_final"
        path = model_path + model_path_checkpoit_fp32
        config = QDQBertConfig.from_pretrained(
            path,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            path,
            use_fast=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model = QDQBertForQuestionAnswering.from_pretrained(
            path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        if model_name == "bert-base":
            model_path_checkpoit = "models/calib/bert-base-uncased/percentile_calib"
            path = model_path + model_path_checkpoit + f"/w8-channel_a16-tensor_calib{model_args.calib_num}"
            quant_trainer_args.aprec = 16
            quant_trainer_args.wprec = 8
        elif model_name == "bert-tiny": #extra 2 epochs fine-tuning for bert-tiny
            path = model_path + f"models/calib/prajjwal1/bert-tiny/extra2epochs/w8_a8_calib{model_args.calib_num}"
            quant_trainer_args.aprec = 8
            quant_trainer_args.wprec = 8
        quant_trainer.set_default_quantizers(quant_trainer_args) 
        config = QDQBertConfig.from_pretrained(
            path,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            path,
            use_fast=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model_quant_8 = QDQBertForQuestionAnswering.from_pretrained(
            path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        if model_name == "bert-base":
            model_path_checkpoit = "models/calib/bert-base-uncased/percentile_calib"
            path = model_path + model_path_checkpoit + f"/w4-channel_a16-tensor_calib{model_args.calib_num}"
            quant_trainer_args.aprec = 16
            quant_trainer_args.wprec = 4
        elif model_name == "bert-tiny": #extra 2 epochs fine-tuning for bert-tiny
            path = model_path + f"models/calib/prajjwal1/bert-tiny/extra2epochs/w4_a8_calib{model_args.calib_num}"
            quant_trainer_args.aprec = 8
            quant_trainer_args.wprec = 4
        quant_trainer.set_default_quantizers(quant_trainer_args)
        config = QDQBertConfig.from_pretrained(
            path,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            path,
            use_fast=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model_quant_4 = QDQBertForQuestionAnswering.from_pretrained(
            path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        
        if model_name == "bert-base":
            model_path_checkpoit = "models/calib/bert-base-uncased/percentile_calib"
            path = model_path + model_path_checkpoit + f"/w2-channel_a16-tensor_calib{model_args.calib_num}"
            quant_trainer_args.aprec = 16
            quant_trainer_args.wprec = 2
        elif model_name == "bert-tiny": #extra 2 epochs fine-tuning for bert-tiny
            path = model_path + f"models/calib/prajjwal1/bert-tiny/extra2epochs/w2_a8_calib{model_args.calib_num}"
            quant_trainer_args.aprec = 8
            quant_trainer_args.wprec = 2
        quant_trainer.set_default_quantizers(quant_trainer_args)
        config = QDQBertConfig.from_pretrained(
            path,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            path,
            use_fast=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model_quant_2 = QDQBertForQuestionAnswering.from_pretrained(
            path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )


        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model_quant_8.to(device)
        model_quant_4.to(device)
        model_quant_2.to(device)

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models at"
            " https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet"
            " this requirement"
        )

    # Preprocessing the datasets.
    # Preprocessing is slighlty different for training and evaluation.
    if training_args.do_train or model_args.do_calib or model_args.do_clado:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval or model_args.save_onnx:
        column_names = raw_datasets["validation"].column_names
    else:
        column_names = raw_datasets["test"].column_names
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # Training preprocessing
    def prepare_train_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    if training_args.do_train or model_args.do_calib or model_args.do_clado:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            # We will select sample from whole data if agument is specified
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        # Create train feature from dataset
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                prepare_train_features,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        if data_args.max_train_samples is not None:
            # Number of samples might increase during Feature Creation, We select only specified max samples
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
            
    # Validation preprocessing
    def prepare_validation_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples["offset_mapping"]

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples


    if training_args.do_eval or model_args.save_onnx or model_args.do_clado:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_examples = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            # We will select sample from whole data
            max_eval_samples = min(len(eval_examples), data_args.max_eval_samples)
            eval_examples = eval_examples.select(range(max_eval_samples))
        # Validation Feature Creation
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_examples.map(
                prepare_validation_features,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
        if data_args.max_eval_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_examples = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            # We will select sample from whole data
            predict_examples = predict_examples.select(range(data_args.max_predict_samples))
        # Predict Feature Creation
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_examples.map(
                prepare_validation_features,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
        if data_args.max_predict_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    # Data collator
    # We have already padded to max length if the corresponding flag is True, otherwise we need to pad in the data
    # collator.
    data_collator = (
        default_data_collator
        if data_args.pad_to_max_length
        else DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
    )

    # Post-processing:
    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=data_args.version_2_with_negative,
            n_best_size=data_args.n_best_size,
            max_answer_length=data_args.max_answer_length,
            null_score_diff_threshold=data_args.null_score_diff_threshold,
            output_dir=training_args.output_dir,
            log_level=log_level,
            prefix=stage,
        )
        # Format the result to the format the metric expects.
        if data_args.version_2_with_negative:
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

        references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    metric = load_metric("squad_v2" if data_args.version_2_with_negative else "squad")
    
    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    # Initialize our Trainer
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train or model_args.do_calib else None,
        eval_dataset=eval_dataset if training_args.do_eval or model_args.save_onnx else None,
        eval_examples=eval_examples if training_args.do_eval or model_args.save_onnx else None,
        calib_num = model_args.calib_num,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
        quant_trainer_args=quant_trainer_args,
    )

    # Clado quantization
    if model_args.do_clado:                
        trainer_clado = QuestionAnsweringTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if model_args.do_clado else None,
            eval_dataset=eval_dataset if model_args.do_clado else None,
            eval_examples=eval_examples if model_args.do_clado else None,
            calib_num = model_args.calib_num,
            tokenizer=tokenizer,
            post_process_function=post_processing_function,
            compute_metrics=compute_metrics,
            quant_trainer_args=quant_trainer_args,
        )
        calib_subset_size = model_args.calib_subset * 8 #(batch_size)
        if model_args.do_clado:
            feintLady(model, model_quant_8, model_quant_4, model_quant_2, model_args.calib_num, calib_subset_size, trainer_clado, model_args.start_batch, model_args.end_batch)

    # Calibration
    if model_args.do_calib:
        logger.info("*** Calibrate ***")
        results = trainer_clado.calibrate()
        trainer.save_model()

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        quant_trainer.configure_model(trainer.model, quant_trainer_args)

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        quant_trainer.configure_model(trainer.model, quant_trainer_args, eval=True)
        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        results = trainer.predict(predict_dataset, predict_examples)
        metrics = results.metrics

        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

    if training_args.push_to_hub:
        kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "question-answering"}
        if data_args.dataset_name is not None:
            kwargs["dataset_tags"] = data_args.dataset_name
            if data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = data_args.dataset_config_name
                kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
            else:
                kwargs["dataset"] = data_args.dataset_name

        trainer.push_to_hub(**kwargs)

    if model_args.save_onnx:
        logger.info("Exporting model to onnx")
        results = trainer.save_onnx(output_dir=training_args.output_dir)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
