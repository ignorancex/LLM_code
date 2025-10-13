# DONE 
import os
import json
import kaggle_utils
import torch
from tqdm import tqdm
from typing import List
import pandas as pd
from sklearn.metrics import average_precision_score
import shutil
import subprocess
from kaggle.api.kaggle_api_extended import KaggleApi
from kaggle_utils import *
import argparse
import pandas as pd


def count_map_neurons(tensor):
    # lang, dataset, layer, neurons
    nlang, ndata, nlayer, nneuron = tensor.shape
    tensor = tensor.double()
    
    result = torch.zeros(nlang, nlayer, nneuron, dtype=torch.float64)
    
    for layer in tqdm(range(nlayer)):
        for neuron in range(nneuron):
            values = []
            labels = []
    
            
            for lang in range(nlang):
                for sample in range(ndata):
                    values.append(tensor[lang, sample, layer, neuron].item())
                    labels.append(lang)  # language ID as label
    
            values = torch.tensor(values, dtype=torch.float64)
            labels = torch.tensor(labels, dtype=torch.long)
    
           
            for lang in range(nlang):
                bin_labels = (labels == lang).double()
                score = average_precision_score(bin_labels.numpy(), values.numpy())
                result[lang, layer, neuron] = score
                # print(f"lang: {lang}, layer: {layer}, neuron: {neuron}")
                # print(f"labels: {bin_labels}")
                # print(f"values: {values}")
                # print(f"scores: {score}")
    return result


def convert_processed_neurons_to_tensor(num_lang, num_layer, num_neurons_perlayer, processed_neurons):
    
    out = torch.zeros((num_lang, num_layer, num_neurons_perlayer), dtype=torch.float32)
    
    # fill it with 1s at the appropriate indices
    for i in range(num_lang):
        for j in range(num_layer):
            indices = processed_neurons[i][j]
            if indices.numel() > 0:
                out[i, j, indices] = 1.0
    return out

def convert_to_per_layer(all_layer_one_lang, num_per_layer, num_layer):
    
    bins = []
    for i in range(num_layer):
        lower = i * num_per_layer
        upper = (i + 1) * num_per_layer
        # print(f"lower: {lower}, upper: {upper}")
        
        
        mask = (all_layer_one_lang >= lower) & (all_layer_one_lang < upper)
        selected = all_layer_one_lang[mask] - lower  # remap to 0–4863
        bins.append(selected)
    return bins

def manual_filter(neuron_xwinograd, processed_neurons, num_layer):
    filtered_manual = []
    num_lang, num_data, num_neurons = neuron_xwinograd.shape 
    num_neurons_perlayer = int(num_neurons/num_layer)
    js_neurons_tensor = convert_processed_neurons_to_tensor(num_lang, num_layer, num_neurons_perlayer, processed_neurons)
    reshaped_proc_tensor = js_neurons_tensor.reshape(num_lang, num_neurons)
    per_lang_mask = reshaped_proc_tensor > 0
    all_neurons = [torch.where(per_lang_mask[i])[0] for i in range(reshaped_proc_tensor.shape[0])]
    
    for i in range(len(all_neurons)):
        neurons_i = set(all_neurons[i].numpy())
        for j in range(len(all_neurons)):
            if j != i:
                # print(f"{j}")
                neurons_j = set(all_neurons[j].numpy())
                neurons_i -= neurons_j
        tensor_i = torch.tensor(sorted(neurons_i), dtype=torch.long)
        filtered_manual.append(tensor_i)
    
    filtered_manual_perlayer = [convert_to_per_layer(filtered_manual[i], num_neurons_perlayer, num_layer) for i in range(len(filtered_manual))]
    return filtered_manual_perlayer



def get_k(scores_tensor, k, is_top):
    n_lang, n_layer, n_neuron = scores_tensor.shape
    topk_values = []
    topk_indices = []
    layer_indices = []
    neuron_indices = []
    
    # scores_flat = scores_tensor.view(n_lang, -1)
    # print(f"scores_flat.shape: {scores_flat.shape}")
    for lang_idx in range(n_lang):
        eps = 1e-8
        scores_flat = scores_tensor[lang_idx].view(-1).float()
        scores_flat = scores_flat + (torch.arange(scores_flat.numel(), dtype=torch.float32) * eps)
        values, indices = torch.topk(scores_flat, k) if is_top else torch.topk(scores_flat, k, largest=False)
        topk_values.append(values)
        topk_indices.append(indices)
    
        layers = indices // n_neuron
        neurons = indices % n_neuron
        layer_indices.append(layers)
        neuron_indices.append(neurons)
    return layer_indices, neuron_indices

def add_to_res(res, n_lang, n_layer, layer_indices, neuron_indices):
    for lang_id in range(n_lang):
        layer_idx = layer_indices[lang_id]       
        neuron_idx = neuron_indices[lang_id]     
    
        for l, n in zip(layer_idx.tolist(), neuron_idx.tolist()):
            res[lang_id][l].append(n)
            
def convert_inner_tensor(res, n_lang, n_layer):
    for lang_id in range(n_lang):
        for l in range(n_layer):
            res[lang_id][l] = torch.tensor(sorted(res[lang_id][l]))

# def get_top_bottom_k(result, k=1000):
#     scores_tensor = result.clone()
#     n_lang, n_layer, n_neuron = scores_tensor.shape
#     layer_indices, neuron_indices = get_k(scores_tensor, k, True)
#     res = [[[] for _ in range(n_layer)] for _ in range(n_lang)]  
#     add_to_res(res, n_lang, n_layer, layer_indices, neuron_indices)
#     layer_indices, neuron_indices = get_k(scores_tensor, k, False)
#     add_to_res(res, n_lang, n_layer, layer_indices, neuron_indices)
#     convert_inner_tensor(res, n_lang, n_layer)
#     return res

def get_top_bottom(tensor, k=1000, modes=[True, False]):
    result = tensor.clone().to(dtype=torch.float64)
    n_lang, n_layer, n_neuron = result.shape
    lang_idx, layer_idx, neuron_idx = torch.meshgrid(
    torch.arange(tensor.size(0)),
    torch.arange(tensor.size(1)),
    torch.arange(tensor.size(2)),
    indexing='ij'
    )

    df = pd.DataFrame({
        'lang': lang_idx.flatten().numpy(),
        'layer': layer_idx.flatten().numpy(),
        'neuron': neuron_idx.flatten().numpy(),
        'score': tensor.flatten().numpy()
    })
    lang_list = sorted(df['lang'].unique())
    lang_to_id = {lang: i for i, lang in enumerate(lang_list)}
    tb_df = [[[] for _ in range(n_layer)] for _ in range(n_lang)]
    for lang in df['lang'].unique():
        for mode in modes:
            df_lang = df[df['lang'] == lang]
            df_lang_sorted = df_lang.sort_values(['score'], ascending=(mode), kind='mergesort').head(k)

            for (l,), group in df_lang_sorted.groupby(['layer']):
                lang_id = lang_to_id[lang]
                neurons_sorted = group['neuron'].tolist()
                tb_df[lang_id][l].extend(neurons_sorted)

    for lang_id in range(n_lang):
            for l in range(n_layer):
                tb_df[lang_id][l] = torch.tensor(sorted(tb_df[lang_id][l]))
    
    return tb_df



def get_map_neurons(tensor, raw_tensor, threshold):
    map_neurons = []
    num_layer = tensor.shape[1]
    for i in range ((tensor.shape[0])):
        per_layer = []
        for j in range((tensor.shape[1])):
            per_layer.append(torch.where(tensor[i][j] > threshold)[0])
        map_neurons.append(per_layer)
    clean_langs = map_neurons.copy()
    filter_manual = manual_filter(raw_tensor, clean_langs, num_layer)
    return filter_manual
 

def map(
    in_kaggle: bool = None,
    dataset_kaggle: str = None,
    filename: str = None, 
    threshold: List[float] = None,
    num_layer: int = 24,
    model_name_inf: str = "Qwen",
    dataset_name_inf: str = "Flores",
    is_last_token: bool = False,
    max_instances: int = None,
    top_bottom_k: int = None,
    top_k: int = None,
    bottom_k: int = None,
    kaggle_dataname_to_save: str =None,
    is_update: bool = None, # is update for kaggle dataset
    parent_dir: str = None,
    result_exist: bool = False,
    data_kaggle_result: str = None,
    res_filename: str = None, # if results already exists and want to compute map threshold or top/bottomk
    subdir_name: str = ""
):

    """
    filename: filename of neuron values from kaggle. tensor is in the shape (num lang, avg/last, num_data, num_total_neurons)
    threshold: from 0.0 to 1.0, threshold for the map 
    is_last_token: using last token aggregation. default is False and use avg aggregation
    result_exist:  bool if results already exists, will prioritize this if even the tensor raw file is also provided (filename)
    """
    
    result = torch.tensor([])

    if result_exist:
        print(f"processing from existing AP tensor")
        if data_kaggle_result:
            download_from_kaggle(data_kaggle_result, res_filename)
            result = torch.load(f"data/{res_filename}", weights_only=True)
        else: result = torch.load(res_filename, weights_only=True)
    else:
        print(f"processing AP tensor from scratch")
        tensor = torch.tensor([])
        if in_kaggle:
            download_from_kaggle(dataset_kaggle, filename)
            tensor = torch.load(f"data/{filename}", weights_only=True)
        else: tensor = torch.load(filename, weights_only=True)
        if tensor.dim() == 4:
            tensor = tensor[:,0,:,:] if not is_last_token else tensor[:,1,:,:] 
        tensor = tensor[:,:max_instances,:] if max_instances else tensor
        print(tensor.shape)
        tensor_per_layer = tensor.reshape(tensor.shape[0], tensor.shape[1],num_layer, int(tensor.shape[2]/num_layer))
        result = count_map_neurons(tensor_per_layer)
    assert result.numel() != 0, "no result or tensor provided to compute map neurons"
    path_res = f"{parent_dir}res/map/{model_name_inf}_{dataset_name_inf}"
    os.makedirs(path_res, exist_ok=True)
    
    if threshold: 
        for i in threshold:
            map_neurons = get_map_neurons(result, tensor, i)
            filename = f"{path_res}/map_{i}_{model_name_inf}_{dataset_name_inf}.pt"
            torch.save(map_neurons, filename)
            print(f"saving: {filename}")
            # torch.save(map_neurons, f"{model_name_inf}_{dataset_name_inf}/map_{i}_{model_name_inf}_{dataset_name_inf}/.pt")
    torch.save(result, f"{path_res}/result_{model_name_inf}_{dataset_name_inf}.pt")
    if top_k:
        print(f"top_k: {top_k}")
        top = get_top_bottom(result, k=top_k, modes=[False])
        print("done doing top")
        filename = f"{path_res}/map_t{top_k}_{model_name_inf}_{dataset_name_inf}.pt"
        torch.save(top, filename)
        print(f"saving: {filename}")
    if bottom_k:
        print(f"bottom_k: {bottom_k}")
        bottom = get_top_bottom(result, k=bottom_k, modes=[True])
        print("done doing  bottom")
        filename = f"{path_res}/map_b{bottom_k}_{model_name_inf}_{dataset_name_inf}.pt"
        torch.save(bottom, filename)
        print(f"saving: {filename}")
    if top_bottom_k:
        print(f"top_bottom_k: {top_bottom_k}")
        top_bottom = get_top_bottom(result, k=top_bottom_k, modes=[True, False])
        print("done doing top bottom")
        filename = f"{path_res}/map_tb{top_bottom_k}_{model_name_inf}_{dataset_name_inf}.pt"
        torch.save(top_bottom, filename)
        print(f"saving: {filename}")
        
    if kaggle_dataname_to_save:
        save_to_kaggle(dataset_name=kaggle_dataname_to_save, data_dir=path_res, is_update=is_update, subdir_name=subdir_name)

def main():
    parser = argparse.ArgumentParser(description="Run the map function with specified parameters.")
    parser.add_argument("--in_kaggle", action='store_true', help="Whether the file is from Kaggle")
    parser.add_argument("--dataset_kaggle", type=str,  help="The dataset name in Kaggle, if not Kaggle then empty")
    parser.add_argument("--filename", type=str, help="The filename to process")
    parser.add_argument("--threshold", type=float, nargs="+",  help="List of threshold float values")
    parser.add_argument("--num_layer", type=int, required=True, help="Number of layers")
    parser.add_argument("--model_name_inf", type=str, required=True, help="Model name")
    parser.add_argument("--dataset_name_inf", type=str, required=True, help="Dataset name")
    parser.add_argument("--is_last_token", action='store_true', help="Set if last token")
    parser.add_argument("--max_instances", type=int, default=None, help="Maximum instances")
    parser.add_argument("--top_bottom_k", type=int, default=None, help="top bottom k")
    parser.add_argument("--top_k", type=int, default=None, help="top  k")
    parser.add_argument("--bottom_k", type=int, default=None, help=" bottom k")
    parser.add_argument("--kaggle_dataname_to_save", type=str, default=None, help="Dataset name for saving to Kaggle NO USERNAME!")
    parser.add_argument("--is_update", action='store_true', help="Flag to update Kaggle dataset")
    parser.add_argument("--parent_dir_to_save", type=str, default=None, help="Parent directory to save like /workspace for runpod")
    parser.add_argument("--result_exist", action='store_true', help="If map result already exist and wants to compute threshold/bottomtopk")
    parser.add_argument("--data_kaggle_result", type=str, default=None, help="dataset name if result_exist nd is in Kaggle dataset")
    parser.add_argument("--res_filename", type=str, default=None, help="filename that saves results")
    parser.add_argument("--subdir_name", type=str, default=None, help="subdir name to save final result")


    args = parser.parse_args()
    parent_dir = args.parent_dir_to_save if args.parent_dir_to_save else ""
    # Ensure correct parsing of 'in_kaggle' (True/False)
    in_kaggle_value = args.in_kaggle  # No need to compare to "True" or "False"
    
    # If you want to handle empty dataset for local files:
    dataset_kaggle_value = args.dataset_kaggle if args.dataset_kaggle != "" else None

    map(
        in_kaggle=in_kaggle_value,
        dataset_kaggle=dataset_kaggle_value,
        filename=args.filename,
        threshold=args.threshold,
        num_layer=args.num_layer,
        model_name_inf=args.model_name_inf,
        dataset_name_inf=args.dataset_name_inf,
        is_last_token=args.is_last_token,
        max_instances=args.max_instances,
        top_bottom_k = args.top_bottom_k,
        top_k = args.top_k,
        bottom_k = args.bottom_k,
        kaggle_dataname_to_save=args.kaggle_dataname_to_save,
        is_update=args.is_update,
        parent_dir=parent_dir,
        result_exist=args.result_exist,
        data_kaggle_result=args.data_kaggle_result,
        res_filename=args.res_filename,
        subdir_name=args.subdir_name
    )


if __name__ == "__main__":
    main()