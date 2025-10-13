langs_code = {
    "en": "eng_Latn",
    "fr": "fra_Latn",
    "jp": "jpn_Jpan",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
    "zh": "zho_Hans",
    "nl": "nld_Latn",
    "id": "ind_Latn",
    "ms": "zsm_Latn",
    "vi": "vie_Latn",
    "et":"est_Latn" ,
    "it": "ita_Latn",
    "ta": "tam_Taml",
    "th": "tha_Thai",
    "tr": "tur_Latn",
    "ht":"hat_Latn",
    "qu": "quy_Latn",
    "sw": "swh_Latn"  
} 
langs_code_rev = {v:k for k,v in langs_code.items()}

import numpy as np
from scipy.stats import entropy
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def get_k_lang_actv_dict(k, full_neurons, method="default", topk=0):
    
    """
    di sini anggap activated neuron (avg token) yang di setiap row di dataset(cuman 2 row) actv valuenya > 0.
    k = num language"""
    activation_dict = {}
    
    full_neurons = full_neurons.transpose(-1,-2)

    if method == "default":
        for i in range (full_neurons.size(0)):
            tensor_lang = full_neurons[i]
            rows_with_both_positive = (tensor_lang > 0).all(dim=-1)
            
            indices = torch.where(rows_with_both_positive)[0]
            activation_dict[i] = indices
            # indices.shape
            # print(indices.shape)
    elif method == "topk":
        if topk==0:
            print(f"topk must not be 0")
        top = (full_neurons.mean(dim=-1).topk(topk).indices)
        for i in range (full_neurons.size(0)):
            activation_dict[i] = top[i]
    # print(activation_dict)
    return activation_dict

# Step 1: Convert lists to probability distributions
def to_probability_distribution(values):
    total = sum(values)
    return [v / total for v in values]


def make_heatmap_neuron_overlap(activation_dict, k, with_label=True, method="default", alpha=1, with_title=False, normalized =False, lang_dict =None, save=False, save_name="overlap_heatmap", modelname="", annot_kws=12):
    # Example dictionary: keys 0-52, values are 1D tensors of activated neuron indices
    # activation_dict = get_k_lang_actv_dict(10)

    overlap_matrix = torch.tensor([])
    if method == "default":
        # Step 1: Create a binary matrix
        max_neuron_index = max(max(indices) for indices in activation_dict.values()) + 1  # Find the maximum neuron index
        binary_matrix = torch.zeros((k, max_neuron_index), dtype=torch.int)  # Initialize binary matrix
        
        for key, indices in activation_dict.items():
            binary_matrix[key, indices] = 1  # Set activated neurons to 1
        
        # Step 2: Compute overlaps (dot product between rows)
        overlap_matrix = torch.matmul(binary_matrix, binary_matrix.T)  # Dot product of binary_matrix with its transpose

    elif method == "jaccard":
        max_neuron_index = max(max(indices) for indices in activation_dict.values()) + 1
        binary_matrix = torch.zeros((k, max_neuron_index), dtype=torch.int)
    
        # Fill binary matrix with activation data
        for key, indices in activation_dict.items():
            binary_matrix[key, indices] = 1  
    
        # Compute Jaccard distance matrix
        overlap_matrix = torch.zeros((k, k))
    
        for i in range(k):
            for j in range(k):
                intersection = (binary_matrix[i] & binary_matrix[j]).sum().item()
                union = (binary_matrix[i] | binary_matrix[j]).sum().item()
                jaccard_similarity = intersection / union if union > 0 else 0
                overlap_matrix[i, j] = jaccard_similarity
        overlap_matrix = overlap_matrix ** alpha
        if normalized:
            overlap_matrix = overlap_matrix / overlap_matrix.sum(axis=1, keepdims=True)

    save_name = f"{save_name}_{method}_{alpha}_{modelname}"
    # Step 3: Visualize the heatmap
    plt.figure(figsize=(18, 18))
    if with_label:
        if lang_dict:
            sns.heatmap(overlap_matrix.numpy(), annot=True, fmt=".1f", annot_kws={"size": annot_kws, "weight": "bold"}, cmap="YlOrRd", linewidths=0.3,cbar =False,
                        xticklabels=[langs_code_rev[lang_dict[i]] for i in range(k)], yticklabels=[langs_code_rev[lang_dict[i]] for i in range(k)])
        else:
            sns.heatmap(overlap_matrix.numpy(), annot=True, fmt=".1f", annot_kws={"size": annot_kws, "weight": "bold"}, cmap="YlOrRd", linewidths=0.3,cbar =False,
                        xticklabels=(range(k)), yticklabels=(range(k)))
        plt.xticks(fontsize=annot_kws+6)
        plt.yticks(fontsize=annot_kws+6)
        
    else:
        sns.heatmap(overlap_matrix.numpy(), fmt=".1f", cmap="YlOrRd",cbar=True)
    # plt.xlabel("Key")
    # plt.ylabel("Key")
    if with_title:
        plt.title(f"Overlap Heatmap of Activated Neurons: {method}")
    if save:
        os.makedirs("../vis", exist_ok=True)
        plt.savefig(f"../vis/{save_name}.pdf") 
    plt.show()
    return overlap_matrix

def make_lsn(num_layers, neurons_per_layer, num_langs, act_dict):
    # num_layers = 24
    # neurons_per_layer = 4864
    # num_langs = 18
    
    reconstructed = []
    
    for lang_id in range(num_langs):
        flat_tensor = act_dict[lang_id]  
        lang_layers = []
        for layer in range(num_layers):
            start = layer * neurons_per_layer
            end = start + neurons_per_layer
            # Extract neurons that belong to this layer and remove the offset
            layer_indices = flat_tensor[(flat_tensor >= start) & (flat_tensor < end)] - start
            lang_layers.append(layer_indices)
        reconstructed.append(lang_layers)  # list of 24 layers per language
    return reconstructed

def visualize_overlap(num_layers, neurons_per_layer, num_lang, tensor, method="default", topk=0, lang_dict=None, alpha=1,save=True, modelname="", annot_kws=12):
    """
    tensor: full neurons 3 dim
    """ 
    activation_dict = get_k_lang_actv_dict(num_lang, tensor, method, topk)
    lsn = make_lsn(num_layers, neurons_per_layer, num_lang, activation_dict)
    make_heatmap_neuron_overlap(activation_dict, num_lang, False)
    annot_kws = annot_kws if annot_kws else 12
    make_heatmap_neuron_overlap(activation_dict, k=num_lang, with_label=True, alpha=alpha, method="default", lang_dict=ld, save=True, modelname=model_name_inf, annot_kws=annot_kws)
    make_heatmap_neuron_overlap(activation_dict, k=num_lang, with_label=True, alpha=alpha, method="jaccard", lang_dict=ld, save=True, modelname=model_name_inf, annot_kws=annot_kws+4)

    return lsn

import argparse
from kaggle_utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--data_kaggle_result", type=str, default=None, help="dataset name if result_exist nd is in Kaggle dataset")
parser.add_argument("--act_filename", type=str, default=None, help="filename that saves act")
parser.add_argument("--ld_filename", type=str, default=None, help="filename that saves ld")
parser.add_argument("--save", action='store_true', help="Flag to save heatmap")
parser.add_argument("--alpha", type=int, default=None, help="alpha for jaccard")
parser.add_argument("--annot_kws", type=int, default=None, help="font size")
parser.add_argument("--n_layer", type=int, default=None, help="number of layer")
parser.add_argument("--parent_dir_to_save", type=str, default=None, help="Parent directory to save like /workspace for runpod")
parser.add_argument("--kaggle_dataname_to_save", type=str, default=None, help="Dataset name for saving to Kaggle NO USERNAME!")


args = parser.parse_args()

data_kaggle_result = args.data_kaggle_result
model_name_inf = data_kaggle_result.split("-")[1]
act_filename = args.act_filename
annot_kws = args.annot_kws if args.annot_kws else 12
download_from_kaggle(data_kaggle_result, act_filename)
lsn = torch.load(f"data/{act_filename}")
num_lang, num_sentences, total_neuron = lsn.shape
num_layer = args.n_layer
num_neuron = total_neuron/num_layer


ld_filename = args.ld_filename
download_from_kaggle(data_kaggle_result, ld_filename)
ld = torch.load(f"data/{ld_filename}")

alpha = args.alpha if args.alpha else 1
lsn = visualize_overlap(num_layer, num_neuron, num_lang, lsn, lang_dict=ld, alpha=alpha, save=args.save, modelname=model_name_inf, annot_kws=annot_kws)
parent_dir = args.parent_dir_to_save
path_res = f"{parent_dir}res/raw_act/{model_name_inf}"
os.makedirs(path_res, exist_ok=True)

torch.save(lsn, f"{path_res}/raw_act_lsn_{model_name_inf}.pt")

if args.kaggle_dataname_to_save:
    save_to_kaggle(dataset_name=args.kaggle_dataname_to_save, data_dir=path_res, is_update=True, subdir_name="")

