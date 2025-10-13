import argparse
import torch
import torch.nn.functional as F
import os
from kaggle_utils import *


parser = argparse.ArgumentParser()


parser.add_argument("--in_kaggle", action='store_true', help="Whether the file is from Kaggle")
parser.add_argument("--dataset_kaggle", type=str, default=None, help="The dataset name in Kaggle, if not Kaggle then empty")
parser.add_argument("--filename", type=str, required=True, help="The filename to process")
parser.add_argument("--model_name_inf", type=str, required=True, help="Model name")
parser.add_argument("--dataset_name_inf", type=str, required=True, help="Dataset name")
parser.add_argument("--kaggle_dataname_to_save", type=str, default=None, help="Dataset name for saving to Kaggle")
parser.add_argument("--is_update", action='store_true', help="Flag to update Kaggle dataset")
parser.add_argument("--parent_dir_to_save", type=str, default=None, help="Parent directory to save like /workspace for runpod")

args = parser.parse_args()
parent_dir = args.parent_dir_to_save if args.parent_dir_to_save else ""

in_kaggle_value = args.in_kaggle  # No need to compare to "True" or "False"

dataset_kaggle_value = args.dataset_kaggle if args.dataset_kaggle != "" else None
n, over_zero = [], []
lang_dict = dict()
# for lang in ['fr', "fr"]:
#     data = torch.load(f'activation.{lang}.train.llama-7b')
#     print(data.keys())
#     n.append(data['n'])
#     over_zero.append(data['over_zero'])


over_zero_list = []
if args.in_kaggle:
    download_from_kaggle(args.dataset_kaggle, args.filename)
    over_zero_list = torch.load(f"data/{args.filename}", weights_only=True)
else: over_zero_list = torch.load(args.filename, weights_only=True)


for i, oz in enumerate(over_zero_list):
    n.append(oz['num'])
    over_zero.append(oz['over_zero'])
    lang_dict[i] = oz['lang']


# print(over_zero[0])
n = torch.tensor(n)
over_zero = torch.stack(over_zero, dim=-1)

num_layers, intermediate_size, lang_num = over_zero.size()


def accumulate(lape):
    all_langs = []
    for lang in range(len(lape)):
        sum_per_lang = 0
        perlang = lape[lang]
        for l in range(len(perlang)):
            sum_per_lang += int(lape[lang][l].size(0))
        all_langs.append(sum_per_lang)
        # print(int(lape[lang][2].size(0)))
        # print(len(lape[lang]))
    return all_langs


def activation():
    top_rate = 0.01
    filter_rate = 0.95
    activation_bar_ratio = 0.95
    activation_probs = over_zero / n # layer x inter x lang_num
    normed_activation_probs = activation_probs / activation_probs.sum(dim=-1, keepdim=True)
    normed_activation_probs[torch.isnan(normed_activation_probs)] = 0
    log_probs = torch.where(normed_activation_probs > 0, normed_activation_probs.log(), 0)
    entropy = -torch.sum(normed_activation_probs * log_probs, dim=-1)
    largest = False
    
    if torch.isnan(entropy).sum():
        print(torch.isnan(entropy).sum())
        raise ValueError
    
    flattened_probs = activation_probs.flatten()
    top_prob_value = flattened_probs.kthvalue(round(len(flattened_probs) * filter_rate)).values.item()
    print(top_prob_value)
    # dismiss the neruon if no language has an activation value over top 90%
    top_position = (activation_probs > top_prob_value).sum(dim=-1)
    entropy[top_position == 0] = -torch.inf if largest else torch.inf

    flattened_entropy = entropy.flatten()
    top_entropy_value = round(len(flattened_entropy) * top_rate)
    _, index = flattened_entropy.topk(top_entropy_value, largest=largest)
    row_index = index // entropy.size(1)
    col_index = index % entropy.size(1)
    selected_probs = activation_probs[row_index, col_index] # n x lang
    # for r, c in zip(row_index, col_index):
    #     print(r, c, activation_probs[r][c])

    # print(selected_probs.size(0), torch.bincount(selected_probs.argmax(dim=-1)))
    selected_probs = selected_probs.transpose(0, 1)
    activation_bar = flattened_probs.kthvalue(round(len(flattened_probs) * activation_bar_ratio)).values.item()
    # print((selected_probs > activation_bar).sum(dim=1).tolist())
    lang, indice = torch.where(selected_probs > activation_bar)

    merged_index = torch.stack((row_index, col_index), dim=-1)
    final_indice = []
    for _, index in enumerate(indice.split(torch.bincount(lang).tolist())):
        lang_index = [tuple(row.tolist()) for row in merged_index[index]]
        lang_index.sort()
        layer_index = [[] for _ in range(num_layers)]
        for l, h in lang_index:
            layer_index[l].append(h)
        for l, h in enumerate(layer_index):
            layer_index[l] = torch.tensor(h).long()
        final_indice.append(layer_index)
    maximum = max(accumulate(final_indice))
    print(f"maximum neurons in a language: {maximum}")
    path_res = f"{parent_dir}res/lape"
    os.makedirs(path_res, exist_ok=True)
    name_to_save = f"{args.model_name_inf}_{args.dataset_name_inf}"
    torch.save(final_indice, f"{path_res}/{name_to_save}")  
    if args.kaggle_dataname_to_save:
        save_to_kaggle(result_neurons=[final_indice, lang_dict], dataset_name=args.kaggle_dataname_to_save, filename=[name_to_save, "lang_dict"], is_update=args.is_update)

activation()
