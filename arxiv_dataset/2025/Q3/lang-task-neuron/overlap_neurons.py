import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator  
from kaggle_utils import *
from get_map_neurons import map


import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--data_kaggle_result", type=str, default=None, help="dataset name if result_exist nd is in Kaggle dataset")
parser.add_argument("--res_filename", type=str, default=None, help="filename that saves results")
parser.add_argument("--lape_filename", type=str, default=None, help="filename that saves lape results")
parser.add_argument("--maximum", type=int, default=None, help="maximum k for topk")
parser.add_argument("--n_layer", type=int, default=None, help="n_layer")
parser.add_argument("--model_name_inf", type=str, default=None, help="Model name")
parser.add_argument("--dataset_name_inf", type=str, default=None, help="Dataset name")
parser.add_argument("--subdir_name", type=str, default=None, help="subdir name to save final result")
parser.add_argument("--kaggle_dataname_to_save", type=str, default=None, help="Dataset name for saving to Kaggle NO USERNAME!")


args = parser.parse_args()




n_layer = args.n_layer
data_kaggle_result = args.data_kaggle_result
lape_filename = args.lape_filename
download_from_kaggle(data_kaggle_result, lape_filename)
lape = torch.load(f"data/{lape_filename}", weights_only=True)
maximum = args.maximum



def plot_small_hist(ax, data_dict, key, n_layer):
    """Plots a compact histogram of activated neurons per layer for a given key on a given axis."""
    if key not in data_dict:
        raise KeyError(f"Key '{key}' not found in the dictionary!")

    layer_tensors = data_dict[key]
    
    if len(layer_tensors) != n_layer:
        raise ValueError(f"Expected {n_layer} layers, but got {len(layer_tensors)}")

    # Compute the number of activated neurons per layer
    num_activated = [tensor.numel() for tensor in layer_tensors]

    ax.bar(range(n_layer), num_activated, color='b', alpha=0.7)

    ax.set_title(f"{key}", fontsize=8)
    ax.set_xticks(range(n_layer))  # Hide x-axis labels for compact display



def plot_everything(lsn_langs):
    # Example: Iterate over all keys and plot 2 per row
    keys = list(lsn_langs.keys())
    num_keys = len(keys)
    num_cols = 2  # 2 plots per row
    num_rows = (num_keys + 1) // num_cols  # Compute needed rows
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, num_rows * 3))  # Adjust figure size
    
    # Flatten axes for easy indexing
    axes = axes.flatten()
    
    for i, key in enumerate(keys):
        plot_small_hist(axes[i], lsn_langs, key)
    
    # Hide any unused subplots (if keys are not a multiple of 2)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()


def overlap(lape, ap, n_layer):
    assert len(lape) == len(ap)
    num_lang = len(lape)
    overlap_lsn = [[] for _ in range(num_lang)]
    for lang in range(num_lang):
        ll = [set(i.tolist()) for i in (lape[lang])]
        ld = [set(i.tolist()) for i in (ap[lang])]
        intersect = [ll[i].intersection(ld[i]) for i in range(n_layer)]
        # print(f"intersect in lang {lang}: {intersect}")
        perlang = [torch.tensor(list(i)) for i in intersect]
        # print(f"lang: {lang}. num_lang: {num_lang}")
        # print(f"len(overlap_lsn): {len(overlap_lsn)}")
        overlap_lsn[lang] = perlang
        
    return overlap_lsn

lape_langs = {k:v for k,v in enumerate(lape)}




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

maximum = maximum if maximum is not None else max(accumulate(lape))
if not args.maximum: 
    map(
    num_layer=n_layer,
    model_name_inf=args.model_name_inf,
    dataset_name_inf=args.dataset_name_inf,
    top_k = maximum,
    kaggle_dataname_to_save=args.kaggle_dataname_to_save,
    is_update=True,
    result_exist=True,
    data_kaggle_result=data_kaggle_result,
    res_filename=args.res_filename,
    parent_dir=""
)
print(f"opening map top_k")
ap_filename = f"map_t{maximum}_{args.model_name_inf}_{args.dataset_name_inf}.pt" 
download_from_kaggle(data_kaggle_result, ap_filename)
ap = torch.load(f"data/{ap_filename}", weights_only=True)
print(f"overlapping...")

lsn_lang = overlap(lape, ap, n_layer)
path_res = f"res/overlap/{args.model_name_inf}_{args.dataset_name_inf}"
os.makedirs(path_res, exist_ok=True)

filename = f"{path_res}/maplape.pt"
torch.save(lsn_lang, filename)

if args.kaggle_dataname_to_save:
    save_to_kaggle(dataset_name=args.kaggle_dataname_to_save, data_dir=path_res, is_update=True, subdir_name=args.subdir_name)