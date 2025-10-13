import torch
from kaggle_utils import *
import argparse
from tqdm import tqdm
parser = argparse.ArgumentParser(description="Run the map function with specified parameters.")
parser.add_argument("--in_kaggle", action='store_true', help="Whether the file is from Kaggle")
parser.add_argument("--dataset_kaggle", type=str,  help="The dataset name in Kaggle, if not Kaggle then empty")
parser.add_argument("--filename", type=str, help="The filename to process")
parser.add_argument("--kaggle_dataname_to_save", type=str, default=None, help="Dataset name for saving to Kaggle NO USERNAME!")
parser.add_argument("--is_update", action='store_true', help="Flag to update Kaggle dataset")
parser.add_argument("--n_layer", type=int, help="num layer")
parser.add_argument("--parent_dir_to_save", type=str, default=None, help="Parent directory to save like /workspace for runpod")

args = parser.parse_args()




parent_dir_to_save = args.parent_dir_to_save if args.parent_dir_to_save else ""


download_from_kaggle(args.dataset_kaggle, args.filename)
act_file = torch.load(f"data/{args.filename}", weights_only=True)

batch_size = 5  

max_vals_list = []
median_vals_list = []

for i in tqdm(range(0, act_file.shape[0], batch_size)):
    batch = act_file[i:i + batch_size]

    
    max_val, _ = batch.max(dim=1)
    max_val = max_val.reshape(batch.shape[0], args.n_layer, int(batch.shape[-1] / args.n_layer))
    max_vals_list.append(max_val)

    
    median_val, _ = batch.median(dim=1)
    median_val = median_val.reshape(batch.shape[0], args.n_layer, int(batch.shape[-1] / args.n_layer))
    median_vals_list.append(median_val)

print("done calculating max and median")
max_vals = torch.cat(max_vals_list, dim=0)
median_vals = torch.cat(median_vals_list, dim=0)
path_res = f"{parent_dir_to_save}res"
os.makedirs(path_res, exist_ok=True)


torch.save(median_vals, f"{path_res}/median.pt")
torch.save(max_vals, f"{path_res}/max.pt")

if args.kaggle_dataname_to_save:
    save_to_kaggle(dataset_name=args.kaggle_dataname_to_save, data_dir=path_res, is_update=args.is_update)