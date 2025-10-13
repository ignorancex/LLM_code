import numpy as np
import torch
import pickle as pkl
from tqdm import tqdm
import random
import json
import os
import argparse
from sklearn.cluster import KMeans
# from cuml import KMeans
from sklearn.metrics.cluster import homogeneity_completeness_v_measure
from utils.args_utils import dict_to_object, get_model_family


def parse_args():

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, help="Path to JSON config file")
    pre_args, _ = pre_parser.parse_known_args()

    args = {}
    
    # If --config is provided, load arguments from JSON
    if pre_args.config:
        with open(pre_args.config, "r") as f:
            args.update(json.load(f))
    # Otherwise, use normal argparse
    else:
        parser = argparse.ArgumentParser()
        
        parser.add_argument("--model_name", type=str, required=True)
        parser.add_argument("--dataset", type=str, required=True)

        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--version", type=int, default=0)
        parser.add_argument("--repr_version", type=int, default=0)
        parser.add_argument("--is_explicit_helper", type=int, default=0)

        args.update(vars(parser.parse_args()))

    args = dict_to_object(args)
    args.model_family = get_model_family(args)

    return args

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":

    args = parse_args()

    if len(args.model_name.split("||")) > 1:
        args.model_family += args.model_name.split("||")[-1]

    set_seed(args.seed)

    print(f"Computing KMeans clustering results for inconsistent representations of [{args.model_name}] on [{args.dataset}]")

    for report_str in ["reporting_image", "reporting_text"]:

        base_dir = os.path.join(
            "outputs",
            f"precompute_representations_ver{args.repr_version:03d}",
            args.model_name,
            args.dataset,
            "inconsistent",
            f"use_helper_{args.is_explicit_helper}",
            report_str
        )

        # Training data path
        train_path = os.path.join(base_dir, "train", "seed42.pkl")
        print(f"Loading {train_path}")
        with open(train_path, "rb") as f:
            inconsistent_data_train = pkl.load(f)

        # Test data path
        test_path = os.path.join(base_dir, "test", "seed42.pkl")
        print(f"Loading {test_path}")
        with open(test_path, "rb") as f:
            inconsistent_data_test = pkl.load(f)

        all_results = {
            "image_preds": [],
            "caption_preds": []
        }

        n_layers_plus_one = inconsistent_data_train['data']['representations'].shape[1]
        
        for layer_idx in tqdm(range(1, n_layers_plus_one), total = n_layers_plus_one-1):

            kmeans_model = KMeans(n_clusters=inconsistent_data_train['num_classes'], random_state=args.seed, n_init="auto")
            kmeans_model.fit(inconsistent_data_train['data']['representations'][:,layer_idx,:])

            test_preds = kmeans_model.predict(inconsistent_data_test['data']['representations'][:,layer_idx,:])

            kmeans_scores_test_cap = homogeneity_completeness_v_measure(test_preds, inconsistent_data_test['data']['caption_labels'])
            kmeans_scores_test_im = homogeneity_completeness_v_measure(test_preds, inconsistent_data_test['data']['image_labels'])

            all_results['image_preds'].append(kmeans_scores_test_im)
            all_results['caption_preds'].append(kmeans_scores_test_cap)

        output_dir = os.path.join(
            "outputs",
            f"clustering_results_ver{args.version:03d}",
            args.model_family,
            args.dataset,
            report_str
        )

        os.makedirs(output_dir, exist_ok=True)

        # Save to file
        with open(os.path.join(output_dir, f'results_seed{args.seed}.json'), 'w') as f:
            json.dump(all_results, f, indent=2)

        del inconsistent_data_train
        del inconsistent_data_test