#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 21:37:00 2024

@author: kcybinski
"""

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

from termcolor import cprint
import os

# Pytorch imports
import torch
import pandas as pd
import numpy as np
import pathlib
from tqdm import tqdm, trange
import argparse
import pickle
from datetime import datetime
import humanize
from matplotlib import cm

from sklearn.decomposition import PCA
from umap import UMAP

from Loaders import Importer
from src.architecture import CNN_SSH
from src.cam_utils import get_CAM, get_preds_from_model, CAMs_plot_gen_final, generate_CAMs, viz_final

device = torch.device("cuda:0" if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else
                          "cpu")

load_path = pathlib.Path("./00_demonstrations_data/datasets")
models_root = pathlib.Path("./00_demonstrations_data/models/")

disorderless_ds_path = load_path.joinpath("disorderless")
disordered_001_ds_path = load_path.joinpath("disordered_W=0.01")
disordered_005_ds_path = load_path.joinpath("disordered_W=0.05")
disordered_015_ds_path = load_path.joinpath("disordered_W=0.15")
disordered_050_ds_path = load_path.joinpath("disordered_W=0.5")
disordered_100_ds_path = load_path.joinpath("disordered_W=1.0")
disordered_200_ds_path = load_path.joinpath("disordered_W=2.0")

if __name__ == "__main__":
    main_root = pathlib.Path("./00_demonstrations_data/models")
    main_root.mkdir(parents=True, exist_ok=True)

    # Get rid of the annoying numba deprecation warning
    tmp_umap = UMAP(n_neighbors=30, min_dist=0.0, n_components=2, metric='euclidean')
    umap_tmp = tmp_umap.fit_transform(np.random.rand(100, 2500))
    del tmp_umap

    parser = argparse.ArgumentParser(description="CNN model analysis")

    parser.add_argument("-bs", "--batch_size", type=int, default=64)
    parser.add_argument("--CAM", type=int, default=0, help="0: Skip CAM generation, 1: Generate just CAMs", dest="cam")
    parser.add_argument("--perturb", type=int, default=1, help="0: Models trained on disorderless data, 1: Models trained with slight disorder")
    parser.add_argument("--exclude_transition", type=int, default=1, help="0: Don't exclude, 1: Exclude")

    
    args = parser.parse_args()
    print(args)
    perturb = bool(args.perturb)
    exclude_transition = bool(args.exclude_transition)

    do_cam = bool(args.cam)

    # ==============================================================================
    # Here we only have two classes, defined by Winding number:
    # Winding number = 0
    # Winding number = 1
    # ==============================================================================
    num_classes = 2
    class_names = ["Trivial", "Topological"]
    perturb_list = [disordered_001_ds_path, disordered_005_ds_path]
    used_ds_names = [disorderless_ds_path] + perturb_list if perturb else [disorderless_ds_path]
    used_ds_names = [str(x) for x in used_ds_names]

    trained_count = 4
    

    with open(main_root.joinpath("parameters.txt"), "r") as f:
        parameters = f.readlines()
        parameters = [x.strip() for x in parameters]
        parameters = [x.split(":") for x in parameters]
        parameters = {x[0].strip(): x[1].strip() for x in parameters}

    used_ds_names = parameters["Datasets"].split(", ")
    device = torch.device("cuda:0" if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else
                          "cpu")

    processing_limits = [0, trained_count]

    cprint("Current working directory: ", "green", end=""); cprint(os.getcwd(), end="\n", flush=True)
    cprint("Loading from: ", "green", end="")
    print(main_root)

    cprint("Range beg: ", "cyan", end=""); cprint(str(processing_limits[0]), end="\n", flush=True)
    cprint("Range end: ", "cyan", end=""); cprint(str(processing_limits[1]), end="\n", flush=True)

    cprint("\nParameters:", "cyan", end="\n")
    cprint("Trained count: ", "green", end=""); cprint(parameters["Trained count"], end="\n", flush=True)
    cprint("Epochs: ", "green", end=""); cprint(parameters["Epochs"], end="\n", flush=True)
    cprint("Batch size: ", "green", end=""); cprint(parameters["Batch size"], end="\n", flush=True)
    cprint("Perturbative dataset: ", "green", end=""); cprint(parameters["Perturbative dataset"], end="\n", flush=True)
    cprint("Datasets: ", "green", end=""); cprint(parameters["Datasets"], end="\n", flush=True)
    cprint("Device: ", "green", end=""); cprint(parameters["Device"], end="\n", flush=True)
    cprint("Learning rate: ", "green", end=""); cprint(parameters["Learning rate"], end="\n", flush=True)
    cprint("Momentum: ", "green", end=""); cprint(parameters["Momentum"], end="\n", flush=True)
    cprint("Weight decay: ", "green", end=""); cprint(parameters["Weight decay"], end="\n", flush=True)
    cprint("Early stopping: ", "green", end=""); cprint(parameters["Early stopping"], end="\n", flush=True)

    
    processed_range = np.arange(0, 4, 1)
    model_folders = [
        models_root.joinpath('well_generalizing_good_CAM'),
        models_root.joinpath('well_generalizing_misleading_CAM'),
        models_root.joinpath('poorly_generalizing_good_CAM'),
        models_root.joinpath('poorly_generalizing_misleading_CAM'),
    ]
    # Progress bar for the models loop
    models_pbar = tqdm(total=processed_range.shape[0], desc="Models", position=0, leave=True, unit="model", colour="red")

    model_folders = [x for x in model_folders if x.exists()]

    # Clustering 
    with torch.no_grad():
        clustering_batch_size = -1
        n_comp = 2

        umap_params = {
            "n_neighbors": 30,
            "min_dist": 0.0,
            "n_components": n_comp,
            "metric": "euclidean",
            # "verbose": True,
        }
        
        test_paths = [
            disorderless_ds_path, 
            disordered_015_ds_path, 
            disordered_050_ds_path, 
            disordered_100_ds_path, 
            disordered_200_ds_path,
        ]

        layer_names = ['clear', 'conv2d1', 'conv2d2', 'cnn_seq_1', 'conv2d3', 'cnn_seq_2', 'avg_pool', 'avg_pool_predicted']
        w_tab = ["0", "0.15", "0.5", "1.0", "2.0", "3.0"]


        for folder in model_folders:
            model = CNN_SSH()
            model.load_state_dict(torch.load(folder.joinpath("trained_model.dict"), map_location=device))
            model.float().to(device)
            model.eval()

            models_pbar.set_description(f"Models | {folder.name}")
            if not do_cam:
                ds_000 = Importer(disorderless_ds_path, clustering_batch_size)
                ds_001 = Importer(disordered_001_ds_path, clustering_batch_size)
                ds_005 = Importer(disordered_005_ds_path, clustering_batch_size)

                test_loader_000 = ds_000.get_test_loader()
                test_loader_001 = ds_001.get_test_loader()
                test_loader_005 = ds_005.get_test_loader()

                val_loader_000 = ds_000.get_val_loader()
                val_loader_001 = ds_001.get_val_loader()
                val_loader_005 = ds_005.get_val_loader()

                tr_loader_000 = ds_000.get_train_loader()
                tr_loader_001 = ds_001.get_train_loader()
                tr_loader_005 = ds_005.get_train_loader()

                pca_pbar = tqdm(total=len(test_paths), desc="PCA", position=1, leave=False, unit="dataset", colour="blue")
                folder_pca = folder.joinpath("PCA")
                folder_pca.mkdir(parents=True, exist_ok=True)
                # PCA
                # %%
                generate_pca = True
                try:
                    # Get the list of all subdirectories
                    subdirectories = [x[0] for x in os.walk(folder_pca)]
                    # Check if all subdirectories contain the file 'all_generated.txt'
                    all_contain_file = all(os.path.isfile(os.path.join(subdir, 'all_generated.txt')) for subdir in subdirectories)

                    if all_contain_file:
                        generate_pca = False
                    else:
                        pass
                except:
                    generate_pca = True

                if generate_pca:
                    for num, test_path in enumerate(test_paths):
                        ds = Importer(test_path, clustering_batch_size)
                        test_loader = ds.get_test_loader()

                        W = ds.W_tab[0]
                        if W == 0:
                            W = f"{W:.0f}"
                        elif 0 < W < 0.5 :
                            W = f"{W:.2f}"
                        else:
                            W = f"{W:.1f}"

                        p = folder_pca.joinpath(f"W={W}")
                        p.mkdir(parents=True, exist_ok=True)

                        pca_pbar.set_description(f"PCA | W = {W}")

                        activation = {}
                        activation_trn = {}
                        activation_val = {}
                        def getActivation(name, dset_name='test'):
                            # the hook signature
                            if dset_name == 'test':
                                def hook(model, input, output):
                                    activation[name] = output.detach().cpu().numpy()
                            elif dset_name == 'trn':
                                def hook(model, input, output):
                                    activation_trn[name] = output.detach().cpu().numpy()
                            elif dset_name == 'val':
                                def hook(model, input, output):
                                    activation_val[name] = output.detach().cpu().numpy()
                            return hook

                        hook_conv2d1 = model.cnn_layers_1[0].register_forward_hook(getActivation('cnn_layers_1_condv2d1'))
                        hook_conv2d2 = model.cnn_layers_1[4].register_forward_hook(getActivation('cnn_layers_1_condv2d2'))
                        hook_cnn_seq_1 = model.cnn_layers_1.register_forward_hook(getActivation('cnn_layers_1'))
                        hook_conv2d3 = model.cnn_layers_2[0].register_forward_hook(getActivation('cnn_layers_2_condv2d1'))
                        hook_cnn_seq_2 = model.cnn_layers_2.register_forward_hook(getActivation('cnn_layers_2'))
                        hook_avgpool = model.avg_pool[0].register_forward_hook(getActivation('avg_pool'))

                        hooks_list = [hook_conv2d1, hook_conv2d2, hook_cnn_seq_1, hook_conv2d3, hook_cnn_seq_2, hook_avgpool]

                        conv2d1_list, conv2d2_list, cnn_seq_1_list, conv2d3_list, cnn_seq_2_list, avgpool_list = [], [], [], [], [], []
                        conv2d1_list_val, conv2d2_list_val, cnn_seq_1_list_val, conv2d3_list_val, cnn_seq_2_list_val, avgpool_list_val = [], [], [], [], [], []
                        conv2d1_list_trn, conv2d2_list_trn, cnn_seq_1_list_trn, conv2d3_list_trn, cnn_seq_2_list_trn, avgpool_list_trn = [], [], [], [], [], []

                        if num == 0 and perturb:
                            # Test data

                            x_000 = test_loader_000.dataset.X
                            y_000 = test_loader_000.dataset.Y

                            x_001 = test_loader_001.dataset.X
                            y_001 = test_loader_001.dataset.Y

                            x_005 = test_loader_005.dataset.X
                            y_005 = test_loader_005.dataset.Y

                            ds_len = y_000.shape[0]

                            if exclude_transition:
                                x_tot = torch.cat([
                                    torch.cat([x_000[:(ds_len//2 - ds_len//10)], x_000[(ds_len//2 + ds_len//10):]],axis=0),
                                    torch.cat([x_001[:(ds_len//2 - ds_len//10)], x_001[(ds_len//2 + ds_len//10):]],axis=0),
                                    torch.cat([x_005[:(ds_len//2 - ds_len//10)], x_005[(ds_len//2 + ds_len//10):]],axis=0),
                                    ], axis=0)
                                y_tot = torch.cat([
                                    torch.cat([y_000[:(ds_len//2 - ds_len//10)], y_000[(ds_len//2 + ds_len//10):]],axis=0),
                                    torch.cat([y_001[:(ds_len//2 - ds_len//10)], y_001[(ds_len//2 + ds_len//10):]],axis=0),
                                    torch.cat([y_005[:(ds_len//2 - ds_len//10)], y_005[(ds_len//2 + ds_len//10):]],axis=0),
                                    ], axis=0)
                            else:
                                x_tot = torch.cat([x_000, x_001, x_005], axis=0)
                                y_tot = torch.cat([y_000, y_001, y_005], axis=0)

                            t_l = [(x_tot.clone(), y_tot.clone())]

                            # Validation data

                            x_000 = val_loader_000.dataset.X
                            y_000 = val_loader_000.dataset.Y

                            x_001 = val_loader_001.dataset.X
                            y_001 = val_loader_001.dataset.Y

                            x_005 = val_loader_005.dataset.X
                            y_005 = val_loader_005.dataset.Y

                            ds_len = y_000.shape[0]

                            if exclude_transition:
                                x_tot = torch.cat([
                                    torch.cat([x_000[:(ds_len//2 - ds_len//10)], x_000[(ds_len//2 + ds_len//10):]],axis=0),
                                    torch.cat([x_001[:(ds_len//2 - ds_len//10)], x_001[(ds_len//2 + ds_len//10):]],axis=0),
                                    torch.cat([x_005[:(ds_len//2 - ds_len//10)], x_005[(ds_len//2 + ds_len//10):]],axis=0),
                                    ], axis=0)
                                y_tot = torch.cat([
                                    torch.cat([y_000[:(ds_len//2 - ds_len//10)], y_000[(ds_len//2 + ds_len//10):]],axis=0),
                                    torch.cat([y_001[:(ds_len//2 - ds_len//10)], y_001[(ds_len//2 + ds_len//10):]],axis=0),
                                    torch.cat([y_005[:(ds_len//2 - ds_len//10)], y_005[(ds_len//2 + ds_len//10):]],axis=0),
                                    ], axis=0)
                            else:
                                x_tot = torch.cat([x_000, x_001, x_005], axis=0)
                                y_tot = torch.cat([y_000, y_001, y_005], axis=0)

                            val_l = [(x_tot.clone(), y_tot.clone())]

                            # Training data

                            x_000 = tr_loader_000.dataset.X
                            y_000 = tr_loader_000.dataset.Y

                            x_001 = tr_loader_001.dataset.X
                            y_001 = tr_loader_001.dataset.Y

                            x_005 = tr_loader_005.dataset.X
                            y_005 = tr_loader_005.dataset.Y

                            ds_len = y_000.shape[0]

                            if exclude_transition:
                                x_tot = torch.cat([
                                    torch.cat([x_000[:(ds_len//2 - ds_len//10)], x_000[(ds_len//2 + ds_len//10):]],axis=0),
                                    torch.cat([x_001[:(ds_len//2 - ds_len//10)], x_001[(ds_len//2 + ds_len//10):]],axis=0),
                                    torch.cat([x_005[:(ds_len//2 - ds_len//10)], x_005[(ds_len//2 + ds_len//10):]],axis=0),
                                    ], axis=0)
                                y_tot = torch.cat([
                                    torch.cat([y_000[:(ds_len//2 - ds_len//10)], y_000[(ds_len//2 + ds_len//10):]],axis=0),
                                    torch.cat([y_001[:(ds_len//2 - ds_len//10)], y_001[(ds_len//2 + ds_len//10):]],axis=0),
                                    torch.cat([y_005[:(ds_len//2 - ds_len//10)], y_005[(ds_len//2 + ds_len//10):]],axis=0),
                                    ], axis=0)
                            else:
                                x_tot = torch.cat([x_000, x_001, x_005], axis=0)
                                y_tot = torch.cat([y_000, y_001, y_005], axis=0)

                            trn_l = [(x_tot.clone(), y_tot.clone())]
                        else:
                            t_l = test_loader
                            trn_l = tr_loader_000
                            val_l = val_loader_000

                        # Test data

                        for X, y in t_l:
                            ds_len = y.shape[0]

                            if num == 0 and perturb:
                                X_subset = X
                                y_subset = y
                            else:
                                if exclude_transition:
                                    X_subset = torch.concatenate([X[:(ds_len//2 - ds_len//10)], X[(ds_len//2 + ds_len//10):]], axis=0)
                                    y_subset = torch.concatenate([y[:(ds_len//2 - ds_len//10)], y[(ds_len//2 + ds_len//10):]], axis=0)
                                else:
                                    X_subset = X
                                    y_subset = y

                            X_subset = X_subset.to(device)
                            y_subset = y_subset.to(device)
                            out = model(X_subset)

                            conv2d1_list.append(activation['cnn_layers_1_condv2d1'])
                            conv2d2_list.append(activation['cnn_layers_1_condv2d2'])
                            cnn_seq_1_list.append(activation['cnn_layers_1'])
                            conv2d3_list.append(activation['cnn_layers_2_condv2d1'])
                            cnn_seq_2_list.append(activation['cnn_layers_2'])
                            avgpool_list.append(activation['avg_pool'])

                        for hook in hooks_list:
                            hook.remove()

                        # Validation data

                        val_hook_conv2d1 = model.cnn_layers_1[0].register_forward_hook(getActivation('cnn_layers_1_condv2d1', 'val'))
                        val_hook_conv2d2 = model.cnn_layers_1[4].register_forward_hook(getActivation('cnn_layers_1_condv2d2', 'val'))
                        val_hook_cnn_seq_1 = model.cnn_layers_1.register_forward_hook(getActivation('cnn_layers_1', 'val'))
                        val_hook_conv2d3 = model.cnn_layers_2[0].register_forward_hook(getActivation('cnn_layers_2_condv2d1', 'val' ))
                        val_hook_cnn_seq_2 = model.cnn_layers_2.register_forward_hook(getActivation('cnn_layers_2', 'val'))
                        val_hook_avgpool = model.avg_pool[0].register_forward_hook(getActivation('avg_pool', 'val'))

                        val_hooks_list = [val_hook_conv2d1, val_hook_conv2d2, val_hook_cnn_seq_1, val_hook_conv2d3, val_hook_cnn_seq_2, val_hook_avgpool]

                        for X, y in val_l:
                            ds_len = y.shape[0]

                            if num == 0 and perturb:
                                X_subset_val = X
                                y_subset_val = y
                            else:
                                if exclude_transition:
                                    X_subset_val = torch.concatenate([X[:(ds_len//2 - ds_len//10)], X[(ds_len//2 + ds_len//10):]], axis=0)
                                    y_subset_val = torch.concatenate([y[:(ds_len//2 - ds_len//10)], y[(ds_len//2 + ds_len//10):]], axis=0)
                                else:
                                    X_subset_val = X
                                    y_subset_val = y

                            X_subset_val = X_subset_val.to(device)
                            y_subset_val = y_subset_val.to(device)
                            out_val = model(X_subset_val)

                            conv2d1_list_val.append(activation_val['cnn_layers_1_condv2d1'])
                            conv2d2_list_val.append(activation_val['cnn_layers_1_condv2d2'])
                            cnn_seq_1_list_val.append(activation_val['cnn_layers_1'])
                            conv2d3_list_val.append(activation_val['cnn_layers_2_condv2d1'])
                            cnn_seq_2_list_val.append(activation_val['cnn_layers_2'])
                            avgpool_list_val.append(activation_val['avg_pool'])

                        for hook in val_hooks_list:
                            hook.remove()

                        # Training data

                        trn_hook_conv2d1 = model.cnn_layers_1[0].register_forward_hook(getActivation('cnn_layers_1_condv2d1', 'trn'))
                        trn_hook_conv2d2 = model.cnn_layers_1[4].register_forward_hook(getActivation('cnn_layers_1_condv2d2', 'trn'))
                        trn_hook_cnn_seq_1 = model.cnn_layers_1.register_forward_hook(getActivation('cnn_layers_1', 'trn'))
                        trn_hook_conv2d3 = model.cnn_layers_2[0].register_forward_hook(getActivation('cnn_layers_2_condv2d1', 'trn' ))
                        trn_hook_cnn_seq_2 = model.cnn_layers_2.register_forward_hook(getActivation('cnn_layers_2', 'trn'))
                        trn_hook_avgpool = model.avg_pool[0].register_forward_hook(getActivation('avg_pool', 'trn'))

                        trn_hooks_list = [trn_hook_conv2d1, trn_hook_conv2d2, trn_hook_cnn_seq_1, trn_hook_conv2d3, trn_hook_cnn_seq_2, trn_hook_avgpool]

                        for X, y in trn_l:
                            ds_len = y.shape[0]

                            if num == 0 and perturb:
                                X_subset_trn = X
                                y_subset_trn = y
                            else:
                                if exclude_transition:
                                    X_subset_trn = torch.concatenate([X[:(ds_len//2 - ds_len//10)], X[(ds_len//2 + ds_len//10):]], axis=0)
                                    y_subset_trn = torch.concatenate([y[:(ds_len//2 - ds_len//10)], y[(ds_len//2 + ds_len//10):]], axis=0)
                                else:
                                    X_subset_trn = X
                                    y_subset_trn = y

                            X_subset_trn = X_subset_trn.to(device)
                            y_subset_trn = y_subset_trn.to(device)
                            out_trn = model(X_subset_trn)

                            conv2d1_list_trn.append(activation_trn['cnn_layers_1_condv2d1'])
                            conv2d2_list_trn.append(activation_trn['cnn_layers_1_condv2d2'])
                            cnn_seq_1_list_trn.append(activation_trn['cnn_layers_1'])
                            conv2d3_list_trn.append(activation_trn['cnn_layers_2_condv2d1'])
                            cnn_seq_2_list_trn.append(activation_trn['cnn_layers_2'])
                            avgpool_list_trn.append(activation_trn['avg_pool'])

                        for hook in trn_hooks_list:
                            hook.remove()

                        # Join along the first axis
                        conv2d1 = np.concatenate(np.array(conv2d1_list), axis=0)
                        conv2d2 = np.concatenate(np.array(conv2d2_list), axis=0)
                        conv2d3 = np.concatenate(np.array(conv2d3_list), axis=0)
                        cnn_seq_1 = np.concatenate(np.array(cnn_seq_1_list), axis=0)
                        cnn_seq_2 = np.concatenate(np.array(cnn_seq_2_list), axis=0)
                        avg_pool = np.concatenate(np.array(avgpool_list), axis=0)

                        color_names = ['Trivial', 'Topological']
                        classes = [color_names[i] for i in y_subset.detach().cpu().numpy()]
                        preds = [color_names[i] for i in out.argmax(axis=1).detach().cpu().numpy()]


                        # Val data

                        val_conv2d1 = np.concatenate(np.array(conv2d1_list_val), axis=0)
                        val_conv2d2 = np.concatenate(np.array(conv2d2_list_val), axis=0)
                        val_conv2d3 = np.concatenate(np.array(conv2d3_list_val), axis=0)
                        val_cnn_seq_1 = np.concatenate(np.array(cnn_seq_1_list_val), axis=0)
                        val_cnn_seq_2 = np.concatenate(np.array(cnn_seq_2_list_val), axis=0)
                        val_avg_pool = np.concatenate(np.array(avgpool_list_val), axis=0)

                        val_color_names = ['Val_Trivial', 'Val_Topological']
                        val_classes = [val_color_names[i] for i in y_subset_val.detach().cpu().numpy()]
                        val_preds = [val_color_names[i] for i in out_val.argmax(axis=1).detach().cpu().numpy()]

                        # Trn data
                        
                        trn_conv2d1 = np.concatenate(np.array(conv2d1_list_trn), axis=0)
                        trn_conv2d2 = np.concatenate(np.array(conv2d2_list_trn), axis=0)
                        trn_conv2d3 = np.concatenate(np.array(conv2d3_list_trn), axis=0)
                        trn_cnn_seq_1 = np.concatenate(np.array(cnn_seq_1_list_trn), axis=0)
                        trn_cnn_seq_2 = np.concatenate(np.array(cnn_seq_2_list_trn), axis=0)
                        trn_avg_pool = np.concatenate(np.array(avgpool_list_trn), axis=0)

                        trn_color_names = ['Trn_Trivial', 'Trn_Topological']
                        trn_classes = [trn_color_names[i] for i in y_subset_trn.detach().cpu().numpy()]
                        trn_preds = [trn_color_names[i] for i in out_trn.argmax(axis=1).detach().cpu().numpy()]

                        # **clear data**

                        if num == 0:
                            pca_clear = PCA(n_components=n_comp)

                            ds_clear = pca_clear.fit_transform(X_subset.reshape(len(classes), -1).detach().cpu().numpy())
                        else:
                            ds_clear = pca_clear.transform(X_subset.reshape(len(classes), -1).detach().cpu().numpy())


                        if n_comp == 2:
                            df = pd.DataFrame(ds_clear, columns=['x', 'y'])
                        else:
                            df = pd.DataFrame(ds_clear, columns=['x', 'y', 'z'])
                        df['class'] = classes
                        df["pred"] = preds

                        val_ds = pca_clear.transform(X_subset_val.reshape(len(val_classes), -1).detach().cpu().numpy())
                        trn_ds = pca_clear.transform(X_subset_trn.reshape(len(trn_classes), -1).detach().cpu().numpy())

                        df_val = pd.DataFrame(val_ds, columns=['val_x', 'val_y'])

                        df_val["val_class"] = val_classes
                        df_val["val_pred"] = val_preds

                        df_trn = pd.DataFrame(trn_ds, columns=['trn_x', 'trn_y'])

                        df_trn["trn_class"] = trn_classes
                        df_trn["trn_pred"] = trn_preds

                        df = pd.concat([df, df_val, df_trn], axis=1)

                        df.to_csv(p.joinpath(f"clear_PCA_W={W}.csv"), index=False)

                        # **conv2d1**

                        if num == 0:
                            pca_conv2d1 = PCA(n_components=n_comp)

                            ds_conv2d1 = pca_conv2d1.fit_transform(conv2d1.reshape(len(classes), -1))

                        else:
                            ds_conv2d1 = pca_conv2d1.transform(conv2d1.reshape(len(classes), -1))


                        if n_comp == 2:
                            df = pd.DataFrame(ds_conv2d1, columns=['x', 'y'])
                        else:
                            df = pd.DataFrame(ds_conv2d1, columns=['x', 'y', 'z'])
                        df['class'] = classes
                        df["pred"] = preds

                        val_ds = pca_conv2d1.transform(val_conv2d1.reshape(len(val_classes), -1))
                        trn_ds = pca_conv2d1.transform(trn_conv2d1.reshape(len(trn_classes), -1))

                        df_val = pd.DataFrame(val_ds, columns=['val_x', 'val_y'])

                        df_val["val_class"] = val_classes
                        df_val["val_pred"] = val_preds

                        df_trn = pd.DataFrame(trn_ds, columns=['trn_x', 'trn_y'])

                        df_trn["trn_class"] = trn_classes
                        df_trn["trn_pred"] = trn_preds

                        df = pd.concat([df, df_val, df_trn], axis=1)

                        df.to_csv(p.joinpath(f"conv2d1_PCA_W={W}.csv"), index=False)

                        # **conv2d2**
                        if num == 0:
                            pca_conv2d2 = PCA(n_components=n_comp)

                            ds_conv2d2 = pca_conv2d2.fit_transform(conv2d2.reshape(len(classes), -1))
                        else:
                            ds_conv2d2 = pca_conv2d2.transform(conv2d2.reshape(len(classes), -1))
                        
                        if n_comp == 2:
                            df = pd.DataFrame(ds_conv2d2, columns=['x', 'y'])
                        else:
                            df = pd.DataFrame(ds_conv2d2, columns=['x', 'y', 'z'])
                        df['class'] = classes
                        df["pred"] = preds

                        val_ds = pca_conv2d2.transform(val_conv2d2.reshape(len(val_classes), -1))
                        trn_ds = pca_conv2d2.transform(trn_conv2d2.reshape(len(trn_classes), -1))

                        df_val = pd.DataFrame(val_ds, columns=['val_x', 'val_y'])

                        df_val["val_class"] = val_classes
                        df_val["val_pred"] = val_preds

                        df_trn = pd.DataFrame(trn_ds, columns=['trn_x', 'trn_y'])

                        df_trn["trn_class"] = trn_classes
                        df_trn["trn_pred"] = trn_preds

                        df = pd.concat([df, df_val, df_trn], axis=1)

                        df.to_csv(p.joinpath(f"conv2d2_PCA_W={W}.csv"), index=False)

                        # **cnn_seq_1**

                        if num == 0:
                            pca_seq_1 = PCA(n_components=n_comp)

                            ds_seq_1 = pca_seq_1.fit_transform(cnn_seq_1.reshape(len(classes), -1))
                        else:
                            ds_seq_1 = pca_seq_1.transform(cnn_seq_1.reshape(len(classes), -1))

                        if n_comp == 2:
                            df = pd.DataFrame(ds_seq_1, columns=['x', 'y'])
                        else:
                            df = pd.DataFrame(ds_seq_1, columns=['x', 'y', 'z'])
                        df['class'] = classes
                        df["pred"] = preds

                        val_ds = pca_seq_1.transform(val_cnn_seq_1.reshape(len(val_classes), -1))
                        trn_ds = pca_seq_1.transform(trn_cnn_seq_1.reshape(len(trn_classes), -1))

                        df_val = pd.DataFrame(val_ds, columns=['val_x', 'val_y'])

                        df_val["val_class"] = val_classes
                        df_val["val_pred"] = val_preds

                        df_trn = pd.DataFrame(trn_ds, columns=['trn_x', 'trn_y'])

                        df_trn["trn_class"] = trn_classes
                        df_trn["trn_pred"] = trn_preds

                        df = pd.concat([df, df_val, df_trn], axis=1)

                        df.to_csv(p.joinpath(f"cnn_seq_1_PCA_W={W}.csv"), index=False)

                        # **conv2d3**

                        if num == 0:
                            pca_conv2d3 = PCA(n_components=n_comp)

                            ds_conv2d3 = pca_conv2d3.fit_transform(conv2d3.reshape(len(classes), -1))
                        else:
                            ds_conv2d3 = pca_conv2d3.transform(conv2d3.reshape(len(classes), -1))

                        if n_comp == 2:
                            df = pd.DataFrame(ds_conv2d3, columns=['x', 'y'])
                        else:
                            df = pd.DataFrame(ds_conv2d3, columns=["x", "y", "z"])
                        df["class"] = classes
                        df["pred"] = preds

                        val_ds = pca_conv2d3.transform(val_conv2d3.reshape(len(val_classes), -1))
                        trn_ds = pca_conv2d3.transform(trn_conv2d3.reshape(len(trn_classes), -1))

                        df_val = pd.DataFrame(val_ds, columns=['val_x', 'val_y'])

                        df_val["val_class"] = val_classes
                        df_val["val_pred"] = val_preds

                        df_trn = pd.DataFrame(trn_ds, columns=['trn_x', 'trn_y'])

                        df_trn["trn_class"] = trn_classes
                        df_trn["trn_pred"] = trn_preds

                        df = pd.concat([df, df_val, df_trn], axis=1)

                        df.to_csv(p.joinpath(f"conv2d3_PCA_W={W}.csv"), index=False)

                        # **cnn_seq_2**

                        if num == 0:
                            pca_seq_2 = PCA(n_components=n_comp)

                            ds_seq_2 = pca_seq_2.fit_transform(cnn_seq_2.reshape(len(classes), -1))
                        else:
                            ds_seq_2 = pca_seq_2.transform(cnn_seq_2.reshape(len(classes), -1))

                        if n_comp == 2:
                            df = pd.DataFrame(ds_seq_2, columns=['x', 'y'])
                        else:
                            df = pd.DataFrame(ds_seq_2, columns=["x", "y", "z"])
                        df["class"] = classes
                        df["pred"] = preds

                        val_ds = pca_seq_2.transform(val_cnn_seq_2.reshape(len(val_classes), -1))
                        trn_ds = pca_seq_2.transform(trn_cnn_seq_2.reshape(len(trn_classes), -1))

                        df_val = pd.DataFrame(val_ds, columns=['val_x', 'val_y'])

                        df_val["val_class"] = val_classes
                        df_val["val_pred"] = val_preds

                        df_trn = pd.DataFrame(trn_ds, columns=['trn_x', 'trn_y'])

                        df_trn["trn_class"] = trn_classes
                        df_trn["trn_pred"] = trn_preds

                        df = pd.concat([df, df_val, df_trn], axis=1)


                        df.to_csv(p.joinpath(f"cnn_seq_2_PCA_W={W}.csv"), index=False)

                        # **avg_pool**

                        if num == 0:
                            pca_avg_pool = PCA(n_components=n_comp)

                            ds_avg_pool = pca_avg_pool.fit_transform(avg_pool.reshape(len(classes), -1))
                        else:
                            ds_avg_pool = pca_avg_pool.transform(avg_pool.reshape(len(classes), -1))

                        if n_comp == 2:
                            df = pd.DataFrame(ds_avg_pool, columns=['x', 'y'])
                        else:
                            df = pd.DataFrame(ds_avg_pool, columns=["x", "y", "z"])
                        df["class"] = classes
                        df["pred"] = preds

                        val_ds = pca_avg_pool.transform(val_avg_pool.reshape(len(val_classes), -1))
                        trn_ds = pca_avg_pool.transform(trn_avg_pool.reshape(len(trn_classes), -1))

                        df_val = pd.DataFrame(val_ds, columns=['val_x', 'val_y'])

                        df_val["val_class"] = val_classes
                        df_val["val_pred"] = val_preds

                        df_trn = pd.DataFrame(trn_ds, columns=['trn_x', 'trn_y'])

                        df_trn["trn_class"] = trn_classes
                        df_trn["trn_pred"] = trn_preds

                        df = pd.concat([df, df_val, df_trn], axis=1)

                        df.to_csv(p.joinpath(f"avg_pool_PCA_W={W}.csv"), index=False)

                        for hook in hooks_list:
                            hook.remove()

                        with open(p.joinpath("all_generated.txt"), "w") as f:
                            f.write("1")


                        pca_pbar.refresh()
                        pca_pbar.update(1)
                
                    del pca_clear, pca_conv2d1, pca_conv2d2, pca_seq_1, pca_conv2d3, pca_seq_2, pca_avg_pool
                    del ds, ds_clear, ds_conv2d1, ds_conv2d2, ds_seq_1, ds_conv2d3, ds_seq_2, ds_avg_pool
                    del t_l, trn_l, val_l
                    del X, y, X_subset, y_subset, X_subset_val, y_subset_val, X_subset_trn, y_subset_trn
                    del trn_classes, trn_preds, trn_ds, trn_color_names, trn_conv2d1, trn_conv2d2, trn_cnn_seq_1, trn_conv2d3, trn_cnn_seq_2, trn_avg_pool
                    del val_classes, val_preds, val_ds, val_color_names, val_conv2d1, val_conv2d2, val_cnn_seq_1, val_conv2d3, val_cnn_seq_2, val_avg_pool
                    del trn_hook_avgpool, trn_hook_cnn_seq_1, trn_hook_cnn_seq_2, trn_hook_conv2d1, trn_hook_conv2d2, trn_hook_conv2d3
                    del val_hook_avgpool, val_hook_cnn_seq_1, val_hook_cnn_seq_2, val_hook_conv2d1, val_hook_conv2d2, val_hook_conv2d3
                    del val_hooks_list, trn_hooks_list, hooks_list
                    del avgpool_list_trn, avgpool_list_val, cnn_seq_1_list_trn, cnn_seq_1_list_val, cnn_seq_2_list_trn, cnn_seq_2_list_val
                    del conv2d1_list_trn, conv2d1_list_val, conv2d2_list_trn, conv2d2_list_val, conv2d3_list_trn, conv2d3_list_val
                    del avgpool_list, cnn_seq_1_list, cnn_seq_2_list, conv2d1_list, conv2d2_list, conv2d3_list
                    del avg_pool, cnn_seq_1, cnn_seq_2, conv2d1, conv2d2, conv2d3
                    del activation, activation_trn, activation_val

                

                fig, axs = plt.subplots(
                    nrows=8, ncols=6,
                    figsize=(30, 24),
                )

                plt.suptitle(f"Internal Clustering {n_comp}D | {folder} | PCA", fontsize=20)
                plt.subplots_adjust(top=0.95)

                pca_pbar.set_description("PCA | Plotting")
                
                for nrow, layer_name in enumerate(layer_names):
                    for ncol, w in enumerate(w_tab):
                        if layer_name == 'avg_pool_predicted':
                            p_tmp = folder_pca.joinpath(f"W={w}").joinpath(f"{'avg_pool'}_PCA_W={w}.csv")
                            p_0 = folder_pca.joinpath(f"W=0").joinpath(f"{'avg_pool'}_PCA_W=0.csv")
                        else:
                            p_tmp = folder_pca.joinpath(f"W={w}").joinpath(f"{layer_name}_PCA_W={w}.csv")
                            p_0 = folder_pca.joinpath(f"W=0").joinpath(f"{layer_name}_PCA_W=0.csv")
                        df = pd.read_csv(p_tmp)
                        df_0 = pd.read_csv(p_0)

                        if layer_name != 'avg_pool_predicted':
                            df['colors'] = df['class'].map({0: 'red', 1: 'blue'})
                            df_to = df.loc[df['class'] == 'Topological']
                            df_tr = df.loc[df['class'] == 'Trivial']
                            df_to_0 = df_0.loc[df_0['class'] == 'Topological']
                            df_tr_0 = df_0.loc[df_0['class'] == 'Trivial']

                            df_trn_to = df.loc[df['trn_class'] == 'Trn_Topological']
                            df_trn_tr = df.loc[df['trn_class'] == 'Trn_Trivial']

                            df_val_to = df.loc[df['val_class'] == 'Val_Topological']
                            df_val_tr = df.loc[df['val_class'] == 'Val_Trivial']
                        else:
                            df['colors'] = df['pred'].map({0: 'red', 1: 'blue'})
                            df_to = df.loc[df['pred'] == 'Topological']
                            df_tr = df.loc[df['pred'] == 'Trivial']
                            df_to_0 = df_0.loc[df_0['pred'] == 'Topological']
                            df_tr_0 = df_0.loc[df_0['pred'] == 'Trivial']

                            df_trn_to = df.loc[df['trn_pred'] == 'Trn_Topological']
                            df_trn_tr = df.loc[df['trn_pred'] == 'Trn_Trivial']

                            df_val_to = df.loc[df['val_pred'] == 'Val_Topological']
                            df_val_tr = df.loc[df['val_pred'] == 'Val_Trivial']
                        
                        
                    

                        ax = axs[nrow, ncol]

                        ax.scatter(df_trn_to['trn_x'], df_trn_to['trn_y'], marker='P', s=1, c="darkred", label="Topological_trn")
                        ax.scatter(df_trn_tr['trn_x'], df_trn_tr['trn_y'], marker='*', s=1, c="navy", label="Trivial_trn")

                        ax.scatter(df_val_to['val_x'], df_val_to['val_y'], marker='P', s=1, c="darkred", label="Topological_val")
                        ax.scatter(df_val_tr['val_x'], df_val_tr['val_y'], marker='*', s=1, c="navy", label="Trivial_val")

                        ax.scatter(df_to['x'], df_to['y'], s=1, c="red", label="Topological")
                        ax.scatter(df_tr['x'], df_tr['y'], s=1, c="blue", label="Trivial")
                        ax.scatter(df_to_0['x'], df_to_0['y'], s=1, c="orange", alpha=0.5, label="Topological_0")
                        ax.scatter(df_tr_0['x'], df_tr_0['y'], s=1, c="cyan", alpha=0.5, label="Trivial_0")
                        ax.set_title(f"{layer_name} W={w}")
                        ax.set_xlim(np.min(pd.concat([df_0[['x']], df[['x']]]).dropna().to_numpy())*1.1, np.max(pd.concat([df_0[['x']], df[['x']]]).dropna().to_numpy())*1.1)
                        ax.set_ylim(np.min(pd.concat([df_0[['y']], df[['y']]]).dropna().to_numpy())*1.1, np.max(pd.concat([df_0[['y']], df[['y']]]).dropna().to_numpy())*1.1)
                        # ax.set_xticks([])
                        # ax.set_yticks([])
                        if ncol == len(w_tab)-1:
                            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

                        pca_pbar.set_description(f"PCA | Plotting | {layer_name} W={w}")
                        pca_pbar.refresh()

                pca_pbar.set_description(f"PCA | Plotting | Saving")
                pca_pbar.refresh()
                plt.savefig(folder.joinpath(f"PCA.png"), bbox_inches='tight')
                plt.savefig(folder.joinpath(f"PCA.pdf"), bbox_inches='tight')
                plt.close()
                pca_pbar.set_description(f"PCA | Plotting | Done")  
                pca_pbar.refresh()
                pca_pbar.close()

                # %%

                # UMAP
                umap_pbar = tqdm(total=len(test_paths), desc="UMAP", position=1, leave=False, unit="dataset", colour="blue")
                folder_umap = folder.joinpath("UMAP")
                folder_umap.mkdir(parents=True, exist_ok=True)

                p = folder_umap

                umap_pbar.set_description("UMAP | Initializing")

                generate_umap = True
                try:
                    # Get the list of all subdirectories
                    subdirectories = [x[0] for x in os.walk(folder_umap)]
                    # Check if all subdirectories contain the file 'all_generated.txt'
                    all_contain_file = all(os.path.isfile(os.path.join(subdir, 'all_generated.txt')) for subdir in subdirectories)

                    if all_contain_file:
                        generate_umap = False
                    else:
                        pass
                except:
                    generate_umap = True

                if generate_umap:
                    activation = {}
                    def getActivation(name):
                        # the hook signature
                        def hook(model, input, output):
                            activation[name] = output.detach().cpu().numpy()
                        return hook

                    # %%
                    hook_conv2d1 = model.cnn_layers_1[0].register_forward_hook(getActivation('cnn_layers_1_condv2d1'))
                    hook_conv2d2 = model.cnn_layers_1[4].register_forward_hook(getActivation('cnn_layers_1_condv2d2'))
                    hook_cnn_seq_1 = model.cnn_layers_1.register_forward_hook(getActivation('cnn_layers_1'))
                    hook_conv2d3 = model.cnn_layers_2[0].register_forward_hook(getActivation('cnn_layers_2_condv2d1'))
                    hook_cnn_seq_2 = model.cnn_layers_2.register_forward_hook(getActivation('cnn_layers_2'))
                    hook_avgpool = model.avg_pool[0].register_forward_hook(getActivation('avg_pool'))

                    hooks_list = [hook_conv2d1, hook_conv2d2, hook_cnn_seq_1, hook_conv2d3, hook_cnn_seq_2, hook_avgpool]

                    # %%
                    conv2d1_list, conv2d2_list, cnn_seq_1_list, conv2d3_list, cnn_seq_2_list, avgpool_list = [], [], [], [], [], []

                    if perturb:
                        # Test data
                        x_000 = test_loader_000.dataset.X
                        y_000 = test_loader_000.dataset.Y

                        x_001 = test_loader_001.dataset.X
                        y_001 = test_loader_001.dataset.Y

                        x_005 = test_loader_005.dataset.X
                        y_005 = test_loader_005.dataset.Y

                        ds_len = y_000.shape[0]

                        if exclude_transition:
                            x_tot = torch.cat([
                                torch.cat([x_000[:(ds_len//2 - ds_len//10)], x_000[(ds_len//2 + ds_len//10):]],axis=0),
                                torch.cat([x_001[:(ds_len//2 - ds_len//10)], x_001[(ds_len//2 + ds_len//10):]],axis=0),
                                torch.cat([x_005[:(ds_len//2 - ds_len//10)], x_005[(ds_len//2 + ds_len//10):]],axis=0),
                                ], axis=0)
                            y_tot = torch.cat([
                                torch.cat([y_000[:(ds_len//2 - ds_len//10)], y_000[(ds_len//2 + ds_len//10):]],axis=0),
                                torch.cat([y_001[:(ds_len//2 - ds_len//10)], y_001[(ds_len//2 + ds_len//10):]],axis=0),
                                torch.cat([y_005[:(ds_len//2 - ds_len//10)], y_005[(ds_len//2 + ds_len//10):]],axis=0),
                                ], axis=0)
                        else:
                            x_tot = torch.cat([x_000, x_001, x_005], axis=0)
                            y_tot = torch.cat([y_000, y_001, y_005], axis=0)

                        t_l = [(x_tot.clone(), y_tot.clone())]
                    else:
                        ds_cl = Importer(disorderless_ds_path, clustering_batch_size)
                        t_l = ds_cl.get_test_loader()


                    ds_list = []
                    test_loader_list = [t_l]
                    w_tab = ["0"]
                    desc_umap = "UMAP"
                    for num, test_path in enumerate(test_paths[1:]):
                        ds_list.append(Importer(test_path, clustering_batch_size))
                        test_loader_list.append(ds_list[-1].get_test_loader())
                        W = ds_list[-1].W_tab[0]
                        if W == 0:
                            W = f"{W:.0f}"
                        elif 0 < W < 0.5 :
                            W = f"{W:.2f}"
                        else:
                            W = f"{W:.1f}"
                        w_tab.append(W)

                    X_tot = torch.tensor([])
                    y_tot = torch.tensor([])
                    lab_list = []
                    pred_list = []
                    class_names = ['Trivial', 'Topological']
                    lab_uniq = [class_names[i]+f"_W={W}" for W in w_tab for i in [0, 1]]
                    for loader, w_tmp in zip(test_loader_list, w_tab):
                        for X, y in loader:
                            ds_len = y.shape[0]
                            if exclude_transition:
                                X_tmp = torch.cat([
                                    X[:(ds_len//2 - ds_len//10)], # We remove 10% of the data on both sides of transition
                                    X[(ds_len//2 + ds_len//10):]
                                ], axis=0)
                                y_tmp = torch.cat([
                                    y[:(ds_len//2 - ds_len//10)],
                                    y[(ds_len//2 + ds_len//10):]
                                ], axis=0)
                            else:
                                X_tmp = X
                                y_tmp = y
                            out_tmp = model(X_tmp.to(device))

                            X_tot = torch.cat([
                                X_tot, 
                                X_tmp
                            ], axis=0)
                            y_tot = torch.cat([
                                y_tot, 
                                y_tmp
                            ], axis=0)

                            lab_list.append(np.array([class_names[int(i)]+f"_W={w_tmp}" for i in y_tmp.detach().cpu().numpy()]))
                            pred_list.append(np.array([class_names[int(i)]+f"_W={w_tmp}" for i in out_tmp.argmax(axis=1).detach().cpu().numpy()]))
                    lab_list = np.concatenate(lab_list, axis=0)
                    pred_list = np.concatenate(pred_list, axis=0)


                    X_tot = X_tot.to(device)
                    y_tot = y_tot.to(device)
                    out = model(X_tot)

                    conv2d1_list.append(activation['cnn_layers_1_condv2d1'])
                    conv2d2_list.append(activation['cnn_layers_1_condv2d2'])
                    cnn_seq_1_list.append(activation['cnn_layers_1'])
                    conv2d3_list.append(activation['cnn_layers_2_condv2d1'])
                    cnn_seq_2_list.append(activation['cnn_layers_2'])
                    avgpool_list.append(activation['avg_pool'])


                    # %%
                    # Join along the first axis
                    conv2d1 = np.concatenate(np.array(conv2d1_list), axis=0)
                    conv2d2 = np.concatenate(np.array(conv2d2_list), axis=0)
                    conv2d3 = np.concatenate(np.array(conv2d3_list), axis=0)
                    cnn_seq_1 = np.concatenate(np.array(cnn_seq_1_list), axis=0)
                    cnn_seq_2 = np.concatenate(np.array(cnn_seq_2_list), axis=0)
                    avg_pool = np.concatenate(np.array(avgpool_list), axis=0)

                    classes = lab_list
                    preds = pred_list

                    # UMAP clean

                    umap_clear = UMAP(**umap_params)

                    ds_clear_umap = umap_clear.fit_transform(X_tot.reshape(len(classes), -1).detach().cpu().numpy())

                    df = pd.DataFrame({'x':ds_clear_umap[:,0], 'y':ds_clear_umap[:,1], 'class':classes, 'pred':preds})

                    df.to_csv(p.joinpath("clear_UMAP.csv"), index=False)

                    umap_pbar.set_description("UMAP | clear")
                    umap_pbar.refresh()
                    umap_pbar.update(1)

                    # %% [markdown]
                    # **conv2d1**

                    umap_conv2d1 = UMAP(**umap_params)

                    ds_conv2d1_umap = umap_conv2d1.fit_transform(conv2d1.reshape(len(classes), -1))
                    
                    if n_comp == 2:
                        df = pd.DataFrame({'x':ds_conv2d1_umap[:,0], 'y':ds_conv2d1_umap[:,1], 'class':classes, 'pred':preds})
                    else:
                        df = pd.DataFrame({'x':ds_conv2d1_umap[:,0], 'y':ds_conv2d1_umap[:,1], 'z':ds_conv2d1_umap[:,2], 'class':classes, 'pred':preds})

                    df.to_csv(p.joinpath("conv2d1_UMAP.csv"), index=False)

                    umap_pbar.set_description("UMAP | conv2d1")
                    umap_pbar.refresh()
                    umap_pbar.update(1)

                    # %% [markdown]
                    # **conv2d2**

                    # %%
                    umap_conv2d2 = UMAP(**umap_params)

                    ds_conv2d2_umap = umap_conv2d2.fit_transform(conv2d2.reshape(len(classes), -1))
                    if n_comp == 2:
                        df = pd.DataFrame({'x':ds_conv2d2_umap[:,0], 'y':ds_conv2d2_umap[:,1], 'class':classes, 'pred':preds})
                    else:
                        df = pd.DataFrame({'x':ds_conv2d2_umap[:,0], 'y':ds_conv2d2_umap[:,1], 'z':ds_conv2d2_umap[:,2], 'class':classes, 'pred':preds})

                    df.to_csv(p.joinpath("conv2d2_UMAP.csv"), index=False)

                    umap_pbar.set_description("UMAP | conv2d2")
                    umap_pbar.refresh()
                    umap_pbar.update(1)

                    # %% [markdown]
                    # **cnn_seq_1**

                    # %%
                    umap_seq_1 = UMAP(**umap_params)

                    ds_seq_1_umap = umap_seq_1.fit_transform(cnn_seq_1.reshape(len(classes), -1))
                    
                    if n_comp == 2:
                        df = pd.DataFrame({'x':ds_seq_1_umap[:,0], 'y':ds_seq_1_umap[:,1], 'class':classes, 'pred':preds})
                    else:
                        df = pd.DataFrame({'x':ds_seq_1_umap[:,0], 'y':ds_seq_1_umap[:,1], 'z':ds_seq_1_umap[:,2], 'class':classes, 'pred':preds})

                    df.to_csv(p.joinpath("cnn_seq_1_UMAP.csv"), index=False)

                    umap_pbar.set_description("UMAP | cnn_seq_1")
                    umap_pbar.refresh()
                    umap_pbar.update(1)

                    # %% [markdown]
                    # **conv2d3**

                    # %%
                    umap_conv2d3 = UMAP(**umap_params)

                    ds_conv2d3_umap = umap_conv2d3.fit_transform(conv2d3.reshape(len(classes), -1))

                    if n_comp == 2:
                        df = pd.DataFrame({'x':ds_conv2d3_umap[:,0], 'y':ds_conv2d3_umap[:,1], 'class':classes, 'pred':preds})
                    else:
                        df = pd.DataFrame({'x':ds_conv2d3_umap[:,0], 'y':ds_conv2d3_umap[:,1], 'z':ds_conv2d3_umap[:,2], 'class':classes, 'pred':preds})

                    df.to_csv(p.joinpath("conv2d3_UMAP.csv"), index=False)

                    umap_pbar.set_description("UMAP | conv2d3")
                    umap_pbar.refresh()
                    umap_pbar.update(1)

                    # %% [markdown]
                    # **cnn_seq_2**

                    # %%
                    umap_seq_2 = UMAP(**umap_params)

                    ds_seq_2_umap = umap_seq_2.fit_transform(cnn_seq_2.reshape(len(classes), -1))

                    if n_comp == 2:
                        df = pd.DataFrame({'x':ds_seq_2_umap[:,0], 'y':ds_seq_2_umap[:,1], 'class':classes, 'pred':preds})
                    else:
                        df = pd.DataFrame({'x':ds_seq_2_umap[:,0], 'y':ds_seq_2_umap[:,1], 'z':ds_seq_2_umap[:,2], 'class':classes, 'pred':preds})

                    df.to_csv(p.joinpath("cnn_seq_2_UMAP.csv"), index=False)

                    umap_pbar.set_description("UMAP | cnn_seq_2")
                    umap_pbar.refresh()
                    umap_pbar.update(1)

                    # %% [markdown]
                    # **avg_pool**

                    # %%
                    umap_avg_pool = UMAP(**umap_params)

                    ds_avg_pool_umap = umap_avg_pool.fit_transform(avg_pool.reshape(len(classes), -1))

                    if n_comp == 2:
                        df = pd.DataFrame({'x':ds_avg_pool_umap[:,0], 'y':ds_avg_pool_umap[:,1], 'class':classes, 'pred':preds})
                    else:
                        df = pd.DataFrame({'x':ds_avg_pool_umap[:,0], 'y':ds_avg_pool_umap[:,1], 'z':ds_avg_pool_umap[:,2], 'class':classes, 'pred':preds})

                    df.to_csv(p.joinpath("avg_pool_UMAP.csv"), index=False)

                    umap_pbar.set_description("UMAP | avg_pool")
                    umap_pbar.refresh()
                    umap_pbar.update(1)


                    # %%
                    for hook in hooks_list:
                        hook.remove()

                    with open(p.joinpath("all_generated.txt"), "w") as f:
                            f.write("1")

                    umap_pbar.refresh()

                    del X_tot, y_tot, out, conv2d1_list, conv2d2_list, cnn_seq_1_list, conv2d3_list, cnn_seq_2_list, avgpool_list, loader
                    del ds_clear_umap, ds_conv2d1_umap, ds_conv2d2_umap, ds_seq_1_umap, ds_conv2d3_umap, ds_seq_2_umap, ds_avg_pool_umap
                    del umap_clear, umap_conv2d1, umap_conv2d2, umap_seq_1, umap_conv2d3, umap_seq_2, umap_avg_pool
                    del activation, hook, hook_conv2d1, hook_conv2d2, hook_cnn_seq_1, hook_conv2d3, hook_cnn_seq_2, hook_avgpool, hooks_list
                    del ds_list, test_loader_list, w_tab, desc_umap, X, y, out_tmp, X_tmp, y_tmp, ds_len
                    del avg_pool, cnn_seq_1, cnn_seq_2, conv2d1, conv2d2, conv2d3
                    del t_l, lab_list, pred_list, lab_uniq

                umap_pbar.set_description("UMAP | Plotting")
                umap_pbar.refresh()

                fig, axs = plt.subplots(
                    nrows=8, ncols=6,
                    figsize=(30, 20),
                )

                layer_names = ['clear', 'conv2d1', 'conv2d2', 'cnn_seq_1', 'conv2d3', 'cnn_seq_2', 'avg_pool', 'avg_pool_predicted']
                w_tab = ["0", "0.15", "0.5", "1.0", "2.0", "3.0"]

                plt.suptitle(f"Internal Clustering | {folder} | UMAP", fontsize=20)
                plt.subplots_adjust(top=0.95)

                for nrow, layer_name in enumerate(tqdm(layer_names, desc="Layer", leave=False, colour="red")):
                    for ncol, w in enumerate(w_tab):
                        umap_pbar.set_description(f"UMAP | Plotting | {layer_name} W={w}")
                        umap_pbar.refresh()
                        p_tmp = folder_umap.joinpath(f"{layer_name if layer_name != 'avg_pool_predicted' else 'avg_pool'}_UMAP.csv")

                        df = pd.read_csv(p_tmp)
                        
                        if layer_name != 'avg_pool_predicted':
                            df_to = df.loc[df['class'] == f'Topological_W={w}']
                            df_tr = df.loc[df['class'] == f'Trivial_W={w}']
                            df_to_0 = df.loc[df['class'] == 'Topological_W=0']
                            df_tr_0 = df.loc[df['class'] == 'Trivial_W=0']
                        else:
                            df_to = df.loc[df['pred'] == f'Topological_W={w}']
                            df_tr = df.loc[df['pred'] == f'Trivial_W={w}']
                            df_to_0 = df.loc[df['pred'] == 'Topological_W=0']
                            df_tr_0 = df.loc[df['pred'] == 'Trivial_W=0']

                        ax = axs[nrow, ncol]
                        if ncol == 0:
                            ax.scatter(df_to['x'], df_to['y'], s=1, c="orange")
                            ax.scatter(df_tr['x'], df_tr['y'], s=1, c="cyan")
                        else:
                            ax.scatter(df_to['x'], df_to['y'], s=1, c="red")
                            ax.scatter(df_tr['x'], df_tr['y'], s=1, c="blue")
                        ax.scatter(df_to_0['x'], df_to_0['y'], s=1, c="orange")
                        ax.scatter(df_tr_0['x'], df_tr_0['y'], s=1, c="cyan")
                        ax.set_title(f"{layer_name} W={w}")
                        # ax.set_xticks([])
                        # ax.set_yticks([])
                        if ncol == len(w_tab)-1:
                            ax.legend(['Topological', 'Trivial', 'Topological_0', 'Trivial_0'], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

                umap_pbar.set_description(f"UMAP | Plotting | Saving")
                plt.savefig(folder.joinpath(f"UMAP.png"), bbox_inches='tight')
                plt.savefig(folder.joinpath(f"UMAP.pdf"), bbox_inches='tight')
                plt.close("all")
                umap_pbar.set_description(f"UMAP | Plotting | Done")
                umap_pbar.refresh()

                umap_pbar.close()

                del x_000, x_001, x_005
                del y_000, y_001, y_005
                del ds_000, ds_001, ds_005
                del df, df_0, df_to, df_tr, df_trn, df_val, df_to_0, df_tr_0, df_trn_to, df_trn_tr, df_val_to, df_val_tr
                del f, fig, ax, axs
                del test_loader_000, test_loader_001, test_loader_005
                del tr_loader_000, tr_loader_001, tr_loader_005
                del val_loader_000, val_loader_001, val_loader_005

            if do_cam:

                # CAM generation
                test_ds_path = pathlib.Path(f"./Datasets/M=50/viz_models/W=0.00")
                test_ds = Importer(test_ds_path, -1)
                test_loader = test_ds.get_test_loader(realization=0)
                cam_pbar = tqdm(total=test_loader.dataset.Y.shape[0], desc="CAM", position=1, leave=False, unit="sample", colour="blue")
                cams_folder = folder.joinpath("CAM")
                cams_folder.mkdir(parents=True, exist_ok=True)
                n_reals = 1

                model_name = folder.joinpath("trained_model")
                # Just one real, so this loop is redundant, but it's here for backward compatibility
                for real in range(n_reals):
                    cams_arr, preds = generate_CAMs(model, test_loader, class_names, device, test_ds)
                    cams_arr = np.array(cams_arr).squeeze()
                    cams_arr_np = np.zeros(shape=(cams_arr.shape[0], 2, 50, 50))
                    for r_no, (el1, el2) in enumerate(cams_arr):
                        cams_arr_np[r_no] = np.array([np.array(el1), np.array(el2)])
                    del el1, el2
                    del cams_arr
                    viz_tuple = (cams_arr_np, preds)
                    with open(cams_folder.joinpath(f"viz_tuple.pkl"), "wb") as f:
                        pickle.dump(viz_tuple, f)
                    del f

                    for img_ind in range(test_loader.dataset.Y.shape[0]):
                        viz_final(cams_arr_np, preds, img_ind, folder, test_ds, test_ds_path, test_loader, save=cams_folder.joinpath(f"ind={img_ind}_v={((img_ind+1)/100):.2f}.png"))
                        cam_pbar.update(1)
                cam_pbar.close()
            plt.close("all")

            models_pbar.update(1)



    
    





