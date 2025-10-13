import os
from pathlib import Path
import copy
import argparse
import torch
from torch import nn
import torchvision
from torchvision import models, transforms, utils
import numpy as np
from thingsvision import get_extractor, get_extractor_from_model

from group_cc import ModelWrapper
from imnet_val import get_imnet_val_acts
from stream_inspect import get_activations


def saveTopN(imgs, lista, neuron_id, n=9, path="", save_inh=False):
    topil = transforms.ToPILImage()
    
    neuron_path = os.path.join(path, neuron_id)
    Path(neuron_path).mkdir(exist_ok=True)

    grids_path = os.path.join(path, "all_grids")
    Path(grids_path).mkdir(exist_ok=True)

    exc_imgs = []
    for i in range(n):
        img = imgs[int(lista[i])]
        exc_imgs.append(img)
        img = topil(img)
        img.save(os.path.join(neuron_path, f"{i}.png"))

    grid = utils.make_grid(exc_imgs, nrow=3)
    grid = torchvision.transforms.ToPILImage()(grid)
    grid.save(os.path.join(grids_path, f"{neuron_id}.png"))

    if save_inh:
        inh_grids_path = os.path.join(path, "inh_grids")
        Path(inh_grids_path).mkdir(exist_ok=True)

        inh_imgs = []
        for i in range(n):
            img = imgs[int(lista[-(i+1)])]
            inh_imgs.append(img)
            img = topil(img)
            img.save(os.path.join(neuron_path, f"{i}_bottom.png"))

        grid = utils.make_grid(inh_imgs, nrow=3)
        grid = torchvision.transforms.ToPILImage()(grid)
        grid.save(os.path.join(inh_grids_path, f"{neuron_id}.png"))


def compare_tuning_curves(extractor, module_name, savedir, valdir):
    all_images, _, unrolled_acts, _, _, _ = \
        get_imnet_val_acts(extractor, module_name, valdir, sort_acts=False)

    unrolled_acts = np.array(unrolled_acts)
    if len(unrolled_acts.shape) == 2:
        unrolled_acts = np.transpose(unrolled_acts, (1, 0))

        for i in range(unrolled_acts.shape[0]):
            unrolled_act = unrolled_acts[i].tolist()
            all_ord_list = np.arange(len(all_images)).tolist()
            all_act_list, all_ord_sorted = zip(*sorted(zip(unrolled_act, all_ord_list), reverse=True))

            saveTopN(all_images, all_ord_sorted, f"{module_name}_neuron{i}", path=savedir, save_inh=True)

            np.save(os.path.join(savedir, f"{module_name}_unit{i}_unrolled_act.npy"),
                    np.array(unrolled_act))
            np.save(os.path.join(savedir, f"{module_name}_unit{i}_all_act_list.npy"),
                    np.array(list(all_act_list)))
            np.save(os.path.join(savedir, f"{module_name}_unit{i}_all_ord_sorted.npy"),
                    np.array(list(all_ord_sorted)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str)
    parser.add_argument('--layer', type=str)
    parser.add_argument('--imnet_val_dir', type=str)
    parser.add_argument('--savedir', type=str)
    args = parser.parse_args()

    savedir = os.path.join(args.savedir, args.network, args.layer)
    Path(savedir).mkdir(parents=True, exist_ok=True)
    
    extractor = get_extractor(
        model_name=args.network,
        source='torchvision',
        device="cuda" if torch.cuda.is_available() else "cpu",
        pretrained=True,
        model_parameters={'weights': 'IMAGENET1K_V2'}
    )

    compare_tuning_curves(extractor, args.layer, savedir, args.imnet_val_dir)