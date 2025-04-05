#!/usr/bin/env python
import os
from os.path import join
import gc

import joblib
import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms
import torchvision

from lolip.extra_ood_utils import get_ood_data_paths
from lolip.models.torch_utils import archs
from lolip.variables import auto_var


MODELPATH = "./models/out_of_sample"

def get_reprs(model, dset, batch_size=64, device="cuda"):
    model.eval().to(device)
    if isinstance(dset, np.ndarray):
        dset = torch.utils.data.TensorDataset(
            torch.from_numpy(dset.transpose(0, 3, 1, 2)).float(),
            torch.ones(len(dset))
        )

    loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=24)

    ret = []
    for (x, _) in tqdm(loader):
        with torch.no_grad():
            output = model.get_repr(x.to(device))
        ret.append(output.cpu().numpy())
    del loader
    return np.concatenate(ret, axis=0)

def calc_no_wo(dset_name, reprs_file):
    if os.path.exists(reprs_file):
        reprs = joblib.load(reprs_file)
    else:
        reprs = {}

    if "cifar" in dset_name:
        image_files, ood_names = get_ood_data_paths(dset_name, "./data/cifar-ood/")
    elif dset_name == "imgnet":
        image_files, ood_names = get_ood_data_paths(dset_name, "/tmp2/")

    model_names = [
        'natural',
    ]

    if "cifar" in dset_name:
        if dset_name == "cifar10":
            ds_name = f'cifar10'
        elif dset_name == "cifar100":
            ds_name = f'cifar100coarse'
        model_paths = [
            f"./models/out_of_sample/pgd-64-{ds_name}-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0-ep0070.pt",
        ]
    elif dset_name == "imgnet":
        ds_name = f'aug10-imgnet100'
        model_paths = [
            join(MODELPATH, f"pgd-128-{ds_name}-70-1.0-0.01-ce-vtor2-ResNet50Norm01-0.0-2-adam-0-0.0-ep0070.pt"),
        ]

    print(ds_name)

    trnX, trny, tstX, _, _ = auto_var.get_var_with_argument("dataset", ds_name)


    for model_name, model_path in zip(model_names, model_paths):
        if not os.path.exists(model_path):
            print(f"`{model_path}` does not exist. skipping...")
            continue
        key = (ds_name, model_name)
        reprs.setdefault(key, {})

        arch_name = model_path.split("-")[model_path.split("-").index("vtor2") + 1]
        model = getattr(archs, arch_name)(n_classes=len(np.unique(trny)), n_channels=3)
        model.load_state_dict(torch.load(model_path)['model_state_dict'])

        trn_reprs = get_reprs(model, trnX)
        reprs[key]['trn'] = trn_reprs
        tst_reprs = get_reprs(model, tstX)
        reprs[key]['tst'] = tst_reprs

        for ood_name, image_file in zip(ood_names, image_files):
            if ood_name in reprs[key]:
                continue
            if "cifar10" in dset_name:
                tX = (np.load(image_file) / 255).astype(np.float32)
            elif dset_name == "imgnet":
                dset = torchvision.datasets.ImageFolder(
                    image_file,
                    transforms.Compose([transforms.Resize(128), transforms.CenterCrop(112), transforms.ToTensor(), ])
                )
                _loader = torch.utils.data.DataLoader(dset, batch_size=64, shuffle=False, num_workers=24)
                tX = np.concatenate([x.numpy() for (x, _) in _loader], axis=0).transpose(0, 2, 3, 1)
            other_ood_pred = get_reprs(model, tX)
            reprs[key][ood_name] = other_ood_pred

    joblib.dump(reprs, reprs_file)
    del model
    gc.collect()
    print("done.")

def main(dset_name, reprs_file):
    if os.path.exists(reprs_file):
        reprs = joblib.load(reprs_file)
    else:
        reprs = {}

    if "cifar" in dset_name:
        image_files, ood_names = get_ood_data_paths(dset_name, "./data/cifar-ood/")
    elif dset_name == "imgnet":
        image_files, ood_names = get_ood_data_paths(dset_name, "/tmp2/")

    model_names = [
        'natural',
        'TRADES(1)',
        'TRADES(2)',
        'TRADES(4)',
        'TRADES(8)',
        'AT(2)',
        'ball',
    ]

    if "cifar" in dset_name:
        ood_classes = [0, 4, 9]
    elif "imgnet" == dset_name:
        ood_classes = [0, 1, 2]

    for ood_class in ood_classes:
        if "cifar" in dset_name:
            if dset_name == "cifar10":
                ds_name = f'cifar10wo{ood_class}'
            elif dset_name == "cifar100":
                ds_name = f'cifar100coarsewo{ood_class}'
            model_paths = [
                f"./models/out_of_sample/pgd-64-{ds_name}-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0-ep0070.pt",
                f"./models/out_of_sample/pgd-64-{ds_name}-70-1.0-0.01-trades6ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0-ep0070.pt",
                f"./models/out_of_sample/pgd-64-{ds_name}-70-2.0-0.01-trades6ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0-ep0070.pt",
                f"./models/out_of_sample/pgd-64-{ds_name}-70-4.0-0.01-trades6ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0-ep0070.pt",
                f"./models/out_of_sample/pgd-64-{ds_name}-70-8.0-0.01-trades6ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0-ep0070.pt",
                f"./models/out_of_sample/pgd-64-{ds_name}-70-2.0-0.01-advce-vtor2-WRN_40_10-0.0-2-adam-0-0.0-ep0070.pt",
                f"./models/out_of_sample/pgd-64-{ds_name}-70-1.0-0.01-cusgradtrades6v2autwotimesce-vtor2-WRN_40_10-halfclose-0.0-2-adam-0-0.0-ep0070.pt",
            ]
        elif dset_name == "imgnet":
            ds_name = f'aug10-imgnet100wo{ood_class}'
            model_paths = [
                join(MODELPATH, f"pgd-128-{ds_name}-70-1.0-0.01-ce-vtor2-ResNet50Norm01-0.0-2-adam-0-0.0-ep0070.pt"),
                join(MODELPATH, f"pgd-128-{ds_name}-70-1.0-0.01-trades6ce-vtor2-ResNet50Norm01-0.0-2-adam-0-0.0-ep0070.pt"),
                join(MODELPATH, f"pgd-128-{ds_name}-70-2.0-0.01-trades6ce-vtor2-ResNet50Norm01-0.0-2-adam-0-0.0-ep0070.pt"),
                join(MODELPATH, f"pgd-128-{ds_name}-70-4.0-0.01-trades6ce-vtor2-ResNet50Norm01-0.0-2-adam-0-0.0-ep0070.pt"),
                join(MODELPATH, f"pgd-128-{ds_name}-70-8.0-0.01-trades6ce-vtor2-ResNet50Norm01-0.0-2-adam-0-0.0-ep0070.pt"),
                join(MODELPATH, f"pgd-128-{ds_name}-70-2.0-0.01-advce-vtor2-ResNet50Norm01-0.0-2-adam-0-0.0-ep0070.pt"),
                join(MODELPATH, f"pgd-128-{ds_name}-70-1.0-0.01-cusgradtrades6v2autwotimesce-vtor2-ResNet50Norm01-halfclose-0.0-2-adam-0-0.0-ep0070.pt"),
            ]

        print(ds_name)

        if dset_name == "imgnet":
            trnX, trny, tstX, tsty, (ood1X, ood2X) = auto_var.get_var_with_argument("dataset", ds_name)
        else:
            trnX, trny, tstX, tsty, (ood1X, ood2X, _, _) = auto_var.get_var_with_argument("dataset", ds_name)
        oodX = np.concatenate((ood1X, ood2X), axis=0)

        ood_classes = [ood_class]

        if dset_name == "cifar10":
            _, _, _, ori_tsty = auto_var.get_var_with_argument("dataset", dset_name)
        elif dset_name == "cifar100":
            _, _, _, ori_tsty = auto_var.get_var_with_argument("dataset", "cifar100coarse")

        oodXs = []
        if "cifar10" in dset_name:
            ori_tsty = np.tile(ori_tsty, 5)
            for image_file in tqdm(image_files, desc="[get oodXs]"):
                tX = np.load(image_file)
                if dset_name == "cifar10":
                    valid_classes = np.delete(np.arange(10), ood_classes)
                elif dset_name == "cifar100":
                    valid_classes = np.delete(np.arange(20), ood_classes)
                valid_idx = np.array([i for i, c in enumerate(ori_tsty) if c in valid_classes])
                oodXs.append((tX[valid_idx] / 255).astype(np.float32))
        elif dset_name == "imgnet":
            ori_dset = torchvision.datasets.ImageFolder("/tmp2/ImageNet100/ILSVRC2012_img_train/")
            for image_file in tqdm(image_files, desc="[get oodXs]"):
                dset = torchvision.datasets.ImageFolder(
                    image_file,
                    transforms.Compose([transforms.Resize(128), transforms.CenterCrop(112), transforms.ToTensor(), ])
                )
                _loader = torch.utils.data.DataLoader(dset, batch_size=64, shuffle=False, num_workers=24)
                tX = np.concatenate([x.numpy() for (x, _) in _loader], axis=0).transpose(0, 2, 3, 1)
                ty = np.concatenate([y.numpy() for (_, y) in _loader])

                valid_classes = []
                for c in ori_dset.classes:
                    valid_classes.append(dset.class_to_idx[c])
                assert len(valid_classes) == 100
                valid_classes = np.delete(np.array(valid_classes), ood_classes)
                valid_idx = np.array([i for i, c in enumerate(ty) if c in valid_classes])
                oodXs.append(tX[valid_idx])

        for model_name, model_path in zip(model_names, model_paths):
            if not os.path.exists(model_path):
                print(f"`{model_path}` does not exist. skipping...")
                continue
            key = (ds_name, model_name)
            reprs.setdefault(key, {})

            arch_name = model_path.split("-")[model_path.split("-").index("vtor2") + 1]
            model = getattr(archs, arch_name)(n_classes=len(np.unique(trny)), n_channels=3)
            model.load_state_dict(torch.load(model_path)['model_state_dict'])

            trn_reprs = get_reprs(model, trnX)
            reprs[key]['trn'] = trn_reprs
            tst_reprs = get_reprs(model, tstX)
            reprs[key]['tst'] = tst_reprs
            ood_reprs = get_reprs(model, oodX)
            reprs[key]['ncg'] = ood_reprs

            for i, tX in enumerate(oodXs):
                if ood_names[i] in reprs[key]:
                    continue
                other_ood_pred = get_reprs(model, tX)
                reprs[key][ood_names[i]] = other_ood_pred

        joblib.dump(reprs, reprs_file)
        del oodXs
        del model
        gc.collect()

    print("done.")
    joblib.dump(reprs, reprs_file)


if __name__ == "__main__":
    BASE_DIR = "./notebooks/"
    calc_no_wo("imgnet", os.path.join(BASE_DIR, f"nb_results/reprs-no-wo-imgnet100.pkl"))
    #calc_no_wo("cifar10", os.path.join(BASE_DIR, f"nb_results/reprs-no-wo-cifar10.pkl"))
    #calc_no_wo("cifar100", os.path.join(BASE_DIR, f"nb_results/reprs-no-wo-cifar100.pkl"))
    #main("cifar10", os.path.join(BASE_DIR, f"nb_results/reprs-cifar10.pkl"))
    #main("cifar100", os.path.join(BASE_DIR, f"nb_results/reprs-cifar100.pkl"))
    #main("imgnet", os.path.join(BASE_DIR, f"nb_results/reprs-imgnet.pkl"))
