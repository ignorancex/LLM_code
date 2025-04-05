import sys
sys.path.append("../")
from os.path import join
import os
import gc

import faiss
import joblib
import numpy as np
import torch
from tqdm import tqdm
import torchvision
from torchvision import transforms
import torch.nn.functional as F

from lolip.extra_ood_utils import get_ood_data_paths
from lolip.models.torch_utils import archs
from lolip.variables import auto_var

MODELPATH = "./models/out_of_sample"


def get_preds(model, dset, batch_size=64, device="cuda"):
    model.eval().to(device)
    if isinstance(dset, np.ndarray):
        dset = torch.utils.data.TensorDataset(
            torch.from_numpy(dset).float(),
            torch.ones(len(dset))
        )
    loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=24)

    ret = []
    for (x, _) in tqdm(loader, desc="[get_preds]"):
        with torch.no_grad():
            output = F.softmax(model(x.to(device)), dim=1)
        ret.append(output.cpu().numpy())
    del loader
    return np.concatenate(ret, axis=0)


def calc_madry_cifar10():
    _, ood_names = get_ood_data_paths("cifar10", "./data/cifar-ood/")

    if os.path.exists("./notebooks/nb_results/feature-mdary-cifar10.pkl"):
        preds, nnidxs, dists = joblib.load("./notebooks/nb_results/feature-mdary-cifar10.pkl")
    else:
        preds, nnidxs, dists = {}, {}, {}

    ds_name = "CIFAR10"
    model_paths = [
        f"./models/out_of_sample/pgd-128-calcedrepr-cifar10-madry-cifar10-cifar_nat.pt.pkl-70-1.0-0.01-ce-vtor2-LargeMLP-0.9-2-sgd-0-0.0-ep0070.pt",
        f"./models/out_of_sample/pgd-128-calcedrepr-cifar10-madry-cifar10-cifar_nat.pt.pkl-70-2.0-0.01-trades6ce-vtor2-LargeMLP-0.9-2-sgd-0-0.0-ep0070.pt",
        f"./models/out_of_sample/pgd-128-calcedrepr-cifar10-madry-cifar10-cifar_nat.pt.pkl-70-4.0-0.01-trades6ce-vtor2-LargeMLP-0.9-2-sgd-0-0.0-ep0070.pt",
        f"./models/out_of_sample/pgd-128-calcedrepr-cifar10-madry-cifar10-cifar_nat.pt.pkl-70-8.0-0.01-trades6ce-vtor2-LargeMLP-0.9-2-sgd-0-0.0-ep0070.pt",
        f"./models/out_of_sample/pgd-128-calcedrepr-cifar10-madry-cifar10-cifar_nat.pt.pkl-70-1.0-0.01-advce-vtor2-LargeMLP-0.9-2-sgd-0-0.0-ep0070.pt",
    ]
    model_names = [
        'natural',
        'TRADES(2)',
        'TRADES(4)',
        'TRADES(8)',
        'AT(1)',
    ]

    # scripts/calc_pretrained_reprs.py
    repr_path = "./notebooks/nb_results/reprs-madry-cifar10.pkl"
    reprs = joblib.load(repr_path)

    key = ("cifar10", "madry_nat")
    trnX, tstX = reprs[key]["trn"], reprs[key]["tst"]

    _, trny, _, tsty, _ = auto_var.get_var_with_argument("dataset", "cifar10")

    index = faiss.IndexFlatL2(int(np.prod(trnX.shape[1:])))
    index.add(trnX.reshape(len(trnX), -1).astype(np.float32))

    for model_name, model_path in zip(model_names, model_paths):
        if not os.path.exists(model_path):
            print(f"`{model_path}` does not exist. skipping...")
            continue
        key = (ds_name, model_name)
        if key in preds:
            continue
        print(key)
        preds.setdefault(key, {})
        nnidxs.setdefault(key, {})
        dists.setdefault(key, {})

        arch_name = model_path.split("-")[model_path.split("-").index("vtor2") + 1]
        model = getattr(archs, arch_name)(n_classes=len(np.unique(trny)), n_features=(trnX.shape[1], ))
        model.load_state_dict(torch.load(model_path)['model_state_dict'])

        tst_preds = get_preds(model, tstX)
        preds[key]['tst'] = tst_preds
        print(f"test acc: {(tst_preds.argmax(1) == tsty).mean()}")

        for ood_name in tqdm(ood_names, desc="[get oodXs]"):
            tX = reprs[("cifar10", "madry_nat")][ood_name].astype(np.float32)
            if ((ds_name, "natural") in nnidxs) and (ood_name in nnidxs[(ds_name, "natural")]):
                nnidxs[key][ood_name] = nnidxs[(ds_name, "natural")][ood_name]
                dists[key][ood_name] = dists[(ds_name, "natural")][ood_name]
            else:
                D, I = index.search(tX.reshape(len(tX), -1), k=1)
                nnidxs[key][ood_name] = I[:, 0]
                dists[key][ood_name] = np.sqrt(D[:, 0])
            other_ood_pred = get_preds(model, tX)
            preds[key][ood_name] = other_ood_pred

    joblib.dump((preds, nnidxs, dists), "./notebooks/nb_results/feature-mdary-cifar10.pkl")
    print("done.")


def calc_imgnet100_wo():
    image_folders, ood_names = get_ood_data_paths("imgnet", "/tmp2")

    model_names = [
        'natural',
        'TRADES(2)',
        #'TRADES(4)',
        #'TRADES(8)',
        #'AT(.5)',
    ]

    if os.path.exists("./notebooks/nb_results/feature-imgnet100.pkl"):
        preds, nnidxs, dists = joblib.load("./notebooks/nb_results/feature-imgnet100.pkl")
    else:
        preds, nnidxs, dists = {}, {}, {}

    for ood_class in [0]:
        ds_name = f'calcedrepr-aug10-imgnet100wo{ood_class}-cwl2-128-aug10-imgnet100wo{ood_class}-70-1.0-0.01-ce-vtor2-ResNet50Norm01-0.0-2-adam-0-0.0.pkl'
        model_paths = [
            join(MODELPATH, f"pgd-128-{ds_name.replace('cwl2', 'pgd')}-70-1.0-0.01-ce-vtor2-LargeMLPv4-0.9-2-sgd-0-0.0-ep0070.pt"),
            join(MODELPATH, f"pgd-128-{ds_name.replace('cwl2', 'pgd')}-70-2.0-0.01-trades6ce-vtor2-LargeMLPv4-0.9-2-sgd-0-0.0-ep0070.pt"),
            join(MODELPATH, f"pgd-128-{ds_name.replace('cwl2', 'pgd')}-70-4.0-0.01-trades6ce-vtor2-LargeMLPv4-0.9-2-sgd-0-0.0-ep0070.pt"),
            join(MODELPATH, f"pgd-128-{ds_name.replace('cwl2', 'pgd')}-70-8.0-0.01-trades6ce-vtor2-LargeMLPv4-0.9-2-sgd-0-0.0-ep0070.pt"),
            join(MODELPATH, f"pgd-128-{ds_name.replace('cwl2', 'pgd')}-70-2.0-0.01-advce-vtor2-LargeMLPv4-0.9-2-sgd-0-0.0-ep0070.pt"),
        ]

        trnX, trny, tstX, _, (ood1X, ood2X) = auto_var.get_var_with_argument("dataset", ds_name)
        oodX = np.concatenate((ood1X, ood2X), axis=0)

        index = faiss.IndexFlatL2(int(np.prod(trnX.shape[1:])))
        index.add(trnX.reshape(len(trnX), -1).astype(np.float32))
        oodD, oodI = index.search(oodX.reshape(len(oodX), -1), k=1)

        # scripts/calc_ood_reprs.py
        #repr_path = "./notebooks/nb_results/reprs-no-wo-imgnet100.pkl"
        #reprs = joblib.load(repr_path)

        for model_name, model_path in zip(model_names, model_paths):
            if not os.path.exists(model_path):
                print(f"`{model_path}` does not exist. skipping...")
                continue
            key = (ds_name, model_name)
            print(key)
            preds.setdefault(key, {})
            nnidxs.setdefault(key, {})
            dists.setdefault(key, {})

            arch_name = "LargeMLPv4"
            model = getattr(archs, arch_name)(n_classes=len(np.unique(trny)),
                    n_channels=3, n_features=(trnX.shape[1], ))
            model.load_state_dict(torch.load(model_path)['model_state_dict'])

            if "tst" not in preds[key]:
                tst_preds = get_preds(model, tstX)
                preds[key]["tst"] = tst_preds

            if "ncg" not in preds[key]:
                ood_preds = get_preds(model, oodX)
                preds[key]["ncg"] = ood_preds
                nnidxs[key]["ncg"] = oodI[:, 0]
                dists[key]["ncg"] = np.sqrt(oodD[:, 0])

        joblib.dump((preds, nnidxs, dists), "./notebooks/nb_results/feature-imgnet100.pkl")

    joblib.dump((preds, nnidxs, dists), "./notebooks/nb_results/feature-imgnet100.pkl")
    print("done.")

def calc_imgnet100():
    image_folders, ood_names = get_ood_data_paths("imgnet", "/tmp2")

    model_names = [
        'natural',
        'TRADES(2)',
        'TRADES(4)',
        'TRADES(8)',
        'AT(.5)',
    ]

    if os.path.exists("./notebooks/nb_results/feature-imgnet-c.pkl"):
        preds, nnidxs, dists = joblib.load("./notebooks/nb_results/feature-imgnet-c.pkl")
    else:
        preds, nnidxs, dists = {}, {}, {}

    ds_name = 'calcedrepr-aug10-imgnet100-cwl2-128-aug10-imgnet100-70-1.0-0.01-ce-vtor2-ResNet50Norm01-0.0-2-adam-0-0.0.pkl'
    model_paths = [
        join(MODELPATH, f"pgd-128-{ds_name.replace('cwl2', 'pgd')}-70-1.0-0.01-ce-vtor2-LargeMLPv4-0.9-2-sgd-0-0.0-ep0070.pt"),
        join(MODELPATH, f"pgd-128-{ds_name.replace('cwl2', 'pgd')}-70-2.0-0.01-trades6ce-vtor2-LargeMLPv4-0.9-2-sgd-0-0.0-ep0070.pt"),
        join(MODELPATH, f"pgd-128-{ds_name.replace('cwl2', 'pgd')}-70-4.0-0.01-trades6ce-vtor2-LargeMLPv4-0.9-2-sgd-0-0.0-ep0070.pt"),
        join(MODELPATH, f"pgd-128-{ds_name.replace('cwl2', 'pgd')}-70-8.0-0.01-trades6ce-vtor2-LargeMLPv4-0.9-2-sgd-0-0.0-ep0070.pt"),
        join(MODELPATH, f"pgd-128-{ds_name.replace('cwl2', 'pgd')}-70-2.0-0.01-advce-vtor2-LargeMLPv4-0.9-2-sgd-0-0.0-ep0070.pt"),
    ]

    trnX, trny, tstX, _, _ = auto_var.get_var_with_argument("dataset", ds_name)

    index = faiss.IndexFlatL2(int(np.prod(trnX.shape[1:])))
    index.add(trnX.reshape(len(trnX), -1).astype(np.float32))

    # scripts/calc_ood_reprs.py
    repr_path = "./notebooks/nb_results/reprs-no-wo-imgnet100.pkl"
    reprs = joblib.load(repr_path)

    for model_name, model_path in zip(model_names, model_paths):
        if not os.path.exists(model_path):
            print(f"`{model_path}` does not exist. skipping...")
            continue
        key = (ds_name, model_name)
        print(key)
        preds.setdefault(key, {})
        nnidxs.setdefault(key, {})
        dists.setdefault(key, {})

        arch_name = "LargeMLPv4"
        model = getattr(archs, arch_name)(n_classes=len(np.unique(trny)),
                n_channels=3, n_features=(trnX.shape[1], ))
        model.load_state_dict(torch.load(model_path)['model_state_dict'])

        if "tst" not in preds[key]:
            tst_preds = get_preds(model, tstX)
            preds[key]["tst"] = tst_preds

        for ood_name in tqdm(ood_names):
            if ood_name in preds[key]:
                continue
            tX = reprs[("aug10-imgnet100", "natural")][ood_name].astype(np.float32)
            other_ood_pred = get_preds(model, tX)
            preds[key][ood_name] = other_ood_pred
            if ((ds_name, "natural") in nnidxs) and (ood_name in nnidxs[(ds_name, "natural")]):
                nnidxs[key][ood_name] = nnidxs[(ds_name, "natural")][ood_name]
                dists[key][ood_name] = dists[(ds_name, "natural")][ood_name]
            else:
                D, I = index.search(tX.reshape(len(tX), -1), k=1)
                nnidxs[key][ood_name] = I[:, 0]
                dists[key][ood_name] = np.sqrt(D[:, 0])

        joblib.dump((preds, nnidxs, dists), "./notebooks/nb_results/feature-imgnet-c.pkl")

    joblib.dump((preds, nnidxs, dists), "./notebooks/nb_results/feature-imgnet-c.pkl")
    print("done.")


def calc_cifar(dset_name):
    _, ood_names = get_ood_data_paths(dset_name, "./data/cifar-ood/")

    model_names = [
        'natural',
        #'TRADES(.25)',
        #'TRADES(1)',
        'TRADES(2)',
        'TRADES(4)',
        'TRADES(8)',
        'AT(1)',
        #'ball',
    ]

    # preds, nnidxs, dists = joblib.load(f"./notebooks/nb_results/{dset_name}-c.pkl")
    preds, nnidxs, dists = {}, {}, {}

    if dset_name == "cifar10":
        ds_name = f'calcedrepr-cifar10-cwl2-64-cifar10-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0.pkl'
    elif dset_name == "cifar100":
        ds_name = f'calcedrepr-cifar100coarse-cwl2-64-cifar100coarse-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0.pkl'
    model_paths = [
        join(MODELPATH, f"pgd-128-{ds_name.replace('cwl2', 'pgd')}-70-1.0-0.01-ce-vtor2-LargeMLP-0.9-2-sgd-0-0.0-ep0070.pt"),
        join(MODELPATH, f"pgd-128-{ds_name.replace('cwl2', 'pgd')}-70-2.0-0.01-trades6ce-vtor2-LargeMLP-0.9-2-sgd-0-0.0-ep0070.pt"),
        join(MODELPATH, f"pgd-128-{ds_name.replace('cwl2', 'pgd')}-70-4.0-0.01-trades6ce-vtor2-LargeMLP-0.9-2-sgd-0-0.0-ep0070.pt"),
        join(MODELPATH, f"pgd-128-{ds_name.replace('cwl2', 'pgd')}-70-8.0-0.01-trades6ce-vtor2-LargeMLP-0.9-2-sgd-0-0.0-ep0070.pt"),
        join(MODELPATH, f"pgd-128-{ds_name.replace('cwl2', 'pgd')}-70-1.0-0.01-advce-vtor2-LargeMLP-0.9-2-sgd-0-0.0-ep0070.pt"),
    ]
    assert len(model_paths) == len(model_names)

    print(ds_name)

    trnX, trny, tstX, _, _ = auto_var.get_var_with_argument("dataset", ds_name)

    index = faiss.IndexFlatL2(int(np.prod(trnX.shape[1:])))
    index.add(trnX.reshape(len(trnX), -1).astype(np.float32))

    # scripts/calc_ood_reprs.py
    repr_path = f"./notebooks/nb_results/reprs-no-wo-{dset_name}.pkl"
    reprs = joblib.load(repr_path)

    for model_name, model_path in zip(model_names, model_paths):
        if not os.path.exists(model_path):
            print(f"`{model_path}` does not exist. skipping...")
            continue
        if dset_name == "cifar10":
            key = (dset_name, model_name)
        elif dset_name == "cifar100":
            key = ("cifar100coarse", model_name)
        print(key)
        preds.setdefault(key, {})
        nnidxs.setdefault(key, {})
        dists.setdefault(key, {})

        arch_name = "LargeMLP"
        model = getattr(archs, arch_name)(n_classes=len(np.unique(trny)), n_features=(trnX.shape[1], ))
        model.load_state_dict(torch.load(model_path)['model_state_dict'])

        if "tst" not in preds[key]:
            tst_preds = get_preds(model, tstX)
            preds[key]["tst"] = tst_preds

        for ood_name in tqdm(ood_names, total=len(ood_names)):
            if ood_name in preds[key]:
                continue

            if dset_name == "cifar10":
                tX = reprs[(dset_name, "natural")][ood_name].astype(np.float32)
            elif dset_name == "cifar100":
                tX = reprs[("cifar100coarse", "natural")][ood_name].astype(np.float32)
            D, I = index.search(tX.reshape(len(tX), -1), k=1)
            other_ood_pred = get_preds(model, tX)
            preds[key][ood_name] = other_ood_pred
            nnidxs[key][ood_name] = I[:, 0]
            dists[key][ood_name] = np.sqrt(D[:, 0])

    joblib.dump((preds, nnidxs, dists), f"./notebooks/nb_results/feature-{dset_name}-c.pkl")
    print("done.")


def main():
    #calc_madry_cifar10()
    #calc_cifar("cifar10")
    #calc_cifar("cifar100")
    calc_imgnet100_wo()
    #calc_imgnet100()


if __name__ == "__main__":
    main()
