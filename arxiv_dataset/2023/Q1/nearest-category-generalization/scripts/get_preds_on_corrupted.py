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
            torch.from_numpy(dset.transpose(0, 3, 1, 2)).float(),
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
    import dill
    image_files, ood_names = get_ood_data_paths("cifar10", "./data/cifar-ood/")

    if os.path.exists("./notebooks/nb_results/madry-cifar10.pkl"):
        preds, nnidxs, dists = joblib.load("nb_results/madry-cifar10-c.pkl")
    else:
        preds, nnidxs, dists = {}, {}, {}

    ds_name = "CIFAR10"
    model_paths = [
        f"./notebooks/pretrained/madry_robustness/cifar_nat.pt",
        f"./notebooks/pretrained/madry_robustness/cifar_l2_0_25.pt",
        f"./notebooks/pretrained/madry_robustness/cifar_l2_0_5.pt",
        f"./notebooks/pretrained/madry_robustness/cifar_l2_1_0.pt",
    ]
    model_names = [
        'natural',
        'AT(0.25)',
        'AT(0.5)',
        'AT(1.0)',
    ]

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                    std=[0.2023, 0.1994, 0.2010])
    def np_normalize(X):
        X = X.transpose(0, 3, 1, 2)
        return normalize(torch.from_numpy(X)).numpy().transpose(0, 2, 3, 1)

    trnX, trny, tstX, tsty = auto_var.get_var_with_argument("dataset", "cifar10")
    trnX = np_normalize(trnX)
    tstX = np_normalize(tstX)

    index = faiss.IndexFlatL2(int(np.prod(trnX.shape[1:])))
    index.add(trnX.reshape(len(trnX), -1).astype(np.float32))

    oodXs = []
    for image_file in tqdm(image_files, desc="[get oodXs]"):
        tX = np.load(image_file)
        oodXs.append((tX / 255).astype(np.float32))

    for model_name, model_path in zip(model_names, model_paths):

        if not os.path.exists(model_path):
            print(f"`{model_path}` does not exist. skipping...")
            continue
        key = (ds_name, model_name)
        print(key)
        preds.setdefault(key, {})
        nnidxs.setdefault(key, {})
        dists.setdefault(key, {})

        model = torchvision.models.resnet50()
        res = torch.load(model_path, pickle_module=dill)
        state_dict = {}
        sd = res['model'] if 'model' in res else res['state_dict']
        for k, v in sd.items():
            if "module.model." in k:
                state_dict[k.replace("module.model.", "")] = v
        model.load_state_dict(state_dict)

        tst_preds = get_preds(model, tstX)
        preds[key]['tst'] = tst_preds
        print(f"test acc: {(tst_preds.argmax(1) == tsty).mean()}")

        for i, tX in tqdm(enumerate(oodXs), total=len(oodXs)):
            if ood_names[i] in preds[key]:
                continue
            D, I = index.search(tX.reshape(len(tX), -1), k=1)
            other_ood_pred = get_preds(model, tX)
            preds[key][ood_names[i]] = other_ood_pred
            nnidxs[key][ood_names[i]] = I[:, 0]
            dists[key][ood_names[i]] = np.sqrt(D[:, 0])
    print("done.")

def calc_madry_imgnet100():
    image_folders, ood_names = get_ood_data_paths("imgnet", "/tmp2")

    model_names = [
        'natural',
        'AT_l2(3)',
        'AT_linf(4)',
        'AT_linf(8)',
    ]

    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)
    def np_normalize(X):
        return (X - mean) / std

    if os.path.exists("./notebooks/nb_results/madry-imgnet.pkl"):
        preds, nnidxs, dists = joblib.load("./notebooks/nb_results/madry-imgnet.pkl")
    else:
        preds, nnidxs, dists = {}, {}, {}

    ds_name = f'aug12-imgnet100'
    model_path = "./notebooks/pretrained/"
    model_paths = [
        join(model_path, "madry_robustness", "imagenet_resnet50-0676ba61.pth"),
        join(model_path, "madry_robustness", "imagenet_l2_3_0.pt"),
        join(model_path, "madry_robustness", "imagenet_linf_4_0.pt"),
        join(model_path, "madry_robustness", "imagenet_linf_8_0.pt"),
    ]

    trnX, _, tstX, _, _ = auto_var.get_var_with_argument("dataset", ds_name)

    index = faiss.IndexFlatL2(int(np.prod(trnX.shape[1:])))
    index.add(np_normalize(trnX).reshape(len(trnX), -1).astype(np.float32))
    del trnX
    gc.collect()

    for model_name, model_path in zip(model_names, model_paths):
        if not os.path.exists(model_path):
            print(f"`{model_path}` does not exist. skipping...")
            continue
        key = (ds_name, model_name)
        print(key)
        preds.setdefault(key, {})
        nnidxs.setdefault(key, {})
        dists.setdefault(key, {})

        model = torchvision.models.resnet50(pretrained=False)
        model.load_state_dict(torch.load(model_path))

        if "tst" not in preds[key]:
            tst_preds = get_preds(model, tstX)
            preds[key]["tst"] = tst_preds

        for i, image_folder in tqdm(enumerate(image_folders), total=len(image_folders), desc="[get oodXs]"):
            if ood_names[i] in preds[key]:
                continue

            dset = torchvision.datasets.ImageFolder(
                image_folder,
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                ])
            )
            _loader = torch.utils.data.DataLoader(dset, batch_size=64, shuffle=False, num_workers=24)

            tX = np.concatenate([x.numpy() for (x, _) in _loader], axis=0).transpose(0, 2, 3, 1)
            D, I = index.search(np_normalize(tX).reshape(len(tX), -1), k=1)
            del tX
            gc.collect()

            other_ood_pred = get_preds(model, _loader)
            preds[key][ood_names[i]] = other_ood_pred
            nnidxs[key][ood_names[i]] = I[:, 0]
            dists[key][ood_names[i]] = np.sqrt(D[:, 0])

    print("done.")
    joblib.dump((preds, nnidxs, dists), "./notebooks/nb_results/mdary-imgnet.pkl")


def calc_imgnet100():
    image_folders, ood_names = get_ood_data_paths("imgnet", "/tmp2")

    model_names = [
        'natural',
        'TRADES(2)',
        'TRADES(4)',
        'TRADES(8)',
        'AT(2)',
    ]

    ori_dset = torchvision.datasets.ImageFolder("/tmp2/ImageNet100/ILSVRC2012_img_train/")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    def np_normalize(X):
        X = X.transpose(0, 3, 1, 2)
        return normalize(torch.from_numpy(X)).numpy().transpose(0, 2, 3, 1)


    if os.path.exists("./notebooks/nb_results/nowo-imgnet100-c.pkl"):
        preds, nnidxs, dists = joblib.load("./notebooks/nb_results/nowo-imgnet100-c.pkl")
    else:
        preds, nnidxs, dists = {}, {}, {}

    ds_name = f'aug10-imgnet100'
    model_paths = [
        join(MODELPATH, f"pgd-128-{ds_name}-70-1.0-0.01-ce-vtor2-ResNet50Norm01-0.0-2-adam-0-0.0-ep0070.pt"),
        join(MODELPATH, f"pgd-128-{ds_name}-70-2.0-0.01-trades6ce-vtor2-ResNet50Norm01-0.0-2-adam-0-0.0-ep0070.pt"),
        join(MODELPATH, f"pgd-128-{ds_name}-70-4.0-0.01-trades6ce-vtor2-ResNet50Norm01-0.0-2-adam-0-0.0-ep0070.pt"),
        join(MODELPATH, f"pgd-128-{ds_name}-70-8.0-0.01-trades6ce-vtor2-ResNet50Norm01-0.0-2-adam-0-0.0-ep0070.pt"),
        join(MODELPATH, f"pgd-128-{ds_name}-70-2.0-0.01-advce-vtor2-ResNet50Norm01-0.0-2-adam-0-0.0-ep0070.pt"),
    ]

    trnX, trny, tstX, _, _ = auto_var.get_var_with_argument("dataset", ds_name)

    index = faiss.IndexFlatL2(int(np.prod(trnX.shape[1:])))
    index.add(np_normalize(trnX).reshape(len(trnX), -1).astype(np.float32))

    oodXs = []
    for image_folder in tqdm(image_folders, desc="[get oodXs]"):
        dset = torchvision.datasets.ImageFolder(
            image_folder,
            transforms.Compose([transforms.Resize(128), transforms.CenterCrop(112), transforms.ToTensor(), ])
        )
        _loader = torch.utils.data.DataLoader(dset, batch_size=64, shuffle=False, num_workers=24)
        tX = np.concatenate([x.numpy() for (x, _) in _loader], axis=0).transpose(0, 2, 3, 1)
        oodXs.append(tX)

    for model_name, model_path in zip(model_names, model_paths):
        if not os.path.exists(model_path):
            print(f"`{model_path}` does not exist. skipping...")
            continue
        key = (ds_name, model_name)
        print(key)
        preds.setdefault(key, {})
        nnidxs.setdefault(key, {})
        dists.setdefault(key, {})

        arch_name = model_path.split("-")[model_path.split("-").index("vtor2") + 1]
        model = getattr(archs, arch_name)(n_classes=len(np.unique(trny)), n_channels=3)
        model.load_state_dict(torch.load(model_path)['model_state_dict'])

        if "tst" not in preds[key]:
            tst_preds = get_preds(model, tstX)
            preds[key]["tst"] = tst_preds

        for i, tX in enumerate(oodXs):
            if ood_names[i] in preds[key]:
                continue
            D, I = index.search(np_normalize(tX).reshape(len(tX), -1), k=1)
            other_ood_pred = get_preds(model, tX)
            preds[key][ood_names[i]] = other_ood_pred
            nnidxs[key][ood_names[i]] = I[:, 0]
            dists[key][ood_names[i]] = np.sqrt(D[:, 0])

    joblib.dump((preds, nnidxs, dists), "./notebooks/nb_results/nowo-imgnet100-c.pkl")
    print("done.")



def calc_imgnet100():
    image_folders, ood_names = get_ood_data_paths("imgnet", "/tmp2")

    model_names = [
        'natural',
        'TRADES(1)',
        'TRADES(2)',
        'TRADES(4)',
        'TRADES(8)',
        'AT(2)',
        'ball',
    ]

    ori_dset = torchvision.datasets.ImageFolder("/tmp2/ImageNet100/ILSVRC2012_img_train/")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    def np_normalize(X):
        X = X.transpose(0, 3, 1, 2)
        return normalize(torch.from_numpy(X)).numpy().transpose(0, 2, 3, 1)


    preds, nnidxs, dists = joblib.load("./notebooks/nb_results/imgnet.pkl")

    for ood_class in [0, 1, 2]:
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

        trnX, trny, tstX, tsty, (ood1X, ood2X) = auto_var.get_var_with_argument("dataset", ds_name)
        oodX = np.concatenate((ood1X, ood2X), axis=0)

        ood_classes = [ood_class]

        index = faiss.IndexFlatL2(int(np.prod(trnX.shape[1:])))
        index.add(np_normalize(trnX).reshape(len(trnX), -1).astype(np.float32))
        oodD, oodI = index.search(np_normalize(oodX).reshape(len(oodX), -1), k=1)

        oodXs = []
        for image_folder in tqdm(image_folders, desc="[get oodXs]"):
            dset = torchvision.datasets.ImageFolder(
                image_folder,
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
            print(key)
            preds.setdefault(key, {})
            nnidxs.setdefault(key, {})
            dists.setdefault(key, {})

            arch_name = model_path.split("-")[model_path.split("-").index("vtor2") + 1]
            model = getattr(archs, arch_name)(n_classes=len(np.unique(trny)), n_channels=3)
            model.load_state_dict(torch.load(model_path)['model_state_dict'])

            if "tst" not in preds[key]:
                tst_preds = get_preds(model, tstX)
                preds[key]["tst"] = tst_preds

            if "ncg" not in preds[key]:
                ood_preds = get_preds(model, oodX)
                preds[key]["ncg"] = ood_preds
                nnidxs[key]["ncg"] = oodI[:, 0]
                dists[key]["ncg"] = np.sqrt(oodD[:, 0])

            for i, tX in enumerate(oodXs):
                if ood_names[i] in preds[key]:
                    continue
                D, I = index.search(np_normalize(tX).reshape(len(tX), -1), k=1)
                other_ood_pred = get_preds(model, tX)
                preds[key][ood_names[i]] = other_ood_pred
                nnidxs[key][ood_names[i]] = I[:, 0]
                dists[key][ood_names[i]] = np.sqrt(D[:, 0])

        joblib.dump((preds, nnidxs, dists), "./notebooks/nb_results/imgnet.pkl")

    print("done.")
    joblib.dump((preds, nnidxs, dists), "./notebooks/nb_results/imgnet.pkl")


def calc_nowo_imgnet100():
    image_folders, ood_names = get_ood_data_paths('imgnet', "/tmp2/")

    model_names = [
        'natural',
        'TRADES(2)',
        'TRADES(4)',
        'TRADES(8)',
        'AT(2)',
    ]

    ori_dset = torchvision.datasets.ImageFolder("/tmp2/ImageNet100/ILSVRC2012_img_train/")
    dset = torchvision.datasets.ImageFolder(
        image_folders[0],
        transforms.Compose([transforms.Resize(128), transforms.CenterCrop(112), transforms.ToTensor(), ])
    )
    _loader = torch.utils.data.DataLoader(dset, batch_size=64, shuffle=False, num_workers=24)
    ty = np.concatenate([y.numpy() for (_, y) in _loader])

    valid_classes = []
    for c in ori_dset.classes:
        valid_classes.append(dset.class_to_idx[c])
    assert len(valid_classes) == 100
    valid_idx = np.array([i for i, c in enumerate(ty) if c in valid_classes])
    #valid_idx = np.arange(50000)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    def np_normalize(X):
        X = X.transpose(0, 3, 1, 2)
        return normalize(torch.from_numpy(X)).numpy().transpose(0, 2, 3, 1)

    if os.path.exists("./notebooks/nb_results/nowo-imgnet100-c.pkl"):
        preds, nnidxs, dists = joblib.load("./notebooks/nb_results/nowo-imgnet100-c.pkl")
    #if os.path.exists("./notebooks/nb_results/nowo-imgnet-c.pkl"):
    #    preds, nnidxs, dists = joblib.load("./notebooks/nb_results/nowo-imgnet-c.pkl")
    else:
        preds, nnidxs, dists = {}, {}, {}

    ds_name = f'aug10-imgnet100'
    model_paths = [
        join(MODELPATH, f"pgd-128-{ds_name}-70-1.0-0.01-ce-vtor2-ResNet50Norm01-0.0-2-adam-0-0.0-ep0070.pt"),
        join(MODELPATH, f"pgd-128-{ds_name}-70-2.0-0.01-trades6ce-vtor2-ResNet50Norm01-0.0-2-adam-0-0.0-ep0070.pt"),
        join(MODELPATH, f"pgd-128-{ds_name}-70-4.0-0.01-trades6ce-vtor2-ResNet50Norm01-0.0-2-adam-0-0.0-ep0070.pt"),
        join(MODELPATH, f"pgd-128-{ds_name}-70-8.0-0.01-trades6ce-vtor2-ResNet50Norm01-0.0-2-adam-0-0.0-ep0070.pt"),
        join(MODELPATH, f"pgd-128-{ds_name}-70-2.0-0.01-advce-vtor2-ResNet50Norm01-0.0-2-adam-0-0.0-ep0070.pt"),
    ]

    trnX, trny, tstX, _, _ = auto_var.get_var_with_argument("dataset", ds_name)

    index = faiss.IndexFlatL2(int(np.prod(trnX.shape[1:])))
    index.add(np_normalize(trnX).reshape(len(trnX), -1).astype(np.float32))

    for model_name, model_path in zip(model_names, model_paths):
        if not os.path.exists(model_path):
            print(f"`{model_path}` does not exist. skipping...")
            continue
        key = (ds_name, model_name)
        #if key in preds:
        #    continue
        print(key)
        preds.setdefault(key, {})
        nnidxs.setdefault(key, {})
        dists.setdefault(key, {})

        #arch_name = model_path.split("-")[model_path.split("-").index("vtor2") + 1]
        #model = getattr(archs, arch_name)(n_classes=len(np.unique(trny)), n_channels=3)
        #model.load_state_dict(torch.load(model_path)['model_state_dict'])

        #if "tst" not in preds[key]:
        #    tst_preds = get_preds(model, tstX)
        #    preds[key]["tst"] = tst_preds

        if "tst" not in dists[key]:
            D, I = index.search(tstX.reshape(len(tstX), -1), k=1)
            nnidxs[key]['tst'] = I[:, 0]
            dists[key]['tst'] = np.sqrt(D[:, 0])

        #for ood_name, image_folder in tqdm(zip(ood_names, image_folders), total=len(ood_names)):
        #    if ood_name in preds[key]:
        #        continue
        #    dset = torchvision.datasets.ImageFolder(
        #        image_folder,
        #        transforms.Compose([transforms.Resize(128), transforms.CenterCrop(112), transforms.ToTensor(), ])
        #    )
        #    _loader = torch.utils.data.DataLoader(dset, batch_size=64, shuffle=False, num_workers=24)
        #    tX = np.concatenate([x.numpy() for (x, _) in _loader], axis=0).transpose(0, 2, 3, 1)
        #    tX = tX[valid_idx]
        #    other_ood_pred = get_preds(model, tX)
        #    preds[key][ood_name] = other_ood_pred
        #    if ((ds_name, "natural") in nnidxs) and (ood_name in nnidxs[(ds_name, "natural")]):
        #        nnidxs[key][ood_name] = nnidxs[(ds_name, "natural")][ood_name]
        #        dists[key][ood_name] = dists[(ds_name, "natural")][ood_name]
        #    else:
        #        D, I = index.search(tX.reshape(len(tX), -1), k=1)
        #        nnidxs[key][ood_name] = I[:, 0]
        #        dists[key][ood_name] = np.sqrt(D[:, 0])

        joblib.dump((preds, nnidxs, dists), "./notebooks/nb_results/nowo-imgnet100-c.pkl")
        #joblib.dump((preds, nnidxs, dists), "./notebooks/nb_results/nowo-imgnet-c.pkl")

    joblib.dump((preds, nnidxs, dists), "./notebooks/nb_results/nowo-imgnet100-c.pkl")
    #joblib.dump((preds, nnidxs, dists), "./notebooks/nb_results/nowo-imgnet-c.pkl")
    print("done.")


def calc_nowo_cifar(dset_name):
    image_files, ood_names = get_ood_data_paths(dset_name, "./data/cifar-ood/")

    model_names = [
        'natural',
        'TRADES(2)',
        #'TRADES(4)',
        #'TRADES(8)',
        'AT(2)',
    ]

    # preds, nnidxs, dists = joblib.load(f"./notebooks/nb_results/{dset_name}-c.pkl")
    preds, nnidxs, dists = {}, {}, {}


    if dset_name == "cifar10":
        ds_name = f'cifar10'
    elif dset_name == "cifar100":
        ds_name = f'cifar100coarse'
    model_paths = [
        join(MODELPATH, f"pgd-64-{ds_name}-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0-ep0070.pt"),
        join(MODELPATH, f"pgd-64-{ds_name}-70-2.0-0.01-trades6ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0-ep0070.pt"),
        #join(MODELPATH, f"pgd-64-{ds_name}-70-4.0-0.01-trades6ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0-ep0070.pt"),
        #join(MODELPATH, f"pgd-64-{ds_name}-70-8.0-0.01-trades6ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0-ep0070.pt"),
        join(MODELPATH, f"pgd-64-{ds_name}-70-2.0-0.01-advce-vtor2-WRN_40_10-0.0-2-adam-0-0.0-ep0070.pt"),
    ]
    assert len(model_paths) == len(model_names)

    print(ds_name)

    trnX, trny, tstX, _, _ = auto_var.get_var_with_argument("dataset", ds_name)

    index = faiss.IndexFlatL2(int(np.prod(trnX.shape[1:])))
    index.add(trnX.reshape(len(trnX), -1).astype(np.float32))

    oodXs = []
    for image_file in tqdm(image_files, desc="[get oodXs]"):
        tX = np.load(image_file)
        oodXs.append((tX / 255).astype(np.float32))

    for model_name, model_path in zip(model_names, model_paths):
        if not os.path.exists(model_path):
            print(f"`{model_path}` does not exist. skipping...")
            continue
        key = (ds_name, model_name)
        print(key)
        preds.setdefault(key, {})
        nnidxs.setdefault(key, {})
        dists.setdefault(key, {})

        arch_name = model_path.split("-")[model_path.split("-").index("vtor2") + 1]
        model = getattr(archs, arch_name)(n_classes=len(np.unique(trny)), n_channels=3)
        model.load_state_dict(torch.load(model_path)['model_state_dict'])

        if "tst" not in preds[key]:
            tst_preds = get_preds(model, tstX)
            preds[key]["tst"] = tst_preds
            D, I = index.search(tstX.reshape(len(tstX), -1), k=1)
            nnidxs[key]["tst"] = I[:, 0]
            dists[key]["tst"] = np.sqrt(D[:, 0])

        for i, tX in enumerate(oodXs):
            if ood_names[i] in preds[key]:
                continue
            D, I = index.search(tX.reshape(len(tX), -1), k=1)
            other_ood_pred = get_preds(model, tX)
            preds[key][ood_names[i]] = other_ood_pred
            nnidxs[key][ood_names[i]] = I[:, 0]
            dists[key][ood_names[i]] = np.sqrt(D[:, 0])

    joblib.dump((preds, nnidxs, dists), f"./notebooks/nb_results/nowo-{dset_name}-c.pkl")
    print("done.")

def calc_cifar(dset_name):
    image_files, ood_names = get_ood_data_paths(dset_name, "./data/cifar-ood/")

    model_names = [
        #'natural',
        #'TRADES(.25)',
        #'TRADES(1)',
        #'TRADES(2)',
        #'TRADES(4)',
        #'TRADES(8)',
        #'AT(2)',
        #'ball',
        'ellipsoid',
        'sub-voronoi',
    ]

    preds, nnidxs, dists = joblib.load(f"./notebooks/nb_results/{dset_name}-c.pkl")
    #preds, nnidxs, dists = {}, {}, {}


    for ood_class in [9]:
        if dset_name == "cifar10":
            ds_name = f'cifar10wo{ood_class}'
        elif dset_name == "cifar100":
            ds_name = f'cifar100coarsewo{ood_class}'
        model_paths = [
            #join(MODELPATH, f"pgd-64-{ds_name}-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0-ep0070.pt"),
            #join(MODELPATH, f"pgd-64-{ds_name}-70-0.25-0.01-trades6ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0-ep0070.pt"),
            #join(MODELPATH, f"pgd-64-{ds_name}-70-1.0-0.01-trades6ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0-ep0070.pt"),
            #join(MODELPATH, f"pgd-64-{ds_name}-70-2.0-0.01-trades6ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0-ep0070.pt"),
            #join(MODELPATH, f"pgd-64-{ds_name}-70-4.0-0.01-trades6ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0-ep0070.pt"),
            #join(MODELPATH, f"pgd-64-{ds_name}-70-8.0-0.01-trades6ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0-ep0070.pt"),
            #join(MODELPATH, f"pgd-64-{ds_name}-70-2.0-0.01-advce-vtor2-WRN_40_10-0.0-2-adam-0-0.0-ep0070.pt"),
            #join(MODELPATH, f"pgd-64-{ds_name}-70-1.0-0.01-cusgradtrades6v2autwotimesce-vtor2-WRN_40_10-halfclose-0.0-2-adam-0-0.0-ep0070.pt"),
            join(MODELPATH, f"pgd-64-{ds_name}-70-1.0-0.01-trades6ce-vtor2-WRN_40_10-pcaellipbatchada-0.0-2-adam-0-0.0.pkl"),
            join(MODELPATH, f"pgd-64-{ds_name}-70-1.0-0.01-cusgradtrades6v2autwotimesce-vtor2-WRN_40_10-halfclose-0.0-2-adam-0-0.0-ep0070.pt"),
        ]
        assert len(model_paths) == len(model_names)

        print(ds_name)

        trnX, trny, tstX, _, (ood1X, ood2X, _, _) = auto_var.get_var_with_argument("dataset", ds_name)
        oodX = np.concatenate((ood1X, ood2X), axis=0)

        ood_classes = [ood_class]

        index = faiss.IndexFlatL2(int(np.prod(trnX.shape[1:])))
        index.add(trnX.reshape(len(trnX), -1).astype(np.float32))
        oodD, oodI = index.search(oodX.reshape(len(oodX), -1), k=1)

        if dset_name == "cifar10":
            _, _, _, ori_tsty, _ = auto_var.get_var_with_argument("dataset", "cifar10")
        elif dset_name == "cifar100":
            _, _, _, ori_tsty, _ = auto_var.get_var_with_argument("dataset", "cifar100coarse")
        ori_tsty = np.tile(ori_tsty, 5)
        oodXs = []
        for image_file in tqdm(image_files, desc="[get oodXs]"):
            tX = np.load(image_file)

            if dset_name == "cifar10":
                valid_classes = np.delete(np.arange(10), ood_classes)
            elif dset_name == "cifar100":
                valid_classes = np.delete(np.arange(20), ood_classes)

            valid_idx = np.array([i for i, c in enumerate(ori_tsty) if c in valid_classes])

            if dset_name == "cifar10":
                assert len(valid_idx) == 45000
            elif dset_name == "cifar100":
                assert len(valid_idx) == 47500

            oodXs.append((tX[valid_idx] / 255).astype(np.float32))

        for model_name, model_path in zip(model_names, model_paths):
            if not os.path.exists(model_path):
                print(f"`{model_path}` does not exist. skipping...")
                continue
            key = (ds_name, model_name)
            print(key)
            preds.setdefault(key, {})
            nnidxs.setdefault(key, {})
            dists.setdefault(key, {})

            arch_name = model_path.split("-")[model_path.split("-").index("vtor2") + 1]
            model = getattr(archs, arch_name)(n_classes=len(np.unique(trny)), n_channels=3)
            model.load_state_dict(torch.load(model_path)['model_state_dict'])

            if "tst" not in preds[key]:
                tst_preds = get_preds(model, tstX)
                preds[key]["tst"] = tst_preds

            if "ncg" not in preds[key]:
                ood_preds = get_preds(model, oodX)
                preds[key]["ncg"] = ood_preds
                nnidxs[key]["ncg"] = oodI[:, 0]
                dists[key]["ncg"] = np.sqrt(oodD[:, 0])

            for i, tX in enumerate(oodXs):
                if ood_names[i] in preds[key]:
                    continue
                D, I = index.search(tX.reshape(len(tX), -1), k=1)
                other_ood_pred = get_preds(model, tX)
                preds[key][ood_names[i]] = other_ood_pred
                nnidxs[key][ood_names[i]] = I[:, 0]
                dists[key][ood_names[i]] = np.sqrt(D[:, 0])

        joblib.dump((preds, nnidxs, dists), f"./notebooks/nb_results/{dset_name}-c.pkl")

    joblib.dump((preds, nnidxs, dists), f"./notebooks/nb_results/{dset_name}-c.pkl")
    print("done.")


def main():
    #calc_madry_imgnet100()
    #calc_nowo_imgnet100()
    #calc_nowo_cifar("cifar10")
    #calc_nowo_cifar("cifar100")
    calc_cifar("cifar10")
    #calc_cifar("cifar100")
    #calc_imgnet100()


if __name__ == "__main__":
    main()
