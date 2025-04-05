import sys
sys.path.append("../")
from os.path import join
import os

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


def get_reprs(model, dset, batch_size=64, device="cuda"):
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
            output = model.get_repr(x.to(device))
        ret.append(output.cpu().numpy())
    del loader
    return np.concatenate(ret, axis=0)


def calc_imgnet100():
    image_folders = []
    ood_names = []
    for i in range(1, 6):
        image_folders += [
            f'/tmp2/ImageNet-c/noise/gaussian_noise/{i}/',
            f'/tmp2/ImageNet-c/noise/impulse_noise/{i}/',
            f'/tmp2/ImageNet-c/noise/shot_noise/{i}/',
            f'/tmp2/ImageNet-c/blur/defocus_blur/{i}/',
            f'/tmp2/ImageNet-c/blur/motion_blur/{i}/',
            f'/tmp2/ImageNet-c/blur/zoom_blur/{i}/',
            f'/tmp2/ImageNet-c/blur/glass_blur/{i}/',

            f'/tmp2/ImageNet-c/weather/snow/{i}/',
            f'/tmp2/ImageNet-c/weather/fog/{i}/',
            f'/tmp2/ImageNet-c/weather/frost/{i}/',
            f'/tmp2/ImageNet-c/weather/brightness/{i}/',
            f'/tmp2/ImageNet-c/digital/contrast/{i}/',
            f'/tmp2/ImageNet-c/digital/pixelate/{i}/',
            f'/tmp2/ImageNet-c/digital/jpeg_compression/{i}/',
            f'/tmp2/ImageNet-c/digital/elastic_transform/{i}/',
        ]
        ood_names += [f'gaussian_{i}', f'impulse_{i}', f'shot_{i}', f'defocus_{i}', f'motion_{i}', f'zoom_{i}', f'glass_{i}',
                      f'snow_{i}', f'fog_{i}', f'frost_{i}', f'brightness_{i}', f'contrast_{i}', f'pixelate_{i}', f'jpeg_{i}', f'elastic_{i}']

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

    #for ood_class in [0, 1, 2]:
    for ood_class in [1, 2]:
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


def calc_cifar(dset_name):
    image_files, ood_names = get_ood_data_paths(dset_name, "./data/cifar-ood/")

    model_names = [
        'natural',
        'TRADES(.25)',
        'TRADES(1)',
        'TRADES(2)',
        'TRADES(4)',
        'TRADES(8)',
        'AT(2)',
        'ball',
    ]

    # preds, nnidxs, dists = joblib.load(f"./notebooks/nb_results/{dset_name}-c.pkl")
    preds, nnidxs, dists = {}, {}, {}


    for ood_class in [9]:
        if dset_name == "cifar10":
            ds_name = f'cifar10wo{ood_class}'
        elif dset_name == "cifar100":
            ds_name = f'cifar100coarsewo{ood_class}'
        model_paths = [
            join(MODELPATH, f"pgd-64-{ds_name}-70-1.0-0.01-ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0-ep0070.pt"),
            join(MODELPATH, f"pgd-64-{ds_name}-70-0.25-0.01-trades6ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0-ep0070.pt"),
            join(MODELPATH, f"pgd-64-{ds_name}-70-1.0-0.01-trades6ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0-ep0070.pt"),
            join(MODELPATH, f"pgd-64-{ds_name}-70-2.0-0.01-trades6ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0-ep0070.pt"),
            join(MODELPATH, f"pgd-64-{ds_name}-70-4.0-0.01-trades6ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0-ep0070.pt"),
            join(MODELPATH, f"pgd-64-{ds_name}-70-8.0-0.01-trades6ce-vtor2-WRN_40_10-0.0-2-adam-0-0.0-ep0070.pt"),
            join(MODELPATH, f"pgd-64-{ds_name}-70-2.0-0.01-advce-vtor2-WRN_40_10-0.0-2-adam-0-0.0-ep0070.pt"),
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
            _, _, _, ori_tsty = auto_var.get_var_with_argument("dataset", "cifar10")
        elif dset_name == "cifar100":
            _, _, _, ori_tsty = auto_var.get_var_with_argument("dataset", "cifar100coarse")
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
    #calc_cifar("cifar10")
    calc_cifar("cifar100")
    #calc_imgnet100()


if __name__ == "__main__":
    main()
