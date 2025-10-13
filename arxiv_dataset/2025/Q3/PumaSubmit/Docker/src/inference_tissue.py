import os
import copy
import sys

import toml
import requests
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from typing import List, Union, Tuple
import torch
import numpy as np
import zarr
import zipfile
from numcodecs import Blosc
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from scipy.special import softmax
from src.multi_head_unet import get_model, load_checkpoint
from src.data_utils import NpyDataset, ImageDataset
from src.augmentations import color_augmentations
from src.spatial_augmenter import SpatialAugmenter
from src.constants import TTA_AUG_PARAMS, VALID_WEIGHTS
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inference_main(
    params: dict,
    models,
    augmenter,
    color_aug_fn,
):
    """
    Inference function for a single input file.

    Parameters
    ----------
    params: dict
        Parameter store, defined in initial main
    models: List[torch.nn.Module]
        list of models to run inference with, e.g. multiple folds or a single model in a list
    augmenter: SpatialAugmenter
        Augmentation module for geometric transformations
    color_aug_fn: torch.nn.Sequential
        Color Augmentation module

    Returns
    ----------
    params: dict
        Parameter store, defined in initial main and modified by this function
    z: Union(Tuple[zarr.ZipStore, zarr.ZipStore], None)
        instance and class segmentation results as zarr stores, kept open for further processing. None if inference was skipped.
    """
    fn = params["p"].split(os.sep)[-1].split(params["ext"])[0]
    output_path = params["root"] + params["output"]
    params["output_dir"] = os.path.join(output_path, fn)
    if not os.path.isdir(params["output_dir"]):
        os.makedirs(params["output_dir"])
    params["model_out_p"] = os.path.join(
        params["output_dir"], fn + "_raw_" + str(params["tile_size"])
    )
    prog_path = os.path.join(params["output_dir"], "progress.txt")

    if not torch.cuda.is_available():
        print("trying to run inference on CPU, aborting...")
        print("if this is intended, remove this check")
        raise Exception("No GPU available")

    # create datasets from specified input
    # pth1 = params["p1"]
    # with open(params["p1"], "r") as f:
    #     params["p1"] = f.read()
    # os.remove(pth1)
    # print('file nams is = ', params['p1'])
    params["p1"] = os.path.join(params['p1'],params['file_name'])
    dataset = ImageDataset(
        params["p"],
        params["tile_size"],
        padding_factor=params["overlap"],
        ratio_object_thresh=0.3,
        min_tiss=0.1,
        tissue=True,
        label_path =params["p1"]

    )



    # setup output files to write to, also create dummy file to resume inference if interruped

    z_inst = zarr.open(
        params["model_out_p"] + "_inst.zip",
        mode="w",
        shape=(len(dataset), 3, params["tile_size"], params["tile_size"]),
        chunks=(params["batch_size"], 3, params["tile_size"], params["tile_size"]),
        dtype="f4",
        compressor=Blosc(cname="lz4", clevel=3, shuffle=Blosc.SHUFFLE),
    )
    z_cls = zarr.open(
        params["model_out_p"] + "_cls.zip",
        mode="w",
        shape=(
            len(dataset),
            params["out_channels_cls"],
            params["tile_size"],
            params["tile_size"],
        ),
        chunks=(
            params["batch_size"],
            params["out_channels_cls"],
            params["tile_size"],
            params["tile_size"],
        ),
        dtype="u1",
        compressor=Blosc(cname="lz4", clevel=3, shuffle=Blosc.BITSHUFFLE),
    )

    # creating progress file to restart inference if it was interrupted
    with open(prog_path, "w") as f:
        f.write("0")
    inf_start = 0

    dataloader = DataLoader(
        dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=params["inf_workers"],
        pin_memory=True,
    )


    # IO thread to write output in parallel to inference
    def dump_results(res, z_cls, z_inst, prog_path):
        cls_, inst_, zc_ = res
        if cls_ is None:
            return
        cls_ = (softmax(cls_.astype(np.float32), axis=1) * 255).astype(np.uint8)
        z_cls[zc_ : zc_ + cls_.shape[0]] = cls_
        z_inst[zc_ : zc_ + inst_.shape[0]] = inst_.astype(np.float32)
        with open(prog_path, "w") as f:
            f.write(str(zc_))
        return

    # Separate thread for IO
    with ThreadPoolExecutor(max_workers=params["inf_writers"]) as executor:
        futures = []
        # run inference
        zc = inf_start
        for raw, _,raw1 in tqdm(dataloader):
            raw = raw.to(device, non_blocking=True).float()
            raw = raw.permute(0, 3, 1, 2)  # BHWC -> BCHW
            raw1 = raw1.to(torch.uint8).to(device, non_blocking=True)
            raw1 = raw1.permute(0, 3, 1, 2)  # BHWC -> BCHW

            # pth = params['tissue_type']
            # with open(params['tissue_type'], "r") as f:
            #     params['tissue_type'] = f.read()
            # os.remove(pth)
            # print(params['tissue_type'])
            with torch.inference_mode():
                ct, inst = batch_pseudolabel_ensemb(
                    raw,raw1, models, params["tta"], augmenter, color_aug_fn,params['tissue_type']
                )
                futures.append(
                    executor.submit(
                        dump_results,
                        (ct.cpu().detach().numpy(), inst.cpu().detach().numpy(), zc),
                        z_cls,
                        z_inst,
                        prog_path,
                    )
                )

                zc += params["batch_size"]

        # Block until all data is written
        for _ in concurrent.futures.as_completed(futures):
            pass
    # clean up
    if os.path.exists(prog_path):
        os.remove(prog_path)
    return params, (z_inst, z_cls)


def batch_pseudolabel_ensemb(
    raw: torch.Tensor,
    raw1: torch.Tensor,
    models: List[torch.nn.Module],
    nviews: int,
    aug: SpatialAugmenter,
    color_aug_fn: torch.nn.Sequential,
        tissue_type='metastatis'
):
    """
    Run inference step on batch of images with test time augmentations

    Parameters
    ----------

    raw: torch.Tensor
        batch of input images
    models: List[torch.nn.Module]
        list of models to run inference with, e.g. multiple folds or a single model in a list
    nviews: int
        Number of test-time augmentation views to aggregate
    aug: SpatialAugmenter
        Augmentation module for geometric transformations
    color_aug_fn: torch.nn.Sequential
        Color Augmentation module

    Returns
    ----------

    ct: torch.Tensor
        Per pixel class predictions as a tensor of shape (batch_size, n_classes+1, tilesize, tilesize)
    inst: torch.Tensor
        Per pixel 3 class prediction map with boundary, background and foreground classes, shape (batch_size, 3, tilesize, tilesize)
    """
    tmp_3c_view = []
    tmp_ct_view = []
    aug_params = aug.params
    # ensure that at least one view is run, even when specifying 1 view with many models
    if nviews <= 0:
        out_fast = []
        with torch.inference_mode():
            for mod in models:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out_fast.append(mod(raw))
        out_fast = torch.stack(out_fast, axis=0).nanmean(0)
        ct = out_fast[:, 5:].softmax(1)
        inst = out_fast[:, 2:5].softmax(1)
    else:
        for pp in range(nviews):
            aug.params = {}
            if pp < 3:
                aug.params['mirror'] = aug_params['mirror'][pp]
            if pp > 3:
                aug.params['rotate'] = aug_params['rotate'][pp - 3]
            aug.interpolation = "bilinear"
            view_aug = aug.forward_transform(raw)
            aug.interpolation = "nearest"
            # view_aug = color_aug_fn(view_aug)
            view_aug = torch.clamp(view_aug, 0, 1)
            out_fast = []
            with torch.inference_mode():
                for mod in models:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        out_fast.append(aug.inverse_transform(mod(view_aug)))
            out_fast = torch.stack(out_fast, axis=0).nanmean(0)
            tmp_3c_view.append(out_fast[:, 2:5].softmax(1))
            out_fast[:, 5:] = nuclei_tissue_cof(ct=out_fast[:, 5:], raw1=raw1, tissue_type=tissue_type)
            tmp_ct_view.append(out_fast[:, 5:].softmax(1))
        ct = torch.stack(tmp_ct_view).nanmean(0)
        # ct = nuclei_tissue_cof(ct=ct,raw1=raw1,tissue_type=tissue_type)
        inst = torch.stack(tmp_3c_view).nanmean(0)
    return ct, inst


def get_inference_setup(params):
    """
    get model/ models and load checkpoint, create augmentation functions and set up parameters for inference
    """
    models = []
    pth = params["data_dir"]
    if not os.path.exists(pth):
        print("ERROR: No checkpoint found")
        sys.exit(0)

    checkpoint_path = f"{pth}/train/best_model"
    mod_params = toml.load(f"{pth}/params.toml")
    params["out_channels_cls"] = mod_params["out_channels_cls"]
    params["inst_channels"] = mod_params["inst_channels"]
    model = get_model(
        enc=mod_params["encoder"],
        out_channels_cls=params["out_channels_cls"],
        out_channels_inst=params["inst_channels"],
    ).to(device)
    model = load_checkpoint(model, checkpoint_path, device)
    model.eval()
    models.append(copy.deepcopy(model))

    # create augmentation functions on device
    augmenter = SpatialAugmenter(TTA_AUG_PARAMS).to(device)
    color_aug_fn = color_augmentations(False, rank=device)

    return params, models, augmenter, color_aug_fn

def nuclei_tissue_cof(raw1 = None, ct = None, tissue_type = 'metastatis'):
    # skin_prob_matrix = np.array([
    #     [0.8, 0.1, 1.0, 0.05, 0.8],  # Nuclei_tumor
    #     [1.0, 0.5, 0.8, 0.0, 0.7],  # Nuclei_stroma
    #     [0.7, 0.5, 0.9, 0.8, 1.0],  # Nuclei_apoptosis
    #     [0.0, 0.0, 0.5, 1.0, 0.0],  # Nuclei_epithelium
    #     [0.9, 0.5, 0.8, 0.5, 0.8],  # Nuclei_histiocyte
    #     [1.0, 0.8, 0.9, 0.5, 0.9],  # Nuclei_lymphocyte
    #     [0.7, 1.0, 0.9, 0.2, 1.0],  # Nuclei_neutrophil
    #     [0.5, 1.0, 0.5, 0.0, 0.0],  # Nuclei_endothelium
    #     [0.8, 0.0, 0.2, 1.0, 0.3],  # Nuclei_melanophage
    #     [0.9, 0.7, 0.9, 0.2, 0.9]  # Nuclei_plasma_cell
    # ])
    # skin_prob_matrix = np.array([
    #     [0.8, 0.0, 0.98, 0.0, 0.8],  # Nuclei_tumor
    #     [1.0, 0.0, 0.0, 0.0, 0.7],  # Nuclei_stroma
    #     [0.7, 0.0, 1.0, 0.0, 1.0],  # Nuclei_apoptosis
    #     [0.0, 0.0, 0.0, 1.0, 0.0],  # Nuclei_epithelium
    #     [1.0, 0.0, 1.0, 0.0, 0.8],  # Nuclei_histiocyte
    #     [0.99, 1.0, 1.0, 0.0, 0.9],  # Nuclei_lymphocyte
    #     [1.0, 1.0, 1.0, 0.0, 1.0],  # Nuclei_neutrophil
    #     [1.0, 1.0, 0.5, 0.0, 0.0],  # Nuclei_endothelium
    #     [1.0, 0.0, 1.0, 0.0, 1],  # Nuclei_melanophage
    #     [0.9, 0.0, 0.0, 0, 0.9]  # Nuclei_plasma_cell
    # ])
    skin_prob_matrix = np.array([
        [1, 1, 1, 0.0, 1],
        [1, 0, 1, 0.0, 1],
        [1, 1, 1, 1, 1],
        ])

    other_organs_prob_matrix = np.array([
        [1, 1, 1, 0.0, 1],
        [1, 0, 1, 0.0, 1],
        [1, 1, 1, 0.0, 1],
        ])
    # other_organs_prob_matrix = np.array([
    #     [0.0, 0.0, 0.93, 0.0, 0.93],  # Nuclei_tumor
    #     [1.0, 0.0, 1.0, 0.0, 0.0],  # Nuclei_stroma
    #     [1.0, 0.0, 0.99, 0, 1.0],  # Nuclei_apoptosis
    #     [0.0, 0.0, 0.0, 0, 0.0],  # Nuclei_epithelium
    #     [1.0, 0.0, 0.0, 0, 0.0],  # Nuclei_histiocyte
    #     [1.0, 1.0, 0.95, 0, 0.0],  # Nuclei_lymphocyte
    #     [1.0, 0.0, 1.0, 0, 0.0],  # Nuclei_neutrophil
    #     [1.0, 1.0, 0.0, 0, 0.0],  # Nuclei_endothelium
    #     [1.0, 0.0, 1.0, 0, 0.0],  # Nuclei_melanophage (absent in other organs)
    #     [1.0, 0.0, 0.0, 0, 0.0]  # Nuclei_plasma_cell
    # ])

    # other_organs_prob_matrix = np.array([
    #     [0.8, 0.1, 1.0, 0, 0.8],  # Nuclei_tumor
    #     [1.0, 0.5, 0.8, 0, 0.7],  # Nuclei_stroma
    #     [0.8, 0.5, 1.0, 0, 1.0],  # Nuclei_apoptosis
    #     [0.8, 0.1, 0.8, 0, 0.0],  # Nuclei_epithelium
    #     [0.8, 0.5, 0.8, 0, 0.8],  # Nuclei_histiocyte
    #     [1.0, 0.8, 0.8, 0, 0.8],  # Nuclei_lymphocyte
    #     [0.8, 1.0, 0.8, 0, 1.0],  # Nuclei_neutrophil
    #     [0.5, 1.0, 0.5, 0, 0.0],  # Nuclei_endothelium
    #     [0.0, 0.0, 0.0, 0, 0.0],  # Nuclei_melanophage (absent in other organs)
    #     [0.8, 0.8, 0.8, 0, 0.8]  # Nuclei_plasma_cell
    # ])

    # new_order = [7, 9, 1, 0, 4, 2, 3, 8, 6, 5]  # Indices matching the requested order
    prim_cof = skin_prob_matrix#[new_order, :]
    metas_cof = other_organs_prob_matrix#[new_order, :]




    if tissue_type ==  'metastatis':
        coff = metas_cof
    else:
        coff = prim_cof
        # print(tissue_type)
    coff = coff + 0*np.mean(coff[coff!=0])
    # coff[coff < 0] = 0
    for batch in range(ct.shape[0]):
        # print(batch)
        tisse_label = raw1[batch]
        tisse_label[tisse_label == 255] = 0
        nuclei_prob = ct[batch]
        # plt.imshow(torch.argmax(nuclei_prob, dim=0).cpu())
        # plt.show()
        # nuclei_prob[0] = np.mean(coff) * nuclei_prob[0]
        maxi = torch.abs(nuclei_prob.min()) if torch.abs(nuclei_prob.min()) > nuclei_prob.max() else nuclei_prob.max()
        for i in [4]:
            if (tisse_label == i).sum()>0:
                mask = tisse_label == i
                for j in range(1,4):
                    nuclei_prob[j][mask[0]] = coff[j-1,i-1]*(nuclei_prob[j][mask[0]] + 30)
                # plt.imshow(torch.argmax(nuclei_prob,dim=0).cpu())
                # plt.show()
        ct[batch] = nuclei_prob
    return ct

