import torch
import pandas as pd
from tqdm import tqdm

import json
import os
import pandas as pd
import hydra

from omegaconf import OmegaConf, DictConfig

from solo.args.feat_extract import parse_cfg
from solo.data.feat_extract_dataloader import prepare_data
from solo.methods import METHODS
import torch.nn as nn

import numpy as np



def extract_features(
        device: str,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
    ) -> pd.DataFrame:
    """Extract features from the dataset using the model and save to DataFrame.

    Args:
        device (str): gpu/cpu device.
        model (nn.Module): current model.
        dataloader (torch.utils.data.Dataloader): current dataloader containing data.

    Returns:
        pd.DataFrame: DataFrame containing features and labels.
    """
    features = []
    labels = []

    # set model to eval mode and collect all feature representations
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Collecting features"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            feats = model(x)
            features.append(feats.cpu().numpy())
            labels.append(y.cpu().numpy())
    model.train()

    # convert lists to numpy arrays
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    # create dataframe
    df = pd.DataFrame(features)
    df["label"] = labels

    return df

@hydra.main(version_base="1.2")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    args = parse_cfg(cfg)
    ckpt_path = args.pretrained_feature_extractor
    args_path = os.path.join(os.path.split(args.pretrained_feature_extractor)[0], "args.json")
    device = torch.device(f"cuda:{args.devices[0]}")
    # load arguments
    with open(args_path) as f:
        method_args = json.load(f)
    method_cfg = OmegaConf.create(method_args)
    method_cfg.devices = [0]

    model = (
        METHODS[method_args["method"]].load_from_checkpoint(ckpt_path, map_location=device, strict=False, cfg=method_cfg).backbone
    )
    
    #model = model.to(device)
    loader = prepare_data(cfg, idxs=[], subsampler=False)
    features_df = extract_features(device, model, loader)
    features_df.to_csv("features.csv", index=False)

if __name__ == "__main__":
    main()