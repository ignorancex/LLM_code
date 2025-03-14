from typing import List, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import transformations
from omegaconf import DictConfig, ListConfig, OmegaConf

from cell_imaging.s3dataset import S3Dataset


def load_meta_data():
    PLATE_TO_ID = {"BR00116991": 0, "BR00116993": 1, "BR00117000": 2}
    FIELD_TO_ID = dict(zip([str(i) for i in range(1, 10)], range(9)))
    WELL_TO_ID = {}
    for i in range(16):
        for j in range(1, 25):
            well_loc = f"{chr(ord('A') + i)}{j:02d}"
            WELL_TO_ID[well_loc] = len(WELL_TO_ID)

    WELL_TO_LBL = {}
    # map the well location to the perturbation label
    base_path = "s3://insitro-research-2023-context-vit/jumpcp/platemap_and_metadata"
    PLATE_MAP = {
        "compound": f"{base_path}/JUMP-Target-1_compound_platemap.tsv",
        "crispr": f"{base_path}/JUMP-Target-1_crispr_platemap.tsv",
        "orf": f"{base_path}/JUMP-Target-1_orf_platemap.tsv",
    }
    META_DATA = {
        "compound": f"{base_path}/JUMP-Target-1_compound_metadata.tsv",
        "crispr": f"{base_path}/JUMP-Target-1_crispr_metadata.tsv",
        "orf": f"{base_path}/JUMP-Target-1_orf_metadata.tsv",
    }

    for perturbation in PLATE_MAP.keys():
        df_platemap = pd.read_parquet(PLATE_MAP[perturbation])
        df_metadata = pd.read_parquet(META_DATA[perturbation])
        df = df_metadata.merge(df_platemap, how="inner", on="broad_sample")

        if perturbation == "compound":
            target_name = "target"
        else:
            target_name = "gene"

        codes, uniques = pd.factorize(df[target_name])
        codes += 1  # set none (neg control) to id 0
        assert min(codes) == 0
        print(f"{target_name} has {len(uniques)} unique values")
        WELL_TO_LBL[perturbation] = dict(zip(df["well_position"], codes))

    return PLATE_TO_ID, FIELD_TO_ID, WELL_TO_ID, WELL_TO_LBL


class JUMPCP(S3Dataset):
    """
    Loads preprocessed JUMPCP dataset
    """

    normalize_mean: Union[List[float], None] = None
    normalize_std: Union[List[float], None] = None

    def __init__(
        self,
        cyto_mask_path_list: ListConfig[str],
        split: str,  # train, valid or test
        is_train: bool,
        transform_cfg: DictConfig,
        perturbation_list: ListConfig[str],
        channels: Union[List[int], None],
        normalization_mean=[],
        normalization_std=[],
    ) -> None:
        """Initialize the dataset."""
        super().__init__()

        # read the cyto mask df
        df = pd.concat(
            [pd.read_parquet(path) for path in cyto_mask_path_list], ignore_index=True
        )
        df = self.get_split(df, split)

        self.data_path = list(df["path"])
        self.data_id = list(df["ID"])
        self.well_loc = list(df["well_loc"])

        assert len(perturbation_list) == 1
        self.perturbation_type = perturbation_list[0]

        if type(channels[0]) is str:
            # channel is separated by hyphen
            self.channels = torch.tensor([int(c) for c in channels[0].split("-")])
        else:
            self.channels = torch.tensor([c for c in channels])

        self.transform = getattr(transformations, transform_cfg.name)(
            is_train,
            normalization_mean=normalization_mean,
            normalization_std=normalization_std,
        )

        self.plate2id, self.field2id, self.well2id, self.well2lbl = load_meta_data()

    def get_split(self, df, split_name, seed=0):
        np.random.seed(seed)
        perm = np.random.permutation(df.index)
        m = len(df.index)
        train_end = int(0.6 * m)
        validate_end = int(0.2 * m) + train_end

        if split_name == "train":
            return df.iloc[perm[:train_end]]
        elif split_name == "valid":
            return df.iloc[perm[train_end:validate_end]]
        elif split_name == "test":
            return df.iloc[perm[validate_end:]]
        else:
            raise ValueError("Unknown split")

    def __getitem__(self, index):
        if self.well_loc[index] not in self.well2lbl[self.perturbation_type]:
            # this well is not labeled
            return None

        img_chw = self.get_image(self.data_path[index])
        if img_chw is None:
            return None

        img_hwc = img_chw.transpose(1, 2, 0)
        img_chw = self.transform(img_hwc)

        channels = self.channels.numpy()

        assert type(img_chw) is not list, "Only support jumpcp for supervised training"

        img_chw = img_chw[channels]

        return (img_chw, self.well2lbl[self.perturbation_type][self.well_loc[index]])

    def __len__(self) -> int:
        return len(self.data_path)


class create_jumpcp_DataModule(pl.LightningDataModule):
    def __init__(
        self,
        perturbation_list=["compound"],
        channels=[0, 1, 2, 3, 4, 5, 6, 7],
        cyto_mask_path_list=[
            "s3://insitro-research-2023-context-vit/jumpcp/BR00116991.pq"
        ],
        batch_size: int = 64,
        num_workers: int = 12,
        normalization_mean=[],
        normalization_std=[],
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        transform_cfg = OmegaConf.create({"name": "CellAugmentation"})

        self.dataset_train = JUMPCP(
            split="train",
            is_train=True,
            transform_cfg=transform_cfg,
            perturbation_list=perturbation_list,
            cyto_mask_path_list=cyto_mask_path_list,
            channels=channels,
            normalization_mean=normalization_mean,
            normalization_std=normalization_std,
        )
        self.dataset_val = JUMPCP(
            split="valid",
            is_train=False,
            transform_cfg=transform_cfg,
            perturbation_list=perturbation_list,
            cyto_mask_path_list=cyto_mask_path_list,
            channels=channels,
            normalization_mean=normalization_mean,
            normalization_std=normalization_std,
        )
        self.dataset_test = JUMPCP(
            split="test",
            is_train=False,
            transform_cfg=transform_cfg,
            perturbation_list=perturbation_list,
            cyto_mask_path_list=cyto_mask_path_list,
            channels=channels,
            normalization_mean=normalization_mean,
            normalization_std=normalization_std,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=False,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
            drop_last=False,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
            drop_last=False,
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=False,
            drop_last=False,
        )


def load_DataModule(
    batch_size=32,
    num_workers=12,
    perturbation_list=["compound"],
    channels=[0, 1, 2, 3, 4, 5, 6, 7],
    cyto_mask_path_list=["s3://insitro-research-2023-context-vit/jumpcp/BR00116991.pq"],
    normalization_mean=[],
    normalization_std=[],
) -> pl.LightningDataModule:
    return create_jumpcp_DataModule(
        perturbation_list,
        channels,
        cyto_mask_path_list,
        batch_size,
        num_workers,
        normalization_mean,
        normalization_std,
    )
