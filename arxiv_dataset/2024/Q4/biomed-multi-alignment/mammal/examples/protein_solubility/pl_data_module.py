import os
import shutil
from collections.abc import Callable

import pandas as pd
import pytorch_lightning as pl
import wget
from fuse.data.datasets.dataset_default import DatasetDefault
from fuse.data.ops.ops_read import OpReadDataframe
from fuse.data.pipelines.pipeline_default import PipelineDefault
from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp
from fuse.data.utils.collates import CollateDefault
from torch.utils.data.dataloader import DataLoader

from mammal.keys import *  # noqa


class ProteinSolubilityDataModule(pl.LightningDataModule):
    def __init__(
        self,
        *,
        data_path: str,
        batch_size: int,
        tokenizer_op: ModularTokenizerOp,
        train_dl_kwargs: dict,
        valid_dl_kwargs: dict,
        seed: int,
        data_preprocessing: Callable,
        protein_max_seq_length: int,
        encoder_input_max_seq_len: int,
        labels_max_seq_len: int,
    ) -> None:
        """_summary_
        Args:
            data_path (str): path to the raw data, if not exist, will download the data to the given path.
            batch_size (int): batch size
            tokenizer_op (ModularTokenizerOp): tokenizer op
            encoder_inputs_max_seq_len: max tokenizer sequence length for the encoder inputs,
            labels_max_seq_len: max tokenizer sequence length for the labels,
            train_dl_kwargs (dict): train dataloader constructor parameters
            valid_dl_kwargs (dict): validation dataloader constructor parameters
            seed (int): random seed
        """
        super().__init__()
        self.data_path = data_path
        self.tokenizer_op = tokenizer_op
        self.protein_max_seq_length = protein_max_seq_length
        self.encoder_input_max_seq_len = encoder_input_max_seq_len
        self.labels_max_seq_len = labels_max_seq_len
        self.batch_size = batch_size
        self.train_dl_kwargs = train_dl_kwargs
        self.valid_dl_kwargs = valid_dl_kwargs
        self.seed = seed
        self.data_preprocessing = data_preprocessing

        self.pad_token_id = self.tokenizer_op.get_token_id("<PAD>")

    def setup(self, stage: str) -> None:
        self.ds_dict = load_datasets(self.data_path)

        task_pipeline = [
            (
                # Prepare the input string(s) in modular tokenizer input format
                self.data_preprocessing,
                dict(
                    protein_sequence_key="data.protein",
                    solubility_label_key="data.label",
                    tokenizer_op=self.tokenizer_op,
                    protein_max_seq_length=self.protein_max_seq_length,
                    encoder_input_max_seq_len=self.encoder_input_max_seq_len,
                    labels_max_seq_len=self.labels_max_seq_len,
                ),
            ),
        ]

        for ds in self.ds_dict.values():
            ds.dynamic_pipeline.extend(task_pipeline)

    def train_dataloader(self) -> DataLoader:
        train_loader = DataLoader(
            dataset=self.ds_dict["train"],
            batch_size=self.batch_size,
            collate_fn=CollateDefault(),
            shuffle=True,
            **self.train_dl_kwargs,
        )
        return train_loader

    def val_dataloader(self) -> DataLoader:
        val_loader = DataLoader(
            self.ds_dict["val"],
            batch_size=self.batch_size,
            collate_fn=CollateDefault(),
            **self.valid_dl_kwargs,
        )

        return val_loader

    def test_dataloader(self) -> DataLoader:
        test_loader = DataLoader(
            self.ds_dict["test"],
            batch_size=self.batch_size,
            collate_fn=CollateDefault(),
            **self.valid_dl_kwargs,
        )

        return test_loader

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()


_SOLUBILITY_URL = "https://zenodo.org/api/records/1162886/files-archive"


def load_datasets(data_path: str) -> dict[str, DatasetDefault]:
    """
    Automatically downloads the data and create dataset iterator for "train", "val" and "test".
    paper: https://academic.oup.com/bioinformatics/article/34/15/2605/4938490
    Data retrieved from: https://zenodo.org/records/1162886
    The benchmark requires classifying protein sequences into binary labels - Soluble or Insoluble (1 or 0).
    :param data_path: path to a directory to store the raw data
    :return: dictionary that maps fold name "train", "val" and "test" to a dataset iterator
    """

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    raw_data_path = os.path.join(data_path, "sameerkhurana10-DSOL_rv0.2-20562ad/data")
    if not os.path.exists(raw_data_path):
        wget.download(_SOLUBILITY_URL, data_path)
        file_path = os.path.join(data_path, "1162886.zip")
        shutil.unpack_archive(file_path, extract_dir=data_path)
        inner_file_path = os.path.join(
            data_path, "sameerkhurana10", "DSOL_rv0.2-v0.3.zip"
        )
        shutil.unpack_archive(inner_file_path, extract_dir=data_path)
        assert os.path.exists(
            raw_data_path
        ), f"Error: download complete but {raw_data_path} doesn't exist"

    # read files
    df_dict = {}
    for set_name in ["train", "val", "test"]:
        input_df = pd.read_csv(
            os.path.join(raw_data_path, f"{set_name}_src"), names=["data.protein"]
        )
        labels_df = pd.read_csv(
            os.path.join(raw_data_path, f"{set_name}_tgt"), names=["data.label"]
        )
        df_dict[set_name] = (input_df, labels_df)

    ds_dict = {}
    for set_name in ["train", "val", "test"]:
        input_df, labels_df = df_dict[set_name]
        size = len(labels_df)
        print(f"{set_name} set size is {size}")
        dynamic_pipeline = PipelineDefault(
            "solubility",
            [
                (OpReadDataframe(input_df, key_column=None), dict()),
                (OpReadDataframe(labels_df, key_column=None), dict()),
            ],
        )

        ds = DatasetDefault(sample_ids=size, dynamic_pipeline=dynamic_pipeline)
        ds.create()
        ds_dict[set_name] = ds

    return ds_dict


if __name__ == "__main__":
    load_datasets("data")
