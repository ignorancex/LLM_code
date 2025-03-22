# Copyright 2023 Sean Robertson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# WARNING! This script is intended for use in the Jupyter notebook
# scripts/analyze.ipynb. As such, it expects the run directory to be scripts/.

import csv

from pathlib import Path
from typing import Any, Iterable, Dict, Literal, Sequence, TypeVar, overload

import pandas as pd

from ruamel.yaml import YAML
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pydrobert.param.serialization import register_serializer

from scpc.train import LightningPretrainedFrontendParams as Params

__all__ = [
    "collate_data",
    "check_data",
    "filter_data_in",
    "filter_data_equal",
]

MODEL_BLACKLIST = (
    "cpc.mono",
    "cpc.deft",
    "cpc.tri",
    "superb.fbank",
)


register_serializer("reckless_json")


def collate_data(
    results_from: Literal["zrc", "tb", "asr_training"] = "zrc",
    exp_dir: str = "../exp",
    model_blacklist: Sequence[str] = MODEL_BLACKLIST,
    collapse_distributed: bool = True,
    model_yaml_glob: str = "*/version_????/model.yaml",
):
    """Combine experiment parameters and results"""
    yaml = YAML(typ="safe", pure=True)

    model_data, res_data = [], []
    exp_dir: Path = Path(exp_dir)
    for id, pth in enumerate(exp_dir.glob(model_yaml_glob)):
        model, version = pth.parts[-3:-1]
        if model in model_blacklist:
            continue

        datum = yaml.load(pth)
        datum["id"] = id
        # clobbers useless 'name' field
        datum["name"] = f"{model}/{version}"
        datum["version"] = int(version.split("_")[1])
        model_data.append(datum)

        if results_from == "zrc":
            for zrc_pth in pth.parent.glob(
                "zrc/librispeech/**/scores/score_all_phonetic.csv"
            ):
                pca_style = zrc_pth.parts[-3]
                assert pca_style == "full" or pca_style.startswith("pca_"), zrc_pth
                with zrc_pth.open(newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        row["pca_style"] = pca_style
                        row["score"] = float(row["score"])
                        datum = dict(id=id, zrc=row)
                        res_data.append(datum)
        elif results_from == "tb":
            for event_pth in exp_dir.glob(f'tb_logs/{datum["name"]}/events.*'):
                ea = EventAccumulator(str(event_pth))
                ea.Reload()
                if ({"epoch", "val_loss"} - set(ea.Tags()["scalars"])) != set():
                    continue
                for epoch, val_loss in zip(ea.Scalars("epoch"), ea.Scalars("val_loss")):
                    assert epoch.step == val_loss.step
                    datum = dict(
                        id=id,
                        tb=dict(
                            step=int(epoch.step),
                            epoch=int(epoch.value),
                            val_loss=val_loss.value,
                        ),
                    )
                    res_data.append(datum)
        elif results_from == "asr_training":
            for training_pth in pth.parent.glob(f"baseline/full_v2000/*training.csv"):
                style = "full" if training_pth.name == "training.csv" else "short"
                with training_pth.open(newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        for x in (
                            "epoch",
                            "es_resume_cd",
                            "es_patience_cd",
                            "rlr_resume_cd",
                            "rlr_patience_cd",
                        ):
                            row[x] = int(row[x])
                        for x in ("lr", "train_met", "val_met"):
                            row[x] = float(row[x])
                        row["style"] = style
                        datum = dict(id=id, asr_training=row)
                        res_data.append(datum)
        else:
            raise NotImplementedError

    df = pd.json_normalize(model_data).set_index("id")

    # throw away columns we probably don't care about
    df = df.drop(
        list(df.filter(regex=r"\.name$").columns)
        + [
            "system_description",
            "training.accelerator",
            "training.cpc_loss.speaker_regex",
        ],
        axis=1,
    )

    # for convenience, we remap any rows with context_type == 'id' to
    # context_type == 'csa', but with 0 layers and max_width 1
    # idx = df['context_type'] == 'id'
    df.loc[
        df["context_type"] == "id",
        [
            "context_type",
            "csa.num_layers",
            "csa.num_heads",
            "csa.dim_feedforward",
            "csa.max_width",
        ],
    ] = ["csa", 0, 8, 1024, 1]

    # depopulate the values which were not selected
    latent_types: Sequence[str] = Params.param.latent_type.objects
    for latent_type in latent_types:
        latent_ne_idx = df["latent_type"] != latent_type
        latent_cols = df.filter(regex=f"^{latent_type}\\.").columns
        df.loc[latent_ne_idx, latent_cols] = pd.NA

    context_types: Sequence[str] = Params.param.context_type.objects
    for context_type in context_types:
        context_ne_idx = df["context_type"] != context_type
        context_cols = df.filter(regex=f"^{context_type}\\.").columns
        df.loc[context_ne_idx, context_cols] = pd.NA

    loss_types: Sequence[str] = Params.param.training.class_.param.loss_type.objects
    for loss_type in loss_types:
        loss_ne_idx = df["training.loss_type"] != loss_type
        loss_cols = df.filter(
            regex=f"^training\\.{loss_type.replace('-', '_')}_loss"
        ).columns
        df.loc[loss_ne_idx, loss_cols] = pd.NA

    # fill max_chunks with zeros whenever the chunking policy is none
    no_chunking_idx = df["training.chunking.policy"] == "none"
    df.loc[no_chunking_idx, "training.chunking.max_chunks"] = 0

    if collapse_distributed:
        # make turn task-level sizes into global sizes by multiplying by num_devices and
        # num_nodes
        num_devices = df["training.num_devices"].fillna(1)
        num_nodes = df["training.num_nodes"].fillna(1)
        df["training.data.common.batch_size"] *= (num_devices * num_nodes).astype(
            df["training.data.common.batch_size"].dtype
        )
        chunk_idx = df["training.chunking.max_chunks"].notna()
        df.loc[chunk_idx, "training.chunking.max_chunks"] *= (
            (num_devices * num_nodes)
            .astype(df["training.chunking.max_chunks"].dtype)
            .loc[chunk_idx]
        )

        df = df.drop(["training.num_devices", "training.num_nodes"], axis=1)

    df = pd.json_normalize(res_data).join(df, on="id", validate="m:1")
    df = df.drop(
        ["id", "training.data.common.subset_ids"], axis=1
    )  # we no longer need this

    df = df.dropna(axis=1, how="all")  # remove columns which are all NA

    # make all string-based entries categorical
    df = df.astype(dict((x, "category") for x in df.columns if df[x].dtype.char == "O"))

    return df


def check_data(df: pd.DataFrame, *cols: str) -> None:
    """Check that the provided cols are the only ones moving and there are no N/As"""
    not_cols = set(cols) - set(str(x) for x in df.columns)
    if not_cols:
        raise ValueError(f"df does not contain colums: {not_cols}")

    # check that values in cols are non-null
    for col in cols:
        if df[col].isna().any():
            raise ValueError(f"col '{col}' from var contains N/A value(s)")

    # now check that the only remaining variable columns can be found in cols
    # (except "name" and "version", which are probably varying with the values in cols)
    for col in df.columns:
        if col in cols or col in ("name", "version"):
            continue
        unique_vals = df[col].unique()
        if len(unique_vals) > 1:
            raise ValueError(f"Column '{col}' contains values '{unique_vals}'")


A = TypeVar("A")


def _filter_preamble(
    df: pd.DataFrame,
    args: Sequence[Dict[str, A]],
    kwargs: Dict[str, A],
) -> Dict[str, A]:
    if len(args) > 1 or (len(args) == 1) == (len(kwargs) > 0):
        raise ValueError("Either pass dict or keyword args")

    if len(args) == 1:
        col2x = args[0]
    else:
        col2x = kwargs

    not_cols = set(col2x) - set(str(x) for x in df.columns)
    if not_cols:
        raise ValueError(f"df does not contain colums: {not_cols}")

    return col2x


@overload
def filter_data_in(df: pd.DataFrame, col2seq: Dict[str, Iterable[Any]]) -> pd.DataFrame:
    ...


@overload
def filter_data_in(df: pd.DataFrame, **col_seq: Iterable[Any]) -> pd.DataFrame:
    ...


def filter_data_in(
    df: pd.DataFrame, *args: Dict[str, Iterable[Any]], **kwargs: Iterable[Any]
) -> pd.DataFrame:
    """Filter data with column values matching a value in iterables passed"""
    col2seq = _filter_preamble(df, args, kwargs)

    idx = True
    for col, seq in col2seq.items():
        idx = df[col].isin(list(seq)) & idx

    return df.loc[idx]


@overload
def filter_data_equal(df: pd.DataFrame, col2val: Dict[str, Any]) -> pd.DataFrame:
    ...


@overload
def filter_data_equal(df: pd.DataFrame, **col_val: Any) -> pd.DataFrame:
    ...


def filter_data_equal(
    df: pd.DataFrame, *args: Dict[str, Any], **kwargs: Any
) -> pd.DataFrame:
    """Filter data with column values matching those passed"""
    col2val = _filter_preamble(df, args, kwargs)

    idx = True
    for col, val in col2val.items():
        idx = (df[col] == val) & idx

    return df.loc[idx]
