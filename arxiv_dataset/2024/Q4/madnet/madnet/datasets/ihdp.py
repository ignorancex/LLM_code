import os
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from jax import Array
from jax import numpy as jnp
from jax import random as jr
from tqdm import tqdm

from madnet.logging import get_logger
from madnet.model_selection import get_folds

from .data_model import DataSet, SyntheticDataGenerator

logger = get_logger(__name__)

CLEAN_DATA = "ihdp.parquet"


def download_data(data_dir: str | os.PathLike = "data") -> pl.DataFrame:
    clean_data_path = Path(data_dir, CLEAN_DATA)
    url = "https://raw.githubusercontent.com/victor5as/RieszLearning/main/data/IHDP/sim_data/{fname}"
    autokey = [f"column_{i+1}" for i in range(30)]
    columns = ["a", "y", "y_cf", "mu_0", "mu_1", *[f"x_{i+1}" for i in range(26)]]
    mapping = dict(zip(autokey, columns))
    df = pl.DataFrame()
    for i in (pbar := tqdm(range(1_000))):
        fname = f"Downloading: ihdp_{i+1}.csv"
        pbar.set_description(fname)
        current_df = pl.read_csv(
            url.format(fname=fname), has_header=False, separator=" "
        ).rename(mapping)
        df = pl.concat([df, current_df.with_columns(seed=i)])

    df.write_parquet(clean_data_path)
    logger.info(f"Clean data saved to {clean_data_path.resolve()}")
    return df


def load_ihdp(data_dir: str | os.PathLike) -> SyntheticDataGenerator:
    """Infant Health and Development Program (IHDP).

    This semi-synthetic dataset is constructed from a randomized experiment
    investigating the effect of home visits by specialists on future cognitive scores.
    """

    clean_data_path = Path(data_dir, CLEAN_DATA)

    if not clean_data_path.is_file():
        download_data(data_dir)

    data = pl.read_parquet(clean_data_path)

    max_seed = max(data["seed"])
    n, p, r = 747, 25, 1000
    a = jnp.moveaxis(data["a"].to_numpy().reshape(r, n), 0, -1)
    x = jnp.moveaxis(data[:, 5:-1].to_numpy().reshape(r, n, p), 0, -1)
    y = jnp.moveaxis(data["y"].to_numpy().reshape(r, n), 0, -1)
    mu_0 = jnp.moveaxis(data["mu_0"].to_numpy().reshape(r, n), 0, -1)
    mu_1 = jnp.moveaxis(data["mu_1"].to_numpy().reshape(r, n), 0, -1)
    mu = jnp.where(a, mu_1, mu_0)
    moment = mu_1 - mu_0
    return SyntheticDataGeneratorIHDP(max_seed, a, x, y, mu, moment)


@dataclass(frozen=True)
class SyntheticDataGeneratorIHDP(SyntheticDataGenerator):
    a: Array
    x: Array
    y: Array
    mu: Array
    moment: Array

    def _generate(self, seed: int, n_folds: int = 5) -> DataSet:
        key = jr.key(seed)
        return DataSet(
            a=self.a[..., seed],
            x=self.x[..., seed],
            y=self.y[..., seed],
            fold=get_folds(key, len(self.y), n_folds=n_folds),
            mu=self.mu[..., seed],
            moment=self.moment[..., seed],
        )


if __name__ == "__main__":
    generator = load_ihdp("data")
    df = generator(0)
    splits = df.split()
