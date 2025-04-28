import argparse
import pathlib
from datetime import datetime
from typing import Literal

import jax.numpy as jnp
import jax.random as jr
import polars as pl
import yaml
from tqdm import tqdm

from madnet import estimands, estimators
from madnet.datasets import load_dataset
from madnet.estimators import BaseEstimator
from madnet.estimators._optimization_utils import estimates2dict
from madnet.logging import get_logger

logger = get_logger(__name__)


def print_leaderboard(out: pl.DataFrame):
    print("Top 5 MAE Leaderboard")
    print("=====================")
    print(
        out.group_by(["learner", "estimator"])
        .agg(pl.col("absolute_error").mean())
        .sort("absolute_error")
        .head()
    )


def main(
    cfg: dict, dataset: Literal["ihdp", "bhp"], start_from: int = 0, num_runs: int = 3
):
    out = pl.DataFrame()

    generator = load_dataset(dataset)
    match dataset:
        case "ihdp":
            estimand = estimands.AverageTreatmentEffect()
        case "bhp":
            estimand = estimands.AverageDerivativeEffect()

    if num_runs < 0:
        seeds = range(generator.max_seed + 1)
    else:
        seeds = range(start_from, start_from + num_runs)

    for seed in (pbar := tqdm(seeds)):
        pbar.set_description(f"Seed: {seed}")

        key = jr.key(seed)
        init_key, fit_key = jr.split(key)

        # Generate dataset
        ds = generator(seed)
        # Normalize treatment
        a_sf = 1 / (jnp.max(ds.a) - jnp.min(ds.a))
        a = a_sf * (ds.a - jnp.min(ds.a))
        if isinstance(estimand, estimands.AverageTreatmentEffect):
            # Max min normalization shouldn't alter binary treatment
            assert jnp.array_equal(a, ds.a)
        # Normalize outcome
        y_std = jnp.std(ds.y)
        true_outcome_moment = jnp.mean(ds.moment) / y_std
        y_scl = (ds.y - jnp.mean(ds.y)) / y_std
        # Combine
        w = (a, ds.x, y_scl)

        if cfg["learner"] == "DragonNet":
            model: BaseEstimator = getattr(estimators, cfg["learner"])(
                ds.p, **cfg["init_kwargs"], key=init_key
            )
        else:
            model: BaseEstimator = getattr(estimators, cfg["learner"])(
                estimand, ds.p, **cfg["init_kwargs"], key=init_key
            )

        for fit_kw in cfg["fit_kwargs"]:
            model, logs = model.fit(*w, **fit_kw, key=fit_key)

        df = (
            pl.DataFrame(estimates2dict(model, w))
            .with_columns(
                dataset=pl.lit(dataset),
                seed=seed,
                true_outcome_moment=true_outcome_moment.item(),
                learner=pl.lit(cfg["learner"]),
                y_std=y_std.item(),
                a_sf=a_sf.item(),
            )
            .with_columns(
                absolute_error=pl.col("y_std")
                * pl.col("a_sf")
                * abs(pl.col("true_outcome_moment") - pl.col("estimate"))
            )
        )
        out = pl.concat([out, df])
        out.write_parquet(out_dir / "results.parquet")

        if (seed + 1) % 10 == 0:
            print_leaderboard(out)

    print_leaderboard(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="dataset name", default="ihdp")
    parser.add_argument("-s", "--start", type=int, help="seed to start from", default=0)
    parser.add_argument(
        "-n",
        "--numruns",
        type=int,
        help="number of runs. If less than zero then max runs is used.",
        default=1,
    )
    parser.add_argument(
        "-c", "--config", help="config file", default="paper/madnet.yaml"
    )
    args = parser.parse_args()

    out_dir = pathlib.Path("results", str(datetime.now()))
    out_dir.mkdir(parents=True, exist_ok=True)
    pathlib.Path("results", ".gitignore").write_text("*")
    config = yaml.safe_load(pathlib.Path(args.config).read_text())
    (out_dir / "config.yaml").write_text(yaml.dump(config))

    main(config, args.dataset, args.start, args.numruns)
