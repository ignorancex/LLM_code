import pathlib
from typing import Literal

import altair as alt
import jax.numpy as jnp
import jax.random as jr
import polars as pl
import yaml

from madnet import estimands, estimators
from madnet.datasets import load_dataset
from madnet.estimators import BaseEstimator
from madnet.logging import get_logger

logger = get_logger(__name__)

FIT_KWARGS = dict(batch_size=64, num_epochs=300, validation_size=0.2, patience=300)


def run(
    configs: list[dict],
    dataset: Literal["ihdp", "bhp"] = "ihdp",
    start_seed: int = 0,
    num_runs: int = 20,
):
    out = pl.DataFrame()

    generator = load_dataset(dataset)
    match dataset:
        case "ihdp":
            estimand = estimands.AverageTreatmentEffect()
        case "bhp":
            estimand = estimands.AverageDerivativeEffect()

    for seed in range(start_seed, start_seed + num_runs):
        key = jr.key(seed)
        init_key, fit_key = jr.split(key)
        ds = generator(seed)
        y_std = jnp.std(ds.y)
        y_scl = (ds.y - jnp.mean(ds.y)) / y_std
        true_outcome_moment = jnp.mean(ds.moment) / y_std
        w = (ds.a, ds.x, y_scl)

        for cfg in configs:
            if cfg["learner"] == "DragonNet":
                model: BaseEstimator = getattr(estimators, cfg["learner"])(
                    ds.p, **cfg["init_kwargs"], key=init_key
                )
            else:
                model: BaseEstimator = getattr(estimators, cfg["learner"])(
                    estimand, ds.p, **cfg["init_kwargs"], key=init_key
                )

            for lr in {1.0e-4, 1.0e-5}:
                logger.info(
                    f"seed = {seed}, learner = {cfg['learner']}, learning rate = {lr}"
                )
                model, logs = model.fit(*w, **FIT_KWARGS, learning_rate=lr, key=fit_key)

                aug_logs = logs.with_columns(
                    learning_rate=lr,
                    seed=seed,
                    dataset=pl.lit(dataset),
                    true_outcome_moment=true_outcome_moment.item(),
                )
                out = pl.concat([out, aug_logs])

    return out


def plot(df: pl.DataFrame) -> alt.Chart:
    df = df.to_pandas()
    df["true_gamma_moment"] = 1.0
    df["zero"] = 0.0
    df["outcome_moment_bias"] = df["true_outcome_moment"] - df["outcome_ipw"]
    df["treatment_moment_bias"] = df["true_gamma_moment"] - df["gamma_ipw"]
    base = alt.Chart(df).encode(x="epoch").properties(width=250, height=120)
    zero = base.encode(y="zero").mark_line(color="black", opacity=0.7)

    # Treatment moment estimat, h(a)
    ipw_g_line = base.encode(
        y=alt.Y(
            "mean(treatment_moment_bias)", title="treament moment bias (IPW)"
        ).scale(domain=(-1, 1)),
        color=alt.Color("learner:N"),
    ).mark_line(clip=True, opacity=0.9)
    ipw_g_stderr = base.encode(
        y=alt.Y("treatment_moment_bias", title="").scale(domain=(-1, 1)),
        color=alt.Color("learner:N"),
    ).mark_errorband(extent="stderr", clip=True)
    ipw_g = zero + ipw_g_stderr + ipw_g_line

    # Outcome moment estimate, h(y)
    ipw_y_line = base.encode(
        y=alt.Y("mean(outcome_moment_bias)", title="outcome moment bias (IPW)").scale(
            domain=(-1, 1)
        ),
        color=alt.Color("learner:N"),
    ).mark_line(clip=True, opacity=0.9)
    ipw_y_stderr = base.encode(
        y=alt.Y("outcome_moment_bias", title="").scale(domain=(-1, 1)),
        color=alt.Color("learner:N"),
    ).mark_errorband(extent="stderr", clip=True)
    ipw_y = zero + ipw_y_stderr + ipw_y_line

    return ipw_g.facet(
        column=alt.Column("learning_rate:N", title="learning rate")
    ) & ipw_y.facet(column=alt.Column("learning_rate:N", title=""))


if __name__ == "__main__":
    dir = pathlib.Path("paper")
    madnet = yaml.safe_load((dir / "madnet_ate.yaml").read_text())
    riesznet = yaml.safe_load((dir / "riesznet_ate.yaml").read_text())
    dragonnet = yaml.safe_load((dir / "dragonnet.yaml").read_text())
    configs = [madnet, riesznet, dragonnet]
    out = run(configs)
    out.write_parquet(dir / "ipw_plot.parquet")
    logger.info(f"Saved results to {str(dir / 'ipw_plot.parquet')}")
    chart = plot(out)
    chart.save(str(dir / "ipw_plot.png"), ppi=200)
