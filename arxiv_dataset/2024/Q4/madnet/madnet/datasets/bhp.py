import os
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import polars.selectors as cs
from jax import Array
from jax import numpy as jnp
from jax import random as jr
from jax.scipy.special import expit
from jax.typing import ArrayLike

from madnet.datasets.data_model import DataSet, SyntheticDataGenerator
from madnet.logging import get_logger
from madnet.model_selection import get_folds

logger = get_logger(__name__)

CLEAN_DATA = "bhp.parquet"
SYNTHETIC_DATA = "bhp-synthetic.parquet"


def download_data(data_dir: str | os.PathLike = "data") -> pl.DataFrame:
    """Download and run data preparation script.

    Downloads data hosted at github.com/victor5as/RieszLearning
    Applies preprocessing and filtering as in that repo.
    """
    # The following preprocessing code is taken from this notebook:
    # https://github.com/victor5as/RieszLearning/blob/main/RieszNet_BHP.ipynb

    clean_data_path = Path(data_dir, CLEAN_DATA)
    fname = "data_BHP2.csv"
    url = f"https://raw.githubusercontent.com/victor5as/RieszLearning/main/data/BHP/{fname}"

    logger.info(f"Downloading raw data from {url}")

    df = (
        pl.read_csv(url)
        .lazy()
        .filter(
            # these filters are not mentioned in the paper but are applied in the Riesznet code
            # As a result we have n=3466 instead of n=3640 as in the paper.
            pl.col("log_p").gt(jnp.log(1.2)),
            pl.col("log_y").gt(jnp.log(15000)),
        )
        .drop(["distance_oil1000", "share"])
        .cast({cs.integer(): pl.UInt8})
        .collect()
        .to_dummies("state_fips", drop_first=True)
    )

    df.write_parquet(clean_data_path)
    logger.info(f"Clean data saved to {clean_data_path.resolve()}")
    return df


def train_synthetic_model(
    a: Array, x: Array, pred_path: str | os.PathLike
) -> pl.DataFrame:
    """Fit a random forest regressor and save predictons.

    Synthetic data is saved and also returned.
    """
    logger.info("Fitting Models for BHP synthetic data.")
    try:
        from sklearn.ensemble import RandomForestRegressor as RForest
        from sklearn.model_selection import cross_val_predict
    except ImportError:
        raise ImportError(
            "For BHP synthetic data production 'scikit-learn' must be installed."
        )

    a = a.squeeze()

    params = {"n_estimators": 100, "min_samples_leaf": 50, "random_state": 123}
    treatment_model = RForest(**params)
    variance_model = RForest(max_depth=5, **params)

    logger.info("Fitting Treatment mean model.")
    treatment_model.fit(x, a)

    logger.info("Cross-fitting Treatment mean model to use for variance model.")
    treatment_residual = a - cross_val_predict(treatment_model, x, a)

    logger.info("Fitting Treatment variance model.")
    variance_model.fit(x, treatment_residual**2)

    preds_treatment = treatment_model.predict(x)
    preds_variance = variance_model.predict(x)
    df = pl.DataFrame(
        {
            "treatment_mean": preds_treatment,
            "treatment_std_dev": preds_variance**0.5,
        }
    )
    df.write_parquet(pred_path)
    logger.info(f"Synthetic treament predictions saved to {Path(pred_path).resolve()}")
    return df


def load_bhp(
    data_dir: str | os.PathLike, refit: bool = False
) -> SyntheticDataGenerator:
    """Gasoline demand data from Blundell et al. (2017).

    The version here is the same as that used in the RieszNet paper
    (Chernozhukov et al. 2022).

    Description from Section IV (A) of Blundell et al. (2017):

    The data are from the 2001 National Household Travel
    Survey (NHTS), which surveys the civilian noninstitutionalized population in the United States.
    This is a household-level survey conducted by telephone and complemented by travel diaries and
    odometer readings. We select the sample to minimize heterogeneity as follows.
    We restrict the analysis to households with a white respondent, two or more adults,
    at least one child under age 16, and at least one driver.
    We drop households in the most rural areas, given the relevance of farming activities in these areas.
    We also restrict attention to localities where the state of residence is known and omit
    households in Hawaii due to its different geographic situation compared to the continental states.
    Households where key variables are not reported are excluded, and we restrict attention
    to gasoline-based vehicles (rather than diesel, natural gas, or electricity),
    requiring gasoline demand of at least 1 gallon; we also drop one observation where the reported
    gasoline share is larger than 1. We take vehicle ownership as given and do not investigate how
    changes in gasoline prices affect vehicle purchases or ownership.
    ...
    The resulting sample contains 3,640 observations.


    References
    ----------
    Blundell, R., Horowitz, J., & Parey, M. (2017). Replication data for:
    'Nonparametric Estimation of a Nonseparable Demand Function under the Slutsky
    Inequality Restriction' [dataset]. Harvard Dataverse.
    https://doi.org/10.7910/DVN/0YALNP

    Chernozhukov, V., Newey, W., Quintas-Martinez, V. M., & Syrgkanis, V. (2022).
    RieszNet and ForestRiesz: Automatic Debiased Machine Learning with Neural Nets
    and Random Forests.
    """
    clean_data_path = Path(data_dir, CLEAN_DATA)
    synthetic_data_path = Path(data_dir, SYNTHETIC_DATA)

    # get input data
    if clean_data_path.is_file():
        df = pl.read_parquet(clean_data_path)
    else:
        df = download_data(data_dir)

    # economics note:
    # log_p is log(price)
    # log_q is log(quantity) i.e. demand which we treat as an outcome.
    a = df.select("log_p").to_numpy()
    x = df.drop(["log_p", "log_q"]).to_numpy()

    # get synthetic data
    if not synthetic_data_path.is_file() or refit:
        df_treatment = train_synthetic_model(a, x, synthetic_data_path)
    else:
        df_treatment = pl.read_parquet(synthetic_data_path)

    treatment_mean = df_treatment.select("treatment_mean").to_numpy().squeeze()
    treatment_std_dev = df_treatment.select("treatment_std_dev").to_numpy().squeeze()

    return SyntheticDataGeneratorBHP(None, a, x, treatment_mean, treatment_std_dev)


@dataclass(frozen=True)
class SyntheticDataGeneratorBHP(SyntheticDataGenerator):
    """Create synthetic conditional mean outcome.

    Returns conditional mean outcome and its moment (derivative w.r.t. treatment)
    We use 'Complex f with linear and non-linear confounders' as described by Chernozhukov et al.

    References
    ----------
    Chernozhukov, V., Newey, W., Quintas-Martinez, V. M., & Syrgkanis, V. (2022).
    RieszNet and ForestRiesz: Automatic Debiased Machine Learning with Neural Nets
    and Random Forests.
    """

    a: ArrayLike
    x: ArrayLike
    treatment_mean: ArrayLike
    treatment_std_dev: ArrayLike

    def _get_synthetic_a(self, treatment_key: Array) -> Array:
        """Create synthetic treatment data."""
        treatment_mean = jnp.asarray(self.treatment_mean)
        noise = jr.normal(treatment_key, treatment_mean.shape)
        return treatment_mean + noise * self.treatment_std_dev

    def _get_synthetic_mu_and_moment(
        self, a: Array, mu_key: Array
    ) -> tuple[Array, Array]:
        """Create synthetic conditional mean and moment."""
        # Function based on 'true_f_compl_nonlin_conf' from:
        # https://github.com/victor5as/RieszLearning/blob/main/RieszNet_BHP.ipynb
        b_key, c_key = jr.split(mu_key, 2)
        z = jnp.hstack((a.reshape(-1, 1), self.x))
        b = jr.uniform(b_key, minval=-0.5, maxval=0.5, shape=(20,))
        c = jr.uniform(c_key, minval=-0.2, maxval=0.2, shape=(8,))

        non_linear = 1.5 * (expit(10 * z[:, 6]) + expit(10 * z[:, 8]))
        coef = -(z[:, 1] ** 2 / 10 + jnp.matmul(z[:, 1:9], c) + 0.5) / 6
        linear = coef * z[:, 0] ** 3 + jnp.matmul(z[:, 1:21], b)
        # for this simple function - I cannot be bothered to auto-diff with Jax
        mean = linear + non_linear
        moment = 3 * coef * jnp.square(a)
        return mean, moment

    def _get_synthetic_y(self, mu: Array, outcome_key: Array) -> Array:
        """Create synthetic outcome data."""
        # Function based on 'gen_y' from:
        # https://github.com/victor5as/RieszLearning/blob/main/RieszNet_BHP.ipynb
        noise = jr.normal(outcome_key, shape=mu.shape)
        std_dev = jnp.sqrt(5.6 * jnp.var(mu))
        return mu + std_dev * noise

    def _generate(self, seed: int) -> DataSet:
        # generate synthetic data on the fly
        x = jnp.asarray(self.x)
        treatment_key, outcome_key, mu_key, folds_key = jr.split(jr.key(seed), 4)
        synthetic_a = self._get_synthetic_a(treatment_key)
        synthetic_mu, synthetic_moment = self._get_synthetic_mu_and_moment(
            synthetic_a, mu_key
        )
        synthetic_y = self._get_synthetic_y(synthetic_mu, outcome_key)
        folds = get_folds(folds_key, n=len(x), n_folds=5)

        return DataSet(
            a=synthetic_a,
            x=x,
            y=synthetic_y,
            fold=folds,
            mu=synthetic_mu,
            moment=synthetic_moment,
        )


if __name__ == "__main__":
    generator = load_bhp("data")
    df = generator(0)
