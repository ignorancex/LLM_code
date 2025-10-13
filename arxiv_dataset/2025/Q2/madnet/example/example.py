import jax
from jax import Array
from jax import random as jr

from madnet.estimands import AverageTreatmentEffect
from madnet.estimators import MADNet, RieszNet
from madnet.logging import get_logger

logger = get_logger(__name__)

ATE = AverageTreatmentEffect()


def synthetic_data(n, p, key: Array) -> tuple[Array, Array, Array]:
    x_key, a_key, y_key = jr.split(key, 3)
    x = jr.uniform(x_key, shape=(n, p))
    propensity_score = jax.nn.sigmoid(x[:, 0] ** 2)
    a = jr.binomial(a_key, 1, p=propensity_score, shape=(n,))
    noise = jr.normal(y_key, shape=(n,))
    cate = 0.1 + 0.05 * x[:, 0]
    y = x[:, 0] - x[:, 1] + cate * a + noise
    y = y.reshape((n, 1))
    a = a.reshape((n, 1))
    return y, a, x


if __name__ == "__main__":
    seed = 134
    n = 4000
    p = 6
    num_epochs = 5000

    key = jr.PRNGKey(seed)
    key_train, key_estimation = jr.split(key)

    y, a, x = synthetic_data(n=n, p=p, key=key_train)

    rnet = RieszNet(
        estimand=ATE,
        covariate_dim=p,
        key=key,
    )
    rnet_fitted, rnet_logs = rnet.fit(
        a=a,
        x=x,
        y=y,
        num_epochs=num_epochs,
        key=key,
        optimizer="adam",
        min_delta=1e-6,
        patience=40,
    )
    rnet_est = rnet_fitted.estimate(a=a, x=x, y=y)

    # evaluate estimators
    for name, vals in rnet_est.items():
        logger.info(f"Estimator: {name}: {vals}")

    # Repeat with MADNet
    mnet = MADNet(
        estimand=ATE,
        covariate_dim=p,
        key=key,
    )
    mnet_fitted, mnet_logs = mnet.fit(
        a=a,
        x=x,
        y=y,
        num_epochs=num_epochs,
        key=key,
        optimizer="adam",
        min_delta=1e-6,
        patience=40,
    )
    mnet_est = mnet_fitted.estimate(a=a, x=x, y=y)

    # evaluate estimators
    for name, vals in mnet_est.items():
        logger.info(f"Estimator: {name}: {vals}")
