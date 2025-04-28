import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.mixture._gaussian_mixture import (
    _compute_precision_cholesky,
    _compute_log_det_cholesky,
)


def generate_random_covariance(A, n, seed):
    np.random.seed(seed)
    B = np.random.randn(n, n)
    B = B @ B.T
    A = A + B
    return A


def rmixGaussian(means, covs, weights, n_samples, random_state=0):
    """
    Sample from a Gaussian mixture

    Parameters
    ----------
    means : array-like, shape (n, d)
    covs :  array-like, shape (n, d, d)
    weights : array-like, shape (n,)

    Returns
    -------
    # n_sampels of samples from the Gaussian mixture
    """
    n_components = means.shape[1]
    if n_samples < 1:
        raise ValueError(
            "Invalid value for 'n_samples': %d . The sampling requires at "
            "least one sample." % (n_components))

    rng = np.random.default_rng(random_state)
    n_samples_comp = rng.multinomial(n_samples, weights)
    X = np.vstack([
        rng.multivariate_normal(mean, cov, int(sample))
        for (mean, cov, sample) in zip(means, covs, n_samples_comp)
    ])

    y = np.concatenate([
        np.full(sample, j, dtype=int)
        for j, sample in enumerate(n_samples_comp)
    ])

    return (X, y)


def df(x, mean, sd, f):
    x = x.reshape(-1, 1) - mean.T
    # x = x - mean.T
    x /= sd
    return f.pdf(x) / sd


def dmixf(x, mean, var, w, f):
    """
    Input:
    x: array-like (n,)
    mean: array-like (k, )
    sd: array-like (k, )
    w: array-like (k, )
    Output:
    sum(w*pnorm(x,mean,sd)): array-like (n,)
    """
    sd = np.sqrt(var)
    prob = df(x, mean, sd, f)
    prob *= w
    return prob.sum(1)


def compute_resp(X, means, covs):
    n_samples, n_features = X.shape
    n_components, _ = means.shape
    precisions_chol = _compute_precision_cholesky(covs, "full")
    log_det = _compute_log_det_cholesky(precisions_chol, "full", n_features)
    log_prob = np.empty((n_samples, n_components))
    for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
        y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
        log_prob[:, k] = np.sum(np.square(y), axis=1)
    log_resp = -0.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det
    return log_resp


def label_predict(weights, means, covs, X, return_resp=False):
    resp = compute_resp(X, means, covs)
    # add weight
    resp = np.exp(resp) * weights
    if return_resp:
        return resp, np.argmax(resp, 1)
    else:
        return np.argmax(resp, 1)


def nchoose2(x):
    return x * (x - 1) / 2.0


def ARI(true_label, predicted_label):
    tab = confusion_matrix(true_label, predicted_label)
    a = nchoose2(tab).sum()
    b = nchoose2(tab.sum(0)).sum() - a
    c = nchoose2(tab.sum(1)).sum() - a
    d = nchoose2(tab.sum()) - a - b - c
    ari = (a - (a + b) * (a + c) /
           (a + b + c + d)) / ((a + b + a + c) / 2 - (a + b) * (a + c) /
                               (a + b + c + d))
    return ari
