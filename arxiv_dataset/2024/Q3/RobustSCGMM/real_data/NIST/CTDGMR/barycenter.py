import numpy as np
from scipy import linalg
from .optGMR import GMR_opt_BFGS


def barycenter(
    means,
    covs,
    lambdas=None,
    tol=1e-7,
    mean_init=None,
    cov_init=None,
    ground_distance="W2",
):
    """Compute the barycenter of Gaussian measures.

    Parameters
    ----------
    means : array-like, shape (n, d)
    covs :  array-like, shape (n, d, d)
    lambdas : array-like, shape (n,), weight in barycenter
    ground_distance: string. Options: "W2", "KL", "WKL" ,"Cauchy-Schwartz", "ISE"

    Returns
    -------
    mean and covariance of the Gaussian Wasserstein barycenter.

    """
    m, d = means.shape
    if lambdas is None:
        lambdas = np.ones((m, )) / m
    else:
        lambdas = lambdas / lambdas.sum()
        # weight normalization

    if ground_distance == "KL" or ground_distance == "WKL":
        barycenter_mean = np.sum((lambdas * means.T).T, axis=0)
        barycenter_cov = np.sum(covs * lambdas.reshape((-1, 1, 1)), axis=0)
        diff = means - barycenter_mean
        barycenter_cov += np.dot(lambdas * diff.T, diff)

    elif ground_distance == "average":
        barycenter_mean = np.sum((lambdas * means.T).T, axis=0)
        barycenter_cov = np.sum(covs * lambdas.reshape((-1, 1, 1)), axis=0)
    elif ground_distance == "W2":
        barycenter_mean = np.sum((lambdas * means.T).T, axis=0)
        if d == 1:
            barycenter_cov = np.sum(
                np.sqrt(covs) * lambdas.reshape((-1, 1, 1)))**2
        else:
            # Fixed point iteration for Gaussian barycenter
            barycenter_cov = barycenter(means,
                                        covs,
                                        lambdas,
                                        ground_distance="KL")[1]
            barycenter_cov_next = np.identity(d)
            while np.linalg.norm(barycenter_cov_next - barycenter_cov,
                                 "fro") > tol:
                barycenter_cov = barycenter_cov_next
                sqrt_barycenter_cov = linalg.sqrtm(barycenter_cov)
                barycenter_cov_next = np.zeros((d, d))
                for k in range(m):
                    barycenter_cov_next = barycenter_cov_next + lambdas[
                        k] * linalg.sqrtm(sqrt_barycenter_cov @ covs[k]
                                          @ sqrt_barycenter_cov)
    elif ground_distance == "CS":
        # find the barycenter w.r.t. Cauchy-Schwartz divergence
        # using fixed point iteration
        def compute_sigma(covs, mus, cov, lambdas):
            # find (Sigma_r+Sigma)^{-1}
            covs = covs + cov
            for i, cov in enumerate(covs):
                covs[i] = np.linalg.inv(cov)

            # find (Sigma_r+Sigma)^{-1}(mu_r-mu)
            mu = compute_mean(covs, mus, cov, lambdas)
            mus = mus - mu
            weighted_mus = np.einsum("ijk,ik->ij", covs, mus)
            sandwich = np.einsum("ij,ik->ijk", weighted_mus, weighted_mus)
            return mu, 2 * (
                (covs - sandwich) * lambdas[:, np.newaxis, np.newaxis]).sum(0)

        def compute_mean(precisions, mus, cov, lambdas):
            # precisions are: (Sigma_r+Sigma)^{-1}
            # find sum_{r}lambda_r(Sigma_r+Sigma)^{-1}
            weighted_precisions = precisions * \
                lambdas[:, np.newaxis, np.newaxis]
            # find sum_{r}lambda_r(Sigma_r+Sigma)^{-1}mu_r
            weighted_mus = np.einsum("ijk,ik->ij", weighted_precisions, mus)
            weighted_mus = weighted_mus.sum(0)
            return np.linalg.solve(weighted_precisions.sum(0), weighted_mus)

        # initial value for fixed point iteration
        barycenter_mean, barycenter_cov = barycenter(means,
                                                     covs,
                                                     lambdas,
                                                     ground_distance="KL")
        barycenter_next = compute_sigma(covs, means, barycenter_cov, lambdas)
        barycenter_cov_next = np.linalg.inv(barycenter_next[1])
        n_iter = 0
        while np.linalg.norm(barycenter_cov_next - barycenter_cov,
                             "fro") > tol:
            n_iter += 1
            barycenter_cov = barycenter_cov_next
            barycenter_next = compute_sigma(covs, means, barycenter_cov,
                                            lambdas)
            barycenter_cov_next = np.linalg.inv(barycenter_next[1])
        barycenter_mean = barycenter_next[0]

    elif ground_distance == "ISE":
        # print(mean_init.shape, cov_init.shape)
        reduced_mix = GMR_opt_BFGS(
            means,
            covs,
            lambdas,
            1,
            loss="ISE",
            init_method="user",
            tol=tol,
            means_init=mean_init.reshape((-1, d)),
            covs_init=cov_init.reshape((-1, d, d)),
            weights_init=np.array([1.0]),
            random_state=0,
        )
        reduced_mix.run()

        barycenter_mean = np.squeeze(reduced_mix.reduced_means)
        barycenter_cov = np.squeeze(reduced_mix.reduced_covs)

    else:
        raise ValueError("This ground_distance %s is no implemented." %
                         ground_distance)

    return barycenter_mean, barycenter_cov


# sanity check
if __name__ == "__main__":
    d = 3
    means = np.random.randn(4, d)
    covs = np.empty((4, d, d))
    for i in range(4):
        a = np.random.randn(d, d)
        covs[i] = a @ a.T + 0.5 * np.eye(d)
        # print(np.linalg.eigvals(covs[i]))
    weights = np.ones(4) / 4

    barycenter_mean, barycenter_cov = barycenter(means,
                                                 covs,
                                                 weights,
                                                 ground_distance="KL")
    print(barycenter_mean, barycenter_cov)

    barycenter_mean, barycenter_cov = barycenter(means,
                                                 covs,
                                                 weights,
                                                 ground_distance="L2",
                                                 coeffs=np.array([1, 1]))
    print(barycenter_mean, barycenter_cov)
