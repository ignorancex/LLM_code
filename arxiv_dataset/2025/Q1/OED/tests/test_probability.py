import pandas as pd
from joblib import delayed, Parallel
from matplotlib import pyplot as plt
from numpy.testing import assert_allclose
from scipy.stats import gaussian_kde

from src.probability import Posterior
from tests.examples import *


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_prior_sample_from_coefficients(prior):
    sample, coefficients = prior.sample(return_coeffs=True)
    sample = sample.squeeze()
    sample_from_cefficients = prior.to_time_domain(coefficients.squeeze())
    assert_allclose(sample, sample_from_cefficients)


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_prior_sample_mean(short_prior):
    """Test that sampling the prior we get zero mean."""
    n_sample = 25000
    k = 12
    get_mean = lambda: short_prior.sample(return_coeffs=False, n_sample=n_sample).mean(axis=0)
    sample = sum(Parallel(n_jobs=7)(delayed(get_mean)() for _ in range(k))) / k
    assert sample.shape == (short_prior.N,)

    assert_allclose(sample, np.zeros_like(sample), rtol=0, atol=1e-2)


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_prior_sample_covariance(short_prior):
    """Test the empirical covariance is the same as the matrix representation of the prior.
    TODOs - check eigenvectors and eigenvalues agree???

    We compare the empiric covariance with the operator. The covariance operator C
    is defined for a Hilbert space H and a corresponding measure P as (see Wikipedia)

    C: H -> H , <Cx, y> = \int_H <x,z><y,z> dP(z).

    We think of H as as discretization of L2. H contains functions that
    are constant on every interval [nh, nh+h) and the inner product is

    <x,y> = \int_0^L x(u)y*(u) du = h sum_i=1^N x_iy_i*.

    Thus (\dot is the linear algebraic dot product):

    h sum (Cx)_iy_i* = <Cx, y> = h^2 \int_H x\dotz y \dot z dP(z).

    Taking x = e_i and y = e_j, we get that

    C_ij = h E[z_i z_j].

    On the left - the covariance operator. On the right - the empiric covariance. Note the h factor
    premultiplying the expectation."""
    n_sample = 200000 if short_prior.transform == 'dst' else 50000
    k = 60 if short_prior.transform == 'dst' else 40
    function = lambda: np.cov(short_prior.sample(n_sample=n_sample), rowvar=False)
    empiric_covariance = sum(Parallel(n_jobs=7)(delayed(function)() for _ in range(k))) / k

    assert empiric_covariance.shape == (short_prior.N, short_prior.N)
    assert_allclose(empiric_covariance, empiric_covariance.conjugate().T)

    # See details in the docs above concerning the factor h.
    assert_allclose(short_prior.matrix, empiric_covariance * short_prior.h, rtol=0, atol=1e-4)


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_prior_pointwise_std(prior):
    Sigma = np.einsum('i, ij->ij', prior.multiplier, prior.to_freq_domain(np.eye(prior.N), axis=0))
    Sigma = prior.to_time_domain(Sigma, axis=0)
    std = np.sqrt(np.diag(Sigma))
    plt.plot(std)
    plt.title(f'transform = {prior.transform}')
    plt.xlabel('x')
    plt.ylabel('prior pointwise standard deviation')
    plt.show()


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_posterior_point_utility_positive(posterior):
    assert posterior.point_utility(np.random.randn(5)**2) > 0

    
@pytest.mark.parametrize("transform", TRANSFORMS)
def test_posterior_diag_utility_positive(posterior):
    assert posterior.diag_utility(np.random.uniform(0,L, 8)) > 0


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_posterior(posterior, many_point):
    # meas = np.random.uniform(low=0, high=L, size=33)
    # meas = np.array([prior.L/2, prior.L/2.1])

    u0 = posterior.prior.sample(return_coeffs=False).squeeze()

    uT = posterior.fwd(u0)
    data = many_point(uT) + np.random.normal(scale=np.sqrt(posterior.sigSqr), size=many_point.shape[0])
    mean, pointwise_std = posterior.mean_std(many_point, data)
    
    top_bar, bottom_bar = mean + 2 * pointwise_std, mean - 2 * pointwise_std
    in_bars = np.mean((u0 < top_bar) & (u0 > bottom_bar))

    if in_bars < 0.95:
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6, 16), sharex=True)
        ax[0].plot(posterior.x, u0.real, label='IC')
        ax[0].plot(posterior.x, uT.real, label='FC')
        ax[0].scatter(many_point.measurements, np.dot(posterior.A, posterior.to_freq_domain(u0)).real,
                      label='Matrix FC')
        ax[0].scatter(many_point.measurements, data.real, label='Measurements', marker='*', s=10, color='r',
                      zorder=10)
        line, = ax[0].plot(posterior.x, mean, label='Posterior mean')
        ax[0].plot(posterior.x, top_bar, color=line.get_color(), label='95% Posterior Interval', linestyle=':')
        ax[0].plot(posterior.x, bottom_bar, color=line.get_color(), linestyle=':')
        ax[0].legend()

        ax[1].plot(posterior.x, pointwise_std, label='posterior STD')
        ax[1].scatter(many_point.measurements, np.ones(many_point.shape[0]) * pointwise_std.mean(),
                      label='measurement locations on x-axis')
        ax[1].legend()
        fig.suptitle(f"Transform = {posterior.transform}, in error bars = {in_bars:.4f}")

        # print(np.diag(post.Sigma)[:9])
        # tra = post.to_freq_domain(post.m)
        # plt.close()
        # plt.plot(tra)
        plt.tight_layout()
        plt.show()
        assert False


@pytest.mark.parametrize("transform", TRANSFORMS)
@pytest.mark.parametrize("m", [2, 6, 12])
def test_unique_optimal(posterior, m):
    n_jobs = 7
    res = Parallel(n_jobs=n_jobs)(delayed(posterior.optimize)(m=m) for _ in range(n_jobs))
    successes = [r for r in res if r['success']]
    failures = [r for r in res if not r['success']]
    utilities = [x['utility'] for x in successes]
    
    n_success = len(successes)
    n_failures = len(failures)
    if np.std(utilities) > 1e-2 * np.mean(utilities):
        plt.scatter(np.arange(n_success), [r['utility'] for r in successes], label='success', color='b')
        plt.scatter(np.arange(n_success, n_success + n_failures), [r['utility'] for r in failures], label='failures', color='r')
        plt.ylim(0, 2*np.mean([r['utility'] for r in res]))
        plt.title(f"test_unique_optimal transform={posterior.transform} m={m}. All y-values should equal")
        plt.xlabel("Attempt number")
        plt.ylabel("utility")
        plt.legend()
        plt.show()


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_optimal_n_iterations(posterior):
    posterior.optimize(m=3, n_iterations=5)


def test_default_posterior():
    post = Posterior()
    assert post.fwd is not None
    assert post.fwd is not None
