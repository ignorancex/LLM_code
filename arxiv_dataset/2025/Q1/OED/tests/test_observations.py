from matplotlib import pyplot as plt
from numpy.testing import assert_allclose
from scipy.interpolate import interp1d

from tests.examples import *
from tests.helpers import align_eigenvectors


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_point_observation_measurements(prior, point_observation, fwd):
    """Test we can measure an observable correctly."""
    sample = prior.sample(return_coeffs=False)
    sample = fwd(sample.squeeze())
    measured = interp1d(prior.x, sample)(point_observation.measurements)
    err = np.abs(measured - point_observation(sample)).max()
    if err > 1e-3:
        plt.figure(figsize=(6, 3))
        plt.plot(point_observation.x, sample)
        plt.scatter(point_observation.measurements,
                    point_observation(sample).real,
                    color='r')
        plt.title(f'{prior.transform} max abs err {err:.4f}')
        plt.show()
    assert err < 1e-3


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_diagonal_observation_on_eigenvectors(diag_obs):
    for k, eigenvalue in enumerate(diag_obs.multiplier[:10]):
        calculated = diag_obs._matvec(diag_obs.eigenvector(k))
        calculated[k] = calculated[k] - eigenvalue
        assert_allclose(calculated, np.zeros_like(calculated), rtol=0, atol=1e-11)


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_diagonal_observation_eigenvectors(diag_obs):
    n = diag_obs.shape[0]
    k = int(diag_obs.N // np.pi)
    OstarO = diag_obs.OstarO
    assert_allclose(OstarO, OstarO.conjugate().T, rtol=0, atol=1e-12)

    D, P = np.linalg.eig(diag_obs.OstarO)
    xtra = P[:, n]
    P = P[:, :n].T # So that eigenvectors are in rows
    P = align_eigenvectors(P, k)

    eigs = np.vstack([diag_obs.eigenvector(i) for i in range(n)])
    eigs = align_eigenvectors(eigs, k)

    assert_allclose(np.linalg.norm(eigs, axis=1), 1)
    assert_allclose(np.linalg.norm(P, axis=1), 1)
    assert P.shape == eigs.shape

    errors = np.abs(P - eigs)
    err = np.max(errors)
    if err < 1e-5:
        assert True
    else:
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
        for i in range(n):
            color = COLORS[i]
            p = P[i, :]
            eig = eigs[i, :]
            real_diff = np.max(np.abs(p.real - eig.real))
            imag_diff = np.max(np.abs(p.imag - eig.imag))
            real_diff = f'{i} Error = {real_diff:.4f}'
            imag_diff = f'{i} Error = {imag_diff:.4f}'
            ax[0, 0].plot(diag_obs.x, p.real, color='r', label=i)
            ax[0, 1].plot(diag_obs.x, p.imag, color='r', label=i)
            ax[1, 0].plot(diag_obs.x, eig.real, color=color, linestyle='-', label=real_diff)
            ax[1, 0].plot(diag_obs.x, p.real, color=color, linestyle=':')
            ax[1, 1].plot(diag_obs.x, eig.imag, color=color, linestyle='-', label=imag_diff)
            ax[1, 1].plot(diag_obs.x, p.imag, color=color, linestyle=':')
        ax[0, 0].plot(diag_obs.x, xtra, label='$n+1$', color='k', alpha=0.2)
        ax[0, 1].plot(diag_obs.x, xtra, label='$n+1$', color='k', alpha=0.2)
        ax[1, 0].legend()
        ax[1, 1].legend()
        OstarO = "OstarO"  # r"$\mathcal{O}^{*}\mathcal{O}$"
        # ax[0].set_title(f"First $n={n}$ eigs of a diagonalizable " + OstarO)
        ax[0, 0].set_title(f"First $n+1={n + 1}$ eigs of " + OstarO + '  (real)')
        ax[0, 1].set_title(f"First $n+1={n + 1}$ eigs of " + OstarO + '  (imaginary)')
        ax[1, 0].set_title(f"First $n={n}$ modes of {diag_obs.transform} (real)")
        ax[1, 1].set_title(f"First $n={n}$ modes of {diag_obs.transform} (imaginary)")

        ax[1, 0].scatter(diag_obs.x[k], 0, color='k')
        ax[1, 1].scatter(diag_obs.x[k], 0, color='k')
        # ind = np.where(np.abs(D) > 1e-9)[0]
        # ax[3].plot(np.arange(n), D[:n], label="e_i") # "'$\mathbf{e}_i$')
        # ax[3].plot(np.arange(n), np.zeros(n), label='y=0')
        # ax[3].set_title(f"Nonzero eigenvalues: {ind}")

        plt.tight_layout()
        plt.show()
        assert False


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_diag_observation_singular_values(diag_obs):
    _, S, _ = np.linalg.svd(diag_obs.matrix)
    S = np.sort(S)
    singular_values = np.sort(diag_obs.singular_values())
    assert abs(np.sum(S) - np.sum(singular_values)) < 1e-9
    assert_allclose(S, singular_values, rtol=1e-7, atol=1e-12)


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_point_observation_eigenvalues(many_point_observation):
    m = many_point_observation.shape[0]
    eigenvalues = many_point_observation.eigenvalues()
    err = abs(np.sum(eigenvalues) - m)
    assert err < 1e-3, err
