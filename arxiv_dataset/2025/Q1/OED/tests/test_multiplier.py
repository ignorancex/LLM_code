from matplotlib import pyplot as plt
from numpy.testing import assert_allclose
from scipy.sparse.linalg import eigsh, gmres, cg

from tests.examples import *
from tests.helpers import align_eigenvectors

COLORS = ['r', 'g', 'b', 'k', 'c', 'm', 'y']


@pytest.mark.parametrize('transform', TRANSFORMS)
def test_norms(multiplier):
    K = 20
    norms_block = multiplier.norms()
    norms_naive = np.array([multiplier.norm(multiplier.eigenfunction(k, False)(multiplier.x)) for k in range(K)])
    assert_allclose(norms_block[:K], norms_naive)

    
@pytest.mark.parametrize('transform', TRANSFORMS)
def test_block(multiplier):
    K = 200
    block = multiplier.block(multiplier.x)
    eigen = np.vstack([multiplier.eigenfunction(k, normalize=False)(multiplier.x) for k in range(K)])
    assert_allclose(block[:K,:], eigen, rtol=0, atol=1e-11)

    normalized_block = multiplier.normalized_block(multiplier.x)
    normalized_eigen = np.vstack([multiplier.eigenfunction(k, normalize=True)(multiplier.x) for k in range(K)])
    assert_allclose(normalized_block[:K,:], normalized_eigen, rtol=0, atol=1e-11)

    
@pytest.mark.parametrize('transform', TRANSFORMS)
def test_matvec(multiplier):
    """Test that if we input a constant vector, the output is also constant (???)"""
    v = np.ones(multiplier.N)
    matvec = multiplier(v)
    assert matvec.max() - matvec.min() < 1e-9


@pytest.mark.parametrize('transform', TRANSFORMS)
def test_matmat(multiplier):
    """Test matmat is the same as matvec on every column"""
    M = multiplier.normal(n_sample=10).T
    assert M.shape == (multiplier.N, 10)
    matmat = multiplier(M)
    matvec = np.vstack([multiplier(col) for col in M.T]).T
    assert_allclose(matmat, matvec)


@pytest.mark.parametrize('transform', TRANSFORMS)
def test_matrix_hermitian(multiplier):
    """ Test matrix representation is hermitian"""
    matrix = multiplier.matrix
    assert_allclose(matrix, matrix.conjugate().T)


@pytest.mark.parametrize('transform', TRANSFORMS)
def test_linearoperator(multiplier):
    """Test a multiplier functions as a proper LinearOperator object. We check
    we can solve a linear system with GMRES and conjugate gradients.
    Note that we create b to be in the range of the multiplier,
    otherwise this would not work."""
    b = sum([np.random.randn() * multiplier.eigenvector(i) for i in np.where(multiplier.multiplier)[0]])
    M = multiplier.matrix
    err = np.abs(M - M.conjugate().T).max()
    assert err < 1e-8
    x, _ = cg(multiplier, b, atol=0)
    b_hat = multiplier(x)
    assert_allclose(b, b_hat, rtol=0, atol=1e-7)

    x, _ = gmres(multiplier, b, atol=0)
    b_hat = multiplier(x)
    assert_allclose(b, b_hat, rtol=0, atol=1e-7)


# @pytest.mark.xfail(reason="Need to consider a proper inner product")
@pytest.mark.parametrize('transform', TRANSFORMS)
def test_orthogonality(multiplier):
    """We check the matrices of to_freq_domain and to_time_domain are orthogonal / unitary up to a constant.
    We also check they are each others' inverse.
    """
    h = multiplier.h
    identity = np.eye(multiplier.N)
    forward = multiplier.to_freq_domain(identity, axis=0)
    product = np.dot(forward, forward.conjugate().T) / h
    assert_allclose(product, identity, atol=1e-7, rtol=0)

    inverse = multiplier.to_time_domain(identity, axis=0)
    assert_allclose(np.dot(inverse, inverse.conjugate().T) * h, identity, atol=1e-7, rtol=0)

    assert_allclose(np.dot(inverse, forward), identity, atol=1e-7, rtol=0)
    assert_allclose(np.dot(forward, inverse), identity, atol=1e-7, rtol=0)

    vector = multiplier.normal().squeeze()
    assert_allclose(np.dot(forward, vector), multiplier.to_freq_domain(vector), rtol=0, atol=1e-9)
    assert_allclose(np.dot(inverse, vector), multiplier.to_time_domain(vector), rtol=0, atol=1e-9)


@pytest.mark.parametrize('k', range(12))
@pytest.mark.parametrize('transform', TRANSFORMS)
def test_eigenvector(multiplier, k):
    eigenvalue = multiplier.multiplier[k]
    v = multiplier.eigenvector(k)
    Mv = multiplier(v)
    err = np.max(np.abs(Mv - eigenvalue*v))
    assert err < 1e-12

    
@pytest.mark.parametrize('transform', TRANSFORMS)
def test_eig_on_matrix(multiplier):
    M = multiplier.matrix
    D, P = np.linalg.eig(M)
    assert_allclose(D, multiplier.multiplier, atol=1e-12)

    n = 6
    eigs = multiplier.normalized_block(multiplier.x)[:n, :]
    eigs = align_eigenvectors(eigs, 1)
    P = align_eigenvectors(P[:,:n].T, 1)
    assert P.shape == eigs.shape
    
    err = np.max(np.abs(eigs - P))
    if err > 1e-9:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 9))
        for i in range(n):
            color = COLORS[i]
            p = P[i, :]
            eig = eigs[i, :]
            real_diff = np.max(np.abs(p.real - eig.real))
            imag_diff = np.max(np.abs(p.imag - eig.imag))
            real_diff = f'{i} Error = {real_diff:.4f}'
            imag_diff = f'{i} Error = {imag_diff:.4f}'
            ax[0].plot(multiplier.x, eig.real, color=color, linestyle='-', label=real_diff)
            ax[0].plot(multiplier.x, p.real, color=color, linestyle=':')
            ax[1].plot(multiplier.x, eig.imag, color=color, linestyle='-', label=imag_diff)
            ax[1].plot(multiplier.x, p.imag, color=color, linestyle=':')
        ax[0].legend()
        ax[1].legend()
        ax[0].set_title(f"First $n={n}$ modes of {multiplier.transform} (real)")
        ax[1].set_title(f"First $n={n}$ modes of {multiplier.transform} (imaginary)")
        plt.tight_layout()
        plt.show()
    
    assert err < 1e-9
                
    
@pytest.mark.parametrize('transform', TRANSFORMS)
def test_eigenvectors_agree(multiplier):
    """We check that the eigenvectors we get from transforming the standard basis
    vectors are the same we get from the multiplier object iteslf."""
    class_eigenfunction = multiplier.normalized_block(multiplier.x)
    eigenfunction = multiplier.to_time_domain(np.eye(multiplier.N), axis=1)
    diff = np.abs(eigenfunction - class_eigenfunction)
    assert_allclose(diff, 0, atol=1e-11, rtol=0)


@pytest.mark.parametrize('transform', TRANSFORMS)
def test_basis_matrix_agree(multiplier):
    identity = np.eye(multiplier.N)
    forward = multiplier.to_time_domain(identity, axis=1)
    U = multiplier.normalized_block(multiplier.x)
    assert_allclose(U, forward, atol=1e-9, rtol=0)


@pytest.mark.parametrize('transform', TRANSFORMS)
def test_eigen_norm(multiplier):
    """ Verify norm of eigenfunctions is 1 in the L2 sense and norm
    of eigenvector is 1 in the standaed linear algebraic sense"""
    for i in range(multiplier.N):
        # eigenvector = multiplier.eigenvector(i)
        # assert not np.any(np.isnan(eigenvector))
        # assert abs(np.linalg.norm(eigenvector) - 1) < 1e-9

        eigenfunction = multiplier.eigenfunction(i)(multiplier.x)
        assert abs(multiplier.norm(eigenfunction) - 1) < 1e-9


@pytest.mark.parametrize('transform', TRANSFORMS)
def test_eigenfunction(multiplier):
    """Test that eigenvectors we get from viewing multiplier as a LinearOperator
    (via the eigsh function) agree with the eigenvectors of the Multiplier object
    we define."""

    number_eigenvectors = 4
    D, P = eigsh(multiplier, which='LM', k=number_eigenvectors)
    assert P.shape == (multiplier.N, number_eigenvectors)
    P = P[:, :number_eigenvectors].T
    P = align_eigenvectors(P)

    eigs = multiplier.normalized_block(multiplier.x)
    eigs = eigs[:number_eigenvectors, :]
    eigs = align_eigenvectors(eigs)

    assert P.shape == eigs.shape
    assert_allclose(np.linalg.norm(eigs, axis=1), 1)
    assert_allclose(np.linalg.norm(P, axis=1), 1)

    errors = np.abs(P - eigs)
    err = np.max(errors)
    if err < 1e-12:
        assert True
    else:
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 7))
        for i in range(number_eigenvectors):
            color = COLORS[i]
            p = P[i, :]
            eig = eigs[i, :]
            real_diff = np.max(np.abs(p.real-eig.real))
            imag_diff = np.max(np.abs(p.imag-eig.imag))
            real_diff = f'{i} {real_diff:.4f}'
            imag_diff = f'{i} {imag_diff:.4f}'
            ax[0].plot(multiplier.x, eig.real, color=color, linestyle='-', label=real_diff)
            ax[0].plot(multiplier.x, p.real, color=color, linestyle=':')
            ax[1].plot(multiplier.x, eig.imag, color=color, linestyle='-', label=imag_diff)
            ax[1].plot(multiplier.x, p.imag, color=color, linestyle=':')
        ax[0].legend()
        ax[1].legend()
        ax[0].set_title(f"First $n={number_eigenvectors}$ real modes of {multiplier.transform}")
        ax[1].set_title(f"First $n={number_eigenvectors}$ imaginary modes of {multiplier.transform}")

        # ind = np.where(np.abs(D) > 1e-9)[0]
        # ax[3].plot(np.arange(n), D[:n], label="e_i") # "'$\mathbf{e}_i$')
        # ax[3].plot(np.arange(n), np.zeros(n), label='y=0')
        # ax[3].set_title(f"Nonzero eigenvalues: {ind}")

        plt.tight_layout()
        plt.show()
        assert False


@pytest.mark.parametrize('transform', TRANSFORMS)
def test_transformed_eigenvector_is_standard_basis_vector(multiplier):
    """Test that if we transform the kth eigenvector, we get the corresponding
    standard unit basis vector."""

    fig, axes = plt.subplots(nrows=1, ncols=2)
    success = True
    for k in range(5):
        eigenvector = multiplier.eigenvector(k)
        transformed = multiplier.to_freq_domain(eigenvector)
        if np.linalg.norm(transformed - np.eye(1, multiplier.N, k)) > 1e-12:
            success = False
            axes[0].plot(multiplier.x, eigenvector, label=k)
            axes[1].plot(transformed, label=k)
    if success:
        assert True
    else:
        plt.legend()
        plt.show()
        assert False


@pytest.mark.parametrize('transform', TRANSFORMS)
def test_eigenvector_and_eigenfunction_agree_to_normalization(transform):
    """Test that eigenvectors and eigenfunctions agree up to normalization."""
    multiplier = FourierMultiplier(N=100, L=1.53, transform=transform)
    for i in range(20):
        ef = multiplier.eigenfunction(i)(multiplier.x)
        ev = multiplier.eigenvector(i)
        assert_allclose(ev, ef, rtol=0, atol=1e-9)


@pytest.mark.parametrize('transform', TRANSFORMS)
def test_to_freq_from_right(transform):
    multiplier = FourierMultiplier(N=100, L=1.53, transform=transform)
    identity = np.eye(multiplier.N)
    matrix = multiplier.to_freq_domain(identity, axis=0)

    rand = multiplier.normal(n_sample=multiplier.N//2)
    assert rand.shape == (multiplier.N//2, multiplier.N)
    
    rand_matrix = rand @ matrix
    operator = multiplier.to_freq_domain_from_right(rand)
    assert_allclose(rand_matrix, operator)
