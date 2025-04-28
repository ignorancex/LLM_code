# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

# The following function names: eye, identity, create, destroy, number, basis, projector, charge, position,
# momentum, sigma, are essential quantum operations with definitions that follow from the literature.
# While, for user readability and adaptability, we have used the same namespace as QuTip with these
# functions, the code within is distinct in all cases, save when the operation is so simple and
# universally defined mathematically that alternative syntax is impractical. 

from qcheff import qcheff_config


def eye(
    n: int,
    dtype: type | None = None,
    sparse: bool | None = None,
):
    """

    Identity operator in sparse/dense format

    Parameters:
    -----------

    n: int
        dimension of the Hilbert space

    dtype: Type | None
        Data type of the operator.
        If None, use the default data type.

    sparse: bool | None
        If True, return a sparse operator. If False, return a dense operator.
        If None, return the default format.

    """

    if dtype is None:
        dtype = qcheff_config.default_dtype

    if sparse is None:
        sparse = qcheff_config.sparse

    sparse_eye = qcheff_config._device_scipy_backend.sparse.identity(
        n, dtype=dtype, format="csr"
    )
    if sparse:
        return sparse_eye
    else:
        return sparse_eye.toarray()


def create(
    n: int,
    dtype: type | None = None,
    sparse: bool | None = None,
):
    """
    Creation operator in sparse/dense format.

    Parameters
    ----------
    n : int
        Dimension of the Hilbert space.
    dtype : Type | None
        Data type of the operator. If None, use the default data type.
    sparse : bool | None
        If True, return a sparse operator. If False, return a dense operator.
        If None, return the default format.

    Returns
    -------
    OperatorMatrix
        The creation operator.
    """

    if dtype is None:
        dtype = qcheff_config.default_dtype

    if sparse is None:
        sparse = qcheff_config.sparse

    create_diags = qcheff_config.device_xp_backend.sqrt(
        qcheff_config.device_xp_backend.arange(1, n, dtype=dtype)
    )
    sparse_create = qcheff_config._device_scipy_backend.sparse.diags(
        create_diags, offsets=-1, shape=(n, n), format="csr"
    )
    if sparse:
        return sparse_create
    else:
        return sparse_create.toarray()


def destroy(
    n: int,
    dtype: type | None = None,
    sparse: bool | None = None,
):
    """
    Destruction operator in sparse/dense format.

    Parameters
    ----------
    n : int
        Dimension of the Hilbert space.
    dtype : Type | None
        Data type of the operator. If None, use the default data type.
    sparse : bool | None
        If True, return a sparse operator. If False, return a dense operator.
        If None, return the default format.

    Returns
    -------
    OperatorMatrix
        The destruction operator.
    """

    if dtype is None:
        dtype = qcheff_config.default_dtype

    if sparse is None:
        sparse = qcheff_config.sparse

    destroy_diags = qcheff_config.device_xp_backend.sqrt(
        qcheff_config.device_xp_backend.arange(
            1,
            n,
            dtype=dtype,
        )
    )
    sparse_destroy = qcheff_config._device_scipy_backend.sparse.diags(
        destroy_diags, offsets=1, shape=(n, n), format="csr"
    )
    if sparse:
        return sparse_destroy
    else:
        return sparse_destroy.toarray()


def number(
    n: int,
    dtype: type | None = None,
    sparse: bool | None = None,
):
    """
    Number operator in sparse/dense format

    Parameters
    ----------
    n : int
        Dimension of the Hilbert space.
    dtype : Type | None
        Data type of the operator. If None, use the default data type.
    sparse : bool | None
        If True, return a sparse operator. If False, return a dense operator.
        If None, return the default format.

    Returns
    -------
    OperatorMatrix
        The number operator.
    """

    if dtype is None:
        dtype = qcheff_config.default_dtype

    if sparse is None:
        sparse = qcheff_config.sparse

    sparse_number = qcheff_config._device_scipy_backend.sparse.diags(
        qcheff_config.device_xp_backend.arange(n),
        shape=(n, n),
        format="csr",
        dtype=dtype,
    )
    if sparse:
        return sparse_number
    else:
        return sparse_number.toarray()


def identity(
    n: int,
    dtype: type | None = None,
    sparse: bool | None = None,
):
    """
    Identity operator in sparse/dense format.

    Parameters
    ----------
    n : int
        Dimension of the Hilbert space.
    dtype : Type | None
        Data type of the operator. If None, use the default data type.
    sparse : bool | None
        If True, return a sparse operator. If False, return a dense operator.
        If None, return the default format.

    Returns
    -------
    OperatorMatrix
        The identity operator.
    """

    if dtype is None:
        dtype = qcheff_config.default_dtype

    if sparse is None:
        sparse = qcheff_config.sparse

    sparse_identity = qcheff_config._device_scipy_backend.sparse.identity(
        n, dtype=dtype, format="csr"
    )
    if sparse:
        return sparse_identity
    else:
        return sparse_identity.toarray()


def basis(
    n: int,
    i: int,
    dtype: type | None = None,
    sparse: bool | None = None,
):
    """
    Basis vector in sparse/dense format.

    Parameters
    ----------
    n : int
        Dimension of the Hilbert space.
    i : int
        Index of the basis.
    dtype : Type | None
        Data type of the operator.
        If None, use the default data type.
    sparse : bool | None
        If True, return a sparse operator. If False, return a dense operator.
        If None, return the default format.

    Returns
    -------
    OperatorMatrix
        The basis vector.
    """

    if dtype is None:
        dtype = qcheff_config.default_dtype

    if sparse is None:
        sparse = qcheff_config.sparse

    basis_vector = qcheff_config.device_xp_backend.zeros(n, dtype=dtype)
    basis_vector[i] = 1

    sparse_basis = qcheff_config._device_scipy_backend.sparse.csr_matrix(
        basis_vector.reshape(-1, 1), dtype=dtype
    )
    if sparse:
        return sparse_basis
    else:
        return sparse_basis.toarray()


def projector(
    n: int,
    i: int,
    j: int,
    dtype: type | None = None,
    sparse: bool | None = None,
):
    """
    Projector operator in sparse/dense format

    Parameters
    ----------
    n : int
        dimension of the Hilbert space
    i : int
        Index of the basis
    j : int
        Index of the basis
    dtype : Type | None, optional
        Data type of the operator.
        If None, use the default data type. By default None
    sparse : bool | None, optional
        If True, return a sparse operator. If False, return a dense operator.
        If None, return the default format. By default None
    """

    return (
        basis(n, i, dtype=dtype, sparse=sparse)
        @ basis(n, j, dtype=dtype, sparse=sparse).T.conj()
    )


def charge(
    n: int,
    dtype: type | None = None,
    sparse: bool | None = None,
):
    """
    Charge operator in sparse/dense format

    Parameters
    ----------
    n : int
        Dimension of the Hilbert space
    dtype : Type | None, optional
        Data type of the operator. If None, use the default data type.
    sparse : bool | None, optional
        If True, return a sparse operator. If False, return a dense operator.
        If None, return the default format.

    Returns
    -------
    OperatorMatrix
        The charge operator.

    """

    if dtype is None:
        dtype = qcheff_config.default_dtype

    if sparse is None:
        sparse = qcheff_config.sparse

    xp = qcheff_config.device_xp_backend
    xpsparse = qcheff_config._device_scipy_backend.sparse

    sparse_charge = xpsparse.diags(
        xp.arange(-n, n + 1), shape=(2 * n + 1, 2 * n + 1), format="csr", dtype=dtype
    )
    if sparse:
        return sparse_charge
    else:
        return sparse_charge.toarray()


def position(
    n: int,
    dtype: type | None = None,
    sparse: bool | None = None,
):
    """
    Position operator in sparse/dense format

    Parameters
    ----------
    n : int
        Dimension of the Hilbert space
    dtype : Type | None, optional
        Data type of the operator. If None, use the default data type.
    sparse : bool | None, optional
        If True, return a sparse operator. If False, return a dense operator.
        If None, return the default format.

    Returns
    -------
    OperatorMatrix
        The position operator
    """

    if dtype is None:
        dtype = qcheff_config.default_dtype

    if sparse is None:
        sparse = qcheff_config.sparse

    sparse_position = (
        create(n, dtype=dtype, sparse=True) + destroy(n, dtype=dtype, sparse=True)
    ) / qcheff_config.device_xp_backend.sqrt(2)

    if sparse:
        return sparse_position
    else:
        return sparse_position.toarray()


def momentum(
    n: int,
    dtype: type | None = None,
    sparse: bool | None = None,
):
    """
    Momentum operator in sparse/dense format

    Parameters
    ----------
    n : int
        Dimension of the Hilbert space
    dtype : Type | None, optional
        Data type of the operator. If None, use the default data type.
    sparse : bool | None, optional
        If True, return a sparse operator. If False, return a dense operator.
        If None, return the default format.

    Returns
    -------
    OperatorMatrix
        The momentum operator
    """

    if dtype is None:
        dtype = qcheff_config.default_dtype

    if sparse is None:
        sparse = qcheff_config.sparse

    sparse_momentum = (
        1j
        * (create(n, dtype=dtype, sparse=True) - destroy(n, dtype=dtype, sparse=True))
        / qcheff_config.device_xp_backend.sqrt(2)
    )

    if sparse:
        return sparse_momentum
    else:
        return sparse_momentum.toarray()


def sigmax(
    dtype: type | None = None,
    sparse: bool | None = None,
):
    """
    Pauli-X operator in sparse/dense format

    Parameters
    ----------
    dtype : Type | None
        Data type of the operator.
        If None, use the default data type.
    sparse : bool | None
        If True, return a sparse operator. If False, return a dense operator.
        If None, return the default format.

    Returns
    -------
    OperatorMatrix
        The Pauli-X operator
    """

    if dtype is None:
        dtype = qcheff_config.default_dtype

    if sparse is None:
        sparse = qcheff_config.sparse

    xpsparse = qcheff_config._device_scipy_backend.sparse

    sparse_sigmax = xpsparse.diags(
        [1, 1],
        offsets=[1, -1],
        shape=(2, 2),
        format="csr",
        dtype=dtype,
    )
    if sparse:
        return sparse_sigmax
    else:
        return sparse_sigmax.toarray()


def sigmay(
    dtype: type | None = None,
    sparse: bool | None = None,
):
    """
    Pauli-Y operator in sparse/dense format.

    Parameters
    ----------
    dtype : Type | None
        Data type of the operator.
        If None, use the default data type.
    sparse : bool | None
        If True, return a sparse operator. If False, return a dense operator.
        If None, return the default format.

    Returns
    -------
    OperatorMatrix
        The Pauli-Y operator
    """

    if dtype is None:
        dtype = qcheff_config.default_dtype

    if sparse is None:
        sparse = qcheff_config.sparse

    xpsparse = qcheff_config._device_scipy_backend.sparse

    sparse_sigmay = xpsparse.diags(
        [-1j, 1j],
        offsets=[1, -1],
        shape=(2, 2),
        format="csr",
        dtype=dtype,
    )
    if sparse:
        return sparse_sigmay
    else:
        return sparse_sigmay.toarray()


def sigmaz(
    dtype: type | None = None,
    sparse: bool | None = None,
):
    """

    Pauli-Z operator in sparse/dense format

    Parameters:
    -----------

    dtype: Type | None
        Data type of the operator.
        If None, use the default data type.

    sparse: bool | None
        If True, return a sparse operator. If False, return a dense operator.
        If None, return the default format.

    """

    if dtype is None:
        dtype = qcheff_config.default_dtype

    if sparse is None:
        sparse = qcheff_config.sparse

    xpsparse = qcheff_config._device_scipy_backend.sparse

    sparse_sigmaz = xpsparse.diags(
        [1, -1],
        shape=(2, 2),
        format="csr",
        dtype=dtype,
    )
    if sparse:
        return sparse_sigmaz
    else:
        return sparse_sigmaz.toarray()


def sigmap(
    dtype: type | None = None,
    sparse: bool | None = None,
):
    """
    Raising operator in sparse/dense format

    Parameters
    ----------
    dtype : Type | None
        Data type of the operator.
        If None, use the default data type.
    sparse : bool | None
        If True, return a sparse operator. If False, return a dense operator.
        If None, return the default format.

    Returns
    -------
    OperatorMatrix
        The raising operator.
    """

    if dtype is None:
        dtype = qcheff_config.default_dtype

    if sparse is None:
        sparse = qcheff_config.sparse

    xpsparse = qcheff_config._device_scipy_backend.sparse

    sparse_sigmap = xpsparse.diags(
        [1],
        offsets=[1],
        shape=(2, 2),
        format="csr",
        dtype=dtype,
    )
    if sparse:
        return sparse_sigmap
    else:
        return sparse_sigmap.toarray()


def sigmam(
    dtype: type | None = None,
    sparse: bool | None = None,
):
    """
    Lowering operator in sparse/dense format.

    Parameters
    ----------
    dtype : Type | None
        Data type of the operator.
        If None, use the default data type.
    sparse : bool | None
        If True, return a sparse operator. If False, return a dense operator.
        If None, return the default format.

    Returns
    -------
    OperatorMatrix
        The lowering operator.
    """

    if dtype is None:
        dtype = qcheff_config.default_dtype

    if sparse is None:
        sparse = qcheff_config.sparse

    xpsparse = qcheff_config._device_scipy_backend.sparse

    sparse_sigmam = xpsparse.diags(
        [1],
        offsets=[-1],
        shape=(2, 2),
        format="csr",
        dtype=dtype,
    )
    if sparse:
        return sparse_sigmam
    else:
        return sparse_sigmam.toarray()
