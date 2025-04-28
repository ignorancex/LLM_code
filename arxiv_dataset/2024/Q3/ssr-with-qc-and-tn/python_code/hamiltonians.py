from type_aliases import *
from laminate import Laminate

import numpy as np
from qiskit.quantum_info import SparsePauliOp


def identity(num_qb: int, coeff: float = 1.) -> SparsePauliOp:
    """Generate the identity operator on `num_qb`

    Args:
        num_qb (int): number of qubits
        coeff (:obj:`float`):  coefficient with which the
            identity operator is multiplied. Defaults to `1.`

    Returns:
        SparsePauliOp: The identity operator

    """
    return SparsePauliOp.from_list([('I' * num_qb, coeff)])


def generate_loss_hamiltonian(laminate: Laminate, target: Parameters, encoding: Encoding,
                              dyad_ops: DyadOps) -> SparsePauliOp:
    """Generate the Hamiltonian corresponding to the loss function

    Args:
        laminate (Laminate): the specified laminate
        target (Parameters): the target lamination parameters
        encoding (Encoding): the encoding of ply-angle indices to qubits
        dyad_ops (DyadOps): A list[SparsePauliOp] where `dyad_ops[angle_idx]` is the outer
            product of the state corresponding to the encoding `encoding[angle_idx]`

    Returns:
        SparsePauliOp: The resulting loss-function Hamiltonian
    """
    qb_per_qd = len(encoding[0])
    num_qubits = qb_per_qd * laminate.num_plies

    individual_ops = []

    for w_idx, weights in enumerate(laminate.weights):
        for f_idx, funcs in enumerate(laminate.funcs.transpose()):
            func_op = sum(f * dyop for f, dyop in zip(funcs, dyad_ops)).simplify()
            # if w_idx == 0:
            #     print(func_op)
            # operator for lamination parameters
            lp_op = sum(
                w * (identity(n * qb_per_qd) ^ func_op ^ identity((laminate.num_plies - n - 1) * qb_per_qd))
                for n, w in enumerate(weights)
            ).simplify()

            # loss with target
            individual_ops.append(((lp_op - identity(num_qubits, target[w_idx, f_idx])) ** 2).simplify())

    # return
    return sum(individual_ops).simplify()


def generate_penalty_hamiltonian(
    laminate: Laminate, constraint_matrix: NDArray[bool | int], encoding: Encoding, dyad_ops: DyadOps, strength: float = 1.
) -> SparsePauliOp:
    """Generate the penalty Hamiltonian for the disorientation constraint or other nearest-neighbor constraints

    Args:
        laminate (Laminate): the specified laminate
        constraint_matrix (NDArray[bool | int]): A numpy array of shape `(num_angles,num_angles)` where
            `constraint_matrix[s1,s2]` is `True` or `1` if the constraint violated for ply-angle indices `s1` and `s2`
            on neighboring plies, and `False` or `0` if the constraint is satisfied.
        encoding (Encoding): the encoding of ply-angle indices to qubits
        dyad_ops (DyadOps): A list[SparsePauliOp] where `dyad_ops[angle_idx]` is the outer
            product of the state corresponding to the encoding `encoding[angle_idx]`
        strength (:obj:`float`, optional): pactor to scale the weight of the penalty. Defaults to `1.`

    Returns:
        SparsePauliOp: The resulting penalty Hamiltonian

    """
    index_pairs = list(zip(*np.where(constraint_matrix)))
    qb_per_qd = len(encoding[0])
    return strength * sum(
        sum(
            identity(
                n * qb_per_qd
            ) ^ dyad_ops[s1] ^ dyad_ops[s2] ^ identity(
                (laminate.num_plies - n - 2) * qb_per_qd
            ) for s1,s2 in index_pairs
        ) for n in range(laminate.num_plies - 1)
    ).simplify()