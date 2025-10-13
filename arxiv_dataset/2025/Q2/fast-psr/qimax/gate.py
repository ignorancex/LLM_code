import numpy as np
from . import tensor, utilities


class Gate:
    def __init__(self, name: str = "", U=None, indices=None, param: float = 0.0):
        # Only save U part, not including the control part
        # Check name is in dictionary
        # check if U is provided
        self.name = name
        if U is None:
            self.U = gate[name[1:] if name[0] == "C" else name]
        else:
            self.U = U
        # print(self.U)
        # Check if indices is int or list of int
        if indices is None:
            raise ValueError("Indices is not provided")
        if isinstance(indices, int):
            self.indices = [indices]
        else:
            self.indices = indices
        # only care about one-parameter gate
        self.param = param

    def __str__(self):
        return f"Name: {self.name}, U: {self.U}, Indices: {self.indices}, Param: {self.param}"


def gates_to_matrix(gates: list[Gate], num_qubits: int) -> np.ndarray:
    """Return the matrix representation of a circuit composed of a list of gates acting on `num_qubits` qubits.

    Args:
        gates (list[Gate]): List of gates in the circuit
        num_qubits (int): # of qubits in the system

    Returns:
        np.ndarray: Matrix representation of the circuit
    """

    # Init tensor form as : I \times I \times ... \times I (num_qubits times)
    tensor_form = [["I"] * num_qubits]
    params_form = [0] * num_qubits
    for gate in gates:
        if len(gate.param) > 0:
            if len(gate.indices) == 1:
                params_form[gate.indices[0]] = gate.param
            else:
                params_form[gate.indices[1]] = gate.param
        if len(gate.indices) == 1:
            tensor_form[0][gate.indices[0]] = gate.name
        else:

            tensor_form = utilities.duplicate_xss(tensor_form)
            tensor_form[0][gate.indices[0]] = "M"
            tensor_form[1][gate.indices[0]] = "P"
            tensor_form[1][gate.indices[1]] = gate.name
    left, right = np.array([1]), np.array([1])
    for i in range(0, len(tensor_form[0])):
        if len(tensor_form[0][i]) == 2:
            left = getattr(tensor, f"A{tensor_form[0][i]}")(left, params_form[i])
        else:
            left = getattr(tensor, f"A{tensor_form[0][i]}")(left)
        if len(tensor_form[1][i]) == 2:
            right = getattr(tensor, f"A{tensor_form[1][i]}")(left, params_form[i])
        else:
            right = getattr(tensor, f"A{tensor_form[1][i]}")(right)
    return left + right


def I(xs: np.ndarray):
    return xs


def CX(xs: np.ndarray):
    # only swap two last elements
    return np.array([xs[0], xs[1], xs[3], xs[2]])


def X(xs: np.ndarray):
    return np.array([xs[1], xs[0]])


def Y(xs: np.ndarray):
    return np.array([xs[1], -xs[0] * 1j])


def Z(xs: np.ndarray):
    return np.array([xs[0], -xs[1]])


def H(xs: np.ndarray):
    return np.array([xs[0] + xs[1], xs[0] - xs[1]]) / np.sqrt(2)


gate = {
    "I": np.array([[1, 0], [0, 1]], dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
    "RX": lambda theta: np.array(
        [
            [np.cos(theta / 2), -1j * np.sin(theta / 2)],
            [-1j * np.sin(theta / 2), np.cos(theta / 2)],
        ],
        dtype=np.complex128,
    ),
    "RY": lambda theta: np.array(
        [
            [np.cos(theta / 2), -np.sin(theta / 2)],
            [np.sin(theta / 2), np.cos(theta / 2)],
        ],
        dtype=np.complex128,
    ),
    "RZ": lambda theta: np.array(
        [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=np.complex128
    ),
    "H": np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2),
    "S": np.array([[1, 0], [0, 1j]], dtype=np.complex128),
    "II": np.array(
        [
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
        ],
        dtype=np.complex128,
    ),
    "IX": np.array(
        [
            [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
        ],
        dtype=np.complex128,
    ),
    "IY": np.array(
        [
            [0.0 + 0.0j, 0.0 - 1.0j, 0.0 + 0.0j, 0.0 - 0.0j],
            [0.0 + 1.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 - 0.0j, 0.0 + 0.0j, 0.0 - 1.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 1.0j, 0.0 + 0.0j],
        ],
        dtype=np.complex128,
    ),
    "IZ": np.array(
        [
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, -1.0 + 0.0j, 0.0 + 0.0j, -0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, -0.0 + 0.0j, 0.0 + 0.0j, -1.0 + 0.0j],
        ],
        dtype=np.complex128,
    ),
    "IH": np.array(
        [
            [0.70710678 + 0.0j, 0.70710678 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.70710678 + 0.0j, -0.70710678 + 0.0j, 0.0 + 0.0j, -0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.70710678 + 0.0j, 0.70710678 + 0.0j],
            [0.0 + 0.0j, -0.0 + 0.0j, 0.70710678 + 0.0j, -0.70710678 + 0.0j],
        ],
        dtype=np.complex128,
    ),
    "XI": np.array(
        [
            [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        ],
        dtype=np.complex128,
    ),
    "XX": np.array(
        [
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        ],
        dtype=np.complex128,
    ),
    "XY": np.array(
        [
            [0.0 + 0.0j, 0.0 - 0.0j, 0.0 + 0.0j, 0.0 - 1.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 1.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 - 1.0j, 0.0 + 0.0j, 0.0 - 0.0j],
            [0.0 + 1.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        ],
        dtype=np.complex128,
    ),
    "XZ": np.array(
        [
            [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, -0.0 + 0.0j, 0.0 + 0.0j, -1.0 + 0.0j],
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, -1.0 + 0.0j, 0.0 + 0.0j, -0.0 + 0.0j],
        ],
        dtype=np.complex128,
    ),
    "XH": np.array(
        [
            [0.0 + 0.0j, 0.0 + 0.0j, 0.70710678 + 0.0j, 0.70710678 + 0.0j],
            [0.0 + 0.0j, -0.0 + 0.0j, 0.70710678 + 0.0j, -0.70710678 + 0.0j],
            [0.70710678 + 0.0j, 0.70710678 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.70710678 + 0.0j, -0.70710678 + 0.0j, 0.0 + 0.0j, -0.0 + 0.0j],
        ],
        dtype=np.complex128,
    ),
    "YI": np.array(
        [
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 - 1.0j, 0.0 - 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 - 0.0j, 0.0 - 1.0j],
            [0.0 + 1.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 1.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        ],
        dtype=np.complex128,
    ),
    "YX": np.array(
        [
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 - 0.0j, 0.0 - 1.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 - 1.0j, 0.0 - 0.0j],
            [0.0 + 0.0j, 0.0 + 1.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 1.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        ],
        dtype=np.complex128,
    ),
    "YY": np.array(
        [
            [0.0 + 0.0j, 0.0 - 0.0j, 0.0 - 0.0j, -1.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 1.0 - 0.0j, 0.0 - 0.0j],
            [0.0 + 0.0j, 1.0 - 0.0j, 0.0 + 0.0j, 0.0 - 0.0j],
            [-1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        ],
        dtype=np.complex128,
    ),
    "YZ": np.array(
        [
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 - 1.0j, 0.0 - 0.0j],
            [0.0 + 0.0j, -0.0 + 0.0j, 0.0 - 0.0j, 0.0 + 1.0j],
            [0.0 + 1.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, -0.0 - 1.0j, 0.0 + 0.0j, -0.0 + 0.0j],
        ],
        dtype=np.complex128,
    ),
    "YH": np.array(
        [
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 - 0.70710678j, 0.0 - 0.70710678j],
            [0.0 + 0.0j, -0.0 + 0.0j, 0.0 - 0.70710678j, 0.0 + 0.70710678j],
            [0.0 + 0.70710678j, 0.0 + 0.70710678j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.70710678j, -0.0 - 0.70710678j, 0.0 + 0.0j, -0.0 + 0.0j],
        ],
        dtype=np.complex128,
    ),
    "ZI": np.array(
        [
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, -1.0 + 0.0j, -0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, -0.0 + 0.0j, -1.0 + 0.0j],
        ],
        dtype=np.complex128,
    ),
    "ZX": np.array(
        [
            [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, -0.0 + 0.0j, -1.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, -1.0 + 0.0j, -0.0 + 0.0j],
        ],
        dtype=np.complex128,
    ),
    "ZY": np.array(
        [
            [0.0 + 0.0j, 0.0 - 1.0j, 0.0 + 0.0j, 0.0 - 0.0j],
            [0.0 + 1.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 - 0.0j, -0.0 + 0.0j, 0.0 + 1.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, -0.0 - 1.0j, -0.0 + 0.0j],
        ],
        dtype=np.complex128,
    ),
    "ZZ": np.array(
        [
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, -1.0 + 0.0j, 0.0 + 0.0j, -0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, -1.0 + 0.0j, -0.0 + 0.0j],
            [0.0 + 0.0j, -0.0 + 0.0j, -0.0 + 0.0j, 1.0 - 0.0j],
        ],
        dtype=np.complex128,
    ),
    "ZH": np.array(
        [
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 - 0.70710678j, 0.0 - 0.70710678j],
            [0.0 + 0.0j, -0.0 + 0.0j, 0.0 - 0.70710678j, 0.0 + 0.70710678j],
            [0.0 + 0.70710678j, 0.0 + 0.70710678j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.70710678j, -0.0 - 0.70710678j, 0.0 + 0.0j, -0.0 + 0.0j],
        ],
        dtype=np.complex128,
    ),
    "CX": np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex128
    ),
    "SWAP": np.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.complex128
    ),
    "CZ": np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=np.complex128
    ),
    "CRX": lambda theta: np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, np.cos(theta / 2), -1j * np.sin(theta / 2)],
            [0, 0, -1j * np.sin(theta / 2), np.cos(theta / 2)],
        ],
        dtype=np.complex128,
    ),
    "CRY": lambda theta: np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, np.cos(theta / 2), -np.sin(theta / 2)],
            [0, 0, np.sin(theta / 2), np.cos(theta / 2)],
        ],
        dtype=np.complex128,
    ),
    "CRZ": lambda theta: np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, np.exp(-1j * theta / 2), 0],
            [0, 0, 0, np.exp(1j * theta / 2)],
        ],
        dtype=np.complex128,
    ),
}
