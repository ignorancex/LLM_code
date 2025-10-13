import qimax.constant as constant
import qimax.gate as gate
import numpy as np
import tensor

def cx(num_qubits: int, control: int, target: int) -> np.ndarray:
    """Return the matrix representation of a general controlled-X gate acting on `num_qubits` qubits.

    Args:
        num_qubits (int): # of qubits in the system
        control (int): index of the control qubit
        target (int): index of the target qubit

    Raises:
        ValueError: Control and target qubits must be different
        ValueError: Control qubit index out of range

    Returns:
        np.ndarray: Matrix representation of the controlled-X gate
    """
    if control == target:
        raise ValueError('Control and target qubits must be different')
    if control < 0 or control >= num_qubits:
        raise ValueError('Control qubit index out of range')
    if control == 0:
        cx_left = constant.outer00
        cx_right = constant.outer11
    else:
        cx_left = gate.gate['I']
        cx_right = gate.gate['I']
    for i in range(1, num_qubits + 1):
        if i == control:
            cx_left = np.kron(cx_left, constant.outer00)
            cx_right = np.kron(cx_right, constant.outer11)
        elif i == target:
            cx_left = tensor.AI(cx_left, 2)
            cx_right = tensor.AX(cx_right)
        else:
            cx_left = tensor.AI(cx_left, 2)
            cx_right = tensor.AI(cx_right, 2)
        
    return cx_left + cx_right
