import numpy as np

def P(i, n = 2):
    result = np.zeros((n,n))
    result[i // n, i % n] = 1
    return result

def state0(num_qubits):
    state = np.zeros(2**num_qubits)
    state[0] = 1
    return state

r = 1/2
epsilon = np.pi/2

constant_gate = {
    "I": np.array([[1, 0], [0, 1]], dtype=np.complex128),
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
    "CX": np.array([[1, 0,0,0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex128),
    "S": np.array([[1, 0], [0, 1j]], dtype=np.complex128),
}
