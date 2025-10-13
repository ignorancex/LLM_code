from qimax import gate, constant
import numpy as np
# def test_gate_to_tensor():
#     cx13 = gate.Gate('X', indices = [1, 3])
#     ry2 = gate.Gate('RY', U = gate.gate['RY'], indices = 2, param = np.pi)
#     test_result = gate.gates_to_matrix([cx13, ry2], 5)
#     I = gate.gate['I']
#     X = gate.gate['X']
#     true_result = np.kron(I, np.kron(constant.P(0), np.kron(gate.gate['RY'](np.pi), np.kron(I, I)))) + np.kron(I, np.kron(constant.P(3), np.kron(I, np.kron(X, I))))
#     assert np.allclose(true_result, test_result)