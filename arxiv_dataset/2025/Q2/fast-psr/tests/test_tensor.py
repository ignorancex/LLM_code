import pytest
import sys
sys.path.insert(0, '..')
from qimax import tensor, gate, constant
import numpy as np

def test_P0A():
    A = np.random.rand(10, 10)
    true_result = np.kron(constant.P(0), A)
    test_result = tensor.MA(A)
    assert np.allclose(true_result, test_result)
    
def test_AP0():
    A = np.random.rand(10, 10)
    true_result = np.kron(A, constant.P(0))
    test_result = tensor.AM(A)
    assert np.allclose(true_result, test_result)
    
def test_P3A():
    A = np.random.rand(10, 10)
    true_result = np.kron(constant.P(3), A)
    test_result = tensor.PA(A)
    assert np.allclose(true_result, test_result)
    
def test_AP3():
    A = np.random.rand(10, 10)
    true_result = np.kron(A, constant.P(3))
    test_result = tensor.AP(A)
    assert np.allclose(true_result, test_result)
    
def test_IA():
    A = np.random.rand(10, 10)
    k = np.random.randint(1, 10)
    true_result = np.kron(np.eye(k), A)
    test_result = tensor.IA(A, k)
    assert np.allclose(true_result, test_result)
    
def testAI():
    A = np.random.rand(10, 10)
    k = np.random.randint(1, 10)
    true_result = np.kron(A, np.eye(k))
    test_result = tensor.AI(A, k)
    assert np.allclose(true_result, test_result)
    
def test_XA():
    A = np.random.rand(10, 10)
    true_result = np.kron(gate.gate['X'], A)
    test_result = tensor.XA(A)
    assert np.allclose(true_result, test_result)

def testAX():
    A = np.random.rand(10, 10)
    true_result = np.kron(A, gate.gate['X'])
    test_result = tensor.AX(A)
    assert np.allclose(true_result, test_result)
    
def test_ZA():
    A = np.random.rand(10, 10)
    true_result = np.kron(gate.gate['Z'], A)
    test_result = tensor.ZA(A)
    assert np.allclose(true_result, test_result)
    
def testAZ():
    A = np.random.rand(10, 10)
    true_result = np.kron(A, gate.gate['Z'])
    test_result = tensor.AZ(A)
    assert np.allclose(true_result, test_result)
    
def testAH():
    A = np.random.rand(10, 10)
    true_result = np.kron(A, gate.gate['H'])
    test_result = tensor.AH(A)
    assert np.allclose(true_result, test_result)

def testHA():
    A = np.random.rand(10, 10)
    true_result = np.kron(gate.gate['H'], A)
    test_result = tensor.HA(A)
    assert np.allclose(true_result, test_result)
    
def testRXA():
    A = np.random.rand(10, 10)
    theta = np.random.uniform(0, 2*np.pi)
    true_result = np.kron(gate.gate['RX'](theta), A)
    test_result = tensor.RXA(A, theta)
    assert np.allclose(true_result, test_result)
    
def testARX():
    A = np.random.rand(10, 10)
    theta = np.random.uniform(0, 2*np.pi)
    true_result = np.kron(A, gate.gate['RX'](theta))
    test_result = tensor.ARX(A, theta)
    assert np.allclose(true_result, test_result)
    
def testRYA():
    A = np.random.rand(10, 10)
    theta = np.random.uniform(0, 2*np.pi)
    true_result = np.kron(gate.gate['RY'](theta), A)
    test_result = tensor.RYA(A, theta)
    assert np.allclose(true_result, test_result)

def testARY():
    A = np.random.rand(10, 10)
    theta = np.random.uniform(0, 2*np.pi)
    true_result = np.kron(A, gate.gate['RY'](theta))
    test_result = tensor.ARY(A, theta)
    assert np.allclose(true_result, test_result)
    
def testRZA():
    A = np.random.rand(10, 10)
    theta = np.random.uniform(0, 2*np.pi)
    true_result = np.kron(gate.gate['RZ'](theta), A)
    test_result = tensor.RZA(A, theta)
    assert np.allclose(true_result, test_result)
    
def testARZ():
    A = np.random.rand(10, 10)
    theta = np.random.uniform(0, 2*np.pi)
    true_result = np.kron(A, gate.gate['RZ'](theta))
    test_result = tensor.ARZ(A, theta)
    assert np.allclose(true_result, test_result)
    
# def testACRX():
#     A = np.random.rand(10, 10)
#     theta = np.random.uniform(0, 2*np.pi)
#     true_result = np.kron(A.conj(), gate.gate['CRX'](theta))
#     test_result = tensor.ACRX(A, theta)
#     assert np.allclose(true_result, test_result)
    
# def testACRY():
#     A = np.random.rand(10, 10)
#     theta = np.random.uniform(0, 2*np.pi)
#     true_result = np.kron(A.conj(), gate.gate['CRY'](theta))
#     test_result = tensor.ACRY(A, theta)
#     assert np.allclose(true_result, test_result)
    
# def testACRZ():
#     A = np.random.rand(10, 10)
#     theta = np.random.uniform(0, 2*np.pi)
#     true_result = np.kron(A.conj(), gate.gate['CRZ'](theta))
#     test_result = tensor.ACRZ(A, theta)
#     assert np.allclose(true_result, test_result)
    
# def testCRXA():
#     A = np.random.rand(10, 10)
#     theta = np.random.uniform(0, 2*np.pi)
#     true_result = np.kron(gate.gate['CRX'](theta), A)
#     test_result = tensor.CRXA(A, theta)
#     assert np.allclose(true_result, test_result)
    
# def testCRYA():
#     A = np.random.rand(10, 10)
#     theta = np.random.uniform(0, 2*np.pi)
#     true_result = np.kron(gate.gate['CRY'](theta), A)
#     test_result = tensor.CRYA(A, theta)
#     assert np.allclose(true_result, test_result)
    
# def testCRZA():
#     A = np.random.rand(10, 10)
#     theta = np.random.uniform(0, 2*np.pi)
#     true_result = np.kron(gate.gate['CRZ'](theta), A)
#     test_result = tensor.CRZA(A, theta)
#     assert np.allclose(true_result, test_result)
    
# def test_mutate():
#     a = 4 + 4
#     if a > 5:
#         a = 5
#     assert a == 5