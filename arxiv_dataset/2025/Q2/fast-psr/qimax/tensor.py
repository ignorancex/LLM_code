import numpy as np
from . import gate, constant

def AM(A): # AP0
    if A.ndim == 1:
        return A*constant.P(0)
    m, n = A.shape
    kron_result = np.zeros((2 * m, 2 * n), dtype=A.dtype)
    kron_result[0::2, 0::2] = A
    return kron_result

def MA(A): # P0A
    if A.ndim == 1:
        return A*constant.P(0)
    m, n = A.shape
    kron_result = np.zeros((2 * m, 2 * n), dtype=A.dtype)
    kron_result[:m, :n] = A
    return kron_result

def AP(A): # AP3
    if A.ndim == 1:
        return A*constant.P(3)
    rows, cols = A.shape
    kron_result = np.zeros((2 * rows, 2 * cols), dtype=A.dtype)
    kron_result[1::2, 1::2] = A
    return kron_result

def PA(A): # P3A
    if A.ndim == 1:
        return A*constant.P(3)
    m, n = A.shape
    kron_result = np.zeros((2 * m, 2 * n), dtype=A.dtype)
    kron_result[m:, n:] = A
    return kron_result

def HA(A):
    # sqrt2_times_A = 1/np.sqrt(2) * A
    # m, n = A.shape
    # out = np.zeros((2, m, 2, n), dtype=A.dtype)
    
    # out[0, :, 0:1, :] = sqrt2_times_A
    # out[0, :, 1, :] = sqrt2_times_A
    # out[1, :, 0, :] = sqrt2_times_A
    # out[1, :, 1, :] = -sqrt2_times_A
    # out.shape = (2 * m, 2 * n)
    # return out
    if A.ndim == 1:
        return A*gate.gate['H']
    return np.kron(gate.gate['H'], A) # This way is still faster

def AH(A):
    if A.ndim == 1:
        return A*gate.gate['H']
    sqrt2_times_A = 1/np.sqrt(2) * A
    m, n = A.shape
    out = np.zeros((m, 2, n, 2), dtype=A.dtype)
    
    out[:, 0, :, 0] = sqrt2_times_A
    out[:, 0, :, 1] = sqrt2_times_A
    out[:, 1, :, 0] = sqrt2_times_A
    out[:, 1, :, 1] = -sqrt2_times_A
    
    out.shape = (2 * m, 2 * n)
    return out

def IA(A, N = 2):
    if A.ndim == 1:
        return A*gate.gate['I']
    m, n = A.shape
    out = np.zeros((N, m, N, n), dtype=A.dtype)
    r = np.arange(N)
    out[r, :, r, :] = A
    out.shape = (m*N, n*N)
    return out


def AI(A, N = 2):
    if A.ndim == 1:
        return A*gate.gate['I']
    m,n = A.shape
    out = np.zeros((m,N,n,N),dtype=A.dtype)
    r = np.arange(N)
    out[:,r,:,r] = A
    out.shape = (m*N,n*N)
    
    return out


def XA(A):
    if A.ndim == 1:
        return A*gate.gate['X']
    m, n = A.shape
    out = np.zeros((2, m, 2, n), dtype=A.dtype)
    out[0, :, 1, :] = A
    out[1, :, 0, :] = A
    out.shape = (2 * m, 2 * n)
    return out


def AX(A):
    if A.ndim == 1:
        return A*gate.gate['X']
    m, n = A.shape
    out = np.zeros((m, 2, n, 2), dtype=A.dtype)
    out[:, 0, :, 1] = A
    out[:, 1, :, 0] = A
    out.shape = (2 * m, 2 * n)
    return out

def ZA(A):
    if A.ndim == 1:
        return A*gate.gate['Z']
    m, n = A.shape
    out = np.zeros((2, m, 2, n), dtype=A.dtype)
    out[0, :, 0, :] = A
    out[1, :, 1, :] = -A
    out.shape = (2 * m, 2 * n)
    return out

def AZ(A):
    if A.ndim == 1:
        return A*gate.gate['Z']
    m, n = A.shape
    out = np.zeros((m, 2, n, 2), dtype=A.dtype)
    out[:, 0, :, 0] = A
    out[:, 1, :, 1] = -A
    out.shape = (2 * m, 2 * n)
    return out


def RXA(A, theta):
    if A.ndim == 1:
        return A * gate.gate['RX'](theta)
    cos_theta_2_times_A = np.cos(theta / 2) * A
    sin_theta_2_times_A = -1j * np.sin(theta / 2) * A
    m, n = A.shape
    out = np.zeros((2, m, 2, n), dtype=np.complex64)
    
    out[0, :, 0, :] = cos_theta_2_times_A
    out[0, :, 1, :] = sin_theta_2_times_A
    out[1, :, 0, :] = sin_theta_2_times_A
    out[1, :, 1, :] = cos_theta_2_times_A
    
    out.shape = (2 * m, 2 * n)
    return out

def ARX(A, theta):
    if A.ndim == 1:
        return A * gate.gate['RX'](theta)
    cos_theta_2_times_A = np.cos(theta / 2) * A
    sin_theta_2_times_A = -1j * np.sin(theta / 2) * A
    m, n = A.shape
    out = np.zeros((m, 2, n, 2), dtype=np.complex64)
    out[:, 0, :, 0] = cos_theta_2_times_A
    out[:, 0, :, 1] = sin_theta_2_times_A
    out[:, 1, :, 0] = sin_theta_2_times_A
    out[:, 1, :, 1] = cos_theta_2_times_A
    out.shape = (2 * m, 2 * n)
    return out

def RYA(A, theta):
    if A.ndim == 1:
        return A * gate.gate['RY'](theta)
    cos_theta_2_times_A = np.cos(theta / 2) * A
    sin_theta_2_times_A = np.sin(theta / 2) * A
    m, n = A.shape
    out = np.zeros((2, m, 2, n), dtype=np.complex64)
    
    out[0, :, 0, :] = cos_theta_2_times_A
    out[0, :, 1, :] = -sin_theta_2_times_A
    out[1, :, 0, :] = sin_theta_2_times_A
    out[1, :, 1, :] = cos_theta_2_times_A
    
    out.shape = (2 * m, 2 * n)
    return out

def ARY(A, theta):
    if A.ndim == 1:
        return A * gate.gate['RY'](theta)
    cos_theta_2_times_A = np.cos(theta / 2) * A
    sin_theta_2_times_A = np.sin(theta / 2) * A
    m, n = A.shape
    out = np.zeros((m, 2, n, 2), dtype=np.complex64)
    out[:, 0, :, 0] = cos_theta_2_times_A
    out[:, 0, :, 1] = -sin_theta_2_times_A
    out[:, 1, :, 0] = sin_theta_2_times_A
    out[:, 1, :, 1] = cos_theta_2_times_A
    out.shape = (2 * m, 2 * n)
    return out

def ARZ(A, theta):
    if A.ndim == 1:
        return A * gate.gate['RZ'](theta)
    phase = 1j * theta / 2
    m, n = A.shape
    out = np.zeros((m, 2, n, 2), dtype=np.complex64)
    out[:, 0, :, 0] = np.exp(-phase) * A
    out[:, 0, :, 1] = 0
    out[:, 1, :, 0] = 0
    out[:, 1, :, 1] = np.exp(phase) * A
    out.shape = (2 * m, 2 * n)
    return out

def RZA(A, theta):
    if A.ndim == 1:
        return A * gate.gate['RZ'](theta)
    phase = 1j * theta / 2
    m, n = A.shape
    out = np.zeros((2, m, 2, n), dtype=A.dtype)
    out[0, :, 0, :] = np.exp(-phase) * A
    out[0, :, 1, :] = 0
    out[1, :, 0, :] = 0
    out[1, :, 1, :] = np.exp(phase) * A
    out.shape = (2 * m, 2 * n)
    return out

# def CXA(A):
#     if A.ndim == 1:
#         return A*gate.gate['CX']
    
#     m, n = A.shape
#     out = np.zeros((4, m, 4, n), dtype=A.dtype)
    
#     out[0, :, 0, :] = A
#     out[1, :, 1, :] = A
#     out[2, :, 3, :] = A
#     out[3, :, 2, :] = A 
#     out.shape = (4 * m, 4 * n)
#     return out

# def ACX(A):
#     if A.ndim == 1:
#         return A*gate.gate['CX']
#     m, n = A.shape
#     out = np.zeros((m, 4, n, 4), dtype=A.dtype)

#     out[:, 0, :, 0] = A
#     out[:, 1, :, 1] = A
#     out[:, 2, :, 3] = A
#     out[:, 3, :, 2] = A
    
#     out.shape = (4 * m, 4 * n)
#     return out

# def ACRX(A, theta):
#     cos_theta_2_times_A = np.cos(theta / 2) * A
#     sin_theta_2_times_A = -1j * np.sin(theta / 2) * A
#     if A.ndim == 1:
#         return np.array([[cos_theta_2_times_A, sin_theta_2_times_A],
#                          [sin_theta_2_times_A, cos_theta_2_times_A]])
#     m, n = A.shape
#     out = np.zeros((m, 4, n, 4), dtype=np.complex64)
#     out[:, 0, :, 0] = A
#     out[:, 1, :, 1] = A
#     out[:, 2, :, 2] = cos_theta_2_times_A
#     out[:, 2, :, 3] = sin_theta_2_times_A
#     out[:, 3, :, 2] = sin_theta_2_times_A
#     out[:, 3, :, 3] = cos_theta_2_times_A
#     out.shape = (4 * m, 4 * n)
#     return out

# def CRXA(A, theta):
#     cos_theta_2_times_A = np.cos(theta / 2) * A
#     sin_theta_2_times_A = -1j * np.sin(theta / 2) * A
#     if A.ndim == 1:
#         return np.array([[cos_theta_2_times_A, sin_theta_2_times_A],
#                          [sin_theta_2_times_A, cos_theta_2_times_A]])
#     m, n = A.shape
#     out = np.zeros((4, m, 4, n), dtype=np.complex64)
#     out[0, :, 0, :] = A
#     out[1, :, 1, :] = A
#     out[2, :, 2, :] = cos_theta_2_times_A
#     out[2, :, 3, :] = sin_theta_2_times_A
#     out[3, :, 2, :] = sin_theta_2_times_A
#     out[3, :, 3, :] = cos_theta_2_times_A
#     out.shape = (4 * m, 4 * n)
#     return out

# def ACRY(A, theta):
#     cos_theta_2_times_A = np.cos(theta / 2) * A
#     sin_theta_2_times_A = np.sin(theta / 2) * A
#     if A.ndim == 1:
#         return np.array([[cos_theta_2_times_A, -sin_theta_2_times_A],
#                          [sin_theta_2_times_A, cos_theta_2_times_A]])
#     m, n = A.shape
#     out = np.zeros((m, 4, n, 4), dtype=np.complex64)
    
#     out[:, 0, :, 0] = A
#     out[:, 1, :, 1] = A
#     out[:, 2, :, 2] = cos_theta_2_times_A
#     out[:, 2, :, 3] = -sin_theta_2_times_A
#     out[:, 3, :, 2] = sin_theta_2_times_A
#     out[:, 3, :, 3] = cos_theta_2_times_A
    
#     out.shape = (4 * m, 4 * n)
#     return out

# def CRYA(A, theta):
#     cos_theta_2_times_A = np.cos(theta / 2) * A
#     sin_theta_2_times_A = np.sin(theta / 2) * A
#     if A.ndim == 1:
#         return np.array([[cos_theta_2_times_A, -sin_theta_2_times_A],
#                          [sin_theta_2_times_A, cos_theta_2_times_A]])
#     m, n = A.shape
#     out = np.zeros((4, m, 4, n), dtype=np.complex64)
    
#     out[0, :, 0, :] = A
#     out[1, :, 1, :] = A
#     out[2, :, 2, :] = cos_theta_2_times_A
#     out[2, :, 3, :] = -sin_theta_2_times_A
#     out[3, :, 2, :] = sin_theta_2_times_A
#     out[3, :, 3, :] = cos_theta_2_times_A
    
#     out.shape = (4 * m, 4 * n)
#     return out


# def CRZA(A, theta):
#     phase = 1j * theta / 2
#     if A.ndim == 1:
#         A = A[0]
#         return np.array([[A, A],
#                          [np.exp(-phase) * A, np.exp(phase) * A]])
#     m, n = A.shape
#     out = np.zeros((4, m, 4, n), dtype=np.complex64)
#     out[0, :, 0, :] = A
#     out[1, :, 1, :] = A
#     out[2, :, 2, :] = np.exp(-phase) * A
#     out[3, :, 3, :] = np.exp(phase) * A
#     out.shape = (4 * m, 4 * n)
#     return out

# def ACRZ(A, theta):
#     phase = 1j * theta / 2
#     # check type A is int
#     if A.ndim == 1:
#         return A*np.array([[A, A],
#                          [np.exp(-phase) * A, np.exp(phase) * A]])
#     m, n = A.shape
#     out = np.zeros((m, 4, n, 4), dtype=np.complex64)
#     out[:, 0, :, 0] = A
#     out[:, 1, :, 1] = A
#     out[:, 2, :, 2] = np.exp(-phase) * A
#     out[:, 3, :, 3] = np.exp(phase) * A
#     out.shape = (4 * m, 4 * n)
#     return out
