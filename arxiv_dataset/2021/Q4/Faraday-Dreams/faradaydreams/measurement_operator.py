import numpy as np
import optimusprimal.linear_operators as linear_operators
import logging

logger = logging.getLogger('Faraday Dreams')


def power_method(m_op, tol=1e-4):
    nu, sol = linear_operators.power_method(m_op, np.ones((m_op.cols)), tol)
    return nu


def create_faraday_matrix(lambda2, phi, weights):
    A = np.zeros((len(lambda2), len(phi)), dtype=np.complex)
    for r in range(len(lambda2)):
        A[r, :] = np.exp(2 * lambda2[r] * phi * 1j) * \
            weights[r]
    return A


def svd_op(A, y, weights):
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    A_svd = (np.diag(s) @ vh)
    y_svd = np.conj(u.T) @ y
    y_svd = y_svd[:len(s)]
    return y_svd, A_svd


def svd_op_cube(A, y, weights):
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    A_svd = (np.diag(s) @ vh)
    y_svd = np.einsum('ij,jkm->ikm', np.conj(u.T), y)
    y_svd = y_svd[:len(s), :, :]
    return y_svd, A_svd


class faraday_operator:
    """
    Operator that simulates a Faraday rotation observation and finite channel width.

    INPUT:
        lambda2 - channel locations in m^2
        phi - pixel locations in rad/m^2
        weights - weights for each channel in m^2
        lambda2_width - channel widths in m^2
        spectral_axis - axis number for performing the calculation
        nufft - to use NUFFT algorithm over matrix operation
        lambda2_channel_averaging_windows - pass a matrix where each row contains the RM sensitivity as a function of phi for the chosen channel
    """
    def __init__(self,
                 lambda2,
                 phi,
                 weights,
                 lambda2_width=None,
                 spectral_axis=-1,
                 nufft=False,
                 lambda2_channel_averaging_windows=None):
        self.spectral_axis = spectral_axis
        self.rows = len(lambda2)
        self.cols = len(phi)
        if (np.all(lambda2_width != None)):
            assert lambda2.shape == lambda2_width.shape
        if lambda2_channel_averaging_windows is None:
            if lambda2_width is None:
                lambda2_channel_averaging_windows = 1
            else:
                lambda2_channel_averaging_windows = np.sinc(
                    phi[np.newaxis, :] * lambda2_width[:, np.newaxis] / np.pi)
        if nufft == False:
            A = create_faraday_matrix(
                lambda2, phi, weights) * lambda2_channel_averaging_windows
            self.wrap_matrix(A)
        else:
            from pynufft import NUFFT
            nufft = NUFFT()
            Nd = phi.shape[0]
            Jd = 7
            Kd = Nd * 2
            dphi = np.abs(phi[0] - phi[1])
            nufft.plan(-lambda2.reshape((len(lambda2), 1)) * dphi * 2, (Nd, ),
                       (Kd, ), (Jd, ))
            self.dir_op_1d = lambda x: nufft.forward(x) * weights
            self.adj_op_1d = lambda x: nufft.adjoint(x * np.conj(weights)) * Kd

    def wrap_matrix(self, A):
        """Takes the Fourier matrix and wraps it into a lambda"""
        A_H = np.conj(A.T)
        self.dir_op_1d = lambda x: A @ x
        self.adj_op_1d = lambda x: A_H @ x

    def dir_op(self, x):
        """Applies the forward operator along the spectral axis"""
        return np.apply_along_axis(self.dir_op_1d, self.spectral_axis, x)

    def adj_op(self, x):
        """Applies the forward operator along the spectral axis"""
        return np.apply_along_axis(self.adj_op_1d, self.spectral_axis, x)


def phi_parameters(lambda2, lambda2_width):
    """Estimates the sampling parameters in Faraday depth using Nyquist sampling assumption."""
    dphi = 1. / np.max(lambda2) * np.pi / 2.
    phi_max = np.max(np.abs(1. / lambda2_width))
    phi_min = np.min(np.abs(1. / lambda2_width))
    logger.info("\u03C6 resolution %s rad/m^2.", dphi)
    logger.info("\u03C6 max range is %s rad/m^2.", phi_max)
    logger.info("\u03C6 min range is %s rad/m^2.", phi_min)
    return phi_max, phi_min, dphi


def create_phi_array(Nd, dphi):
    return np.linspace(-Nd * dphi / 2, Nd * dphi / 2, Nd, endpoint=False)


def create_lambda2_array(Nd, dphi):
    return np.fft.fftshift(np.fft.fftfreq(Nd, dphi) * np.pi)
