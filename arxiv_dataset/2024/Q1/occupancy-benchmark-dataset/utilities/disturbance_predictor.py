import numpy as np
from scipy import linalg
from typing import Callable

from .kalman_filter import KalmanFilter


class DisturbancePredictor:
    """
    Implements a disturbance predictor with a state space approximated Gaussian Process as disturbance model.
    The prediction implementation is based on the Kalman Filter.

    Attributes:
        kalman_filter: reference to the underlying KalmanFilter object.
    """
    kalman_filter: KalmanFilter

    def __init__(self, KalmanFilterObject: Callable, Kernel: Callable, hyperparameters: list, likelihood, dt):
        """
        Initializes the internal KalmanFilter object

        Arguments:
            KalmanFilterObject: Class reference to the used Kalman Filter.
            Kernel: Constructor function for the state space representation.
            hyperparameters: List of input arguments of Kernel.
            likelihood: The likelihood parameter for the Kalman Filter (eg. R).
            dt: sample time.
        """
        F, Pinf, _, H, _ = Kernel(*hyperparameters)
        Ak = linalg.expm(F * dt)
        Qk = Pinf - Ak @ Pinf @ Ak.T
        m0 = np.zeros([F.shape[0], 1])
        P0 = np.eye(F.shape[0])
        self.kalman_filter = KalmanFilterObject(dt, likelihood, Qk, F, H, m0, P0)

    def get_prediction(self, y: np.ndarray, length: int) -> np.ndarray:
        """
        Updates the predictor with the current value y and computes a disturbance trajectory.

        Arguments:
            y (np.ndarray): Current values for the estimated disturbance
            length (int): Length of the prediction trajectory

        Returns:
            prediction (np.ndarray): Prediction trajectory
        """
        self.kalman_filter.update(y)
        prediction = self.kalman_filter.predict_loop(length)
        self.kalman_filter.predict()
        return prediction
