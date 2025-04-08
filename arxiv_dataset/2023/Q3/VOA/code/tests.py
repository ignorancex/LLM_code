import numpy as np
from kalmanFilter import KalmanFilter
from prm import*
from util import*


def VOAInPlanning(sol: List[NodePRM], start: List[float], help_step: int = -1, voa_sigma: float = 0.2, Sigma: np.matrix = np.matrix('0 0 ; 0 0'), route_point: int = -1):

    # initialization Kalman Filter
    A = np.matrix('1 0; 0 1')
    B = np.matrix('1 0; 0 1')
    C = np.matrix('1 0; 0 1')
    R = np.matrix([[voa_sigma, 0], [0, voa_sigma]])
    Q = np.matrix('0.01 0; 0 0.01')
    x_0 = np.matrix([[start.pos.bounds[0]], [start.pos.bounds[1]]])
    Sigma_0 = Sigma
    mu_0 = x_0
    k = 11.82  # value for 3Sigma meaning 99.74% probability
    KF = KalmanFilter(A, B, C, R, Q, mu_0, Sigma_0, x_0, k)

    # run simulation
    action_speed = 0.5
    mu_final, Sigma_final, Belief, rout_point = KF.priorySimWithOneCourseCorrection(
        action_speed, sol, help_step, route_point)

    if help_step >= 0:
        return Belief, mu_final, Sigma_final, rout_point
    else:
        return Belief, -1, -1, -1


def calcVOIScenarioT(sol: List[NodePRM], start: List[float], voi_sigma: float, help_step: int = -1):
    # initialization Kalman Filter
    A = np.matrix('1 0; 0 1')
    B = np.matrix('1 0; 0 1')
    C = np.matrix('1 0; 0 1')
    R = np.matrix([[voi_sigma, 0], [0, voi_sigma]])
    Q = np.matrix('0.01 0; 0 0.01')
    x_0 = np.matrix([[start.pos.bounds[0]], [start.pos.bounds[1]]])
    Sigma_0 = np.matrix('0 0 ; 0 0')
    mu_0 = x_0
    k = 11.82  # value for 3Sigma meaning 99.74% probability
    KF = KalmanFilter(A, B, C, R, Q, mu_0, Sigma_0, x_0, k)

    action_speed = 0.5
    path_with_cor, path_no_cor = KF.runSimWithOneCourseCorrection(
        action_speed, sol, help_step)

    return path_with_cor, path_no_cor
