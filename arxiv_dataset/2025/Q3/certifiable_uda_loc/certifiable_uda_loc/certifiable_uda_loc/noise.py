import numpy as np
import pymlg


def sample_so2_langevin(mode: np.ndarray, kappa: float) -> np.ndarray:
    # Sections 2.2 and Appendix A of
    # https://arxiv.org/pdf/1612.07386#page=7.72
    d = mode.shape[0]
    R = None
    if d == 2:
        th = np.random.vonmises(0, 2 * kappa)
        R = pymlg.SO2.Exp(th)
    # R = np.eye(2)
    return R
