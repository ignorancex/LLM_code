import numpy as np

from src.multiplier import FourierMultiplier

class Heat(FourierMultiplier):
    """Run forward heat equation for some time.
    The heat equation is defined as:
        u_t = \alpha \Delta u

    with various boundary conditions. We utilize the homogeneous Dirichlet or Neumann boundary.
    Homogeneous Dirichlet boundary conditions require the "heat" at the edges to be zero at all times.
    Homogeneous Neumann boundary conditions require the derivative of the "heat" in the direction orthogonal to
    the boundary to be zero at all times.
    """

    def __init__(self,
                 time=3e-3,
                 alpha=0.6,
                 **kwargs):

        """
        Parameters:

        time: float
            run time of the heat equation, in arbitrary units.
        alpha: float
            coefficient of the laplace operator in the heat equation.
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.time = time

        ## In the frequency domain, running the heat equation for the given time amounts to
        ## multiplying by this Fourier multiplier
        self.multiplier = np.exp(-self.alpha * np.pi ** 2 * self.time * self.freqs ** 2)

    def __str__(self):
        return f'Heat operator alpha {self.alpha} time {self.time}'
