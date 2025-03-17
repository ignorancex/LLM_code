import numpy as np
from pyparc.parc import PARC

from slpwampc.misc.regions import Polytope


class Parc(PARC):
    def get_partition(self, A: np.ndarray, b: np.ndarray) -> list[Polytope]:
        """Return the partition as a collection of regions within the space Ax <= b.

        Parameters
        ----------
        A : np.ndarray
            The matrix A in the inequality Ax <= b
        b : np.ndarray
            The vector b in the inequality Ax <= b

        Returns
        -------
        list[Polytope]
            The regions in the partition."""
        nx = self.nx
        ind = np.arange(2, nx, dtype=int)
        values = np.zeros(nx - 2)

        omega = self.omega
        gamma = self.gamma + (omega[:, ind] @ values).ravel()
        omega = np.delete(omega, ind, axis=1)
        xbar = np.delete(self.xbar, ind, axis=1)
        K = self.K

        # Plot PWL partition
        A_ = np.vstack((A, np.zeros((K - 1, 2))))
        b_ = np.vstack((b, np.zeros((K - 1, 1))))
        regions = list()

        for j in range(0, K):
            i = b.shape[0]
            for h in range(0, K):
                if h != j:
                    A_[i, :] = omega[h, :] - omega[j, :]
                    b_[i] = -gamma[h] + gamma[j]
                    i += 1
            regions.append(Polytope(A_, b_))
        return regions
