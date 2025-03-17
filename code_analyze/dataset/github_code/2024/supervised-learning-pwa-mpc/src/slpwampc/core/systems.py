import numpy as np


class PwaSystem:
    """Class to represent a piecewise affine system."""

    def __init__(self, system_dict: dict) -> None:
        """Initialise the PWA system. Simply takes a dictionary
        and assigns the values to the class attributes.

        Parameters
        ----------
        system_dict : dict
            A dictionary containing lists of A, B ,c, S, R, T matrices, such that x(k+1) = A(i)x(k) + B(i)u(k) + c(i)
            when S(i)x(k) + R(i)u(k) <= T(i).
            Also contains bound matrices D, E, F, G matrices, such that Dx <= E and Fu <= G.
        """
        self.A = system_dict["A"]
        self.B = system_dict["B"]
        self.c = system_dict["c"]
        self.S = system_dict["S"]
        self.R = system_dict["R"]
        self.T = system_dict["T"]
        self.D = system_dict["D"]
        self.E = system_dict["E"]
        self.F = system_dict["F"]
        self.G = system_dict["G"]

        if any([o.any() for o in self.R]):
            self.region_depends_on_u = True
        else:
            self.region_depends_on_u = False

    def next_state(self, x: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, int]:
        """Get the next state of the system given the current state and control input.

        Parameters
        ----------
        x : np.ndarray
            The current state.
        u : np.ndarray
            The control input.

        Returns
        -------
        tuple[np.ndarray, int]
            The next state and the region index.
        """
        for i in range(len(self.S)):
            if (self.S[i] @ x + self.R[i] @ u <= self.T[i]).all():
                return self.A[i] @ x + self.B[i] @ u + self.c[i], i
        raise ValueError("No region found for state and action")
