import gurobipy as gp
import numpy as np
from dmpcpwa.mpc.mpc_mld import MpcMld


class ThisMpcMld(MpcMld):
    """A mixed-integer MPC for a PWA system with MLD model."""

    def __init__(
        self,
        system: dict,
        N: int,
        nx: int,
        nu: int,
        X_f: tuple[np.ndarray, np.ndarray] | None = None,
        verbose: bool = False,
    ) -> None:
        """Initialize the MPC,

        Parameters
        ----------
        system : dict
            The PWA dictionary.
        N : int
            The prediction horizon.
        nx : int
            The state dimension.
        nu : int
            The input dimension.
        X_f : tuple[np.ndarray, np.ndarray], optional
            The terminal region A <= b (A, b), by default None.
        verbose : bool, optional
            Whether to print the solver output, by default False.
        """
        # dynamics, state, and input constraints built in here with MLD model conversion
        super().__init__(system, N, verbose=verbose)

        Q_x = np.eye(nx)
        Q_u = np.eye(nu)

        obj = 0
        obj += sum(
            [
                self.min_1_norm(self.x[:, [k]], Q_x)
                + self.min_1_norm(self.u[:, [k]], Q_u)
                for k in range(N)
            ]
        )
        obj += self.min_1_norm(self.x[:, [N]], Q_x)

        if X_f is not None:
            A, b = X_f
            self.mpc_model.addConstr(A @ self.x[:, [N]] <= b)
        self.mpc_model.setObjective(obj, gp.GRB.MINIMIZE)


class ThisTightenedMpcMld(ThisMpcMld):
    """A mixed-integer MPC for a PWA system with MLD model. The first state constraint is tightened."""

    def __init__(
        self,
        system: dict,
        N: int,
        nx: int,
        nu: int,
        eps: float,
        X_f: tuple[np.ndarray, np.ndarray] | None = None,
        verbose: bool = False,
    ) -> None:
        """Initialize the MPC,

        Parameters
        ----------
        system : dict
            The PWA dictionary.
        N : int
            The prediction horizon.
        nx : int
            The state dimension.
        nu : int
            The input dimension.
        eps : float
            The tightening parameter.
        verbose : bool, optional
            Whether to print the solver output, by default False.
        """
        super().__init__(system, N, nx, nu, X_f, verbose)
        self.mpc_model.addConstrs(
            system["D"] @ self.x[:, [k]]
            <= system["E"] - eps * np.linalg.norm(system["D"], ord=1, axis=1)
            for k in range(N + 1)
        )
