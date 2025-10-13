import numpy as np


# pylint: skip-file
class Toy1D:
    class sim:
        # initial u guess
        u_0 = np.array([[0]])

        # simulation length
        n_steps = 200

    class sys:
        # system dimensions
        m = 1
        p = 1

        u_opt = np.array([[-2]])

        # input constraints
        A_u = np.array(
            [
                [1],
                [-1],
            ]
        )
        b_u = np.array(
            [
                [0],
                [2],
            ]
        )

        # output constraints
        A_y = np.array(
            [
                [1],
            ]
        )
        b_y = np.array(
            [
                [1.2],
            ]
        )

        # nonlinear dynamics
        def h(u):
            return np.array([[2 * u[0, 0] ** 2 + u[0, 0] ** 3]])

        def du_h(u):
            return np.array([[4 * u[0, 0] + 3 * u[0, 0] ** 2]])

    class opt:
        # input objective cost function
        lin_u = np.array([[0.2]])

    class opt_prim(opt):
        name = r"II-A Projected Primal"
        alpha = 0.15

    class opt_dualhprox_cent(opt):
        name = r"III-C Cent. PRIME-H"
        rho = 0.15
        gamma_u = 1
        gamma_z = 1
        centralized = True
