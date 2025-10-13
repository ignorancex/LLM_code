import numpy as np


# pylint: skip-file
class ConvexToy:
    class sim:
        u_opt = np.array([[0.25], [-1]])

        # simulation length
        n_steps = 10

        # measurement noise
        noise_seed = 42
        noise_y_std = 1e-2

    class sys:
        # system dimensions
        m = 2
        p = 1

        # input constraints
        A_u = np.array(
            [
                [1, 0],
                [-1, 0],
                [0, 1],
                [0, -1],
            ]
        )
        b_u = np.array(
            [
                [1],
                [1],
                [1],
                [1],
            ]
        )

        # output constraints
        A_y = np.array(
            [
                [1],
                [-1],
            ]
        )
        b_y = np.array(
            [
                [1],
                [0],
            ]
        )

        # nonlinear dynamics
        def h(u):
            u_1 = u[0, 0]
            u_2 = u[1, 0]
            return np.array([[2 * u_2 - u_1 + 1.5]])

        def du_h(u):
            # u_1 = u[0,0]
            # u_2 = u[1, 0]
            return np.array([[-1, 2]])

    class opt:
        # input objective cost function
        quad_u = np.diag([20, 2])
        lin_u = np.array([[1], [1]])

        # output objective cost function
        quad_y = np.array([[0.5]])
        lin_y = np.array([[-1]])

    class opt_prim(opt):
        name = r"II-A Projected Primal"
        alpha = 0.03

    class opt_dualyprox(opt):
        name = r"III-A PRIME-Y"
        rho = 2

    class opt_dualhprox_dist(opt):
        name = r"III-C Dist. PRIME-H"
        rho = 2
        centralize = False

    class opt_dualhprox_cent(opt):
        name = r"III-C Cent. PRIME-H"
        rho = 2
        centralize = True
