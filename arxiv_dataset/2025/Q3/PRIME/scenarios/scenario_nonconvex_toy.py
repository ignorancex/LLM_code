import numpy as np


# pylint: skip-file
class NonConvexToy:
    class sim:
        # initial u guess
        u_0 = np.array([[-0.5], [0.5]])
        u_opt = np.array([[-0.166625], [0.395009]])

        # simulation length
        n_steps = 50

        # measurement noise
        noise_seed = 42
        # noise_y_std = 0.05

    class sys:
        # system dimensions
        m = 2
        p = 1

        u_opt = np.array([[-0.5], [1]])

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
            return np.array([[u_2**3 + u_1 - u_2 + 0.5]])

        def du_h(u):
            # u_1 = u[0,0]
            u_2 = u[1, 0]
            return np.array([[1, 3 * u_2**2 - 1]])

    class opt:
        # input objective cost function
        quad_u = np.diag([1, 1])
        lin_u = -np.array([[0.5], [0.5]])

        # output objective cost function
        lin_y = np.array([[5]])

    class opt_prim(opt):
        name = r"Algo. 1, Projected Primal"
        alpha = 0.3

    class opt_dualyprox_dist(opt):
        name = r"Algo. 3, PRIME-Y"
        rho = 0.3
        gamma_u = 10
        centralized = True

    class opt_dualhprox_dist(opt):
        name = r"Algo. 4, PRIME-H"
        rho = 0.3
        gamma_u = 5
        gamma_z = 5
        centralized = False
