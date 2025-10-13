import numpy as np


# pylint: skip-file
class UnicornNoise:
    class sim:
        # optimal u
        u_opt = np.array([6.63761286, 3.94908578, 2.0, 0.5]).reshape(-1, 1)

        # simulation length
        n_steps = 1_000

        # measurement noise
        noise_seed = 0
        noise_y_std = (1.05 - 0.95) * 1.5 / 100

    class sys:
        # Electrical Network
        net_path = "data/unicorn_56.json"

    class opt:
        # input objective cost function
        quad_u = 0.1 * np.eye(4)
        lin_u = np.array([[0.1], [0.1], [0], [0]])

    class opt_prim(opt):
        name = r"Algo. 1, Projected Primal"
        alpha = 0.1

    class opt_dualy(opt):
        name = r"Algo. 2, Primal-Dual"
        alpha = 4
        beta = 8

    class opt_dualh(opt):
        name = r"Algo. -, Primal-Dual"
        alpha = 4
        beta = 2

    class opt_dualyprox_dist(opt):
        name = r"Algo. 4, PRIME-Y"
        rho = 1e3
        gamma_u = 20
        centralized = False

    class opt_dualhprox_dist(opt):
        name = r"Algo. 4, PRIME-H"
        rho = 1e3
        gamma_u = 20
        gamma_z = 20
        centralized = False
