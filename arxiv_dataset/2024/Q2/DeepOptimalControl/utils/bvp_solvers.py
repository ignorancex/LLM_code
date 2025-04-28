import numpy as np
from scipy.integrate import solve_bvp
from problems.problem import ControlProblem


def space_march_fix_forward(problem: ControlProblem, x0, k_try=60, init_guess=None):
    tolerance, max_nodes = problem.data_tol, problem.max_nodes
    try_array = np.linspace(0.01, 1, k_try)
    try_x0s = x0 * try_array
    x0_scale = try_x0s[:, 0:1]
    try:
        t = init_guess['t']
        x_aug = init_guess['X_aug']
    except:
        t = np.linspace(0, problem.tT)
        x_aug = np.vstack(
            (np.zeros((x0_scale.shape[0], t.size)), np.zeros((x0_scale.shape[0], t.size))))

    i, fail_time = 0, 0
    while (fail_time < 5) and (i < k_try):
        x0_scale = try_x0s[:, i:i+1]
        bc = problem.make_bc(x0_scale.reshape(-1))
        SOL = solve_bvp(problem.aug_dynamics, bc, t, x_aug,
                        tol=tolerance, max_nodes=max_nodes, verbose=0)
        if SOL.success:
            t = SOL.x
            x_aug = SOL.y
            fail_time = 0
        else:
            print(
                '*********** The march scale {} failed!'.format(try_array[i]))
            fail_time += 1
        i += 1
    return SOL


def direct_fix(problem: ControlProblem, x0, init_guess=None):
    tolerance, max_nodes = problem.data_tol, problem.max_nodes
    try:
        t = init_guess['t']
        x_aug = init_guess['X_aug']
    except:
        t = np.linspace(0, problem.tT)
        x_aug = np.vstack((x0.repeat(t.size, axis=1),
                           np.zeros((x0.shape[0], t.size))))

    bc = problem.make_bc(x0.reshape(-1))
    SOL = solve_bvp(problem.aug_dynamics, bc, t, x_aug,
                    tol=tolerance, max_nodes=max_nodes, verbose=0)

    return SOL
