from navlie.lib.states import SE2State
import numpy as np
import pymlg
import certifiable_uda_loc.gauss_newton_baseline as gn_baseline
import navlie


def test_deadreckoning():
    residuals = [
        gn_baseline.PriorResidualFro(
            keys=["T0"],
            prior_state=SE2State(
                value=np.array(
                    [
                        [-0.9220918, -0.38697119, -0.80217284],
                        [0.38697119, -0.9220918, -0.44887781],
                        [0.0, 0.0, 1.0],
                    ]
                )
            ),
            weight_rot=100,
            weight_pos=100,
        ),
        gn_baseline.RelativePoseResidualFro(
            ["T0", "T1"],
            dT=np.array(
                [
                    [-0.43802382, 0.89896336, -0.31759656],
                    [-0.89896336, -0.43802382, 0.52265442],
                    [0.0, 0.0, 1.0],
                ]
            ),
            weight_pos=10,
            weight_rot=100,
        ),
        gn_baseline.RelativePoseResidualFro(
            ["T1", "T2"],
            dT=np.array(
                [
                    [-0.33944808, -0.94062479, -0.37895997],
                    [0.94062479, -0.33944808, 2.57541416],
                    [0.0, 0.0, 1.0],
                ]
            ),
            weight_pos=10,
            weight_rot=100,
        ),
    ]
    se2_states = [
        SE2State(
            stamp=0,
            state_id="T0",
            direction="right",
            value=np.array(
                [
                    [-0.86713957, -0.49806522, -0.80217284],
                    [0.49806522, -0.86713957, -0.44887781],
                    [0.0, 0.0, 1.0],
                ]
            ),
        ),
        SE2State(
            stamp=1,
            state_id="T1",
            direction="right",
            value=np.array(
                [
                    [0.82757018, -0.56136228, -0.78708828],
                    [0.56136228, 0.82757018, -1.06027594],
                    [0.0, 0.0, 1.0],
                ]
            ),
        ),
        SE2State(
            stamp=2,
            state_id="T2",
            direction="right",
            value=np.array(
                [
                    [-0.80894838, -0.58787968, -2.54644461],
                    [0.58787968, -0.80894838, 0.85832618],
                    [0.0, 0.0, 1.0],
                ]
            ),
        ),
    ]

    # Case: Prior State only. This seems to work.
    problem = navlie.batch.problem.Problem(solver="GN", step_tol=1e-10, max_iters=300)
    for var in [se2_states[0]]:
        var: SE2State = var
        problem.add_variable(var.state_id, var)
    for resid in [residuals[0]]:
        problem.add_residual(resid)
    result = problem.solve()

    problem = navlie.batch.problem.Problem(solver="GN", step_tol=1e-10, max_iters=300)
    for var in se2_states[:2]:
        var: SE2State = var
        problem.add_variable(var.state_id, var)
    for resid in residuals[:2]:
        problem.add_residual(resid)
    result = problem.solve()
    assert result["summary"].cost[-1] < 1e-12


def test_jacobians():
    # TODO: Different weights.
    test_state = SE2State(pymlg.SE2.random(), direction="right")
    prior_state = SE2State(pymlg.SE2.random(), direction="right")
    res = gn_baseline.PriorResidualFro(["T0"], prior_state, 10, 100)
    jac_fd = res.jacobian_fd([test_state])
    _, jac_list = res.evaluate([test_state], [True])

    assert np.linalg.norm((jac_list[0] - jac_fd[0]), "fro") < 1e-5

    test_state_k = SE2State(pymlg.SE2.random(), direction="right")
    test_state_kp = SE2State(pymlg.SE2.random(), direction="right")
    dT = pymlg.SE2.random()
    res = gn_baseline.RelativePoseResidualFro(["T0"], dT, 100, 100)
    jac_fd = res.jacobian_fd([test_state_k, test_state_kp])
    _, jac_list = res.evaluate([test_state_k, test_state_kp], [True, True])

    assert np.linalg.norm((jac_list[0] - jac_fd[0]), "fro") < 1e-4
    assert np.linalg.norm((jac_list[1] - jac_fd[1]), "fro") < 1e-4


if __name__ == "__main__":
    test_jacobians()
    test_deadreckoning()
