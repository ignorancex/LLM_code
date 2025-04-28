import numpy as np
from scipy.integrate import quad
import pytest

from qcd.splitting_functions import P_qq, P_gq, P_qg, P_gg
from qcd.splitting_functions import Phat_qq, Phat_gq, Phat_qg, Phat_gg

EPSILON = 1e-5  # Regularization parameter


@pytest.mark.parametrize(
    "splitting_function, expected_integral, description",
    [
        (lambda x: P_qq(x, epsilon=EPSILON, regularize=True),
         0, "P_qq normalization"),
        (lambda x: P_gg(x, epsilon=EPSILON, regularize=True,
                        num_flavors=0),
         0, "P_gg normalization"),
    ],
)
def test_integral_normalization(splitting_function, expected_integral,
                                description):
    """Test normalization of splitting functions."""
    integral = quad(splitting_function, 0, EPSILON)[0]
    integral += quad(splitting_function, EPSILON, 1-EPSILON)[0]
    integral += quad(splitting_function, 1-EPSILON, 1)[0]
    assert np.isclose(
        integral, expected_integral, atol=1e-2
    ), f"{description} test failed with integral={integral:.5f}"


# @pytest.mark.parametrize(
#     "description",
#     ["momentum conservation"],
# )
# def test_momentum_conservation(description):
#     """Test momentum conservation sum rule."""
#     P_qq_momentum = quad(lambda x: x * P_qq(x, epsilon=EPSILON, regularize=True), 0, 1)[0]
#     P_qg_momentum = quad(lambda x: x * P_qg(x, epsilon=EPSILON, regularize=True), 0, 1)[0]
#     P_gq_momentum = quad(lambda x: x * P_gq(x, epsilon=EPSILON, regularize=True), 0, 1)[0]
#     P_gg_momentum = quad(lambda x: x * P_gg(x, epsilon=EPSILON, regularize=True), 0, 1)[0]

#     total_quark_momentum = P_qq_momentum + P_gq_momentum
#     total_gluon_momentum = P_qg_momentum + P_gg_momentum

#     assert np.isclose(
#         total_quark_momentum, 1, atol=1e-2
#     ), f"Quark {description} test failed with {total_quark_momentum} != 1"
#     assert np.isclose(
#         total_gluon_momentum, 1, atol=1e-2
#     ), f"Gluon {description} test failed with {total_gluon_momentum} != 1"


@pytest.mark.parametrize(
    "N, expected_function, splitting_function, description",
    [
        (3, Phat_gq, P_gq, "P_gq moment N=3"),
        (3, Phat_qq, P_qq, "P_qq moment N=3"),
        (3, Phat_qg, P_qg, "P_qg moment N=3"),
        (3, Phat_gg, P_gg, "P_gg moment N=3"),
    ],
)
def test_moments(N, expected_function, splitting_function,
                 description):
    """Test specific moments of splitting functions."""
    # Compute the Nth moment numerically
    opts = {}
    if splitting_function == P_gg:
        opts = {'num_flavors': 0}

    # Regularized moment
    reg_integrand = lambda x: x**(N-1) * splitting_function(x,
                                                    epsilon=EPSILON,
                                                    regularize=True,
                                                    **opts)
    reg_moment = quad(reg_integrand, 0, 1)[0]

    # Closed form for moment
    expected_moment = expected_function(N, **opts)

    assert np.isclose(
        reg_moment, expected_moment, atol=1e-2
    ), f"{description} test failed with "\
        f"regularized_momment={reg_moment:.5f} and "\
        f"expected_moment={expected_moment:.5f}"


if __name__ == "__main__":
    pytest.main()
