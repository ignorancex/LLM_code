from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

from tests.examples import *


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_heat(fwd, prior, point_observation):
    """Check that analytic and numerical solution are identical.
    Check that we can measure correctly also."""
    # IC
    u0, coeffs0 = prior.sample(return_coeffs=True)
    u0 = np.squeeze(u0)

    # Analytic
    coeffsT = coeffs0 * fwd.multiplier
    uT = prior.to_time_domain(coeffsT).squeeze()
    # uT = np.squeeze(uT)
    # Numeric solution
    uT_numeric = fwd(u0)

    numeric_success = np.allclose(uT, uT_numeric, rtol=0, atol=1e-3)
    inversion_success = np.allclose(prior(prior.inverse(uT)), prior.inverse(prior(uT)), rtol=0, atol=1e-3)

    interpolant = interp1d(fwd.x, uT)
    measure_success = np.allclose(point_observation(uT),
                                  interpolant(point_observation.measurements),
                                  atol=1e-2,
                                  rtol=0)

    success = numeric_success and inversion_success and measure_success
    if success:
        assert success
    else:
        plt.plot(fwd.x, u0.real, label='IC')
        plt.plot(fwd.x, uT.real, label='Analytic FC')
        plt.plot(fwd.x, uT_numeric.real + 0.025, label='Numeric FC')
        plt.plot(prior.x, prior(prior.inverse(uT)).real + 0.05, label=str(prior) + prior.invstr + 'FC')
        plt.plot(prior.x, prior.inverse(prior(uT)).real + 0.075, label=prior.invstr + str(prior) + 'FC')
        plt.plot(prior.x, prior(uT).real, label=str(prior) + 'FC')
        # plt.plot(prior.x, prior.inverse(uT), label= prior.inv_str + 'FC')
        plt.scatter(point_observation.measurements,
                    point_observation(uT).real,
                    label='Measurements of FC',
                    marker='*',
                    s=10,
                    color='w',
                    zorder=10)
        plt.legend()
        title = f'{fwd.transform} successes: numeric {numeric_success} inversion {inversion_success} measure {measure_success}'
        plt.title(title)
        plt.show()
        assert success

