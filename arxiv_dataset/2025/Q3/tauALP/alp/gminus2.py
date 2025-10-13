import numpy as np
from scipy import integrate


def sigma_quadrature(measure_1, measure_2, digits=0):
    sigma = np.sqrt(measure_1[1] ** 2 + measure_2[1] ** 2)
    if digits < 1:
        return sigma
    coeff_digits = np.power(10, digits - 1)
    power_normalise = np.power(10, -np.floor(np.log10(sigma)))
    return (
        np.rint(sigma * coeff_digits * power_normalise) / coeff_digits / power_normalise
    )


def delta_amu(measure_1, measure_2, digits=0):
    return np.array(
        [
            measure_1[0] - measure_2[0],
            sigma_quadrature(measure_1, measure_2, digits=digits),
        ]
    )


def weighted_average(measure_1, measure_2):
    """Compute the weighted average between two measures, expressed such that (value +- error) <==> (measure_i[0] +- measure_i[1])"""
    vals = np.array([measure_1[0], measure_2[0]])
    sigs = np.array([measure_1[1], measure_2[1]])
    variance = np.sum(1 / np.power(sigs, 2))
    value = np.sum(vals / np.power(sigs, 2)) / variance
    return np.array([value, np.power(variance, -0.5)])


M_MU = 105.66e-3  # GeV, PDG

SM_VALUE = np.array([116591810e-11, 43.0e-11])
SM_2025_VALUE = np.array([116592033e-11, 62.0e-11])  # SM value for 2025

BNL_VALUE = np.array([116592089e-11, 63.0e-11])
FNAL_VALUE = np.array([116592040e-11, 54.0e-11])
FNAL_2023_VALUE = np.array([116592059e-11, 22.0e-11])
FNAL_2025_VALUE = np.array([116592059e-11, 22.0e-11 / 2])

BMW_VALUE = np.array([116591954e-11, 55.0e-11])
CMD_DISPERSIVE = np.array([116591810e-11 + 379.35e-10 - 364.5e-10, 43.0e-11])
WORLD_AVERAGE = weighted_average(FNAL_VALUE, BNL_VALUE)
WORLD_AVERAGE_2023 = weighted_average(FNAL_2023_VALUE, BNL_VALUE)
WORLD_AVERAGE_2025 = weighted_average(FNAL_2025_VALUE, BNL_VALUE)

DELTA_BNL = delta_amu(BNL_VALUE, SM_VALUE, digits=2)
DELTA_FNAL = delta_amu(FNAL_VALUE, SM_VALUE, digits=2)
DELTA_FNAL_2023 = delta_amu(FNAL_2023_VALUE, SM_VALUE, digits=2)

DELTA_BNL_SM_2025 = delta_amu(BNL_VALUE, SM_2025_VALUE, digits=2)
DELTA_FNAL_SM_2025 = delta_amu(FNAL_VALUE, SM_2025_VALUE, digits=2)
DELTA_FNAL_2023_SM_2025 = delta_amu(FNAL_2023_VALUE, SM_2025_VALUE, digits=2)

DELTA_COMBINED = delta_amu(WORLD_AVERAGE, SM_VALUE, digits=2)
DELTA_COMBINED_SM_2025 = delta_amu(WORLD_AVERAGE, SM_2025_VALUE, digits=2)
DELTA_BMW = delta_amu(WORLD_AVERAGE, BMW_VALUE, digits=2)
DELTA_CMD = delta_amu(WORLD_AVERAGE, CMD_DISPERSIVE, digits=2)

DELTA_COMBINED_2023 = delta_amu(WORLD_AVERAGE_2023, SM_VALUE, digits=2)
DELTA_COMBINED_2023_SM_2025 = delta_amu(WORLD_AVERAGE_2023, SM_2025_VALUE, digits=2)
DELTA_BMW_2023 = delta_amu(WORLD_AVERAGE_2023, BMW_VALUE, digits=2)
DELTA_CMD_2023 = delta_amu(WORLD_AVERAGE_2023, CMD_DISPERSIVE, digits=2)

DELTA_BMW_2025 = delta_amu(WORLD_AVERAGE_2025, BMW_VALUE, digits=2)
DELTA_CMD_2025 = delta_amu(WORLD_AVERAGE_2025, CMD_DISPERSIVE, digits=2)
DELTA_COMBINED_2025 = delta_amu(WORLD_AVERAGE_2025, SM_2025_VALUE, digits=2)


DELTA_a_electron_Berkeley = np.array([-0.88e-12, 0.36e-12])
DELTA_a_electron_LKB = np.array([0.48e-12, 0.30e-12])


def delta_amu_fnal(
    mz,
    delta_amu,
    error,
    factor,
    upper_limit=False,
    higher_point=1e100,
    significance=1.0,
):
    """Accepts 4 arguments:
      - mz: array of mass values of dark photon where the kinetic mixing coupling should be computed;
      - delta_amu: value of delta_a_mu from FNAL;
      - error: if it is a number, it is treated as the symmetric error on delta_a_mu, if it is an array of shape (2,), then it is treated as an asymmetric error on delta_a_mu;
      - factor: it is the factor such that delta_amu = factor * coupling^2 * m_ratio^2 * integrand_function(m_ratio) with m_ratio = m_mu/mz
      - upper_limit: if True it computes the upper limit from the error (it takes the [0] element if it is an array), multiplying it by the significance;
      - higher_point: it is the higher limit, given so that it is possible to consistently fill between the lines in the upper limit case.
      - significance: significance to multiply to the error or to consider for upper limit
    The return is a list of two arrays y_low, y_up, representing the kinetic mixing values evaluated at the mz values (y_up is an array of higher_point in case upper_limit == True)
    """
    if upper_limit:
        if isinstance(error, list) or isinstance(error, tuple):
            error = error[0]
        bounds = [delta_amu + significance * error, np.inf]
    elif (isinstance(error, list) or isinstance(error, tuple)) and len(error) == 2:
        bounds = [
            delta_amu - significance * error[0],
            delta_amu + significance * error[1],
        ]
    elif isinstance(error, float) or isinstance(error, int):
        bounds = [delta_amu - significance * error, delta_amu + significance * error]
    else:
        raise ValueError("Invalid value of 'error' argument.")
    y_values = []
    for delta_amu_at_bound in bounds:
        if np.isinf(delta_amu_at_bound):
            y_values.append(np.broadcast_to(higher_point, len(mz)))
            continue
        y_values.append(
            np.array(
                [
                    kinetic_mixing_from_delta_amu(
                        delta_amu=delta_amu_at_bound, mz=mass_z, factor=factor
                    )
                    for mass_z in mz
                ]
            )
        )
    return y_values


def delta_amu_fnal_central(mz, delta_amu, factor):
    """Accepts 3 arguments:
      - mz: array of mass values of dark photon where the kinetic mixing coupling should be computed;
      - delta_amu: value of delta_a_mu from FNAL;
      - factor: it is the factor such that delta_amu = factor * coupling^2 * m_ratio^2 * integrand_function(m_ratio) with m_ratio = m_mu/mz
    The return is an array containing the value of the kinetic mixing values evaluated at the mz values to reproduce delta_amu
    """
    return np.array(
        [
            kinetic_mixing_from_delta_amu(delta_amu=delta_amu, mz=mass_z, factor=factor)
            for mass_z in mz
        ]
    )


def plot_gminus2_region(
    ax, mz, gminus2_down, gminus2_up, sigma, color, fill=True, **kwargs
):
    """Plot the g-2 region according to pre-computed y_down and y_up values of the band of a certain sigma:
    - ax: the Axes instance to plot on;
    - mz: array of values of Z' mass;
    - gminus2_down, gminus2_up: arrays computed with delta_amu_fnal with upper_limit == False and significance == sigma;
    - sigma: the 'significance' parameter of the computation above, it will appear in the legend entry;
    - color: color of the band and of the edges;
    - kwargs: other keyword arguments to tune plotting parameters.
    """
    bound_up = None
    bound_down = None
    if sigma == 3:
        bound_up = ax.plot(
            mz, gminus2_up, color="darkgreen", alpha=1, lw=0.75
        )  # ,zorder=10)
        bound_down = ax.plot(
            mz, gminus2_down, color="darkgreen", alpha=1, lw=0.75
        )  # ,zorder=10)
    if not fill:
        return [bound_up, bound_down]
    return ax.fill_between(
        mz,
        gminus2_down,
        gminus2_up,
        facecolor=color,
        edgecolor="None",
        alpha=0.7,
        label=r"$\Delta a_\mu \pm " + "{:1.0f}".format(sigma) + r"\sigma$",
        **kwargs
    )


def compute_and_plot_gminus2_region(
    ax, mz, delta_amu, error, factor, sigmas, colors, fill=True, **kwargs
):
    """Compute and plot the g-2 region according to the passed parameters:
      - ax: the Axes instance to plot on;
      - mz, delta_amu, error, factor: parameters passed to the inner call of delta_amu_fnal, refer to it for documentation;
      - sigmas, colors: array containing the value of the argument 'significance' of delta_amu_fnal and of the argument 'sigma' of plot_gminus2_region for each call;
      - colors: array containing the value of the argument 'color' of plot_gminus2_region for each call, it needs to be of the same lenght as the sigmas parameter;
      - kwargs: other keyword arguments to tune plotting parameters, passed to plot_gminus2_region.
    It returns a list of dictionaries composed of the entries 'value_down', 'value_up' (returned from delta_amu_fnal), 'plot' (returned from plot_gminus2_region), 'sigma' (the value of the sigma parameter the computation and plotting are referring to).
    """
    return_vals = []
    for c, s in zip(colors, sigmas):
        y_down, y_up = delta_amu_fnal(
            mz=mz,
            delta_amu=delta_amu,
            error=error,
            factor=factor,
            upper_limit=False,
            significance=s,
        )
        band = plot_gminus2_region(ax, mz, y_down, y_up, s, c, fill=fill, **kwargs)
        return_vals.append(
            {"value_down": y_down, "value_up": y_up, "sigma": s, "plot": band}
        )
    return return_vals


def compute_and_plot_gminus2_central(ax, mz, delta_amu, factor, **kwargs):
    """Compute and plot the g-2 central value (with significance == 0 and sigma == 0) according to the passed parameters:
      - ax: the Axes instance to plot on;
      - mz, delta_amu, factor: parameters passed to the inner call of delta_amu_fnal_central, refer to it for documentation;
      - kwargs: other keyword arguments to tune plotting parameters, passed to an inner call of Axes.plot.
    It returns a dictionary composed of the entries 'value' (returned from delta_amu_fnal_central), 'plot' (returned from a Axes.plot call), 'sigma' = 0 for consistencies.
    """
    y_central = delta_amu_fnal(mz=mz, delta_amu=delta_amu, factor=factor)
    line = ax.plot(mz, y_central, **kwargs)
    return {"value": y_central, "sigma": 0.0, "plot": line}
