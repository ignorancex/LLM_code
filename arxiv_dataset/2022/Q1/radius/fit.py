"""Model fitting and cross-validation."""

import argparse
import copy

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import least_squares

from models import calc_cs, get_b2, get_init, get_radius

N_NORM_PARAMS = 31  # Number of normalization parameters
BEAM_ENERGIES = [180, 315, 450, 585, 720, 855]  # List of beam energies (in MeV)
SPECTROMETERS = ["A", "B", "C"]  # List of spectrometers


def read_cs_data():
    """Read raw cross section data from CrossSections.dat into dictionary."""
    # Read specific columns from data file:
    cols = {
        0: "E",  # Beam energy (in MeV)
        1: "spec",  # Spectrometer used (A, B, or C)
        3: "Q2",  # Four-momentum transfer squared (in GeV^2)
        4: "cs",  # Cross section relative to the dipole one
        5: "delta_cs",  # Point-to-point uncertainty of cross section
        6: "cs_mincut",  # Cross section with the min energy cut
        7: "cs_maxcut",  # Cross section with the max energy cut
        9: "systematic_scale",  # Factor to calculate systematic uncertainties
        10: "norms",  # Specific combination of normalization parameters
    }
    data = pd.read_csv(
        "data/CrossSections.dat", sep=" ", skiprows=1, usecols=cols.keys(), names=cols.values()
    )
    # Format normalization indices as lists:
    data["norms"] = [[int(i) for i in s.split(":")] for s in data["norms"]]
    # Add filler index:
    data["norms"] = [[0] + s if len(s) == 1 else s for s in data["norms"]]
    # Convert to dictionary of numpy arrays:
    data = data.to_dict("series")
    for key in data:
        data[key] = np.array(data[key].values)
    data["norms"] = np.stack(data["norms"])
    assert np.all(data["norms"] <= N_NORM_PARAMS)
    data["cs_sysup"] = data["cs"] * data["systematic_scale"]
    data["cs_syslow"] = data["cs"] / data["systematic_scale"]
    return data


def calc_fit_cov(jacobian):
    """Calculate covariance from Jacobian with Moore-Penrose."""
    _, singular_vals, vt_rsv = scipy.linalg.svd(jacobian, full_matrices=False)
    # Discard very small singular values:
    singular_vals = singular_vals[singular_vals > 1e-10]
    vt_rsv = vt_rsv[: len(singular_vals)]
    cov = np.dot(vt_rsv.T / singular_vals ** 2, vt_rsv)
    return cov


def covariance_bad(cov):
    """Check if covariance matrix is ill-conditioned (too collinear)."""
    sigmas = np.sqrt(np.diagonal(cov))
    # Correlation coefficients
    rhos = cov / (sigmas[None, :] * sigmas[:, None])
    np.fill_diagonal(rhos, 0)
    return np.max(np.abs(rhos)) > 0.998


def fit(train_data, test_data, order, reg_param, norms=None):
    """Fit and evaluate model with given training and test data."""

    def residuals(params, data=train_data, regularization=True):
        """Objective function."""
        energy = data["E"] / 1000  # Beam energy in GeV
        Q2 = data["Q2"]  # Four-momentum transfer squared in GeV^2
        # Normalization factor:
        if norms is None:
            norm_params = np.concatenate([[1], params[:N_NORM_PARAMS]])
        else:
            norm_params = np.concatenate([[1], norms])
        norm = np.prod(norm_params[data["norms"]], axis=1)
        # Model cross section:
        model_cs = calc_cs(energy, Q2, params[-(2 * order + 1) :], order)
        # Renormalized data cross section and its uncertainty:
        data_cs = norm * data["cs"]
        delta_cs = norm * data["delta_cs"]
        result = (data_cs - model_cs) / delta_cs
        if regularization:
            result = np.concatenate([result, np.sqrt(reg_param) * params[-2 * order :]])
        return result

    # Find best-fit parameters:
    if norms is None:
        init_params = np.array([1.0] * N_NORM_PARAMS + get_init(order))  # Initial guess
    else:
        init_params = np.array(get_init(order))  # Initial guess when not fitting normalization
    res = least_squares(residuals, init_params, method="lm", x_scale="jac")
    best_params = res.x
    chi2_train = np.sum(residuals(best_params, regularization=False) ** 2)
    chi2_test = np.sum(residuals(best_params, data=test_data, regularization=False) ** 2)
    L = np.sum(residuals(best_params, regularization=True) ** 2)
    covariance = calc_fit_cov(res.jac)
    return best_params, chi2_train, chi2_test, L, covariance


def split_data(data, indices):
    """Split data dictionaries into train and test by given indices."""
    train_data = {}
    test_data = {}
    for key in data:
        test_data[key] = data[key][tuple(indices)]
        train_data[key] = np.delete(data[key], indices, axis=0)
    return train_data, test_data


def group_validation(data, order, norms, reg_param):
    """Perform 18-fold cross-validation by experimental group."""
    val_indices = []
    for energy in BEAM_ENERGIES:
        for spectrometer in SPECTROMETERS:
            bools = np.logical_and(data["E"] == energy, data["spec"] == spectrometer)
            val_indices.append(list(np.where(bools)))
    running_train = 0
    running_test = 0
    for group in val_indices:
        train_data, test_data = split_data(data, group)
        _, chi2_train, chi2_test, _, _ = fit(train_data, test_data, order, reg_param, norms=norms)
        running_train += chi2_train
        running_test += chi2_test
    print("chi^2_train = {:.0f}, chi^2_test = {:.0f}".format(running_train / 17, running_test))


def fit_systematic_variant(key, data, order, reg_param):
    """Fit experimental systematic variant of data."""
    data_cut = copy.deepcopy(data)
    data_cut["cs"] = data[key]
    datacut_params, _, _, _, datacut_cov = fit(data_cut, data_cut, order, reg_param)
    return datacut_params[N_NORM_PARAMS:], datacut_cov[N_NORM_PARAMS:, N_NORM_PARAMS:]


def calc_systematics(b2, radius, *args):
    """Return max fit variations based on min/max energy cut and min/max systematic range."""
    variants = ["cs_mincut", "cs_maxcut", "cs_sysup", "cs_syslow"]
    b2_diff = []
    radius_diff = []
    for var in variants:
        params, cov = fit_systematic_variant(var, *args)
        b2_var, b2_sigma_var = get_b2(params, cov)
        radius_var, _ = get_radius(b2_var, b2_sigma_var)
        b2_diff.append(b2_var - b2)
        radius_diff.append(np.abs(radius_var - radius))
    return np.max(b2_diff), np.max(radius_diff)


def print_fit_params(fit_params, fit_cov):
    """Print best-fit parameters with uncertainties."""
    uncerts = np.sqrt(fit_cov.diagonal())
    print("\nBest-fit parameters:")
    print("Lambda = {:.3f} +/- {:.3f} GeV".format(fit_params[0], uncerts[0]))
    for i in range(1, (len(fit_params) + 1) // 2):
        print("alpha{} = {:.3f} +/- {:.3f}".format(i, fit_params[2 * i - 1], uncerts[2 * i - 1]))
        print("beta{} = {:.3f} +/- {:.3f}".format(i, fit_params[2 * i], uncerts[2 * i]))


def main(order, reg_param):
    """Run full analysis for given fit settings."""
    print("Model: N = {}, lambda = {}".format(order, reg_param))

    # Read the cross section data:
    data = read_cs_data()

    # Fit the full dataset:
    best_params, chi2, _, L, cov = fit(data, data, order, reg_param)
    normalizations = best_params[:N_NORM_PARAMS]
    fit_params = best_params[N_NORM_PARAMS:]
    fit_cov = cov[N_NORM_PARAMS:, N_NORM_PARAMS:]

    # Perform cross validation:
    print("\n18-fold group cross-validation results:")
    group_validation(data, order, normalizations, reg_param)

    # Print final results:
    print("\nResults obtained using the full dataset:")
    print("L = {:.0f}, chi^2 = {:.0f}".format(L, chi2))
    print("\nBest-fit normalizations:")
    print(normalizations)
    print_fit_params(fit_params, fit_cov)

    # Extract and print the radii:
    b2, b2_sigma = get_b2(fit_params, fit_cov)
    radius, radius_stat = get_radius(b2, b2_sigma)
    b2_syst, radius_syst = calc_systematics(b2, radius, data, order, reg_param)
    print("\nExtracted radii:")
    print("<b1^2> = {:.2f} +/- {:.2f} (stat) +/- {:.2f} (syst) 1/GeV^2".format(b2, b2_sigma, b2_syst))
    print("r_E = {:.3f} +/- {:.3f} (stat) +/- {:.3f} (syst) fm".format(radius, radius_stat, radius_syst))

    if covariance_bad(fit_cov):
        print("\nWarning: Covariance ill-conditioned, statistical uncertainty estimate unreliable")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fit and validate models to cross section data.")
    parser.add_argument("--order", type=int, default=5, help="order of form factor expansion (default: N=5)")
    parser.add_argument(
        "--reg_param", type=float, default=0.02, help="regularization parameter (default: lambda=0.02)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    ARGS = parse_args()
    main(ARGS.order, ARGS.reg_param)
