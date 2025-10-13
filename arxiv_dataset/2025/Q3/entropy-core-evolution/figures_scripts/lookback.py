import numpy as np
from astropy.cosmology import FlatLambdaCDM, Planck18
from scipy.optimize import root_scalar


def redshift_from_lookback_time(
    lookback_time: float, H0: float = 67.7, Om0: float = 0.307
) -> float:
    """
    Calculate the redshift corresponding to a given lookback time and cosmological parameters.

    Parameters:
    - lookback_time (float): Lookback time in gigayears (Gyr).
    - H0 (float): Hubble constant in km/s/Mpc (default is 67.7 for Planck18).
    - Om0 (float): Matter density parameter (default is 0.307 for Planck18).

    Returns:
    - float: Corresponding redshift for the given lookback time.
    """
    # Define the cosmology with provided parameters
    cosmology = FlatLambdaCDM(H0=H0, Om0=Om0)

    # Define function to solve for redshift
    def time_difference(z):
        return cosmology.lookback_time(z).value - lookback_time

    # Use a root-finding method to solve for redshift
    solution = root_scalar(time_difference, bracket=[0, 20], method="bisect")

    if solution.converged:
        return solution.root
    else:
        raise ValueError(
            "Redshift calculation did not converge. Check the input parameters."
        )


def time_diff_from_redshift_diff(z1, z2, H0: float = 67.7, Om0: float = 0.307):
    # Define the cosmology with provided parameters
    cosmology = FlatLambdaCDM(H0=H0, Om0=Om0)

    # Define function to solve for redshift
    diff = cosmology.lookback_time(z1) - cosmology.lookback_time(z2)

    return diff.value


def inverse_hubble_time(redshift: float, H0: float = 67.7, Om0: float = 0.307) -> float:
    # Define the cosmology with provided parameters
    cosmology = FlatLambdaCDM(H0=H0, Om0=Om0)

    return cosmology.H(redshift).to("1/Gyr").value


# Example usage:
# lookback_time = 5.0  # in Gyr
# redshift = redshift_from_lookback_time(lookback_time)
# print(f"The redshift for a lookback time of {lookback_time} Gyr is approximately {redshift:.3f}.")

# for z in np.linspace(0, 10, 10):
#     print(z, inverse_hubble_time(z))
