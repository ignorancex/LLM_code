from typing import Any

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from galpy.orbit import Orbit
from scipy.stats import norm
from tqdm import tqdm

from plato.utils import get_abspath

KINEMATICS_TABLE = pd.read_csv(
    get_abspath() + "data/external/kinematic_characteristics.csv"
)


def get_kinematic_parameter(
    Z: float | np.ndarray,
    R: float | np.ndarray,
    progress: bool = True,
) -> pd.DataFrame:
    """
    Get the kinematic characteristics of the Galactic components
    Thin Disk, Thick Disk, and Halo at a given position in the
    Galactic plane (Z, R), following Chen2021.

    Parameters
    ----------
    Z : float | np.ndarray
        The distance from the Galactic plane in kpc.
    R : float | np.ndarray
        The distance from the Galactic center in the
        Galactic plane in kpc.
    progress : bool, optional
        If True, show a progress bar, by default True.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the kinematic characteristics
        of the Galactic component at the given position. The
        characteristics are:
            - sigma_U: The velocity dispersion in U in km/s.
            - sigma_V: The velocity dispersion in V in km/s.
            - sigma_W: The velocity dispersion in W in km/s.
            - V_asym: The mean velocity in V in km/s.
            - X: The relative fraction of the component.
        The output dataframe contains each of these values
        for the Thin Disk, Thick Disk, and Halo components, with
        the suffixes _D, _TD, and _H, respectively.

    """
    # check input types
    match (Z, R):
        case (np.ndarray(), np.ndarray()) if Z.shape == R.shape:
            pass  # both are arrays of the same shape
        case (float() | int(), float() | int()):
            pass  # both are scalars
        case (float() | int(), np.ndarray()):
            Z = np.full(R.shape, Z)  # make Z an array of same shape as R
        case (np.ndarray(), float() | int()):
            R = np.full(Z.shape, R)  # make R an array of same shape as Z
        case _:
            raise TypeError(
                "Z and R must be either both arrays of the same shape or scalar."
            )

    # clip Z and R to the min and max values in the kinematics table.
    Z = np.clip(
        np.abs(Z),
        a_min=KINEMATICS_TABLE["Z"].min(),
        a_max=KINEMATICS_TABLE["Z"].max(),
    )
    R = np.clip(
        R,
        a_min=KINEMATICS_TABLE["R"].min(),
        a_max=KINEMATICS_TABLE["R"].max(),
    )

    # find the correct bin in the kinematics table
    kinematics_rows = []
    for z, r in tqdm(
        np.broadcast(Z, R),
        total=len(Z),
        desc=f"Retrieving Kinematic Parameter: ",
        disable=not progress,
    ):
        matching_row = KINEMATICS_TABLE[
            (KINEMATICS_TABLE["Z"] <= z) & (KINEMATICS_TABLE["R"] <= r)
        ].iloc[-1]

        kinematics_rows.append(matching_row)

    kinematics_table = pd.DataFrame(kinematics_rows).reset_index(drop=True)
    return kinematics_table


def calculate_galactic_quantities(
    dataframe: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the Galactic velocities
        - U: The heliocentric velocity in the direction of the Galactic center
        - V: The heliocentric velocity in the direction of the Galactic rotation
        - W: The heliocentric velocity in the direction of the North Galactic Pole
    and the Galactic Coordinates
        - R: The distance from the Galactic center in the Galactic plane
        - Z: The distance from the Galactic plane
    using galpy.

    The input dataframe should contain the following columns:
        - ra: Right ascension in degrees
        - dec: Declination in degrees
        - parallax: Parallax in mas
        - pmra: Proper motion in RA in mas/yr
        - pmdec: Proper motion in Dec in mas/yr
        - radial_velocity: Radial velocity in km/s


    Parameters
    ----------
    dataframe : pd.DataFrame
        The input dataframe.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple containing the Galactic velocities
        U, V, and W in the heliocentric frame in km/s, and the
        Galactic coordinates R and Z in kpc.

    """
    for key in ["ra", "dec", "parallax", "pmra", "pmdec", "radial_velocity"]:
        if key not in dataframe.columns:
            raise ValueError(f"Dataframe must contain {key!r}.")

    coord = SkyCoord(
        ra=dataframe["ra"].to_numpy() * u.degree,
        dec=dataframe["dec"].to_numpy() * u.degree,
        distance=(1000 / dataframe["parallax"].to_numpy()) * u.pc,
        pm_ra_cosdec=dataframe["pmra"].to_numpy() * u.mas / u.yr,
        pm_dec=dataframe["pmdec"].to_numpy() * u.mas / u.yr,
        radial_velocity=dataframe["radial_velocity"].to_numpy() * u.km / u.s,
    )

    orbits = Orbit(
        vxvv=[
            coord.ra,
            coord.dec,
            coord.distance,
            coord.pm_ra_cosdec,
            coord.pm_dec,
            coord.radial_velocity,
        ],
        radec=True,
    )

    return orbits.U(), orbits.V(), orbits.W(), orbits.z(), orbits.R()


def component_probability(
    U: float | np.ndarray,
    V: float | np.ndarray,
    W: float | np.ndarray,
    sigma_U: float | np.ndarray,
    sigma_V: float | np.ndarray,
    sigma_W: float | np.ndarray,
    V_asym: float | np.ndarray,
    log: bool = False,
) -> float | np.ndarray:
    """
    Calculate the probability of a star with velocities
    U, V, and W to belong to a given component characterized
    by the parameters sigma_U, sigma_V, sigma_W, and V_asym.

    Parameters
    ----------
    U : float | np.ndarray
        The Galactic velocity U (pointed towards the galactic center)
        in km/.
    V : float | np.ndarray
        The Galactic velocity V (pointed in the direction of
        the galactic rotation) in km/s.
    W : float | np.ndarray
        The Galactic velocity W (pointed towards the
        North Galactic Pole) in km/s.
    sigma_U : float | np.ndarray
        The velocity dispersion in U in km/s.
    sigma_V : float | np.ndarray
        The velocity dispersion in V in km/s.
    sigma_W : float | np.ndarray
        The velocity dispersion in W in km/s.
    V_asym : float | np.ndarray
        The mean velocity in V in km/s.
    log : bool, optional
        If True, return the log of the probability instead,
        by default False

    Returns
    -------
    float | np.ndarray
        The probability of the star to belong to the component.

    """
    if log:
        U_prob = norm.logpdf(U, loc=0, scale=sigma_U)
        V_prob = norm.logpdf(V, loc=V_asym, scale=sigma_V)
        W_prob = norm.logpdf(W, loc=0, scale=sigma_W)
        return U_prob + V_prob + W_prob

    U_prob = norm.pdf(U, loc=0, scale=sigma_U)
    V_prob = norm.pdf(V, loc=V_asym, scale=sigma_V)
    W_prob = norm.pdf(W, loc=0, scale=sigma_W)
    return U_prob * V_prob * W_prob


def relative_probability(
    U: float | np.ndarray,
    V: float | np.ndarray,
    W: float | np.ndarray,
    kinematic_parameter: pd.DataFrame,
    component_1: str,
    component_2: str,
) -> float | np.ndarray:
    """
    Calculate the relative probability of a star to belong to
    component 1 with respect to component 2. The probability is
    calculated as the ratio of the probability of the star to belong
    to component 1 and the probability of the star to belong to component 2,
    weighted by the relative fraction of the components.

    The components must be either provided by strings found in the
    component_dict or by stand-alone dictionaries with the necessary
    parameters.

    Classification based on Bensby et al. (2003) and Bensby et al. (2014).

    If component_dict is None, the default components are:
    - Thin Disk: X=0.94, sigma_U=35, sigma_V=20, sigma_W=16, V_asym=-15
    - Thick Disk: X=0.06, sigma_U=67, sigma_V=38, sigma_W=35, V_asym=-46
    - Halo: X=0.0015, sigma_U=160, sigma_V=90, sigma_W=90, V_asym=-220

    Parameters
    ----------
    U : float | np.ndarray
        The Galactic velocity U in km/s.
    V : float | np.ndarray
        The Galactic velocity V in km/s.
    W : float | np.ndarray
        The Galactic velocity W in km/s.
    kinematic_parameter : pd.DataFrame
        The kinematic parameters of the Galactic components
        at the given position (Z, R), given as a dataframe.
        The dataframe should contain the following columns:
            - sigma_U{component}: The velocity dispersion in U in km/s
            - sigma_V{component}: The velocity dispersion in V in km/s
            - sigma_W{component}: The velocity dispersion in W in km/s
            - V_asym{component}: The mean velocity in V in km/s
            - X{component}: The relative fraction of the component
        for each component, where {component} is one of
            - _D for the Thin Disk
            - _TD for the Thick Disk
            - _H for the Halo
    component_1 : str
        The first component, must be "Thin Disk", "Thick Disk", or "Halo".
    component_2 : str
        The second component, must be "Thin Disk", "Thick Disk", or "Halo".

    Returns
    -------
    float | np.ndarray
        The relative probability of the star to belong to component 1
        with respect to component 2.

    """
    component_map = {
        "thin disk": "_D",
        "thick disk": "_TD",
        "halo": "_H",
    }

    for component in [component_1, component_2]:
        if component.lower() not in component_map.keys():
            raise ValueError(
                f"Component {component!r} not found. Must be one of "
                f"{list(component_map.keys())}."
            )
    component_1 = component_map[component_1.lower()]
    component_2 = component_map[component_2.lower()]

    probs = []
    for comp in [component_1, component_2]:
        prob = np.asarray(
            component_probability(
                U,
                V,
                W,
                sigma_U=kinematic_parameter[f"sigma_U{comp}"].to_numpy(),
                sigma_V=kinematic_parameter[f"sigma_V{comp}"].to_numpy(),
                sigma_W=kinematic_parameter[f"sigma_W{comp}"].to_numpy(),
                V_asym=kinematic_parameter[f"V_asym{comp}"].to_numpy(),
            )
        )
        probs.append(prob)
    prob_1, prob_2 = probs

    # calculate relative probabilities, deal with division by zero
    relative_probabilities = np.full_like(prob_1, np.nan, dtype=float)

    relative_probabilities[(prob_1 == 0.0) | (prob_2 > 0.0)] = 0
    relative_probabilities[(prob_1 > 0.0) & (prob_2 == 0.0)] = np.inf
    relative_probabilities[(prob_1 != 0.0) & (prob_2 != 0.0)] = (
        kinematic_parameter[f"X{component_1}"].to_numpy()
        / kinematic_parameter[f"X{component_2}"].to_numpy()
        * prob_1[(prob_1 != 0.0) & (prob_2 != 0.0)]
        / prob_2[(prob_1 != 0.0) & (prob_2 != 0.0)]
    )
    return relative_probabilities


def classify_stars(
    dataframe: pd.DataFrame,
    lower_cut: float = 0.1,
    upper_cut: float = 10,
    include_galactic_quantities: bool = False,
    include_probabilities: bool = False,
    overwrite: bool = False,
    raise_errors: bool = True,
) -> pd.DataFrame:
    """
    Classify stars into different populations based on their Galactic
    velocities. A new column 'Population' is added to the dataframe,
    containing the classification. The classification is based on
    the relative probabilities of the stars to belong to the Thin Disk,
    Thick Disk, and Halo components.

    The input dataframe should contain the following columns:
        - ra: Right ascension in degrees
        - dec: Declination in degrees
        - parallax: Parallax in mas
        - pmra: Proper motion in RA in mas/yr
        - pmdec: Proper motion in Dec in mas/yr
        - radial_velocity: Radial velocity in km/s

    The classes are defined as follows:
        - Thin Disk
        - Thick Disk Candidate
        - Thick Disk
        - Halo Candidate
        - Halo

    The classification is based on the relative probabilities
    of the stars to belong to the Thin Disk, Thick Disk, and
    Halo components. Specifically, the the stars are classified
    to a component if the relative probability is below a lower
    cut or above an upper cut. For in-between values, the stars
    are classified as candidates.

    For Halo stars and Halo candidates, the stars are checked
    if they are classified as Thick Disk beforehand. If so, an
    error is raised. Deactivate this check by setting raise_errors
    to False.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input dataframe.
    lower_cut : float, optional
        The lower cut for the relative probability, if
        e.g. the relative probabiltiy Thick Disk/Thin
        Disk is below this value, the star is classified
        as Thin Disk, by default 0.1.
    upper_cut : float, optional
        The upper cut for the relative probability, if
        e.g. the relative probabiltiy Thick Disk/Thin
        Disk is above this value, the star is classified
        as Thick Disk, by default 10.
    include_galactic_quantities : bool, optional
        If True, include the Galactic velocities U, V,
        and W, and the total non-circular velocity
        UW = sqrt(U^2 + W^2), as well as the Galactic
        coordinates R and Z in the output dataframe,
        by default False.
    include_probabilities : bool, optional
        If True, include the relative probabilities
        Thick Disk/Thin Disk and Thick Disk/Halo in the
        output dataframe, by default False. If True, the
        columns names are 'TD/D' and 'TD/H'.
    overwrite : bool, optional
        If True, overwrite the 'Population' column if it
        already exists, by default False.
    raise_errors : bool, optional
        If True, raise errors if some stars are not classified
        or if some Halo stars or Halo candidates are classified
        as Thick Disk beforehand, by default True.

    Returns
    -------
    pd.DataFrame
        The input dataframe with a new column 'Population'
        containing the classification.

    Raises
    ------
    ValueError
        If the dataframe already contains a 'Population' column
        and overwrite is False.
    RuntimeError
        If some some Halo stars or Halo candidates
        are not classified as Thick Disk beforehand.
    RuntimeError
        If some stars are not classified.
    """

    if "Population" in dataframe.columns and not overwrite:
        raise ValueError(
            "Dataframe already contains a 'Population' column, "
            "use overwrite=True to overwrite."
        )

    new_df = dataframe.copy()

    # calculate Galactic velocities
    u, v, w, z, r = calculate_galactic_quantities(dataframe)

    # get kinematic parameters
    kinematic_parameter = get_kinematic_parameter(z, r)

    # calculate relative probabilities between Thick Disk and Thin Disk
    td_d = relative_probability(
        u,
        v,
        w,
        kinematic_parameter,
        "Thick Disk",
        "Thin Disk",
    )
    # calculate relative probabilities between Thick Disk and Halo
    h_td = relative_probability(
        u,
        v,
        w,
        kinematic_parameter,
        "Halo",
        "Thick Disk",
    )

    if include_galactic_quantities:
        new_df["U"] = u
        new_df["V"] = v
        new_df["W"] = w
        new_df["UW"] = np.sqrt(u**2 + w**2)
        new_df["R"] = r
        new_df["Z"] = z

    if include_probabilities:
        new_df["TD/D"] = td_d
        new_df["TD/H"] = np.where(h_td == 0, np.inf, 1 / h_td)

    # classify stars
    populations = {
        "Thin Disk": (td_d <= lower_cut),
        "Thick Disk Candidate": ((td_d > lower_cut) & (td_d < upper_cut)),
        "Thick Disk": (td_d >= upper_cut),
        "Halo Candidate": ((h_td > lower_cut) & (h_td < upper_cut)),
        "Halo": (h_td >= upper_cut),
    }

    new_df["Population"] = np.nan
    new_df["Population"] = new_df["Population"].astype("str")
    for population, mask in populations.items():
        # Halo and Halo Candidate stars should be subset
        # of Thick Disk Candidate stars that were already classified
        if population in ["Halo Candidate", "Halo"]:
            if (
                {"Thick Disk", "Halo Candidate"}
                < set(new_df["Population"][mask].unique())
            ) and raise_errors:
                raise RuntimeError(
                    "Some Halo stars or Halo candidates are not classified "
                    "as Thick Disk beforehand."
                )
        new_df.loc[mask, "Population"] = population

    # check if all stars are classified
    if (new_df["Population"].isna()).any() and raise_errors:
        raise RuntimeError(
            "Some stars are not classified. A common cause are "
            "NaN values in the input dataframe. Check probabilities using "
            "include_probabilities=True."
        )
    return new_df


def classify_by_chemistry(
    dataframe: pd.DataFrame,
    metallicity_column: str = "[Fe/H]",
    alpha_column: str = "[alpha/Fe]",
    output_column: str = "Population by Chemistry",
    return_full: bool = True,
    overwrite: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Classify stars into their components based on their metallicity and alpha
    abundance, following Horta2022. This is a convenience function to classify
    stars in a dataframe. For a detailed explanation of the classification,
    see the function chemistry_classifications.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input dataframe, containing the metallicity and alpha columns.
    metallicity_column : str, optional
        Name of the metallicity column, by default "[Fe/H]".
    alpha_column : str, optional
        Name of the alpha abundance column, by default "[alpha/Fe]".
    output_column : str, optional
        Name of the output column, by default "Population by Chemistry".
    return_full : bool, optional
        If True, return the full dataframe with the classification column added,
        otherwise return only the classification column as a separate dataframe,
        by default True.
    overwrite : bool, optional
        If True, overwrite the output column if it already exists, by default True.
    **kwargs
        Additional keyword arguments to pass to the chemistry_classifications function.

    Returns
    -------
    pd.DataFrame
        The output dataframe. Either the full dataframe with the classification
        column added, or only the classification column, depending on the value
        of return_full.
    """

    metallicity = dataframe[metallicity_column]
    alpha = dataframe[alpha_column]

    classification = pd.DataFrame(
        [chemistry_classfications(m, a, **kwargs) for m, a in zip(metallicity, alpha)],
        columns=[output_column],
    )

    if return_full:
        if overwrite and output_column in dataframe.columns:
            dataframe[output_column] = classification
            return dataframe
        else:
            return pd.concat([dataframe, classification], axis=1)
    return classification


def chemistry_classfications(
    metallicity: float,
    alpha: float,
    c: float = 0.14,
) -> str | None:
    """
    Classify a star into their components based on their metallicity and alpha
    abundance, following Horta2022.

    Parameters
    ----------
    metallicity : float
        The [Fe/H] metallicity of the star.
    alpha : float
        The [alpha/Fe] abundance of the star.
    c : float, optional
        Thin disk - thick disk separation [alpha/Fe] value for low metallicity stars.
        The value changes by a linear funtion of metallicity for higher metallicity
        stars, defined as A * [Fe/H] + B. A and B are calculated from the values in
        Horta2022, and hardcoded here for simplicity. The default is 0.13.

    Returns
    -------
    str | None
        The classification of the star, either "Thin Disk", "Thick Disk", or "Halo".
        If the classification is not possible, because metallicity or alpha are NaN,
        return None.

    """
    A = -1 / 6
    B = 0.1 * A + c - 0.05

    if np.isnan(metallicity) or np.isnan(alpha):
        return None

    if metallicity < -1:
        return "Halo"
    elif metallicity < -0.8:
        return "Thick Disk" if alpha > c else "Halo"
    elif metallicity < -0.4:
        return "Thick Disk" if alpha > c else "Thin Disk"
    elif metallicity < 0.2:
        return "Thick Disk" if alpha > A * metallicity + B else "Thin Disk"
    else:
        return "Thick Disk" if alpha > A * 0.2 + B else "Thin Disk"
