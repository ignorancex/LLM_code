import re
from typing import List

import numpy as np
import pandas as pd
import torch
from numpy.typing import ArrayLike

from gravann.util.constants import UNITLESS_TO_ACCELERATION, UNITLESS_TO_METER


def convert_pandas_altitudes(
        result_df: pd.DataFrame, altitudes: List[float], convert_height: bool = False, only_height: bool = False,
) -> pd.DataFrame:
    """Converts the entries of the given pandas DataFrame. Operates in-place!

    Args:
        result_df: the result dataframe for the validation & training containing all information
        altitudes: the altitudes used in the configuration as extra-validation heights, e.g. [0.25, 0.5, 1.0]
        convert_height: if True, replace the normalized heights with the real heights, e.g. 0.05 --> 0.31224...
        only_height: if True, replace the names with only the height, e.g. relRMSE@0.05 --> only 0.05
            (can unambiguous in the end!)


    Returns:
        pandas DataFrame

    """
    # Legacy stuff
    if "Sample" in result_df:
        result_df.rename(columns={"Sample": "sample"}, inplace=True)
    if "Target Sampler Domain" in result_df:
        result_df.rename(columns={"Target Sampler Domain": "sample_domain"}, inplace=True)

    # Iterate over all possible samples
    for sample_name in UNITLESS_TO_METER.keys():
        altitude2metric = UNITLESS_TO_METER[sample_name]
        value2metric = UNITLESS_TO_ACCELERATION[sample_name]
        # First adapt validation results
        for column in [x for x in result_df.columns if x.lower().startswith(("Normalized L1 Loss", "RMSE", "relRMSE"))]:
            result_df.loc[result_df["sample"] == sample_name, column] *= value2metric
        # Second adapt the columns from the sampling domain
        factor = altitude2metric if convert_height else 1.0
        result_df.loc[result_df["sample"] == sample_name, "sample_domain"] = \
            result_df.loc[result_df["sample"] == sample_name, "sample_domain"].apply(
                lambda val: np.fromstring(val.replace('[', '').replace(']', ''), sep=',') * factor
            )
        # Rename columns of altitudes to real values
        for column in [x for x in result_df.columns if "@Altitude_" in x]:
            match = re.search(r"(.*@)Altitude_(\d+)", column) \
                if not only_height else re.search(r"().*@Altitude_(\d+)", column)
            if match is not None:
                altitude_prefix = match.group(1)
                altitude_id = int(match.group(2))
                result_df.rename(columns={
                    column: f"{altitude_prefix}{altitudes[altitude_id] * factor}"
                }, inplace=True)
    return result_df


def convert_altitude(sample: str, altitudes: ArrayLike, forward: bool = True, unit: str = 'm') -> ArrayLike:
    """Converts the unitless altitudes of an ArrayLike to metric.

    Args:
        sample: the sample
        altitudes: the altitudes to convert
        unit: defaults to [m] meter, kilometre also possible [km]
        forward: if true (default) does the above, otherwise the function converts from meter to unitless

    Returns:
        the converted altitudes in meter [m] or [km] depending on unit

    """
    units = {
        'm': 1.0,
        'km': 1000.0
    }
    try:
        conversion_constant = UNITLESS_TO_METER[sample]
        return altitudes * conversion_constant / units[unit] if forward \
            else (altitudes * units[unit]) / conversion_constant
    except KeyError:
        raise NotImplementedError(f"The requested sample {sample} does not yet have a conversion factor!")


def convert_acceleration(sample: str, acceleration: torch.Tensor, forward: bool = True) -> torch.Tensor:
    """Converts the given unitless accelerations to SI units [m/s^2].

    Args:
        sample: the sample body's name
        acceleration: the accelerations to converts (cartesian vector)
        forward: if true (default) does the above, otherwise the function converts from meter to unitless

    Returns:
        the converted acceleration in [m/s^2]

    """
    try:
        conversion_constant = UNITLESS_TO_ACCELERATION[sample]
        return acceleration * conversion_constant if forward else acceleration / conversion_constant
    except KeyError:
        raise NotImplementedError(f"The requested sample {sample} does not yet have a conversion factor!")
