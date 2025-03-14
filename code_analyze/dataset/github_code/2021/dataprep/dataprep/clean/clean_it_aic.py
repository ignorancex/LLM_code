"""
Clean and validate a DataFrame column containing
Italian code for identification of drugs (AICs).
"""

# pylint: disable=too-many-lines, too-many-arguments, too-many-branches
from typing import Any, Union
from operator import itemgetter

import dask.dataframe as dd
import numpy as np
import pandas as pd

from stdnum.it import aic
from ..progress_bar import ProgressBar
from .utils import NULL_VALUES, to_dask


def clean_it_aic(
    df: Union[pd.DataFrame, dd.DataFrame],
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Italian code for identification of drugs (AICs) type data in a DataFrame column.

    Parameters
    ----------
        df
            A pandas or Dask DataFrame containing the data to be cleaned.
        col
            The name of the column containing data of AIC type.
        output_format
            The output format of standardized number string.
            If output_format = 'compact', return string without any separators or whitespace.
            If output_format = 'standard', return string with proper separators and whitespace.
            If output_format = 'base10', convert a BASE32 representation to a BASE10 one.
            If output_format = 'base32', convert a BASE10 representation to a BASE32 one.
            Note: in the case of AIC, the compact format is the same as the standard one.
                And 'compact' may contain both BASE10 and BASE32 represatation.

            (default: "standard")
        inplace
           If True, delete the column containing the data that was cleaned.
           Otherwise, keep the original column.

           (default: False)
        errors
            How to handle parsing errors.
            - ‘coerce’: invalid parsing will be set to NaN.
            - ‘ignore’: invalid parsing will return the input.
            - ‘raise’: invalid parsing will raise an exception.

            (default: 'coerce')
        progress
            If True, display a progress bar.

            (default: True)

    Examples
    --------
    Clean a column of AIC data.

    >>> df = pd.DataFrame({
            "aic": [
            '000307052',
            '999999',]
            })
    >>> clean_it_aic(df, 'aic')
            aic             aic_clean
    0       000307052       000307052
    1       999999          NaN
    """

    if output_format not in {"compact", "standard", "base10", "base32"}:
        raise ValueError(
            f"output_format {output_format} is invalid. "
            'It needs to be "compact", "standard", "base10" or "base32".'
        )

    # convert to dask
    df = to_dask(df)

    # To clean, create a new column "clean_code_tup" which contains
    # the cleaned values and code indicating how the initial value was
    # changed in a tuple. Then split the column of tuples and count the
    # amount of different codes to produce the report
    df["clean_code_tup"] = df[column].map_partitions(
        lambda srs: [_format(x, output_format, errors) for x in srs],
        meta=object,
    )

    df = df.assign(
        _temp_=df["clean_code_tup"].map(itemgetter(0)),
    )

    df = df.rename(columns={"_temp_": f"{column}_clean"})

    df = df.drop(columns=["clean_code_tup"])

    if inplace:
        df[column] = df[f"{column}_clean"]
        df = df.drop(columns=f"{column}_clean")
        df = df.rename(columns={column: f"{column}_clean"})

    with ProgressBar(minimum=1, disable=not progress):
        df = df.compute()

    return df


def validate_it_aic(
    df: Union[str, pd.Series, dd.Series, pd.DataFrame, dd.DataFrame],
    column: str = "",
) -> Union[bool, pd.Series, pd.DataFrame]:
    """
    Validate if a data cell is AIC in a DataFrame column. For each cell, return True or False.

    Parameters
    ----------
    df
            A pandas or Dask DataFrame containing the data to be validated.
    col
            The name of the column to be validated.
    """
    if isinstance(df, (pd.Series, dd.Series)):
        return df.apply(aic.is_valid)
    elif isinstance(df, (pd.DataFrame, dd.DataFrame)):
        if column != "":
            return df[column].apply(aic.is_valid)
        else:
            return df.applymap(aic.is_valid)
    return aic.is_valid(df)


def _format(val: Any, output_format: str = "standard", errors: str = "coarse") -> Any:
    """
    Reformat a number string with proper separators and whitespace.

    Parameters
    ----------
    val
           The value of number string.
    output_format
           If output_format = 'compact', return string without any separators or whitespace.
           If output_format = 'standard', return string with proper separators and whitespace.
           If output_format = 'base10', convert a BASE32 representation to a BASE10 one.
           If output_format = 'base32', convert a BASE10 representation to a BASE32 one.
           Note: in the case of AIC, the compact format is the same as the standard one.
               And 'compact' may contain both BASE10 and BASE32 represatation.
    """
    # pylint: disable=bare-except

    val = str(val)
    result: Any = []

    if val in NULL_VALUES:
        return [np.nan]

    if not validate_it_aic(val):
        if errors == "raise":
            raise ValueError(f"Unable to parse value {val}")
        error_result = val if errors == "ignore" else np.nan
        return [error_result]

    if output_format in {"compact", "standard"}:
        result = [aic.compact(val)] + result
    elif output_format == "base10":
        try:
            aic.validate_base10(val)
            result = [val] + result
        except:  # already know it is a valid AIC, so it can only be BASE32
            result = [aic.from_base32(val)] + result
    elif output_format == "base32":
        try:
            aic.validate_base32(val)
            result = [val] + result
        except:
            result = [aic.to_base32(val)] + result

    return result
