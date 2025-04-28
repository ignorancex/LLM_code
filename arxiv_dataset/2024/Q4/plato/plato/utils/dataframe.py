from typing import Optional

import numpy as np
import pandas as pd


def accumulate_from_sources(
    dataframe: pd.DataFrame,
    column: str,
    source_columns: list[str],
    additional_masks: Optional[dict[str, np.ndarray | pd.Series]] = None,
    include_errors: bool = True,
    error_suffixes: list[str] = ["_lower", "_upper"],
    drop_sources: bool = True,
) -> pd.DataFrame:
    """
    Accumulate a new column from multiple sources in a DataFrame.
    The new columns will be filled with non-null values
    from the source columns, with a priority order given by
    the order of the sources in the source_columns list.
    Also add the source of the value to a new source column.
    If include_errors is True, the error collumns will be
    added as well.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The DataFrame to accumulate the new column in, with the source columns.
    column : str
        The name of the new column to accumulate.
    source_columns : list[str]
        The names of the source columns to accumulate the new column from.
        The order of the sources in the list will determine the priority order.
    additional_masks : dict[str, np.ndarray | pd.Series], optional
        Additional masks to apply to the source columns before accumulating
        the new column, for example quality flags. Additional masks should
        be given as a dictionary with the source column names as keys,
        and the corresponding masks as values, by default None.
    include_errors : bool, optional
        Whether to accumulate the errors in the new column as well.
        The errors columns should have the same name as the source columns
        with the error suffixes appended, e.g. "_lower" and "_upper".
        The new errors columns will be named "e_{column}{error_suffixes[0]}"
        and "e_{column}{error_suffixes[1]}", by default True.
    error_suffixes : list[str], optional
        The suffixes to append to the source columns to get the error columns,
        by default ["_lower", "_upper"].
    drop_sources : bool, optional
        Whether to drop the source columns after accumulating the new column,
        by default True.

    Returns
    -------
    pd.DataFrame
        The DataFrame with the new column accumulated from the sources,
        and the source columns dropped if specified.

    """
    if additional_masks is None:
        additional_masks = {}

    # Initialize the columns for metallicity and its errors
    dataframe[column] = np.nan
    dataframe[f"{column}_source"] = ""
    if include_errors:
        dataframe[f"e_{column}{error_suffixes[0]}"] = np.nan
        dataframe[f"e_{column}{error_suffixes[1]}"] = np.nan

    # Add metallicity in the following priority order
    for source in source_columns:
        mask = dataframe[f"{source}"].notnull() & dataframe[column].isnull()
        if source in additional_masks.keys():
            mask = mask & additional_masks[source]
        dataframe.loc[mask, column] = dataframe[f"{source}"]
        dataframe.loc[mask, f"{column}_source"] = source
        if include_errors:
            dataframe.loc[mask, f"e_{column}{error_suffixes[0]}"] = dataframe[
                f"{source}{error_suffixes[0]}"
            ]
            dataframe.loc[mask, f"e_{column}{error_suffixes[1]}"] = dataframe[
                f"{source}{error_suffixes[1]}"
            ]

    # Drop original columns and their errors
    if drop_sources:
        columns_to_drop = [""] + error_suffixes if include_errors else [""]
        dataframe.drop(
            columns=[
                f"{source}{suffix}"
                for source in source_columns
                for suffix in columns_to_drop
            ],
            inplace=True,
        )

    return dataframe
