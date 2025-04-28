from typing import Optional

import numpy as np
import pandas as pd


def create_grid(
    n: int | tuple[int, int],
    x_limits: tuple[float, float] = (0.1, 10),
    y_limits: tuple[float, float] = (0.1, 10),
    space: str = "linear",
    return_as: str = "meshgrid",
    columns: Optional[list[str]] = None,
) -> (
    tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]
    | tuple[np.ndarray, np.ndarray, list[tuple[float, float]]]
    | tuple[np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, pd.DataFrame]
):
    """_summary_

    Parameters
    ----------
    n : int | tuple[int | int]
        Number of points in the grid. Can be a single
        integer or a tuple of two integers. In the latter
        case, the first integer corresponds to the number
        of points in the x-axis and the second integer to
        the number of points in the y-axis.
    x_limits : tuple[float, float], optional
        Lower and upper limits of the x-axis. By default,
        (0.1, 10).
    y_limits : tuple[float, float], optional
        Lower and upper limits of the y-axis. By default,
        (0.1, 10).
    space : str, optional
        Type of spacing between points. Can be either
        'linear', 'geom' or 'log'. By default, space = 'linear'.
    return_as : str, optional
        Format in which the grid is returned. Can be either
        'meshgrid', 'list', 'array' or 'dataframe'. By default,
        'meshgrid'.
    columns : list[str], optional
        Column names of the dataframe. By default, None.

    Returns
    -------
    tuple
        Returns a tuple with the x-axis limits, the y-axis
        limits and the grid. The format of the grid depends
        on the value of the parameter return_as.


    """
    n = (n, n) if isinstance(n, int) else n

    if space == "linear":
        x_ = np.linspace(*x_limits, n[0])
        y_ = np.linspace(*y_limits, n[1])
    elif space == "geom":
        x_ = np.geomspace(*x_limits, n[0])
        y_ = np.geomspace(*y_limits, n[1])
    elif space == "log":
        x_ = np.logspace(*x_limits, n[0])
        y_ = np.logspace(*y_limits, n[1])
    else:
        raise ValueError("space must be either 'linear', 'geom' or 'log'.")

    x, y = np.meshgrid(x_, y_)

    if return_as == "meshgrid":
        return x_, y_, (x, y)

    parameter_list = list(zip(x.ravel(), y.ravel()))  # type: ignore

    if return_as == "list":
        return x_, y_, parameter_list

    elif return_as == "array":
        return x_, y_, np.array(parameter_list)

    elif return_as == "dataframe":
        df = pd.DataFrame(parameter_list)
        if columns is not None:
            df.columns = columns
        return (x_, y_, df)

    else:
        raise ValueError(
            "return_as must be either 'meshgrid', 'list' 'array' or 'dataframe'."
        )
