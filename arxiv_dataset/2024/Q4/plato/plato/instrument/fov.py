from functools import partial
from multiprocessing import Pool
from typing import Any, Optional

import astropy.units as u
import numpy as np
import polars
from astropy.coordinates import SkyCoord
from pandas import DataFrame as pdDataFrame
from polars import DataFrame as plDataFrame
from tqdm import tqdm

from plato.instrument.platopoint import platopoint  # type: ignore


def count_cameras(source: SkyCoord, **kwargs: Any) -> int:
    """Count the number of Plato cameras that observe particular
    source (for a given plato pointing which
    can be changed with kwargs). Default pointing is the
    LOPS2 field.


    Parameters
    ----------
    source : SkyCoord
        Coordinates of the source.
    kwargs : Any
        Additional arguments to pass to platopoint, including
        Plato pointing.

    Returns
    -------
    int
        Number of CCDs the source falls on.
    """
    # find CCD that the source falls on
    on_CCD = platopoint(targetCoord=source, **kwargs)[0].values()  # type: ignore
    # count number of CCDs
    if any(on_CCD):
        num_CCDs = sum(1 for val in on_CCD if val is not None)
    else:
        num_CCDs = 0

    n_cameras = 6 * num_CCDs
    return n_cameras


def find_targets(
    data: pdDataFrame | plDataFrame,
    field: str | dict = "LOPS2",
    progress: bool = True,
    processes: Optional[int] = None,
    **kwargs: Any
) -> pdDataFrame | plDataFrame:
    """Find the number of Plato CCDs that each source falls on, and add
    this information to the DataFrame as "num_CCDs" column.

    Parameters
    ----------
    data : pandas.DataFrame | polars.DataFrame
        DataFrame containing the sources with 'ra' and 'dec' columns in deg.
    progress : bool, optional
        Show progress bar, by default True.
    processes : int, optional
        Number of processes, by default None (uses all available cores).
    kwargs : Any
        Additional arguments to pass to platopoint, including Plato pointing.

    Returns
    -------
    pdDataFrame | plDataFrame
        DataFrame with 'num_CCDs' column added. Zero corresponds to target
        being outside the Plato field of view.
    """
    if not isinstance(data, (pdDataFrame, plDataFrame)):
        raise ValueError("data must be a (pandas or polars) DataFrame.")

    if kwargs is None:
        kwargs = {}

    if field == "LOPS2":
        kwargs["platformCoord"] = SkyCoord(
            "06:21:14.5 -47:53:13",
            unit=(u.hourangle, u.deg),
        )
        kwargs["rotationAngle"] = np.deg2rad(13.943091582693294)

    elif field == "LOPN1":
        kwargs["platformCoord"] = SkyCoord(
            "18:28:43.2 52:51:34",
            unit=(u.hourangle, u.deg),
        )
        kwargs["rotationAngle"] = np.deg2rad(-13.943091582693294)

    elif isinstance(field, dict):
        # check if field is a dictionary with SkyCoord and rotationAngle
        if not all(key in field.keys() for key in ["platformCoord", "rotationAngle"]):
            raise ValueError(
                "field must be a dictionary with 'platformCoord' and "
                "'rotationAngle' keys."
            )
        assert isinstance(field, dict)
        kwargs["platformCoord"] = field["platformCoord"]
        kwargs["rotationAngle"] = field["rotationAngle"]
    else:
        raise ValueError(
            "field must be 'LOPS2', 'LOPN1', or a dictionary "
            "with 'platformCoord' and 'rotationAngle' keys."
        )

    print(f"Field Center: RA = {kwargs['platformCoord'].ra.deg:.3f} "  # type: ignore
          f"Dec = {kwargs['platformCoord'].dec.deg:.3f} deg.")  # type: ignore
    print(f"Rotation Angle: {np.rad2deg(kwargs["rotationAngle"]):.3f} deg.\n")

    try:
        data_coords = SkyCoord(ra=data["ra"], dec=data["dec"], unit="deg")
    except KeyError | polars.exceptions.ColumnNotFoundError:
        raise ValueError("data must be a DataFrame with 'ra' and 'dec' columns in deg.")

    targets = []

    with Pool(processes) as pool:
        targets = list(
            tqdm(
                pool.imap(partial(count_cameras, **kwargs), data_coords),
                total=len(data_coords),
                disable=not progress,
                desc="Targets: ",
            )
        )

    if isinstance(data, pdDataFrame):
        data = data.assign(n_cameras=targets)
    else:
        data = data.with_columns(n_cameras=polars.Series(targets))
    return data
