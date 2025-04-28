from pathlib import Path

import numpy as np
from numpy import typing as npt


def load_seismic_datasets__salt_model(data_path: Path) -> npt.NDArray[np.int32]:
    """
    Parameters
    ----------
    data_path: Path
        The path to the seismic dataset named "Saltf@@"

        data is described at https://wiki.seg.org/wiki/SEG/EAGE_Salt_and_Overthrust_Models

        data is available at https://s3.amazonaws.com/open.source.geoscience/open_data/seg_eage_models_cd/salt_and_overthrust_models.tar.gz

        example: "./3-D_Salt_Model/VEL_GRIDS/Saltf@@"

    Returns
    -------
    NDArray:
        seismic data of salt models

        seismic_data.dtype == np.int32

        seismic_data.shape == (nz, ny, nx) == (210, 676, 676)
    """

    # ref: https://github.com/pyvista/show-room/blob/13a5ab7bc2315e7e36765868ca488a3b66ed1973/seg-eage-3d-salt-model.ipynb

    nz, ny, nx = 210, 676, 676
    with open(data_path, "r") as file:
        vel = np.fromfile(file, dtype=np.dtype("float32").newbyteorder(">"))

    vel = vel.reshape(nx, ny, nz, order="F")

    # Cast type
    vel = np.asarray(vel, dtype=float)

    # THE SEG/EAGE salt-model uses positive z downwards;
    # here we want positive upwards. Hence:
    vel = np.flip(vel, 2)

    raw_seismic_data = np.transpose(vel, (2, 1, 0))

    assert raw_seismic_data.shape == (nz, ny, nx)
    assert np.min(raw_seismic_data) == 1500.0
    assert np.max(raw_seismic_data) == 4482.0

    seismic_data = raw_seismic_data.astype(np.int32)

    assert np.sum(np.abs(seismic_data - raw_seismic_data)) == 0

    return seismic_data
