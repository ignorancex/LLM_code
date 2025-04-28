from pathlib import Path

import numpy as np
from numpy import typing as npt


def load_seismic_datasets__overthrust_model(data_path: Path) -> npt.NDArray[np.float64]:
    """
    Parameters
    ----------
    data_path: Path
        The path to the seismic dataset named "overthrust.vites"

        data is described at https://wiki.seg.org/wiki/SEG/EAGE_Salt_and_Overthrust_Models

        data is available at https://s3.amazonaws.com/open.source.geoscience/open_data/seg_eage_models_cd/salt_and_overthrust_models.tar.gz

        example: "./3-D_Overthrust_Model_Disk1/3D-Velocity-Grid/overthrust.vites"

    Returns
    -------
    NDArray:
        seismic data of overthrust model

        seismic_data.dtype == np.float64

        seismic_data.shape == (nz, ny, nx) == (187, 801, 801)
    """

    # ref: https://github.com/pyvista/show-room/blob/13a5ab7bc2315e7e36765868ca488a3b66ed1973/seg-eage-3d-salt-model.ipynb

    nz, ny, nx = 187, 801, 801
    with open(data_path, "r") as file:
        vel = np.fromfile(file, dtype=np.dtype("float32").newbyteorder(">"))

    vel = vel.reshape(nx, ny, nz, order="F")

    # Cast type
    vel = np.asarray(vel, dtype=float)

    # THE SEG/EAGE salt-model uses positive z downwards;
    # here we want positive upwards. Hence:
    vel = np.flip(vel, 2)

    seismic_data = np.transpose(vel, (2, 1, 0))[::-1]

    assert seismic_data.shape == (nz, ny, nx)
    assert np.min(seismic_data) == 2178.83447265625
    assert np.max(seismic_data) == 6000.0

    return seismic_data
