import pytest
import os
import tempfile
import shutil
from fixtures_and_helpers import (
    simple_wcs,
    sip_wcs,
    rotated_wcs,
    device,
    create_test_fits_file
)


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Clean up after tests
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="session")
def source_fits_file(test_data_dir, simple_wcs):
    """Create a source FITS file for testing."""
    import numpy as np

    # Create test data - a gradient pattern
    y, x = np.mgrid[0:200, 0:200]
    data = 100.0 * np.exp(-((x - 100) ** 2 + (y - 100) ** 2) / 1000.0)

    # Ensure data has native byte order
    data = np.asarray(data, dtype=np.float32).copy()

    # Create the file
    filename = os.path.join(test_data_dir, "source.fits")
    create_test_fits_file(
        wcs_header=simple_wcs.to_header(relax=True),
        data=data,
        filename=filename
    )
    return filename


@pytest.fixture(scope="session")
def target_fits_file(test_data_dir, sip_wcs):
    """Create a target FITS file for testing."""
    # Create a FITS file with target WCS but empty data
    import numpy as np
    data = np.zeros((200, 200), dtype=np.float32)
    # Ensure data has native byte order
    data = np.asarray(data, dtype=np.float32).copy()
    # Create the file
    filename = os.path.join(test_data_dir, "target.fits")
    create_test_fits_file(
        wcs_header=sip_wcs.to_header(relax=True),
        data=data,
        filename=filename
    )
    return filename