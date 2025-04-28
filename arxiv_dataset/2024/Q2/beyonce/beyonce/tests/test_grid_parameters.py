from beyonce.shallot.grid_parameters import Parameters
from beyonce.shallot.errors import LoadError, InvalidBoundsError
from beyonce.shallot.errors import OriginMissingError

import numpy as np
import pytest
import shutil
import os

TEST_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 
    "test_data/grid_parameters"
)

def test_repr() -> None:
    gp = Parameters(0, 1, 2, 0, 1, 2, 2, 2)
    repr_string = gp.__repr__()
    assert repr_string == ("\nGrid Parameters\n----------------------------\n"
        "dx:   0.00 ->   1.00 (2)\ndy:   0.00 ->   1.00 (2)\nrf:   2.00 ->   "
        "1.00 ->   2.00 (3)\ngrid_shape: (2, 2, 3)")


def test_str() -> None:
    gp = Parameters(0, 1, 2, 0, 1, 2, 2, 2)
    str_string = gp.__str__()
    assert str_string == ("\nGrid Parameters\n----------------------------\n"
        "dx:   0.00 ->   1.00 (2)\ndy:   0.00 ->   1.00 (2)\nrf:   2.00 ->   "
        "1.00 ->   2.00 (3)\ngrid_shape: (2, 2, 3)")


def test_ymin_greater_than_ymax() -> None:
    with pytest.raises(InvalidBoundsError) as ERROR:
        Parameters(0, 1, 2, 2, 1, 2, 2, 2)
    
    assert str(ERROR.value) == ("The max_y argument (1.0000) must be greater"
        " than the min_y argument (2.0000).")


def test_xmin_greater_than_xmax() -> None:
    with pytest.raises(InvalidBoundsError) as ERROR:
        Parameters(1, 0, 2, 0, 1, 2, 2, 2)
    assert str(ERROR.value) == ("The max_x argument (0.0000) must be greater"
        " than the min_x argument (1.0000).")


def test_extendable_init() -> None:
    min_x = 0
    max_x = 1
    num_x = 2
    min_y = 0
    max_y = 1
    num_y = 4
    max_rf = 5
    num_rf = 4
    gp = Parameters(min_x, max_x, num_x, min_y, max_y, num_y, max_rf, 
        num_rf)
    dx = np.linspace(min_x, max_x, num_x)[None, :, None]
    dy = np.linspace(min_y, max_y, num_y)[:, None, None]
    rf = np.linspace(1, max_rf, num_rf)
    rf_array = np.concatenate((np.flip(rf), rf[1:]), 0)
    
    assert np.all(gp.dx == dx)
    assert np.all(gp.dy == dy)
    assert np.all(gp.rf == rf)
    assert np.all(gp.rf_array == rf_array)
    assert gp.grid_shape == (num_y, num_x, 2 * num_rf - 1)
    assert gp.slice_shape == (num_y, num_x)
    assert gp.extendable == True


def test_non_extendable_init() -> None:
    min_x = 0.5
    max_x = 1
    num_x = 2
    min_y = 0
    max_y = 1
    num_y = 4
    max_rf = 5
    num_rf = 4
    gp = Parameters(min_x, max_x, num_x, min_y, max_y, num_y, max_rf, 
        num_rf)
    dx = np.linspace(min_x, max_x, num_x)[None, :, None]
    dy = np.linspace(min_y, max_y, num_y)[:, None, None]
    rf = np.linspace(1, max_rf, num_rf)
    rf_array = np.concatenate((np.flip(rf), rf[1:]), 0)
    
    assert np.all(gp.dx == dx)
    assert np.all(gp.dy == dy)
    assert np.all(gp.rf == rf)
    assert np.all(gp.rf_array == rf_array)
    assert gp.grid_shape == (num_y, num_x, 2 * num_rf - 1)
    assert gp.slice_shape == (num_y, num_x)
    assert gp.extendable == False


def test_get_vectors() -> None:
    min_x = 0
    max_x = 1
    num_x = 2
    min_y = 0
    max_y = 1
    num_y = 5
    max_rf = 5
    num_rf = 4
    gp = Parameters(min_x, max_x, num_x, min_y, max_y, num_y, max_rf, 
        num_rf)
    dx = np.linspace(min_x, max_x, num_x)
    dy = np.linspace(min_y, max_y, num_y)
    rf = np.linspace(1, max_rf, num_rf)
    rf_array = np.concatenate((np.flip(rf), rf[1:]), 0)
    
    dy_get, dx_get, rf_array_get = gp.get_vectors()
    
    assert np.all(dx_get == dx)
    assert np.all(dy_get == dy)
    assert np.all(rf_array_get == rf_array)


def test_extend_grid_invalid() -> None:
    min_x = 0.5
    max_x = 1
    num_x = 2
    min_y = 0
    max_y = 1
    num_y = 4
    max_rf = 5
    num_rf = 4
    gp = Parameters(min_x, max_x, num_x, min_y, max_y, num_y, max_rf, 
        num_rf)

    with pytest.raises(OriginMissingError) as ERROR:
        gp.extend_grid()

    assert str(ERROR.value) == ("The grid parameters can not be extended. "
        "That is only possible when the grid parameters include the origin.")

def test_extend_grid_valid() -> None:
    min_x = 0
    max_x = 1
    num_x = 2
    min_y = 0
    max_y = 1
    num_y = 5
    max_rf = 5
    num_rf = 4
    gp = Parameters(min_x, max_x, num_x, min_y, max_y, num_y, max_rf, 
        num_rf)
    gp.extend_grid()

    dx = np.linspace(-max_x, max_x, 2 * num_x - 1)[None, :, None]
    dy = np.linspace(-max_y, max_y, 2 * num_y - 1)[:, None, None]
    rf = np.linspace(1, max_rf, num_rf)
    rf_array = np.concatenate((np.flip(rf), rf[1:]), 0)
    
    assert np.all(gp.dx == dx)
    assert np.all(gp.dy == dy)
    assert np.all(gp.rf == rf)
    assert np.all(gp.rf_array == rf_array)
    assert gp.grid_shape == (2 * num_y - 1, 2 * num_x - 1, 2 * num_rf - 1)
    assert gp.slice_shape == (2 * num_y - 1, 2 * num_x - 1)
    assert gp.extendable == False

def test_save() -> None:
    min_x = 0
    max_x = 1
    num_x = 2
    min_y = 0
    max_y = 1
    num_y = 5
    max_rf = 5
    num_rf = 4
    gp = Parameters(min_x, max_x, num_x, min_y, max_y, num_y, max_rf, 
        num_rf)

    gp.save(TEST_DIR)

    dx = np.load(f"{TEST_DIR}/dx.npy")
    dy = np.load(f"{TEST_DIR}/dy.npy")
    rf = np.load(f"{TEST_DIR}/rf.npy")
    rf_array = np.load(f"{TEST_DIR}/rf_array.npy")

    shutil.rmtree(TEST_DIR)

    assert np.all(dx == gp.dx)
    assert np.all(dy == gp.dy)
    assert np.all(rf == gp.rf)
    assert np.all(rf_array == gp.rf_array)


def test_load_invalid() -> None:
    directory = "invalid"
    
    with pytest.raises(LoadError) as ERROR:
        Parameters.load(directory)

    message = f"Failed to load parameters from {directory}."
    assert str(ERROR.value) == message


def test_load_valid() -> None:
    if not os.path.exists(TEST_DIR):
        os.mkdir(TEST_DIR)
    
    dx = np.linspace(0, 1, 2)
    dy = np.linspace(0, 2, 3)
    rf = np.linspace(1, 3, 2)
    rf_array = np.concatenate((np.flip(rf), rf[1:]), 0)
    
    np.save(f"{TEST_DIR}/dx", dx)
    np.save(f"{TEST_DIR}/dy", dy)
    np.save(f"{TEST_DIR}/rf", rf)
    np.save(f"{TEST_DIR}/rf_array", rf_array)

    gp = Parameters.load(TEST_DIR)
    shutil.rmtree(TEST_DIR)

    assert np.all(dx == gp.dx)
    assert np.all(dy == gp.dy)
    assert np.all(rf == gp.rf)
    assert np.all(rf_array == gp.rf_array)
    assert gp.grid_shape == (3, 2, 3)
    assert gp.slice_shape == (3, 2)
    assert gp.extendable == True

