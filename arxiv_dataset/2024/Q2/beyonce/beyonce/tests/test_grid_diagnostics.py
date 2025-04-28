"""
This module tests all the class functionality contained by the Diagnostics
class in the shallot.grid_diagnostics module.
"""
# import main packages
import numpy as np
import shutil
import pytest
import os

# import beyonce packages
from beyonce.shallot.grid_diagnostics import Diagnostics
from beyonce.shallot.grid_parameters import Parameters
from beyonce.shallot.errors import LoadError, ValueOutsideRangeError


# initial conditions
GRID_PARAMETERS = Parameters(
    min_x = 0, 
    max_x = 1, 
    num_x = 2, 
    min_y = 0, 
    max_y = 1,
    num_y = 2,
    max_rf = 2, 
    num_rf = 1
)

TEST_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 
    "test_data/grid_diagnostic"
)

def test_repr() -> None:
    diag = Diagnostics(GRID_PARAMETERS)
    repr_string = diag.__repr__()
    assert repr_string == ("\nGrid Diagnostics\n----------------------------"
        "\ndiagnostics saved: 0")


def test_repr_with_diagnostics() -> None:
    diag = Diagnostics(GRID_PARAMETERS)
    diag.save_diagnostic(0, 0, np.zeros(1), np.ones(1))
    repr_string = diag.__repr__()
    assert repr_string == ("\nGrid Diagnostics\n----------------------------"
        "\ndiagnostics saved: 1")


def test_str() -> None:
    diag = Diagnostics(GRID_PARAMETERS)
    str_string = diag.__str__()
    assert str_string == ("\nGrid Diagnostics\n----------------------------"
        "\ndiagnostics saved: 0")


def test_str_with_diagnostics() -> None:
    diag = Diagnostics(GRID_PARAMETERS)
    diag.save_diagnostic(0, 0, np.zeros(1), np.ones(1))
    str_string = diag.__str__()
    assert str_string == ("\nGrid Diagnostics\n----------------------------"
        "\ndiagnostics saved: 1")


def test_generate_key_none() -> None:
    diag = Diagnostics(GRID_PARAMETERS)
    y = 1
    x = 0
    key = diag._generate_key(y, x)
    assert key == f"({y}, {x})"


def test_generate_key_valid() -> None:
    diag = Diagnostics(GRID_PARAMETERS)
    y = 0
    x = 1
    key = diag._generate_key(y, x)
    assert key == f"({y}, {x})"


def test_generate_key_y_invalid() -> None:
    diag = Diagnostics(GRID_PARAMETERS)
    invalid_y = 2
    with pytest.raises(ValueOutsideRangeError) as ERROR:
        diag._generate_key(invalid_y, 0)

    error_message = (f"The y argument ({invalid_y:.4f}) must be"
        " between or equal to 0.0000 and 1.0000.")
    assert str(ERROR.value) == error_message


def test_generate_key_x_invalid() -> None:
    diag = Diagnostics(GRID_PARAMETERS)
    invalid_x = 2
    with pytest.raises(ValueOutsideRangeError) as ERROR:
        diag._generate_key(0, invalid_x)

    error_message = (f"The x argument ({invalid_x:.4f}) must be between or "
        "equal to 0.0000 and 1.0000.")
    assert str(ERROR.value) == error_message


def test_generate_key_both_invalid() -> None:
    diag = Diagnostics(GRID_PARAMETERS)
    invalid_y = 10
    invalid_x = 10
    with pytest.raises(ValueOutsideRangeError) as ERROR:
        diag._generate_key(invalid_y, invalid_x)

    error_message = (f"The y argument ({invalid_y:.4f}) must be between or "
        "equal to 0.0000 and 1.0000.")
    assert str(ERROR.value) == error_message


def test_save_diagnostic() -> None:
    fy = np.linspace(0, 1, 3)
    disk_radius = np.ones_like(fy)

    diag = Diagnostics(GRID_PARAMETERS)
    diag.save_diagnostic(1, 0, fy, disk_radius)

    key = diag._generate_key(1, 0)
    assert np.all(diag._fy_dict[key] == fy)
    assert np.all(diag._disk_radius_dict[key] == disk_radius)


def test_save_diagnostic_valid_with_allowed() -> None:
    fy = np.linspace(0, 1, 3)
    disk_radius = np.ones_like(fy)

    diag = Diagnostics(GRID_PARAMETERS)
    diag.save_diagnostic(1, 0, fy, disk_radius)

    key = diag._generate_key(1, 0)
    assert np.all(diag._fy_dict[key] == fy)
    assert np.all(diag._disk_radius_dict[key] == disk_radius)


def test_get_diagnostic_valid() -> None:
    fy = np.linspace(0, 1, 3)
    disk_radius = np.ones_like(fy)

    diag = Diagnostics(GRID_PARAMETERS)
    diag.save_diagnostic(1, 0, fy, disk_radius)

    fy_get, disk_radius_get = diag.get_diagnostic(1, 0)

    assert np.all(fy_get == fy)
    assert np.all(disk_radius_get == disk_radius)

def test_get_diagnostic_invalid() -> None:
    fy = np.linspace(0, 1, 3)
    disk_radius = np.ones_like(fy)

    diag = Diagnostics(GRID_PARAMETERS)
    diag.save_diagnostic(1, 0, fy, disk_radius)

    with pytest.raises(KeyError) as ERROR:
        diag.get_diagnostic(0, 1)

    assert str(ERROR.value) == "'(0, 1)'"


def test_save_without_allowed() -> None:
    fy = np.linspace(0, 1, 3)
    disk_radius = np.ones_like(fy)

    diag = Diagnostics(GRID_PARAMETERS)
    diag.save_diagnostic(1, 0, fy, disk_radius)
    diag.save(f"{TEST_DIR}")

    fy_saved: dict = np.load(f"{TEST_DIR}/fy_dict.npy", 
        allow_pickle=True).item()
    disk_radius_saved: dict = np.load(f"{TEST_DIR}/disk_radius_dict.npy", 
        allow_pickle=True).item()

    shutil.rmtree(f"{TEST_DIR}")

    for key, value in fy_saved.items():
        assert np.all(value == diag._fy_dict[key])

    for key, value, in disk_radius_saved.items():
        assert np.all(value == diag._disk_radius_dict[key])


def test_save_with_allowed() -> None:
    fy = np.linspace(0, 1, 3)
    disk_radius = np.ones_like(fy)

    diag = Diagnostics(GRID_PARAMETERS)
    diag.save_diagnostic(1, 0, fy, disk_radius)
    diag.save(f"{TEST_DIR}")

    fy_saved: dict = np.load(f"{TEST_DIR}/fy_dict.npy", 
        allow_pickle=True).item()
    disk_radius_saved: dict = np.load(f"{TEST_DIR}/disk_radius_dict.npy",
        allow_pickle=True).item()

    shutil.rmtree(f"{TEST_DIR}")

    for key, value in fy_saved.items():
        assert np.all(value == diag._fy_dict[key])

    for key, value, in disk_radius_saved.items():
        assert np.all(value == diag._disk_radius_dict[key])


def test_extend() -> None:
    diag = Diagnostics(GRID_PARAMETERS)
    diag.extend()
    assert diag._extended == True


def test_extended_save() -> None:
    diag = Diagnostics(GRID_PARAMETERS)
    diag.extend()

    with pytest.raises(RuntimeError) as ERROR:
        diag.save_diagnostic(0, 0, np.zeros(1), np.ones(1))

    error_message = "diagnostics have been extended and can not be changed"
    assert str(ERROR.value) == error_message


def test_generate_key_extended_invalid() -> None:
    diag = Diagnostics(GRID_PARAMETERS)
    diag.save_diagnostic(0, 0, np.zeros(1), np.ones(1))
    diag.extend()
    
    with pytest.raises(KeyError) as ERROR:
        diag.get_diagnostic(0, 0)
    
    assert str(ERROR.value) == "'(2, 2)'"


def test_generate_key_extended_valid() -> None:
    diag = Diagnostics(GRID_PARAMETERS)
    diag.save_diagnostic(1, 1, np.zeros(1), np.ones(1))
    diag.extend()

    key = diag._generate_key(0, 0)
    assert key == "(2, 2)"


def test_load_invalid() -> None:
    directory = "invalid"
    with pytest.raises(LoadError) as ERROR:
        Diagnostics.load(directory)
    
    error_message = f"Failed to load diagnostics from {directory}."
    assert str(ERROR.value) == error_message


def test_load_valid() -> None:
    os.mkdir(TEST_DIR)
    fy_dict = {"(1, 0)": np.zeros(2)}
    disk_radius_dict = {"(1, 0)": np.ones(2)}
    max_x = 2
    max_y = 3
    max_yx = np.array([max_y, max_x])
    np.save(f"{TEST_DIR}/fy_dict", fy_dict)
    np.save(f"{TEST_DIR}/disk_radius_dict", disk_radius_dict)
    np.save(f"{TEST_DIR}/max_yx", max_yx)
    
    diag = Diagnostics.load(TEST_DIR)
    
    shutil.rmtree(TEST_DIR)

    for key, value in fy_dict.items():
        assert np.all(value == diag._fy_dict[key])

    for key, value, in disk_radius_dict.items():
        assert np.all(value == diag._disk_radius_dict[key])

    assert np.all(diag._max_x == max_x)
    assert np.all(diag._max_y == max_y)

    
