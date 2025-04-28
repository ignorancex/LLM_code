from beyonce.shallot.grid_parameters import Parameters
from beyonce.shallot.grid_components import Gradient
from beyonce.shallot.errors import LoadError, InvalidShapeError

import numpy as np
import pytest
import shutil
import os


@pytest.fixture
def grid_gradient() -> Gradient:
    """This function produces a grid gradients object for use in tests."""
    grid_parameters = Parameters(0, 1, 11, 0, 1, 11, 5, 4)
    data = np.arange(11*11*7).reshape((11, 11, 7)).astype(float)
    gg = Gradient( 
        data = data, 
        parameters = grid_parameters,
        position = 0.2
    )
    return gg

TEST_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 
    "test_data/grid_gradient"
)


def test_str(grid_gradient: Gradient) -> None:
    str_string = grid_gradient.__str__()
    assert str_string == ("\nGradient [-] @ pos =  0.2000\n-----------------"
        "-----------\nmin value:           0.0000\nmax value:         "
        "846.0000\nmean value:        423.0000\nmedian value:      "
        "423.0000\n\nGrid Parameters\n----------------------------"
        "\ndx:   0.00 ->   1.00 (11)\ndy:   0.00 ->   1.00 (11)\nrf:   "
        "5.00 ->   1.00 ->   5.00 (7)\ngrid_shape: (11, 11, 7)")


def test_repr(grid_gradient: Gradient) -> None:
    repr_string = grid_gradient.__repr__()
    assert repr_string == ("\nGradient [-] @ pos =  0.2000\n-----------------"
        "-----------\nmin value:           0.0000\nmax value:         "
        "846.0000\nmean value:        423.0000\nmedian value:      "
        "423.0000")


def test_repr_mask(grid_gradient: Gradient) -> None:
    grid_gradient.determine_mask(0.3, 1.2)
    repr_string = grid_gradient.__repr__()

    assert repr_string == ("\nGradient [-] @ pos =  0.2000\n-----------------"
        "-----------\nmin value:           0.0000\nmax value:         "
        "846.0000\nmean value:        423.0000\nmedian value:      "
        "423.0000\n\nmask [out]:          0.1181%\nmeasured gradient:   "
        "0.3000\nmeasured error:      1.0000\norbital scale:       1.2000"
        "\ntransmission change: 1.0000")


def test_set_mask_valid(grid_gradient: Gradient) -> None:
    mask = np.ones_like(grid_gradient.data).astype(bool)
    grid_gradient.set_mask(mask)
    assert np.all(grid_gradient.mask == mask)


def test_set_mask_invalid(grid_gradient: Gradient) -> None:
    mask = np.ones_like((2, 2, 3)).astype(bool)
    
    with pytest.raises(InvalidShapeError) as ERROR:
        grid_gradient.set_mask(mask)
    
    message = ""
    names_list = ["mask", "data"]
    arrays_list = [mask, grid_gradient.data]
    
    for name, array in zip(names_list, arrays_list):
        message = message + f"{name} {str(array.shape)}, "
    message = message[:-2] + " should all have the same shape."
    
    assert str(ERROR.value) == message


def test_get_scaled_gradient(grid_gradient: Gradient) -> None:
    measured_gradient = 0.5
    orbital_scale = 1.2
    transmission_change = 0.9
    grid_gradient.determine_mask(measured_gradient, orbital_scale, 
        transmission_change)
    
    scaled_gradient = grid_gradient.get_scaled_gradient()
    sg = measured_gradient * orbital_scale / transmission_change

    assert scaled_gradient == sg


def test_determine_mask_defaults(grid_gradient: Gradient) -> None:
    measured_gradient = 0.5
    orbital_scale = 1.5
    grid_gradient.determine_mask(measured_gradient, orbital_scale)
    scaled_gradient = grid_gradient.get_scaled_gradient()
    mask = scaled_gradient > grid_gradient.data
    
    assert np.all(grid_gradient.mask == mask)
    assert grid_gradient.measured_gradient == measured_gradient
    assert grid_gradient.orbital_scale == orbital_scale
    assert grid_gradient.measured_error == 1
    assert grid_gradient.transmission_change == 1


def test_determine_mask_invalid(grid_gradient: Gradient) -> None:
    with pytest.raises(ValueError) as ERROR:
        grid_gradient.determine_mask(2, 2, 0.01)
    
    message = ("scaled gradient is greater than one, check the measured "
        "gradient, orbital scale and transmission change")
    assert str(ERROR.value) == message


def test_get_data_unmasked(grid_gradient: Gradient) -> None:
    mask = np.ones_like(grid_gradient.data).astype(bool)
    grid_gradient.set_mask(mask)
    data = grid_gradient.get_data(masked=False)
    
    assert np.all(data == grid_gradient.data)


def test_get_data_masked(grid_gradient: Gradient) -> None:
    mask = grid_gradient.data < 37
    grid_gradient.set_mask(mask)
    data = grid_gradient.get_data(masked=True)
    
    expected_data = grid_gradient.data
    expected_data[mask] = np.nan
    
    assert np.all(np.isnan(data) == np.isnan(expected_data))


def test_get_data_masked_no_mask(grid_gradient: Gradient) -> None:
    data = grid_gradient.get_data(masked=True)
    expected_data = grid_gradient.data
    assert np.all(data == expected_data)


def test_set_contrast_parameters_none(grid_gradient: Gradient) -> None:
    assert grid_gradient.vmin == np.nanmin(grid_gradient.data)
    assert grid_gradient.vmax == np.nanmax(grid_gradient.data)
    assert grid_gradient.color_map == "viridis"
    assert grid_gradient.num_colors == 11


def test_set_contrast_parameters_all_nan(grid_gradient: Gradient) -> None:
    grid_gradient.data = np.nan * grid_gradient.data
    grid_gradient.set_contrast_parameters()
    assert grid_gradient.vmin is None
    assert grid_gradient.vmax is None
    assert grid_gradient.color_map == "viridis"
    assert grid_gradient.num_colors == 11


def test_set_contrast_parameters(grid_gradient: Gradient) -> None:
    grid_gradient.set_contrast_parameters(0, 2, "twilight", 3)
    assert grid_gradient.vmin == 0 
    assert grid_gradient.vmax == 2
    assert grid_gradient.color_map == "twilight"
    assert grid_gradient.num_colors == 3


def test_get_scaled_gradient_none_set(grid_gradient: Gradient) -> None:
    scaled_gradient = grid_gradient.get_scaled_gradient()
    assert scaled_gradient is None


def test_get_scaled_gradient_set_properly(
        grid_gradient: Gradient
    ) -> None:
    grid_gradient.determine_mask(0.2, 1.3, 0.8)
    scaled_gradient = grid_gradient.get_scaled_gradient()
    expected_gradient = 0.325
    assert scaled_gradient == expected_gradient


def test_sorting(grid_gradient: Gradient):
    from copy import deepcopy
    
    other_grid_gradient = deepcopy(grid_gradient)
    other_grid_gradient.position = -0.2
    
    gradients = [grid_gradient, other_grid_gradient]
    positions = [0.2, -0.2]
    
    for gradient, position in zip(gradients, positions):
        assert gradient.position == position

    gradients.sort()
    new_positions = [-0.2, 0.2]

    for gradient, position in zip(gradients, new_positions):
        assert gradient.position == position


def test_save_no_mask(grid_gradient: Gradient) -> None:
    if not os.path.exists(TEST_DIR):
        os.mkdir(TEST_DIR)

    grid_gradient.save_gradient(TEST_DIR)
    
    data = np.load(f"{TEST_DIR}/gradient_{grid_gradient.position:.4f}/"
        f"{grid_gradient.name.name}_{grid_gradient.unit.name}.npy")
    shutil.rmtree(TEST_DIR)
    
    assert np.all(grid_gradient.data == data)


def test_save_with_mask(grid_gradient: Gradient) -> None:
    if not os.path.exists(TEST_DIR):
        os.mkdir(TEST_DIR)
    
    mask = np.ones_like(grid_gradient.data).astype(bool)
    grid_gradient.set_mask(mask)
    grid_gradient.save_gradient(TEST_DIR)
    grid_gradient.position = 0.2
    
    data = np.load(f"{TEST_DIR}/gradient_{grid_gradient.position:.4f}/"
        f"{grid_gradient.name.name}_{grid_gradient.unit.name}.npy")
    mask_load = np.load(f"{TEST_DIR}/gradient_{grid_gradient.position:.4f}/"
        f"{grid_gradient.name.name}_{grid_gradient.unit.name}_mask.npy")
    shutil.rmtree(TEST_DIR)
    
    assert np.all(grid_gradient.data == data)
    assert np.all(grid_gradient.mask == mask_load)


def test_load_invalid() -> None:
    with pytest.raises(LoadError) as ERROR:
        Gradient.load("invalid")

    message = "Failed to load Gradient [None] from invalid."
    assert str(ERROR.value) == message


def test_load_valid_no_mask(grid_gradient: Gradient) -> None:
    if not os.path.exists(TEST_DIR):
        os.mkdir(TEST_DIR)
        
    grid_gradient.save_gradient(TEST_DIR)
    
    grid_gradient_load = Gradient.load(f"{TEST_DIR}/gradient_"
        f"{grid_gradient.position:.4f}/")
    shutil.rmtree(TEST_DIR)

    assert np.all(grid_gradient.data == grid_gradient_load.data)
    assert grid_gradient_load.mask is None


def test_load_valid_with_mask(grid_gradient: Gradient) -> None:
    if not os.path.exists(TEST_DIR):
        os.mkdir(TEST_DIR)

    grid_gradient.determine_mask(0.3, 1.2, 0.9)
    grid_gradient.save_gradient(TEST_DIR)
    
    grid_gradient_load = Gradient.load(f"{TEST_DIR}/gradient_"
        f"{grid_gradient.position:.4f}/")
    shutil.rmtree(TEST_DIR)

    assert np.all(grid_gradient.data == grid_gradient_load.data)
    assert np.all(grid_gradient.mask == grid_gradient_load.mask)