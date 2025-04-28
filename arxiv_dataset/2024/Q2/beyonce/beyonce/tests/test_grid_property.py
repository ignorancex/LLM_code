from beyonce.shallot.grid_names import Name
from beyonce.shallot.grid_units import Unit
from beyonce.shallot.grid_parameters import Parameters
from beyonce.shallot.grid_components import Property
from beyonce.shallot.errors import LoadError, InvalidShapeError

from pytest import MonkeyPatch
import matplotlib.pyplot as plt
import numpy as np
import pytest
import shutil
import os


@pytest.fixture
def grid_property() -> Property:
    grid_parameters = Parameters(0, 1, 11, 0, 1, 11, 5, 4)
    data = np.arange(11 * 11 * 7).reshape((11, 11, 7)).astype(float)
    gp = Property(
        name = Name.DISK_RADIUS, 
        unit = Unit.ECLIPSE_DURATION, 
        data = data, 
        parameters = grid_parameters
    )
    return gp

TEST_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 
    "test_data/grid_property"
)


def test_str(grid_property: Property) -> None:
    str_string = grid_property.__str__()
    assert str_string == ("\nDisk Radius [$t_{ecl}$]\n-----------------"
        "-----------\nmin value:           0.0000\nmax value:         "
        "846.0000\nmean value:        423.0000\nmedian value:      "
        "423.0000\n\nGrid Parameters\n----------------------------"
        "\ndx:   0.00 ->   1.00 (11)\ndy:   0.00 ->   1.00 (11)\nrf:   "
        "5.00 ->   1.00 ->   5.00 (7)\ngrid_shape: (11, 11, 7)")


def test_repr(grid_property: Property) -> None:
    repr_string = grid_property.__repr__()
    assert repr_string == ("\nDisk Radius [$t_{ecl}$]\n-----------------"
        "-----------\nmin value:           0.0000\nmax value:         "
        "846.0000\nmean value:        423.0000\nmedian value:      "
        "423.0000")


def test_repr_mask(grid_property: Property) -> None:
    mask = np.ones_like(grid_property.data).astype(bool)
    grid_property.set_mask(mask)
    repr_string = grid_property.__repr__()

    assert repr_string == ("\nDisk Radius [$t_{ecl}$]\n-----------------"
        "-----------\nmin value:           0.0000\nmax value:         "
        "846.0000\nmean value:        423.0000\nmedian value:      "
        "423.0000\n\nmask [out]:        100.0000%")


def test_set_mask_valid(grid_property: Property) -> None:
    mask = np.ones_like(grid_property.data).astype(bool)
    grid_property.set_mask(mask)
    assert np.all(grid_property.mask == mask)


def test_set_mask_invalid(grid_property: Property) -> None:
    mask = np.ones_like((2, 2, 3)).astype(bool)
    
    with pytest.raises(InvalidShapeError) as ERROR:
        grid_property.set_mask(mask)
    
    message = ""
    names_list = ["mask", "data"]
    arrays_list = [mask, grid_property.data]
    
    for name, array in zip(names_list, arrays_list):
        message = message + f"{name} {str(array.shape)}, "
    message = message[:-2] + " should all have the same shape."
    
    assert str(ERROR.value) == message


def test_get_data_unmasked(grid_property: Property) -> None:
    mask = np.ones_like(grid_property.data).astype(bool)
    grid_property.set_mask(mask)
    data = grid_property.get_data(masked=False)
    
    assert np.all(data == grid_property.data)


def test_get_data_masked(grid_property: Property) -> None:
    mask = grid_property.data < 37
    grid_property.set_mask(mask)
    data = grid_property.get_data(masked=True)
    
    expected_data = grid_property.data
    expected_data[mask] = np.nan
    
    assert np.all(np.isnan(data) == np.isnan(expected_data))


def test_get_data_masked_no_mask(grid_property: Property) -> None:
    data = grid_property.get_data(masked=True)
    expected_data = grid_property.data
    assert np.all(data == expected_data)


def test_set_contrast_parameters_none(grid_property: Property) -> None:
    assert grid_property.vmin == np.nanmin(grid_property.data)
    assert grid_property.vmax == np.nanmax(grid_property.data)
    assert grid_property.color_map == "viridis"
    assert grid_property.num_colors == 11


def test_set_contrast_parameters_all_nan(grid_property: Property) -> None:
    grid_property.data = np.nan * grid_property.data
    grid_property.set_contrast_parameters()
    assert grid_property.vmin is None
    assert grid_property.vmax is None
    assert grid_property.color_map == "viridis"
    assert grid_property.num_colors == 11


def test_set_contrast_parameters(grid_property: Property) -> None:
    grid_property.set_contrast_parameters(0, 2, "twilight", 3)
    assert grid_property.vmin == 0 
    assert grid_property.vmax == 2
    assert grid_property.color_map == "twilight"
    assert grid_property.num_colors == 3


def test_save_no_mask(grid_property: Property) -> None:
    grid_property.save(TEST_DIR)
    
    data = np.load(f"{TEST_DIR}/{grid_property.name.name}_"
        f"{grid_property.unit.name}.npy")
    shutil.rmtree(TEST_DIR)
    
    assert np.all(grid_property.data == data)


def test_save_with_mask(grid_property: Property) -> None:
    mask = np.ones_like(grid_property.data).astype(bool)
    grid_property.set_mask(mask)
    grid_property.save(TEST_DIR)
    
    data = np.load(f"{TEST_DIR}/{grid_property.name.name}_"
        f"{grid_property.unit.name}.npy")
    mask_load = np.load(f"{TEST_DIR}/{grid_property.name.name}_"
        f"{grid_property.unit.name}_mask.npy")
    shutil.rmtree(TEST_DIR)
    
    assert np.all(grid_property.data == data)
    assert np.all(grid_property.mask == mask_load)


def test_load_invalid() -> None:
    with pytest.raises(LoadError) as ERROR:
        Property.load("invalid", 
            Name.DISK_RADIUS, Unit.ECLIPSE_DURATION)

    message = "Failed to load Disk Radius [Eclipse Duration] from invalid."
    assert str(ERROR.value) == message


def test_load_valid_no_mask(grid_property: Property) -> None:
    grid_property.save(TEST_DIR)
    
    grid_property_load = Property.load(TEST_DIR, 
        Name.DISK_RADIUS, Unit.ECLIPSE_DURATION)

    assert np.all(grid_property.data == grid_property_load.data)
    assert grid_property_load.mask is None


def test_load_valid_with_mask(grid_property: Property) -> None:
    mask = np.ones_like(grid_property.data).astype(bool)
    grid_property.set_mask(mask)
    grid_property.save(TEST_DIR)
    
    grid_property_load = Property.load(TEST_DIR, 
        Name.DISK_RADIUS, Unit.ECLIPSE_DURATION)
    shutil.rmtree(TEST_DIR)

    assert np.all(grid_property.data == grid_property_load.data)
    assert np.all(grid_property.mask == grid_property_load.mask)


def test_plot_cube(
        grid_property: Property, 
        monkeypatch: MonkeyPatch
    ) -> None:
    """unit test that 'handles' plt.show()"""
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: True)
    grid_property.plot_cube()
    assert True


def test_plot_slice(grid_property: Property) -> None:
    """unit test that 'handles' plotting"""
    ax, image = grid_property.plot_slice(axis=2, index=0)
    
    renderer = plt.gcf().canvas.get_renderer()
    image_shape = image.make_image(renderer, unsampled=True)[0].shape[:-1]

    assert ax.get_title() == ("Disk Radius [$t_{ecl}$] - $R_f$ = 5.0000 - "
        "horizontal")
    assert grid_property.data.shape[:-1] == image_shape