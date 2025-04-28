from beyonce.shallot.grid_components import Gradient, Property
from beyonce.shallot.shallot import ShallotGrid, TEMP_SAVE_DIR
from beyonce.shallot.grid_parameters import Parameters
from beyonce.shallot.grid_names import Name
from beyonce.shallot.grid_units import Unit

from copy import deepcopy
import filecmp
import shutil
import numpy as np
import logging
import pytest
import os

PARAMETERS = Parameters(0, 1, 6, 0, 2, 11, 5, 5)

ECLIPSE_DURATION = 25
TRANSVERSE_VELOCITY = 2.4
LIMB_DARKENING = 0.8
ECLIPSE_PARAMETERS = (ECLIPSE_DURATION, TRANSVERSE_VELOCITY, LIMB_DARKENING)

TIMES = ECLIPSE_DURATION * np.array([0.3, 0.2, -0.1, -0.4])
GRADIENTS = np.array([0.2, 1.0, 0.5, 0.3])
ERRORS = np.array([0.1, 0.3, 0.2, 0.1])
TRANSMISSION_CHANGES = np.array([0.9, 0.95, 0.8, 0.99])

NUM_FX_FY = 5001

SHALLOT_GRID = ShallotGrid(
    parameters = deepcopy(PARAMETERS), 
    num_fxfy = NUM_FX_FY, 
    logging_level = logging.INFO,
    keep_diagnostics = True,
    intermittent_saving = False
)
SHALLOT_GRID.set_eclipse_parameters(ECLIPSE_DURATION, TRANSVERSE_VELOCITY, LIMB_DARKENING)
SHALLOT_GRID.add_gradients(TIMES, GRADIENTS, ERRORS, TRANSMISSION_CHANGES)

TEST_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 
    "test_data/shallot_grid"
)

SAVE_TEST_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 
    "test_data/test_shallot_save"
)

MOD_TEST_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 
    "test_data/modified_grid"
)

##############################################################################
#################################### FIXTURES ################################
##############################################################################

@pytest.fixture
def grid_parameters() -> Parameters:
    return deepcopy(PARAMETERS)


@pytest.fixture
def shallot_grid_both() -> ShallotGrid:
    return deepcopy(SHALLOT_GRID)

@pytest.fixture
def shallot_grid_diagnostics() -> ShallotGrid:
    # copy grid
    grid = deepcopy(SHALLOT_GRID)

    # remove eclipse parameters
    grid._eclipse_duration = None
    grid._transverse_velocity = None
    grid._limb_darkening = None

    # remove gradients
    grid.gradients = None
    grid.gradient_fit = None

    return grid

@pytest.fixture
def shallot_grid_gradients() -> ShallotGrid:
    # copy grid
    grid = deepcopy(SHALLOT_GRID) 

    # remove diagnostics
    grid.diagnostics = None

    return grid

@pytest.fixture
def shallot_grid() -> ShallotGrid:
    # remove gradient information
    grid = deepcopy(SHALLOT_GRID)
    grid._eclipse_duration = None
    grid._transverse_velocity = None
    grid._limb_darkening = None
    grid.gradients = None
    grid.gradient_fit = None

    # remove diagnostics
    grid.diagnostics = None
    
    return grid

##############################################################################
#################################### FUNCTIONS ###############################
##############################################################################


def determine_shear(parameters: Parameters) -> np.ndarray:
    """determine the shear based on the grid parameters"""
    # extract parameters
    dx = parameters.dx
    dy = parameters.dy
    num_dy, num_dx = parameters.slice_shape

    # determine dx and dy grids
    dx_grid = np.repeat(dx, num_dy, 0)
    dy_grid = np.repeat(dy, num_dx, 1)
    mask = (dy_grid != 0)

    # determine shear
    shear = np.nan * np.ones((num_dy, num_dx))[:, :, None]
    shear[mask] = -dx_grid[mask] / dy_grid[mask]

    # correct origin
    origin = (dy_grid == 0) * (dx_grid == 0)
    shear[origin] = 0

    return shear


def determine_circular_radius(parameters: Parameters) -> np.ndarray:
    """determine the circular radius based on grid parameters"""
    return np.hypot(1/2, parameters.dy)


def load_valid_disk_parameters(directory: str) -> tuple[
        Property, 
        Property,
        Property, 
        Property, 
        Property, 
        Property,
        list[Gradient]
    ]:
    """
    due to the involved and numerical nature of the determination of these
    parameters, they are simply loaded from file
    """
    # load parameter data
    disk_radius = Property.load(directory, Name.DISK_RADIUS, 
        Unit.ECLIPSE_DURATION)
    inclination = Property.load(directory, Name.INCLINATION, 
        Unit.DEGREE)
    tilt = Property.load(directory, Name.TILT, 
        Unit.DEGREE)
    fx_map = Property.load(directory, Name.FX_MAP, 
        Unit.NONE)
    fy_map = Property.load(directory, Name.FY_MAP,
        Unit.NONE)
    diagnostic_map = Property.load(directory, 
        Name.DIAGNOSTIC_MAP, Unit.ECLIPSE_DURATION)

    # gradients
    positions = np.array([-0.3, -0.2, 0.1, 0.4])
    gradients = []
    for position in positions:
        gradient_directory = f"{directory}/gradient_{position:.4f}"
        gradients.append(Gradient.load(gradient_directory)) 

    return disk_radius, inclination, tilt, fx_map, fy_map, diagnostic_map, gradients


def load_combined_mask(directory: str, include_gradients: bool) -> np.ndarray:
    disk_radius, inclination, tilt, *_, gradients = load_valid_disk_parameters(directory)

    all_masks = [disk_radius.mask, inclination.mask, tilt.mask]
    if include_gradients:
        for gradient in gradients:
            all_masks.append(gradient.mask)

    combined_mask = np.zeros(disk_radius.data.shape)
    for mask in all_masks:
        if mask is not None:
            combined_mask += mask.astype(float)

    return combined_mask.astype(bool)


def assert_array_comparisons(array1: np.ndarray, array2: np.ndarray) -> None:
    """This helper function is used to compare arrays"""
    assert np.allclose(array1[~np.isnan(array1)], array2[~np.isnan(array2)])
    assert np.all(np.isnan(array1) == np.isnan(array2))


##############################################################################
###################################### TESTS #################################
##############################################################################


def test_init(
        grid_parameters: Parameters, 
        shallot_grid: ShallotGrid
    ) -> None:  
    """test initialisation by loading known test value"""

    # assert (not) none values
    assert shallot_grid.diagnostics is None
    assert shallot_grid.gradient_fit is None
    assert shallot_grid.gradients is None
    assert shallot_grid.logger is not None

    grid_parameters.extend_grid()
    
    # assert inputs
    assert shallot_grid.parameters == grid_parameters
    assert shallot_grid._num_fxfy == NUM_FX_FY
    
    # check shear
    shear = determine_shear(grid_parameters)
    assert_array_comparisons(shallot_grid.shear, shear)

    # check circular radius
    circular_radius = determine_circular_radius(grid_parameters)
    assert_array_comparisons(shallot_grid.circular_radius, circular_radius)

    # check parameters
    parameters = load_valid_disk_parameters(TEST_DIR)
    shallot_parameters = [shallot_grid.disk_radius, shallot_grid.inclination, 
        shallot_grid.tilt, shallot_grid.fx_map, shallot_grid.fy_map,
        shallot_grid.diagnostic_map]

    for parameter, shallot_parameter in zip(parameters, shallot_parameters):
        assert_array_comparisons(parameter.data, shallot_parameter.data)
    

def test_repr_no_gradients_diagnostics(shallot_grid: ShallotGrid) -> None:
    repr_string = ("\n==========================================\n******** "
        "SHALLOT GRID INFORMATION ********\n"
        "==========================================\n\nGrid Parameters\n"
        "----------------------------\ndx:  -1.00 ->   1.00 (11)\ndy:  -2.00"
        " ->   2.00 (21)\nrf:   5.00 ->   1.00 ->   5.00 (9)\ngrid_shape: "
        "(21, 11, 9)\n\nDisk Radius [$t_{ecl}$]\n----------------------------"
        "\nmin value:           0.5000\nmax value:          12.5000\n"
        "mean value:          5.1105\nmedian value:        4.6861\n\n"
        "Tilt [$^o$]\n----------------------------\n"
        "min value:           0.0000\nmax value:         180.0000\n"
        "mean value:         89.9547\nmedian value:       90.0000\n\n"
        "Inclination [$^o$]\n----------------------------\n"
        "min value:           0.0000\nmax value:          89.2522\n"
        "mean value:         77.9577\nmedian value:       81.8034\n\n"
        "Grid Information\n----------------------------\n"
        "percentage masked: 0.00%\nworst interpolation fit: 0.000000007"
        "\n\n==========================================")
    assert shallot_grid.__repr__() == repr_string


def test_repr_no_gradients(grid_parameters: Parameters) -> None:
    grid = ShallotGrid(grid_parameters, NUM_FX_FY, keep_diagnostics=True)
    repr_string = ("\n==========================================\n"
        "******** SHALLOT GRID INFORMATION ********\n"
        "==========================================\n\nGrid Parameters\n"
        "----------------------------\ndx:  -1.00 ->   1.00 (11)\ndy:  -2.00 "
        "->   2.00 (21)\nrf:   5.00 ->   1.00 ->   5.00 (9)\ngrid_shape: "
        "(21, 11, 9)\n\nDisk Radius [$t_{ecl}$]\n----------------------------"
        "\nmin value:           0.5000\nmax value:          12.5000\n"
        "mean value:          5.1105\nmedian value:        4.6861\n\n"
        "Tilt [$^o$]\n----------------------------\n"
        "min value:           0.0000\nmax value:         180.0000\n"
        "mean value:         89.9547\nmedian value:       90.0000\n\n"
        "Inclination [$^o$]\n----------------------------\n"
        "min value:           0.0000\nmax value:          89.2522\n"
        "mean value:         77.9577\nmedian value:       81.8034\n\n"
        "Grid Diagnostics\n----------------------------\n"
        "diagnostics saved: 66\n\nGrid Information\n"
        "----------------------------\npercentage masked: 0.00%\n"
        "worst interpolation fit: 0.000000007\n\n"
        "==========================================")
    assert grid.__repr__() == repr_string


def test_repr_no_diagnostics(shallot_grid: ShallotGrid) -> None:
    shallot_grid.set_eclipse_parameters(*ECLIPSE_PARAMETERS)
    shallot_grid.add_gradients(TIMES, GRADIENTS)
    repr_string = ("\n==========================================\n"
        "******** SHALLOT GRID INFORMATION ********\n"
        "==========================================\n\nGrid Parameters\n"
        "----------------------------\ndx:  -1.00 ->   1.00 (11)\ndy:  -2.00 ->"
        "   2.00 (21)\nrf:   5.00 ->   1.00 ->   5.00 (9)\ngrid_shape: "
        "(21, 11, 9)\n\nDisk Radius [$t_{ecl}$]\n----------------------------"
        "\nmin value:           0.5000\nmax value:          12.5000\n"
        "mean value:          5.1105\nmedian value:        4.6861\n\n"
        "Tilt [$^o$]\n----------------------------\n"
        "min value:           0.0000\nmax value:         180.0000\n"
        "mean value:         89.9547\nmedian value:       90.0000\n\n"
        "Inclination [$^o$]\n"
        "----------------------------\nmin value:           0.0000\n"
        "max value:          89.2522\nmean value:         77.9577\n"
        "median value:       81.8034\n\nGradient [-] @ pos = -0.3000\n"
        "----------------------------\nmin value:           0.0011\n"
        "max value:           1.0000\nmean value:          0.4063\n"
        "median value:        0.2783\n\nmask [out]:         47.2823%\n"
        "measured gradient:   0.2000\nmeasured error:      1.0000\n"
        "orbital scale:       0.5794\ntransmission change: 1.0000\n\n"
        "Gradient [-] @ pos = -0.2000\n----------------------------\n"
        "min value:           0.0007\nmax value:           0.9997\n"
        "mean value:          0.3910\nmedian value:        0.2032\n\n"
        "mask [out]:         63.7326%\nmeasured gradient:   1.0000\n"
        "measured error:      1.0000\norbital scale:       0.5794\n"
        "transmission change: 1.0000\n\nGradient [-] @ pos =  0.1000\n"
        "----------------------------\nmin value:           0.0004\n"
        "max value:           1.0000\nmean value:          0.3732\n"
        "median value:        0.1111\n\nmask [out]:         59.3074%\n"
        "measured gradient:   0.5000\nmeasured error:      1.0000\n"
        "orbital scale:       0.5794\ntransmission change: 1.0000\n\n"
        "Gradient [-] @ pos =  0.4000\n----------------------------\n"
        "min value:           0.0014\nmax value:           1.0000\n"
        "mean value:          0.4207\nmedian value:        0.3363\n\n"
        "mask [out]:         47.4747%\nmeasured gradient:   0.3000\n"
        "measured error:      1.0000\norbital scale:       0.5794\n"
        "transmission change: 1.0000\n\nGrid Information\n"
        "----------------------------\npercentage masked: 63.73%\n"
        "worst interpolation fit: 0.000000007\n\n"
        "==========================================")
    assert shallot_grid.__repr__() == repr_string


def test_repr(grid_parameters: Parameters) -> None:
    grid = ShallotGrid(grid_parameters, NUM_FX_FY, keep_diagnostics=True)
    grid.set_eclipse_parameters(*ECLIPSE_PARAMETERS)
    grid.add_gradients(TIMES, GRADIENTS)

    repr_string = ("\n==========================================\n"
        "******** SHALLOT GRID INFORMATION ********\n"
        "==========================================\n\nGrid Parameters\n"
        "----------------------------\ndx:  -1.00 ->   1.00 (11)\ndy:  -2.00 ->"
        "   2.00 (21)\nrf:   5.00 ->   1.00 ->   5.00 (9)\ngrid_shape: "
        "(21, 11, 9)\n\nDisk Radius [$t_{ecl}$]\n----------------------------"
        "\nmin value:           0.5000\nmax value:          12.5000\n"
        "mean value:          5.1105\nmedian value:        4.6861\n\n"
        "Tilt [$^o$]\n----------------------------\n"
        "min value:           0.0000\nmax value:         180.0000\n"
        "mean value:         89.9547\nmedian value:       90.0000\n\n"
        "Inclination [$^o$]\n"
        "----------------------------\nmin value:           0.0000\n"
        "max value:          89.2522\nmean value:         77.9577\n"
        "median value:       81.8034\n\nGradient [-] @ pos = -0.3000\n"
        "----------------------------\nmin value:           0.0011\n"
        "max value:           1.0000\nmean value:          0.4063\n"
        "median value:        0.2783\n\nmask [out]:         47.2823%\n"
        "measured gradient:   0.2000\nmeasured error:      1.0000\n"
        "orbital scale:       0.5794\ntransmission change: 1.0000\n\n"
        "Gradient [-] @ pos = -0.2000\n----------------------------\n"
        "min value:           0.0007\nmax value:           0.9997\n"
        "mean value:          0.3910\nmedian value:        0.2032\n\n"
        "mask [out]:         63.7326%\nmeasured gradient:   1.0000\n"
        "measured error:      1.0000\norbital scale:       0.5794\n"
        "transmission change: 1.0000\n\nGradient [-] @ pos =  0.1000\n"
        "----------------------------\nmin value:           0.0004\n"
        "max value:           1.0000\nmean value:          0.3732\n"
        "median value:        0.1111\n\nmask [out]:         59.3074%\n"
        "measured gradient:   0.5000\nmeasured error:      1.0000\n"
        "orbital scale:       0.5794\ntransmission change: 1.0000\n\n"
        "Gradient [-] @ pos =  0.4000\n----------------------------\n"
        "min value:           0.0014\nmax value:           1.0000\n"
        "mean value:          0.4207\nmedian value:        0.3363\n\n"
        "mask [out]:         47.4747%\nmeasured gradient:   0.3000\n"
        "measured error:      1.0000\norbital scale:       0.5794\n"
        "transmission change: 1.0000\n\nGrid Diagnostics\n"
        "----------------------------\ndiagnostics saved: 66\n\n"
        "Grid Information\n----------------------------\n"
        "percentage masked: 63.73%\nworst interpolation fit: 0.000000007\n\n"
        "==========================================")
    assert grid.__repr__() == repr_string


def test_str(shallot_grid: ShallotGrid) -> None:
    assert shallot_grid.__repr__() == shallot_grid.__str__()


def test_load(shallot_grid: ShallotGrid) -> None:
    grid = ShallotGrid.load(TEST_DIR)

    # grid parameters
    grid_parameters = [grid.disk_radius, grid.tilt, grid.inclination,
        grid.fx_map, grid.fy_map, grid.diagnostic_map]

    # check parameters
    shallot_parameters = [shallot_grid.disk_radius, shallot_grid.tilt, 
        shallot_grid.inclination, shallot_grid.fx_map, shallot_grid.fy_map,
        shallot_grid.diagnostic_map]

    for grid_parameter, shallot_parameter in zip(grid_parameters, shallot_parameters):
        assert_array_comparisons(grid_parameter.data, shallot_parameter.data)


def test_save(shallot_grid: ShallotGrid) -> None:
    try:
        shallot_grid.save(SAVE_TEST_DIR)
        result = filecmp.dircmp(SAVE_TEST_DIR, TEST_DIR)
        assert len(result.diff_files) == 0
    finally:
        shutil.rmtree(SAVE_TEST_DIR)


def test_recording_diagnostics() -> None:
    parameters = Parameters(0, 1, 2, 0, 1, 2, 2, 2)
    total_grid_points = np.prod(parameters.slice_shape)
    
    grid = ShallotGrid(parameters, NUM_FX_FY, keep_diagnostics=True)
    assert grid.diagnostics is not None
    assert len(grid.diagnostics._disk_radius_dict) == total_grid_points
    assert len(grid.diagnostics._fy_dict) == total_grid_points


def test_intermittent_saving_complete(shallot_grid: ShallotGrid) -> None:
    shutil.copytree(TEST_DIR, SAVE_TEST_DIR)
    np.save(f"{SAVE_TEST_DIR}/start_y", np.array([0]))

    try:
        grid = ShallotGrid.load(SAVE_TEST_DIR)
    finally:
        shutil.rmtree(SAVE_TEST_DIR)

    # grid parameters
    grid_parameters = [grid.disk_radius, grid.tilt, grid.inclination,
        grid.fx_map, grid.fy_map, grid.diagnostic_map]

    # check parameters
    shallot_parameters = [shallot_grid.disk_radius, shallot_grid.tilt, 
        shallot_grid.inclination, shallot_grid.fx_map, shallot_grid.fy_map,
        shallot_grid.diagnostic_map]

    for grid_parameter, shallot_parameter in zip(grid_parameters, shallot_parameters):
        assert_array_comparisons(grid_parameter.data, shallot_parameter.data)

    
def test_intermittent_saving_incomplete(shallot_grid: ShallotGrid) -> None:
    shutil.copytree(TEST_DIR, SAVE_TEST_DIR)
    np.save(f"{SAVE_TEST_DIR}/start_y", np.array([4]))

    try:
        grid = ShallotGrid.load(SAVE_TEST_DIR)
    finally:
        shutil.rmtree(SAVE_TEST_DIR)

    # grid parameters
    grid_parameters = [grid.disk_radius, grid.tilt, grid.inclination,
        grid.fx_map, grid.fy_map, grid.diagnostic_map]

    # check parameters
    shallot_parameters = [shallot_grid.disk_radius, shallot_grid.tilt, 
        shallot_grid.inclination, shallot_grid.fx_map, shallot_grid.fy_map,
        shallot_grid.diagnostic_map]

    for grid_parameter, shallot_parameter in zip(grid_parameters, shallot_parameters):
        assert_array_comparisons(grid_parameter.data, shallot_parameter.data)


def test_get_disk_radius_unmasked(shallot_grid: ShallotGrid) -> None:
    disk_radius = shallot_grid.get_disk_radius(masked=False)
    # note that the masked here refers to the shallot grid property -> 
    # property_masked
    assert_array_comparisons(disk_radius, 
        shallot_grid.disk_radius.get_data(masked=True))


def test_get_disk_radius_masked(shallot_grid: ShallotGrid) -> None:
    mask = load_combined_mask(TEST_DIR, True)
    disk_radius_masked = shallot_grid.disk_radius.data
    disk_radius_masked[mask] = np.nan
    assert_array_comparisons(disk_radius_masked, 
        shallot_grid.get_disk_radius(masked=True))


def test_get_inclination_unmasked(shallot_grid: ShallotGrid) -> None:
    inclination = shallot_grid.get_inclination(masked=False)
    # note that the masked here refers to the shallot grid property -> 
    # property_masked
    assert_array_comparisons(inclination, 
        shallot_grid.inclination.get_data(masked=True))


def test_get_inclination_masked(shallot_grid: ShallotGrid) -> None:
    mask = load_combined_mask(TEST_DIR, True)
    inclination_masked = shallot_grid.inclination.data
    inclination_masked[mask] = np.nan
    assert_array_comparisons(inclination_masked, 
        shallot_grid.get_inclination(masked=True))


def test_get_tilt_unmasked(shallot_grid: ShallotGrid) -> None:
    tilt = shallot_grid.get_tilt(masked=False)
    # note that the masked here refers to the shallot grid property -> 
    # property_masked
    assert_array_comparisons(tilt, shallot_grid.tilt.get_data(masked=True))


def test_get_tilt_masked(shallot_grid: ShallotGrid) -> None:
    mask = load_combined_mask(TEST_DIR, True)
    tilt_masked = shallot_grid.tilt.data
    tilt_masked[mask] = np.nan
    assert_array_comparisons(tilt_masked, shallot_grid.get_tilt(masked=True))


def test_get_fx_map_unmasked(shallot_grid: ShallotGrid) -> None:
    fx_map = shallot_grid.get_fx_map(masked=False)
    # note that the masked here refers to the shallot grid property -> 
    # property_masked
    assert_array_comparisons(fx_map, 
        shallot_grid.fx_map.get_data(masked=True))


def test_get_fx_map_masked(shallot_grid: ShallotGrid) -> None:
    mask = load_combined_mask(TEST_DIR, True)
    fx_map_masked = shallot_grid.fx_map.data
    fx_map_masked[mask] = np.nan
    assert_array_comparisons(fx_map_masked, 
        shallot_grid.get_fx_map(masked=True))

def test_get_fy_map_unmasked(shallot_grid: ShallotGrid) -> None:
    fy_map = shallot_grid.get_fy_map(masked=False)
    # note that the masked here refers to the shallot grid property -> 
    # property_masked
    assert_array_comparisons(fy_map, 
        shallot_grid.fy_map.get_data(masked=True))


def test_get_fy_map_masked(shallot_grid: ShallotGrid) -> None:
    mask = load_combined_mask(TEST_DIR, True)
    fy_map_masked = shallot_grid.fy_map.data
    fy_map_masked[mask] = np.nan
    assert_array_comparisons(fy_map_masked, 
        shallot_grid.get_fy_map(masked=True))


def test_get_diagnostic_map_unmasked(shallot_grid: ShallotGrid) -> None:
    diagnostic_map = shallot_grid.get_diagnostic_map(masked=False)
    # note that the masked here refers to the shallot grid property -> 
    # property_masked
    assert_array_comparisons(diagnostic_map, 
        shallot_grid.diagnostic_map.get_data(masked=True))


def test_get_diagnostic_map_masked(shallot_grid: ShallotGrid) -> None:
    mask = load_combined_mask(TEST_DIR, True)
    diagnostic_map_masked = shallot_grid.diagnostic_map.data
    diagnostic_map_masked[mask] = np.nan
    assert_array_comparisons(diagnostic_map_masked, 
        shallot_grid.get_diagnostic_map(masked=True))


def test_setting_eclipse_parameters_no_gradients(shallot_grid: ShallotGrid) -> None:
    assert shallot_grid.gradients is None
    
    shallot_grid.set_eclipse_parameters(*ECLIPSE_PARAMETERS)
    
    assert shallot_grid._eclipse_duration == ECLIPSE_DURATION
    assert shallot_grid._transverse_velocity == TRANSVERSE_VELOCITY
    assert shallot_grid._limb_darkening == LIMB_DARKENING
    assert shallot_grid.gradients is None


def test_setting_eclipse_parameters_preexisting_gradients(shallot_grid: ShallotGrid) -> None:
    shallot_grid.set_eclipse_parameters(*ECLIPSE_PARAMETERS)
    shallot_grid.add_gradients(TIMES, GRADIENTS)
    
    eclipse_duration = 12
    transverse_velocity = 1.6
    limb_darkening = 0.3
    eclipse_parameters = (eclipse_duration, transverse_velocity, limb_darkening)
    assert shallot_grid.gradients is not None
    shallot_grid.set_eclipse_parameters(*eclipse_parameters)
    assert shallot_grid.gradients is None
    
    assert shallot_grid._eclipse_duration == eclipse_duration
    assert shallot_grid._transverse_velocity == transverse_velocity
    assert shallot_grid._limb_darkening == limb_darkening
    

def test_add_gradients_defaults(shallot_grid: ShallotGrid) -> None:
    shallot_grid.set_eclipse_parameters(*ECLIPSE_DURATION)
    shallot_grid.add_gradients(TIMES, GRADIENTS)

    *_, gradients = load_valid_disk_parameters(TEST_DIR)
    shallot_gradients = shallot_grid.gradients
    
    for gradient, shallot_gradient in zip(gradients, shallot_gradients):
        assert gradient.position == shallot_gradient.position

        assert_array_comparisons(gradient.data, shallot_gradient.data)
        assert np.all(gradient.mask == shallot_gradient.mask)

        assert gradient.measured_gradient == shallot_gradient.measured_gradient


def test_add_gradients_defaults(shallot_grid: ShallotGrid) -> None:
    shallot_grid.set_eclipse_parameters(*ECLIPSE_PARAMETERS)
    shallot_grid.add_gradients(TIMES, GRADIENTS)

    *_, gradients = load_valid_disk_parameters(TEST_DIR)
    shallot_gradients = shallot_grid.gradients
    
    for gradient, shallot_gradient in zip(gradients, shallot_gradients):
        assert gradient.position == shallot_gradient.position

        assert_array_comparisons(gradient.data, shallot_gradient.data)
        assert np.all(gradient.mask == shallot_gradient.mask)

        assert (gradient.measured_gradient == 
            shallot_gradient.measured_gradient)
        assert gradient.measured_error == shallot_gradient.measured_error
        assert (gradient.transmission_change == 
            shallot_gradient.transmission_change)


def test_del_gradients(shallot_grid: ShallotGrid) -> None:
    shallot_grid.set_eclipse_parameters(*ECLIPSE_PARAMETERS)
    shallot_grid.add_gradients(TIMES, GRADIENTS)
    shallot_grid.remove_gradients(np.array([0, 2]))

    *_, all_gradients = load_valid_disk_parameters(TEST_DIR)
    expected_gradients = [all_gradients[1], all_gradients[3]]
    gradients = (expected_gradients, shallot_grid.gradients)

    assert len(shallot_grid.gradients) == 2
    
    for expected_gradient, grid_gradient in zip(*gradients):
        assert expected_gradient.position == grid_gradient.position

        assert_array_comparisons(expected_gradient.data, grid_gradient.data)
        assert np.all(expected_gradient.mask == grid_gradient.mask)
        
        assert (expected_gradient.measured_gradient == 
            grid_gradient.measured_gradient)
        assert (expected_gradient.measured_error == 
            grid_gradient.measured_error)
        assert (expected_gradient.transmission_change == 
            grid_gradient.transmission_change)




def test_add_gradients_modified(shallot_grid: ShallotGrid) -> None:
    shallot_grid.set_eclipse_parameters(*ECLIPSE_PARAMETERS)
    shallot_grid.add_gradients(TIMES, GRADIENTS, ERRORS, TRANSMISSION_CHANGES)
    *_, gradients = load_valid_disk_parameters(MOD_TEST_DIR)
    shallot_gradients = shallot_grid.gradients
    
    for gradient, shallot_gradient in zip(gradients, shallot_gradients):
        assert gradient.position == shallot_gradient.position

        assert_array_comparisons(gradient.data, shallot_gradient.data)
        assert np.all(gradient.mask == shallot_gradient.mask)

        assert gradient.measured_gradient == shallot_gradient.measured_gradient

def test_temp_save_dir_create(shallot_grid: ShallotGrid) -> None:
    try:
        preexisting = os.path.exists(TEMP_SAVE_DIR)
        shallot_grid.save(TEMP_SAVE_DIR)
        existing = os.path.exists(TEMP_SAVE_DIR)
        assert not preexisting 
        assert existing
    finally:
        shutil.rmtree(TEMP_SAVE_DIR)
    

def test_save_with_y_value(shallot_grid: ShallotGrid) -> None:
    shallot_grid.save(SAVE_TEST_DIR, y_value=1)
    try:
        result = filecmp.dircmp(SAVE_TEST_DIR, TEST_DIR)
        assert len(result.diff_files) == 0
        assert os.path.exists(f"{SAVE_TEST_DIR}/y_value.npy")
    finally:
        shutil.rmtree(SAVE_TEST_DIR)


def test_update_gradient_scaling_none(shallot_grid: ShallotGrid) -> None:
    shallot_grid.set_eclipse_parameters(*ECLIPSE_PARAMETERS)
    shallot_grid.add_gradients(TIMES, GRADIENTS, ERRORS, TRANSMISSION_CHANGES)
    # extract pre update orbital and transmission scales
    orbital_scales = np.zeros_like(GRADIENTS)
    transmission_changes = np.zeros_like(GRADIENTS)
    for k, gradient in enumerate(shallot_grid.gradients):
        orbital_scales[k] = gradient.orbital_scale
        transmission_changes[k] = gradient.transmission_change
    
    shallot_grid.update_gradient_scaling()
    for k, gradient in enumerate(shallot_grid.gradients):
        assert gradient.orbital_scale == orbital_scales[k]
        assert gradient.transmission_change == transmission_changes[k]


def test_update_gradient_scaling_transmission_changes(
        shallot_grid: ShallotGrid
    ) -> None:
    shallot_grid.set_eclipse_parameters(*ECLIPSE_PARAMETERS)
    shallot_grid.add_gradients(TIMES, GRADIENTS, ERRORS, TRANSMISSION_CHANGES)
    # extract pre update orbital and transmission scales
    orbital_scales = np.zeros_like(GRADIENTS)
    transmission_changes = np.zeros_like(GRADIENTS)
    for k, gradient in enumerate(shallot_grid.gradients):
        orbital_scales[k] = gradient.orbital_scale
        transmission_changes[k] = gradient.transmission_change
    
    new_transmission_changes = transmission_changes + 0.003
    shallot_grid.update_gradient_scaling(
        transmission_changes=new_transmission_changes
    )
    for k, gradient in enumerate(shallot_grid.gradients):
        assert gradient.orbital_scale == orbital_scales[k]
        assert gradient.transmission_change == new_transmission_changes[k]


def test_update_gradient_scaling_transverse_velocity(
        shallot_grid: ShallotGrid
    ) -> None:
    shallot_grid.set_eclipse_parameters(*ECLIPSE_PARAMETERS)
    shallot_grid.add_gradients(TIMES, GRADIENTS, ERRORS, TRANSMISSION_CHANGES)
    # extract pre update orbital and transmission scales
    orbital_scales = np.zeros_like(GRADIENTS)
    transmission_changes = np.zeros_like(GRADIENTS)
    for k, gradient in enumerate(shallot_grid.gradients):
        orbital_scales[k] = gradient.orbital_scale
        transmission_changes[k] = gradient.transmission_change
    
    new_transverse_velocity = TRANSVERSE_VELOCITY + 0.03
    change_factor = TRANSVERSE_VELOCITY/new_transverse_velocity
    shallot_grid.update_gradient_scaling(
        transverse_velocity=new_transverse_velocity
    )
    for k, gradient in enumerate(shallot_grid.gradients):
        assert gradient.orbital_scale == orbital_scales[k]*change_factor
        assert gradient.transmission_change == transmission_changes[k]


def test_update_gradient_scaling_limb_darkening(
        shallot_grid: ShallotGrid
    ) -> None:
    shallot_grid.set_eclipse_parameters(*ECLIPSE_PARAMETERS)
    shallot_grid.add_gradients(TIMES, GRADIENTS, ERRORS, TRANSMISSION_CHANGES)
    # extract pre update orbital and transmission scales
    orbital_scales = np.zeros_like(GRADIENTS)
    transmission_changes = np.zeros_like(GRADIENTS)
    for k, gradient in enumerate(shallot_grid.gradients):
        orbital_scales[k] = gradient.orbital_scale
        transmission_changes[k] = gradient.transmission_change
    
    limb_darkening_scale = (
        (6 - 2 * LIMB_DARKENING) / 
        (12 - 12 * LIMB_DARKENING + 3 * np.pi * LIMB_DARKENING)
    )
    new_limb_darkening = LIMB_DARKENING/2
    new_limb_darkening_scale = (
        (6 - 2 * new_limb_darkening) / 
        (12 - 12 * new_limb_darkening + 3 * np.pi * new_limb_darkening)
    )
    change_factor = new_limb_darkening_scale/limb_darkening_scale

    shallot_grid.update_gradient_scaling(
        limb_darkening=new_limb_darkening
    )
    for k, gradient in enumerate(shallot_grid.gradients):
        assert np.round(gradient.orbital_scale, 5) == np.round(orbital_scales[k]*change_factor, 5)
        assert gradient.transmission_change == transmission_changes[k]


def test_update_gradient_scaling_limb_darkening_and_transverse_velocity(
        shallot_grid: ShallotGrid
    ) -> None:
    shallot_grid.set_eclipse_parameters(*ECLIPSE_PARAMETERS)
    shallot_grid.add_gradients(TIMES, GRADIENTS, ERRORS, TRANSMISSION_CHANGES)
    # extract pre update orbital and transmission scales
    orbital_scales = np.zeros_like(GRADIENTS)
    transmission_changes = np.zeros_like(GRADIENTS)
    for k, gradient in enumerate(shallot_grid.gradients):
        orbital_scales[k] = gradient.orbital_scale
        transmission_changes[k] = gradient.transmission_change
    
    limb_darkening_scale = (
        (6 - 2 * LIMB_DARKENING) / 
        (12 - 12 * LIMB_DARKENING + 3 * np.pi * LIMB_DARKENING)
    )
    new_limb_darkening = LIMB_DARKENING/2
    new_limb_darkening_scale = (
        (6 - 2 * new_limb_darkening) / 
        (12 - 12 * new_limb_darkening + 3 * np.pi * new_limb_darkening)
    )

    new_transverse_velocity = TRANSVERSE_VELOCITY + 0.03

    change_factor = (
        (new_limb_darkening_scale * TRANSVERSE_VELOCITY) / 
        (limb_darkening_scale * new_transverse_velocity)
    )
    
    shallot_grid.update_gradient_scaling(
        transverse_velocity=new_transverse_velocity,
        limb_darkening=new_limb_darkening
    )
    for k, gradient in enumerate(shallot_grid.gradients):
        assert np.round(gradient.orbital_scale, 5) == np.round(orbital_scales[k]*change_factor, 5)
        assert gradient.transmission_change == transmission_changes[k]

# grid.diagnostic_map, gradient_fit, _eclipse_duration, _transverse_velocity, _limb_darkening, 
# grid.parameters, num_fxfy, disk_radius, inclination, tilt, fx_map, fy_map, diagnostics, 
# grid.logger, circular_radius, shear, gradients

# consider making logger, num_fxfy, fx_map, fy_map, circular_radius, and shear private variables?

# orbital parameters -> 25, 2.4, 0.8
# gradients @ times = -(-0.2, -0.3, 0.1 0.4) * ecl, (0.2, 1.0, 0.5, 0.3)