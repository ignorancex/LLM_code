# from beyonce.shallot.shallot import ShallotGrid
# from beyonce.shallot.grid_parameters import Parameters


# from copy import deepcopy
# import numpy as np
# import logging
# import pytest
# import filecmp
# import shutil

# ROOT_DIRECTORY = "test_data/"

# PARAMETERS = Parameters(0, 1, 6, 0, 2, 11, 5, 5)

# ECLIPSE_DURATION = 25
# TRANSVERSE_VELOCITY = 2.4
# LIMB_DARKENING = 0.8

# TIMES = ECLIPSE_DURATION * np.array([0.3, 0.2, -0.1, -0.4])
# GRADIENTS = np.array([0.2, 1.0, 0.5, 0.3])
# ERRORS = np.array([0.1, 0.3, 0.2, 0.1])
# TRANSMISSION_CHANGES = np.array([0.9, 0.95, 0.8, 0.99])

# SHALLOT_GRID = ShallotGrid(
#     parameters = deepcopy(PARAMETERS), 
#     num_fxfy = 5001, 
#     logging_level = logging.INFO,
#     keep_diagnostics = True,
#     intermittent_saving = False
# )
# SHALLOT_GRID.set_eclipse_parameters(ECLIPSE_DURATION, TRANSVERSE_VELOCITY, LIMB_DARKENING)
# SHALLOT_GRID.add_gradients(TIMES, GRADIENTS, ERRORS, TRANSMISSION_CHANGES)


# @pytest.fixture
# def grid_parameters() -> Parameters:
#     return deepcopy(PARAMETERS)


# @pytest.fixture
# def shallot_grid_both() -> ShallotGrid:
#     return deepcopy(SHALLOT_GRID)

# @pytest.fixture
# def shallot_grid_diagnostics() -> ShallotGrid:
#     # copy grid
#     grid = deepcopy(SHALLOT_GRID)

#     # remove eclipse parameters
#     grid._eclipse_duration = None
#     grid._transverse_velocity = None
#     grid._limb_darkening = None

#     # remove gradients
#     grid.gradients = None
#     grid.gradient_fit = None

#     return grid

# @pytest.fixture
# def shallot_grid_gradients() -> ShallotGrid:
#     # copy grid
#     grid = deepcopy(SHALLOT_GRID) 

#     # remove diagnostics
#     grid.diagnostics = None

#     return grid

# @pytest.fixture
# def shallot_grid_none() -> ShallotGrid:
#     # remove gradient information
#     grid = deepcopy(SHALLOT_GRID)

#     # remove eclipse parameters
#     grid._eclipse_duration = None
#     grid._transverse_velocity = None
#     grid._limb_darkening = None

#     # remove gradients
#     grid.gradients = None
#     grid.gradient_fit = None

#     # remove diagnostics
#     grid.diagnostics = None
    
#     return grid


# def assert_array_comparisons(array1: np.ndarray, array2: np.ndarray) -> None:
#     """This helper function is used to compare arrays"""
#     assert np.allclose(array1[~np.isnan(array1)], array2[~np.isnan(array2)])
#     assert np.all(np.isnan(array1) == np.isnan(array2))


# def test_init_none(grid_parameters: Parameters, shallot_grid_none: ShallotGrid) -> None:
#     grid = ShallotGrid(grid_parameters, 5001, keep_diagnostics=False)

#     # None / not None properties
#     assert shallot_grid_none.gradient_fit is None
#     assert shallot_grid_none.gradients is None
#     assert shallot_grid_none.logger is not None
    
#     # diagnostics
#     assert shallot_grid_none.diagnostics is None

#     # parameters
#     assert shallot_grid_none._num_fxfy == 5001
#     assert shallot_grid_none.parameters == grid.parameters
    
#     # check shear
#     assert_array_comparisons(shallot_grid_none.shear, grid.shear)
#     assert_array_comparisons(shallot_grid_none.circular_radius, grid.circular_radius)

#     params = [grid.disk_radius, grid.inclination, grid.tilt, grid.fx_map, grid.fy_map, grid.diagnostic_map]
#     shallot_params = [shallot_grid_none.disk_radius, shallot_grid_none.inclination, shallot_grid_none.tilt, shallot_grid_none.fx_map, shallot_grid_none.fy_map, shallot_grid_none.diagnostic_map]

#     for param, shallot_param in zip(params, shallot_params):
#         assert_array_comparisons(param.data, shallot_param.data)


# def test_init_diagnostics(grid_parameters: Parameters, shallot_grid_diagnostics: ShallotGrid) -> None:
#     grid = ShallotGrid(grid_parameters, 5001, keep_diagnostics=True)

#     # None / not None properties
#     assert shallot_grid_diagnostics.gradient_fit is None
#     assert shallot_grid_diagnostics.gradients is None
#     assert shallot_grid_diagnostics.logger is not None
    
#     # diagnostics
#     for (shallot_key, shallot_value), (grid_key, grid_value) in zip(shallot_grid_diagnostics.diagnostics._disk_radius_dict.items(), grid.diagnostics._disk_radius_dict.items()): 
#         assert shallot_key == grid_key
#         assert_array_comparisons(shallot_value, grid_value)

#     for (shallot_key, shallot_value), (grid_key, grid_value) in zip(shallot_grid_diagnostics.diagnostics._fy_dict.items(), grid.diagnostics._fy_dict.items()): 
#         assert shallot_key == grid_key
#         assert_array_comparisons(shallot_value, grid_value)
    
#     assert shallot_grid_diagnostics.diagnostics._extended == grid.diagnostics._extended

#     # parameters
#     assert shallot_grid_diagnostics._num_fxfy == 5001
#     assert shallot_grid_diagnostics.parameters == grid.parameters
    
#     # check shear
#     assert_array_comparisons(shallot_grid_diagnostics.shear, grid.shear)
#     assert_array_comparisons(shallot_grid_diagnostics.circular_radius, grid.circular_radius)

#     params = [grid.disk_radius, grid.inclination, grid.tilt, grid.fx_map, grid.fy_map, grid.diagnostic_map]
#     shallot_params = [shallot_grid_diagnostics.disk_radius, shallot_grid_diagnostics.inclination, shallot_grid_diagnostics.tilt, shallot_grid_diagnostics.fx_map, shallot_grid_diagnostics.fy_map, shallot_grid_diagnostics.diagnostic_map]

#     for param, shallot_param in zip(params, shallot_params):
#         assert_array_comparisons(param.data, shallot_param.data)


# def test_repr(shallot_grid_both: ShallotGrid) -> None:
#     repr_string = ("\n==========================================\n"
#         "******** SHALLOT GRID INFORMATION ********\n"
#         "==========================================\n\nGrid Parameters\n"
#         "----------------------------\ndx:  -1.00 ->   1.00 (11)\ndy:  -2.00 ->"
#         "   2.00 (21)\nrf:   5.00 ->   1.00 ->   5.00 (9)\ngrid_shape: "
#         "(21, 11, 9)\n\nDisk Radius [$t_{ecl}$]\n----------------------------"
#         "\nmin value:           0.5000\nmax value:          12.5000\n"
#         "mean value:          5.1105\nmedian value:        4.6861\n\n"
#         "Tilt [$^o$]\n----------------------------\n"
#         "min value:           0.0000\nmax value:         180.0000\n"
#         "mean value:         89.9547\nmedian value:       90.0000\n\n"
#         "Inclination [$^o$]\n"
#         "----------------------------\nmin value:           0.0000\n"
#         "max value:          89.2522\nmean value:         77.9577\n"
#         "median value:       81.8034\n\nGradient [-] @ pos = -0.3000\n"
#         "----------------------------\nmin value:           0.0011\n"
#         "max value:           1.0000\nmean value:          0.4063\n"
#         "median value:        0.2783\n\nmask [out]:         47.3785%\n"
#         "measured gradient:   0.2000\nmeasured error:      0.1000\n"
#         "orbital scale:       0.5794\ntransmission change: 0.9000\n\n"
#         "Gradient [-] @ pos = -0.2000\n----------------------------\n"
#         "min value:           0.0007\nmax value:           0.9997\n"
#         "mean value:          0.3910\nmedian value:        0.2032\n\n"
#         "mask [out]:         64.3098%\nmeasured gradient:   1.0000\n"
#         "measured error:      0.3000\norbital scale:       0.5794\n"
#         "transmission change: 0.9500\n\nGradient [-] @ pos =  0.1000\n"
#         "----------------------------\nmin value:           0.0004\n"
#         "max value:           1.0000\nmean value:          0.3732\n"
#         "median value:        0.1111\n\nmask [out]:         60.2694%\n"
#         "measured gradient:   0.5000\nmeasured error:      0.2000\n"
#         "orbital scale:       0.5794\ntransmission change: 0.8000\n\n"
#         "Gradient [-] @ pos =  0.4000\n----------------------------\n"
#         "min value:           0.0014\nmax value:           1.0000\n"
#         "mean value:          0.4207\nmedian value:        0.3363\n\n"
#         "mask [out]:         47.4747%\nmeasured gradient:   0.3000\n"
#         "measured error:      0.1000\norbital scale:       0.5794\n"
#         "transmission change: 0.9900\n\nGrid Diagnostics\n"
#         "----------------------------\ndiagnostics saved: 66\n\n"
#         "Grid Information\n----------------------------\n"
#         "percentage masked: 64.31%\nworst interpolation fit: 0.000000007\n\n"
#         "==========================================")
#     assert shallot_grid_both.__repr__() == repr_string


# def test_str(shallot_grid_both: ShallotGrid) -> None:
#     assert shallot_grid_both.__repr__() == shallot_grid_both.__str__()


# def test_save_both(shallot_grid_both: ShallotGrid) -> None:
#     load_directory = f"{ROOT_DIRECTORY}/shallot_grid"
#     save_directory = f"{ROOT_DIRECTORY}/shallot_grid_both"

#     try:
#         shallot_grid_both.save(save_directory)
#         result = filecmp.dircmp(save_directory, load_directory)
#         assert len(result.diff_files) == 0
#     finally:
#         shutil.rmtree(save_directory)

# def test_save_gradients(shallot_grid_gradients: ShallotGrid) -> None:
#     load_directory = f"{ROOT_DIRECTORY}/shallot_grid"
#     save_directory = f"{ROOT_DIRECTORY}/shallot_grid_gradients"

#     try:
#         shallot_grid_gradients.save(save_directory)
#         result = filecmp.dircmp(save_directory, load_directory)
#         assert len(result.diff_files) == 0
#     finally:
#         shutil.rmtree(save_directory)


# def test_save_diagnostics(shallot_grid_diagnostics: ShallotGrid) -> None:
#     load_directory = f"{ROOT_DIRECTORY}/shallot_grid"
#     save_directory = f"{ROOT_DIRECTORY}/shallot_grid_diagnostics"

#     try:
#         shallot_grid_diagnostics.save(save_directory)
#         result = filecmp.dircmp(save_directory, load_directory)
#         assert len(result.diff_files) == 0
#     finally:
#         shutil.rmtree(save_directory)


# def test_save_none(shallot_grid_none: ShallotGrid) -> None:
#     load_directory = f"{ROOT_DIRECTORY}/shallot_grid"
#     save_directory = f"{ROOT_DIRECTORY}/shallot_grid_none"

#     try:
#         shallot_grid_none.save(save_directory)
#         result = filecmp.dircmp(save_directory, load_directory)
#         assert len(result.diff_files) == 0
#     finally:
#         shutil.rmtree(save_directory)

# def test_load(shallot_grid_gradients: ShallotGrid) -> None:
#     grid = ShallotGrid.load("test_data/shallot_grid")

#     # None / not None properties
#     #assert shallot_grid_both.gradient_fit is None
#     #assert shallot_grid_both.gradients is None
#     assert shallot_grid_gradients.logger is not None
#     assert shallot_grid_gradients.diagnostics is None

#     # parameters
#     assert shallot_grid_gradients._num_fxfy == 5001
#     assert shallot_grid_gradients.parameters == grid.parameters
    
#     # check shear
#     assert_array_comparisons(shallot_grid_gradients.shear, grid.shear)
#     assert_array_comparisons(shallot_grid_gradients.circular_radius, grid.circular_radius)

#     params = [grid.disk_radius, grid.inclination, grid.tilt, grid.fx_map, grid.fy_map, grid.diagnostic_map]
#     shallot_params = [shallot_grid_gradients.disk_radius, shallot_grid_gradients.inclination, shallot_grid_gradients.tilt, shallot_grid_gradients.fx_map, shallot_grid_gradients.fy_map, shallot_grid_gradients.diagnostic_map]

#     for param, shallot_param in zip(params, shallot_params):
#         assert_array_comparisons(param.data, shallot_param.data)

#     print(f"{grid.gradients = }")
#     print(f"{shallot_grid_gradients.gradients = }")
#     for gradient, shallot_gradient in zip(grid.gradients, shallot_grid_gradients.gradients):
#         assert_array_comparisons(gradient.data, shallot_gradient.data)
#         assert_array_comparisons(gradient.mask, shallot_gradient.mask)
#         assert gradient.position == shallot_gradient.position


# #def test_load(shallot_grid_both: ShallotGrid) -> None:
