import itertools

import pytest
import torch
from polyhedral_gravity.utility import check_mesh

from gravann.input import sample_reader
from gravann.labels import mascon, polyhedral
from gravann.training.losses import relRMSE
from gravann.util import get_target_point_sampler

# ====================== TEST PARAMETERS ======================
# The tested bodies
TEST_FILENAMES = ["bennu", "churyumov-gerasimenko", "eros", "hollow", "itokawa", "torus"]
# The tested distances to the body center
TEST_DISTANCES = [1e-1, 5e-1, 1.0, 5.0, 1e2, 5e2]
# The test relative error needed for comparison between mascon & polyhedral model for the potential
TEST_EPSILON = 5e-3
# The maximum allowed relative root squared mean error for the accelerations
TEST_relRMSE = 8e-3
# The number of points per body per altitude
TEST_POINTS_PER_BATCH = 100
# The seed for the test cases
TEST_SEED = 0


# ======================= SETUP UTILITY =======================
def get_data(sample: str):
    """Utility function. Gets the parameters for a sample body
    
    Args:
        sample: the name of the sample

    Returns:
        tuple of mascon_points, mascon_masses, vertices, triangle_faces, the name of the sample

    """
    # Get the mascon & mesh data
    mascon_points, mascon_masses = sample_reader.load_mascon_data(sample)
    vertices, triangles = sample_reader.load_polyhedral_mesh(sample)

    # Pack everything together
    return (mascon_points, mascon_masses), (vertices, triangles), sample


# The Data produced by get_data
TEST_DATA = [get_data(file_name) for file_name in TEST_FILENAMES]
# The parameters combined as test input data & the test distances
TEST_PARAMETERS = list(itertools.product(TEST_DATA, TEST_DISTANCES))
# More readable names for the parametrized tests
TEST_NAMES = [f"{name}-{distance}" for name, distance in itertools.product(TEST_FILENAMES, TEST_DISTANCES)]


# ======================== TEST CASE ==========================
@pytest.mark.parametrize("data, distance", TEST_PARAMETERS, ids=TEST_NAMES)
def test_compare_mascon_polyhedral_model(data, distance):
    """
    Compares the mascon model to the polyhedral gravity models results
    """
    mascon_data, mesh_data, body_name = data
    # Set the print precision of torch for more reasonable messages
    torch.set_printoptions(precision=20)
    # Get the Mascon Parameters
    mascon_points, mascon_masses = mascon_data
    # Get the Mesh Parameters
    vertices, triangles = mesh_data

    # First assert that the mass of the mascon is actually normed and equals one (1.0)
    assert torch.sum(mascon_masses) == pytest.approx(1.0)

    # Get a function to sample the input points
    get_target_point = get_target_point_sampler(TEST_POINTS_PER_BATCH,
                                                'altitude',
                                                [distance],
                                                limit_shape_to_asteroid=f"./3dmeshes/{body_name}.pk",
                                                replace=True,
                                                seed=TEST_SEED)
    # The actual sample points
    target_points = get_target_point()

    # Compute the potential and the acceleration with the two model
    mascon_potential = torch.squeeze(mascon.potential(target_points, mascon_points, mascon_masses))
    polyhedral_potential = torch.squeeze(
        polyhedral.potential(target_points, vertices, triangles))
    mascon_acceleration = torch.squeeze(mascon.acceleration(target_points, mascon_points, mascon_masses))
    polyhedral_acceleration = torch.squeeze(
        polyhedral.acceleration(target_points, vertices, triangles))

    # Compare the results
    assert mascon_potential == pytest.approx(polyhedral_potential, rel=TEST_EPSILON)
    root_squared_mean_error = relRMSE(mascon_acceleration, polyhedral_acceleration)
    assert root_squared_mean_error < TEST_relRMSE


@pytest.mark.parametrize("sample", TEST_FILENAMES, ids=TEST_FILENAMES)
def test_check_input_mesh(sample):
    """
    Checks that the input mesh's plane unit normals are outwards pointing
    """
    vertices, triangles = sample_reader.load_polyhedral_mesh(sample)
    assert check_mesh(vertices, triangles)
