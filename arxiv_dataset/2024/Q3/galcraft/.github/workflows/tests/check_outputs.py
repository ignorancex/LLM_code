import numpy as np
from astropy.io import fits
import pytest
from pathlib import Path

# Define the directories for the test files
test_dir = Path(__file__).parent
true_dir = test_dir / "outputs"
# outputs_dir = "/home/runner/work/galcraft/galcraft/galcraft-test_kit/tests/outputs"
outputs_dir = test_dir / "galcraft-test_kit/tests/output/test_kit"


files_npy = ["mass_fraction_array_0.npy", "mass_fraction_array_1.npy",
             "statistic_count0.npy", "statistic_count1.npy"]
@pytest.mark.parametrize("parafile", files_npy)
def test_npy(parafile):
    output_file = outputs_dir / parafile
    true_file = true_dir / parafile
    output = np.load(output_file)
    true = np.load(true_file)
    assert np.all(output == true), f"The files {output_file} and {true_file} are not the same!"


files_fits1 = ["data_cube_0.fits", "data_cube_1.fits"]
@pytest.mark.parametrize("parafile", files_fits1)
def test_fits1(parafile):
    output_file = outputs_dir / parafile
    true_file = true_dir / parafile
    output = fits.open(output_file)
    true = fits.open(true_file)
    assert np.array_equal(output[1].data, true[1].data, equal_nan=True), f"The files {output_file} and {true_file} are not the same!"

# files_fits2 = ["particle_table_0.fits", "particle_table_1.fits"]
# @pytest.mark.parametrize("parafile", files_fits2)
# def test_fits2(parafile):
#     output_file = outputs_dir / parafile
#     true_file = true_dir / parafile
#     output = fits.open(output_file)
#     true = fits.open(true_file)
#     assert np.all(output[1].data == true[1].data), f"The files {output_file} and {true_file} are not the same!"
