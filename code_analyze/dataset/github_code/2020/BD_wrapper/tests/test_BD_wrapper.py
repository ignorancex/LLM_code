import os
import unittest
import numpy as np
from astropy.table import Table
import BD_wrapper.BD_wrapper as bdw


class TestBayesianDistance(unittest.TestCase):
    def setUp(self):
        self.dirname = os.path.dirname(
            os.path.dirname(os.path.realpath(__file__)))
        self.kda_table_test = Table.read(
            os.path.join(self.dirname, 'data', 'kda_table_test.dat'),
            format='ascii')
        self.bdc = bdw.BayesianDistance()

    def test_get_cartesian_coords(self):
        bdc = bdw.BayesianDistance()

        for (glon, glat, dist), solution in zip(
                [(0, 0, 10),
                 (90, 0, 10),
                 (0, 90, 10)],
                [(10.0, 0.0, 0.0),
                 (0.0, 10.0, 0.0),
                 (0.0, 0.0, 10.0)]):
            result = bdc.get_cartesian_coords(glon, glat, dist)
            self.assertEqual(result, solution)

    def test_get_kda(self):
        bdc = bdw.BayesianDistance()
        weight, refs = bdc.get_kda(
            np.array([-0.5, 0.4]), ['right_ref', 'wrong_ref'])
        assert refs == 'right_ref'

    def test_point_in_ellipse(self):
        bdc = bdw.BayesianDistance()
        bdc._size = 'fwhm'
        bdc._threshold_spatial = 0.125

        mask_weight, weight = bdc.point_in_ellipse(
            self.kda_table_test, 30, 2.001)
        self.assertFalse(mask_weight)

        mask_weight, weight = bdc.point_in_ellipse(
            self.kda_table_test, 30, 2)
        self.assertEqual(round(weight[0], 3), 0.125)

        mask_weight, weight = bdc.point_in_ellipse(
            self.kda_table_test, 30, 1)
        self.assertEqual(weight[0], 1)

    def test_get_table_distance_max_probability(self):
        bdc = bdw.BayesianDistance()
        bdc.verbose = False
        bdc.version == '2.4'
        bdc.table_results = Table.read(os.path.join(
            self.dirname, 'data', 'table_pmax_test.dat'), format='ascii')
        bdc.get_table_distance_max_probability(save=False)

        for i, (value, flag) in enumerate(zip(
                np.arange(1, 11, 1), [0, 1, 1, 2, 2, 3, 3, 4, 4, 0])):
            self.assertEqual(bdc.table_results['dist'][i], value)
            self.assertEqual(bdc.table_results['flag'][i], flag)

    # def test_init(self):
    #     Aperture(8, 8, 4, data=self.test_data)
    #     # Non-integer indizes:
    #     Aperture(8.2, 8.3, 4, data=self.test_data)
    #     # Non-interger radius:
    #     with self.assertRaises(ValueError):
    #         Aperture(8.2, 8.3, 3.7, data=self.test_data)
    #
    # def test_crop(self):
    #     aperture = Aperture(8, 8, 4, data=self.test_data, crop=True)
    #     aperture.crop()
    #     # imshow(aperture())
    #     aperture = Aperture(8, 8, 4, data=self.test_data, crop=False)
    #     # imshow(aperture())
    #     aperture.crop()
    #     # imshow(aperture())
    #
    # def test_encircled_energy(self):
    #     aperture = Aperture(8, 8, 4, data=self.test_data)
    #     aperture.get_encircled_energy(saveto='data/test/example_encircled_energy.dat')
    #
    # def test_call(self):
    #     imshow(self.test_data)
    #     test_aperture = Aperture(8, 8, 4, data=self.test_data)
    #     imshow(test_aperture.data)
    #     test_aperture = Aperture(64, 64, 16, data=self.test_data_large)
    #     imshow(test_aperture.data)


if __name__ == "__main__":
    unittest.main()
