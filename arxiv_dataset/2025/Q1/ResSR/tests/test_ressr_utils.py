import unittest
import sys
import numpy as np

sys.path.append('../ResSR/utils/')
from ResSR.utils.ressr_utils import replicate_and_subsample

class ResSRUtilsTests(unittest.TestCase):
    def test_replicate_and_subsample(self):
        # Test case 1: Empty input
        input_list = []
        N_s = 10
        self.assertRaises(ValueError, replicate_and_subsample, input_list, N_s)

        # Test case 2: No subsampled pixels
        input_list = [np.zeros((2, 2, 1)), np.zeros((1, 1, 1))]
        N_s = 0
        self.assertRaises(ValueError, replicate_and_subsample, input_list, N_s)

        # Test case 3: 
        input_list = [np.zeros((2, 2, 1)), np.zeros((1, 1, 1))]
        N_s = 4
        expected_output = np.zeros((4,2))
        self.assertEqual(replicate_and_subsample(input_list, N_s).tolist(), expected_output.tolist())

        # Test case 4: 
        input_list = [np.zeros((2, 2, 1)), np.zeros((1, 1, 1))]
        N_s = 1
        expected_output = np.zeros((1,2))
        self.assertEqual(replicate_and_subsample(input_list, N_s).tolist(), expected_output.tolist())

if __name__ == '__main__':
    unittest.main()