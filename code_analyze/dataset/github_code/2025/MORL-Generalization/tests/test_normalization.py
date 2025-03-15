import numpy as np
import unittest

def get_normalized_vec_returns(all_vec_returns, minmax_range):
    # Convert minmax_range to a NumPy array for efficient operations
    minmax_array = np.array([minmax_range[str(i)] for i in range(all_vec_returns.shape[-1])])
    min_vals = minmax_array[:, 0].reshape(1, 1, -1)  # Reshape for broadcasting
    max_vals = minmax_array[:, 1].reshape(1, 1, -1)

    # Clip values to min and max
    clipped_vec_returns = np.clip(all_vec_returns, min_vals, max_vals)
    
    # Normalize
    normalized_vec_returns = (clipped_vec_returns - min_vals) / (max_vals - min_vals)
    
    return normalized_vec_returns

class TestNormalization(unittest.TestCase):
    def test_get_normalized_vec_returns(self):
        # Example input and expected output
        all_vec_returns = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]], [[0, 1, 2], [11, 12, 13]]])
        minmax_range = {'0': (1, 10), '1': (2, 11), '2': (3, 12)}
        
        # Expected normalization
        expected_output = np.array([[[0., 0., 0.], [0.33333333, 0.33333333, 0.33333333]],
                                    [[0.66666667, 0.66666667, 0.66666667], [1., 1., 1.]],
                                    [[0, 0, 0], [1., 1., 1.]]])
        
        # Perform normalization
        normalized_vec_returns = get_normalized_vec_returns(all_vec_returns, minmax_range)
        
        # Check if the output matches the expected output
        np.testing.assert_almost_equal(normalized_vec_returns, expected_output, decimal=7, 
                                       err_msg="Normalization did not produce expected output.")

    def test_values_below_minimum(self):
        # Test when values are below the minimum range
        all_vec_returns = np.array([[[0, 1, 2]]])
        minmax_range = {'0': (1, 10), '1': (2, 11), '2': (3, 12)}
        
        expected_output = np.array([[[0., 0., 0.]]])
        
        normalized_vec_returns = get_normalized_vec_returns(all_vec_returns, minmax_range)
        np.testing.assert_almost_equal(normalized_vec_returns, expected_output, decimal=7,
                                       err_msg="Values below minimum were not handled correctly.")

    def test_values_above_maximum(self):
        # Test when values are above the maximum range
        all_vec_returns = np.array([[[15, 20, 25]]])
        minmax_range = {'0': (1, 10), '1': (2, 11), '2': (3, 12)}
        
        expected_output = np.array([[[1., 1., 1.]]])
        
        normalized_vec_returns = get_normalized_vec_returns(all_vec_returns, minmax_range)
        np.testing.assert_almost_equal(normalized_vec_returns, expected_output, decimal=7,
                                       err_msg="Values above maximum were not handled correctly.")

    def test_edge_case_single_element(self):
        # Test with a single element
        all_vec_returns = np.array([[[5, 7, 9]]])
        minmax_range = {'0': (0, 10), '1': (2, 12), '2': (3, 15)}
        
        expected_output = np.array([[[0.5, 0.5, 0.5]]])
        
        normalized_vec_returns = get_normalized_vec_returns(all_vec_returns, minmax_range)
        np.testing.assert_almost_equal(normalized_vec_returns, expected_output, decimal=7,
                                       err_msg="Single element case not handled correctly.")

    def test_large_input(self):
        # Test with large input arrays to ensure function scales well
        all_vec_returns = np.random.uniform(0, 100, (100, 100, 3))
        minmax_range = {'0': (0, 100), '1': (0, 100), '2': (0, 100)}
        
        expected_output = all_vec_returns / 100  # Since min is 0 and max is 100
        
        normalized_vec_returns = get_normalized_vec_returns(all_vec_returns, minmax_range)
        np.testing.assert_almost_equal(normalized_vec_returns, expected_output, decimal=7,
                                       err_msg="Large input not handled correctly.")
        
    def test_negative_range(self):
        # Test with a negative range
        all_vec_returns = np.array([[[ -5, 0, 5], [10, -10, 25]]])
        minmax_range = {'0': (-10, 10), '1': (-20, 20), '2': (-5, 25)}
        
        expected_output = np.array([[[0.25, 0.5, 0.33333333], [1., 0.25, 1.]]])
        
        normalized_vec_returns = get_normalized_vec_returns(all_vec_returns, minmax_range)
        np.testing.assert_almost_equal(normalized_vec_returns, expected_output, decimal=7,
                                       err_msg="Negative range not handled correctly.")

    def test_mixed_positive_negative(self):
        # Test with mixed positive and negative values
        all_vec_returns = np.array([[[5, -10, 15], [-5, 10, 0]]])
        minmax_range = {'0': (0, 10), '1': (-20, 20), '2': (0, 30)}
        
        expected_output = np.array([[[0.5, 0.25, 0.5], [0., 0.75, 0.]]])
        
        normalized_vec_returns = get_normalized_vec_returns(all_vec_returns, minmax_range)
        np.testing.assert_almost_equal(normalized_vec_returns, expected_output, decimal=7,
                                       err_msg="Mixed positive and negative range not handled correctly.")

if __name__ == '__main__':
    unittest.main()
