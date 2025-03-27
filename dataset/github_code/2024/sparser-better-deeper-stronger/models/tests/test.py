import unittest
import torch
from models.initializers import orthogonal_matrix, delta_orthogonal, conv_orthogonal, symmetric_projection
from orthogonal.tests.test import are_almost_same, is_col_orthogonal


HIGH_PRECISION = 1e-6
LOW_PRECISION = 1e-3

def random_vector(length: int, entry_std=float) -> torch.Tensor:
    return torch.FloatTensor(length).uniform_(-entry_std, entry_std)

def random_img_tensor(channels: int, size: int, entry_std=float) -> torch.Tensor:
    return torch.FloatTensor(channels, size, size).uniform_(-entry_std, entry_std)

def convolution(x: torch.Tensor, conv: torch.Tensor) -> torch.Tensor:
    _, _, ker_x, _ = conv.shape
    y = torch.nn.functional.conv2d(x, conv, padding=(ker_x - 1))
    return y

def is_conv_orthogonal(conv):  # empirical check if random inputs are not rescaled
    c_in = conv.size(1)
    for n in range(100, 120):
        x = random_img_tensor(c_in, n, entry_std=7.0)
        y = convolution(x, conv)
        return are_almost_same(x.norm() / y.norm(), 1.0, LOW_PRECISION)


class TestMatrixOrthogonal(unittest.TestCase):

    def test_generated_matrices_are_orhtogonal_square(self):
        for n in range(1, 100):
            shape_template = torch.zeros((n, n))
            a = orthogonal_matrix(shape_template)

            self.assertEqual(a.shape, (n, n))
            self.assertTrue(is_col_orthogonal(a))

    def test_generated_matrices_are_orhtogonal_rectangular(self):
        for n in range(1, 50):
            shape_template = torch.zeros((2 * n, n))
            a = orthogonal_matrix(shape_template)

            self.assertEqual(a.shape, (2 * n, n))
            self.assertTrue(is_col_orthogonal(a))


class TestConvOrthogonal(unittest.TestCase):
    def test_is_orthogonal(self):
        for ker_size in [1, 3, 5, 7, 9]:
            for c_in, c_out in [(13, 17), (16, 32), (32, 32)]:
                conv_shape_template = torch.zeros((c_out, c_in, ker_size, ker_size))
                conv = conv_orthogonal(conv_shape_template)

                self.assertEqual(conv.shape, (c_out, c_in, ker_size, ker_size))
                self.assertTrue(is_conv_orthogonal(conv), msg=f"Failed: conv not orthogonal")


class TestDeltaOrthogonal(unittest.TestCase):
    def test_is_orthogonal(self):
        for ker_size in [1, 3, 5, 7, 9]:
            for c_in, c_out in [(13, 17), (16, 32), (32, 32)]:
                conv_shape_template = torch.zeros((c_out, c_in, ker_size, ker_size))
                matrix_shape_template = torch.zeros((c_out, c_in))
                h = orthogonal_matrix(matrix_shape_template)
                conv = delta_orthogonal(conv_shape_template, h)

                self.assertEqual(conv.shape, (c_out, c_in, ker_size, ker_size))
                self.assertTrue(is_conv_orthogonal(conv), msg=f"Failed: conv not orthogonal")


class TestSymmetricProjection(unittest.TestCase):

    def test_properties(self):
        for n in range(1, 100):
            shape_template = torch.zeros((n, n))
            a = symmetric_projection(shape_template)

            self.assertTrue(are_almost_same(a, a.T, HIGH_PRECISION)) # A = A^T

            for _ in range(10):
                x = random_vector(n, entry_std=11.0)
                y = torch.mv(a, x)
                z = torch.mv(a, y)

                self.assertTrue(are_almost_same(y, z, LOW_PRECISION)) # AAx = Ax


if __name__ == '__main__':
    unittest.main()