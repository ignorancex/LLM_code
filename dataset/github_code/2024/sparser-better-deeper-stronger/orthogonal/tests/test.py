import unittest
import torch
from orthogonal.ortho import *


PRECISION = 1e-6

def are_almost_same(value1: torch.Tensor, value2: torch.Tensor, precision: float) -> bool:
    return torch.abs(torch.mean(value1 - value2)) < precision
        
def is_col_orthogonal(matrix: torch.Tensor): # (result, reason)
    for i, col in enumerate(matrix.T):
        if not are_almost_same(col.norm(), 1.0, PRECISION):
            print(f"Col {i} has non-unit length: {col.norm()}.")
            return False
        for j in range(i + 1, len(matrix.T)):
            other_col = matrix.T[j]
            dt = torch.dot(col, other_col)
            if not are_almost_same(dt, 0.0, PRECISION):
                print(f"Cols {i} and {j} have non-zero dot product: {dt}.")
                return False
    return True

class TestGivens(unittest.TestCase):

    def test_givens_composition_is_orthogonal_tiny(self):
        a = identity(2, 2)
        self.assertTrue(is_col_orthogonal(a))
        i, j, alpha = rand_givens(2)
        mul_by_givens(a, i, j, alpha)
        self.assertTrue(is_col_orthogonal(a))

    def test_givens_composition_is_orthogonal_small(self):
        a = identity(10, 10)
        self.assertTrue(is_col_orthogonal(a))
        for _ in range(20):
            i, j, alpha = rand_givens(10)
            mul_by_givens(a, i, j, alpha)
            self.assertTrue(is_col_orthogonal(a))

    def test_givens_composition_is_orthogonal_medium(self):
        a = identity(100, 100)
        self.assertTrue(is_col_orthogonal(a))
        for p in range(600):
            i, j, alpha = rand_givens(100)
            mul_by_givens(a, i, j, alpha)
            self.assertTrue(is_col_orthogonal(a))

    def test_givens_composition_is_orthogonal_rectangular_cols(self):
        a = identity(40, 20) # more rows than cols
        self.assertTrue(is_col_orthogonal(a))
        for p in range(100):
            i, j, alpha = rand_givens(40)
            mul_by_givens(a.T, i, j, alpha)
            self.assertTrue(is_col_orthogonal(a))

    def test_givens_composition_has_not_orthogonal_columns_when_matrix_is_wide(self): # sanity check
        a = identity(20, 40) # more cols than rows
        self.assertFalse(is_col_orthogonal(a))
        for p in range(100):
            i, j, alpha = rand_givens(40)
            mul_by_givens(a, i, j, alpha)
            self.assertFalse(is_col_orthogonal(a))


class TestDensity(unittest.TestCase):

    def test_get_density_small(self):
        a = torch.tensor([
            [1.0, 2.0, 0.0],
            [0.0, 0.0, 4.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, -7.0],
        ])
        self.assertTrue(are_almost_same(get_density(a), torch.tensor(5.0 / 12.0), PRECISION))

    

if __name__ == '__main__':
    unittest.main()