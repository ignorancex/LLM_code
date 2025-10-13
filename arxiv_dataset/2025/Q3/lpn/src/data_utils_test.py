import functools
import unittest

import jax
import jax.numpy as jnp

from src.data_utils import _apply_rotation


class TestDataAugmentation(unittest.TestCase):

    def setUp(self):
        self.grid = jnp.array(
            [[1, 2, 3, 0, 0], [4, 5, 6, 0, 0], [7, 8, 9, 0, 0], [10, 11, 12, 0, 0], [0, 0, 0, 0, 0]]
        )
        self.grid_shape = jnp.array([4, 3])
        self.grid2 = jnp.array(
            [[1, 2, 0, 0, 0], [3, 4, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
        )
        self.grid_shape2 = jnp.array([2, 2])

    def test__apply_rotation_0(self):
        grid, grid_shape = _apply_rotation(self.grid, self.grid_shape, k=0)
        self.assertTrue(jnp.array_equal(grid_shape, self.grid_shape))
        self.assertTrue(jnp.array_equal(grid, self.grid))

    def test__apply_rotation_1(self):
        grid, grid_shape = _apply_rotation(self.grid, self.grid_shape, k=1)
        expected_grid_shape = jnp.array([3, 4])
        expected_grid = jnp.array(
            [[10, 7, 4, 1, 0], [11, 8, 5, 2, 0], [12, 9, 6, 3, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
        )
        self.assertTrue(jnp.array_equal(grid_shape, expected_grid_shape))
        self.assertTrue(jnp.array_equal(grid, expected_grid))

    def test__apply_rotation_2(self):
        grid, grid_shape = _apply_rotation(self.grid, self.grid_shape, k=2)
        expected_grid_shape = jnp.array([4, 3])
        expected_grid = jnp.array(
            [[12, 11, 10, 0, 0], [9, 8, 7, 0, 0], [6, 5, 4, 0, 0], [3, 2, 1, 0, 0], [0, 0, 0, 0, 0]]
        )
        self.assertTrue(jnp.array_equal(grid_shape, expected_grid_shape))
        self.assertTrue(jnp.array_equal(grid, expected_grid))

    def test_vmap__apply_rotation_2(self):
        grids = jnp.stack([self.grid, self.grid2])
        grid_shapes = jnp.stack([self.grid_shape, self.grid_shape2])
        output_grids, output_grid_shapes = jax.vmap(functools.partial(_apply_rotation, k=2))(
            grids, grid_shapes
        )
        expected_grid_shapes = jnp.array([[4, 3], [2, 2]])
        expected_grids = jnp.array(
            [
                [[12, 11, 10, 0, 0], [9, 8, 7, 0, 0], [6, 5, 4, 0, 0], [3, 2, 1, 0, 0], [0, 0, 0, 0, 0]],
                [[4, 3, 0, 0, 0], [2, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            ]
        )
        self.assertTrue(jnp.array_equal(output_grid_shapes, expected_grid_shapes))
        self.assertTrue(jnp.array_equal(output_grids, expected_grids))

    def tearDown(self):
        # Cleanup code to run after each test
        pass


if __name__ == "__main__":
    unittest.main()
