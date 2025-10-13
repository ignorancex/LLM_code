import pytest
import torch
import numpy as np
from scipy import ndimage
import dfreproject.lanczos as lanczos


class TestLanczosKernel:
    """Test the LANCZOS kernel function."""

    def test_kernel_at_zero(self):
        """Test that kernel equals 1 at x=0."""
        result = lanczos.lanczos_kernel(torch.tensor(0.0))
        assert torch.isclose(result, torch.tensor(1.0), atol=1e-6)

    def test_kernel_symmetry(self):
        """Test that kernel is symmetric around 0."""
        x = torch.linspace(-2.5, 2.5, 21)
        result = lanczos.lanczos_kernel(x)
        result_flipped = lanczos.lanczos_kernel(-x)
        assert torch.allclose(result, result_flipped, atol=1e-6)

    def test_kernel_support(self):
        """Test that kernel is zero outside [-3, 3] for a=3."""
        # Test points outside support
        x_outside = torch.tensor([3.1, -3.1, 4.0, -4.0])
        result = lanczos.lanczos_kernel(x_outside)
        expected = torch.zeros_like(x_outside)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_kernel_different_dtypes(self):
        """Test kernel with different data types."""
        x_float32 = torch.tensor(1.0, dtype=torch.float32)
        x_float64 = torch.tensor(1.0, dtype=torch.float64)

        result_32 = lanczos.lanczos_kernel(x_float32)
        result_64 = lanczos.lanczos_kernel(x_float64)

        assert result_32.dtype == torch.float32
        assert result_64.dtype == torch.float64
        assert torch.isclose(result_32.double(), result_64, atol=1e-6)

    def test_kernel_batch_processing(self):
        """Test kernel with batch inputs."""
        x = torch.tensor([[0.0, 1.0, 2.0], [0.5, 1.5, 2.5]])
        result = lanczos.lanczos_kernel(x)
        assert result.shape == (2, 3)
        assert torch.isclose(result[0, 0], torch.tensor(1.0), atol=1e-6)  # x=0 case


class TestLanczosGridSample:
    """Test the LANCZOS grid sampling function."""

    @pytest.fixture
    def simple_image(self):
        """Create a simple test image."""
        # Create a 16x16 image with a simple pattern
        torch.manual_seed(42)
        img = torch.randn(1, 1, 16, 16, dtype=torch.float64)

        # Add a smooth pattern for better testing
        x, y = torch.meshgrid(
            torch.linspace(0, 2 * np.pi, 16),
            torch.linspace(0, 2 * np.pi, 16),
            indexing='ij'
        )
        img[0, 0] += torch.sin(x) * torch.cos(y)
        return img

    @pytest.fixture
    def identity_grid(self):
        """Create an identity grid (no transformation)."""
        H_out, W_out = 16, 16
        i, j = torch.meshgrid(
            torch.linspace(-1, 1, H_out),
            torch.linspace(-1, 1, W_out),
            indexing='ij'
        )
        grid = torch.stack([j, i], dim=-1).unsqueeze(0)  # (1, H_out, W_out, 2)
        return grid.double()

    @pytest.fixture
    def scaling_grid(self):
        """Create a grid that scales down by 0.8."""
        H_out, W_out = 12, 12
        i, j = torch.meshgrid(
            torch.linspace(-0.8, 0.8, H_out),
            torch.linspace(-0.8, 0.8, W_out),
            indexing='ij'
        )
        grid = torch.stack([j, i], dim=-1).unsqueeze(0)
        return grid.double()

    def test_identity_transformation(self, simple_image, identity_grid):
        """Test that identity grid preserves the image."""
        result = lanczos.lanczos_grid_sample(simple_image, identity_grid)

        # Should be very close to original (allowing for interpolation differences at edges)
        center_slice = slice(2, -2)  # Avoid edge effects
        original_center = simple_image[0, 0, center_slice, center_slice]
        result_center = result[0, 0, center_slice, center_slice]

        # Should be reasonably close
        diff = torch.mean(torch.abs(original_center - result_center))
        assert diff < 0.1  # Allow for some interpolation error

    def test_output_shape(self, simple_image):
        """Test that output has correct shape."""
        H_out, W_out = 20, 24
        i, j = torch.meshgrid(
            torch.linspace(-1, 1, H_out),
            torch.linspace(-1, 1, W_out),
            indexing='ij'
        )
        grid = torch.stack([j, i], dim=-1).unsqueeze(0).double()

        result = lanczos.lanczos_grid_sample(simple_image, grid)

        expected_shape = (1, 1, H_out, W_out)
        assert result.shape == expected_shape

    def test_padding_zeros(self):
        """Test that out-of-bounds sampling returns zeros."""
        # Create a small image with distinctive values
        img = torch.zeros(1, 1, 6, 6, dtype=torch.float64)  # Start with zeros
        img[0, 0, 2:4, 2:4] = 5.0  # Put 5.0 only in the center 2x2 region

        # Test 1: Sample way outside the image
        grid_far_out = torch.tensor([[[[1.8, 1.8]]]], dtype=torch.float64)  # Very far outside
        result_far = lanczos.lanczos_grid_sample(img, grid_far_out)

        # Test 2: Sample in the zero region but inside the image
        grid_zero_region = torch.tensor([[[[-0.8, -0.8]]]], dtype=torch.float64)  # Maps to ~(0.3, 0.3)
        result_zero = lanczos.lanczos_grid_sample(img, grid_zero_region)

        print(f"Far outside result: {result_far[0, 0, 0, 0]}")
        print(f"Zero region result: {result_zero[0, 0, 0, 0]}")

        # Far outside should be close to zero
        assert torch.abs(result_far[0, 0, 0, 0]) < 0.5

        # Zero region should be close to zero (but might have tiny contributions from center)
        assert torch.abs(result_zero[0, 0, 0, 0]) < 1.0  # More lenient

    def test_true_zero_padding(self):
        """Test a clearer case of zero padding."""
        # Create image where only center pixel is non-zero
        img = torch.zeros(1, 1, 8, 8, dtype=torch.float64)
        img[0, 0, 4, 4] = 10.0  # Only center pixel has value

        # Sample far from the center pixel
        grid = torch.tensor([[[[0.9, 0.9]]]], dtype=torch.float64)  # Maps to ~(7.65, 7.65)
        result = lanczos.lanczos_grid_sample(img, grid)

        print(f"Result when sampling far from single pixel: {result[0, 0, 0, 0]}")

        # Should be very close to zero since we're far from the single non-zero pixel
        assert torch.abs(result[0, 0, 0, 0]) < 0.1

    def test_different_chunk_sizes(self, simple_image, scaling_grid):
        """Test that different chunk sizes give the same result."""
        result_small_chunks = lanczos.lanczos_grid_sample(simple_image, scaling_grid, chunk_size=4)
        result_large_chunks = lanczos.lanczos_grid_sample(simple_image, scaling_grid, chunk_size=16)

        assert torch.allclose(result_small_chunks, result_large_chunks, atol=1e-10)

    def test_batch_processing(self):
        """Test processing multiple images in a batch."""
        # Create batch of 2 images
        batch_img = torch.randn(2, 1, 8, 8, dtype=torch.float64)

        # Create batch of grids
        H_out, W_out = 6, 6
        i, j = torch.meshgrid(
            torch.linspace(-0.8, 0.8, H_out),
            torch.linspace(-0.8, 0.8, W_out),
            indexing='ij'
        )
        grid = torch.stack([j, i], dim=-1).unsqueeze(0).double()
        batch_grid = grid.repeat(2, 1, 1, 1)

        result = lanczos.lanczos_grid_sample(batch_img, batch_grid)

        assert result.shape == (2, 1, H_out, W_out)

        # Process individually and compare
        result_0 = lanczos.lanczos_grid_sample(batch_img[[0]], batch_grid[[0]])
        result_1 = lanczos.lanczos_grid_sample(batch_img[[1]], batch_grid[[1]])

        assert torch.allclose(result[0:1], result_0, atol=1e-10)
        assert torch.allclose(result[1:2], result_1, atol=1e-10)

    def test_multichannel_image(self):
        """Test processing multi-channel images."""
        # Create RGB-like image
        img = torch.randn(1, 3, 8, 8, dtype=torch.float64)

        H_out, W_out = 6, 6
        i, j = torch.meshgrid(
            torch.linspace(-0.8, 0.8, H_out),
            torch.linspace(-0.8, 0.8, W_out),
            indexing='ij'
        )
        grid = torch.stack([j, i], dim=-1).unsqueeze(0).double()

        result = lanczos.lanczos_grid_sample(img, grid)

        assert result.shape == (1, 3, H_out, W_out)

        # Each channel should be processed independently
        for c in range(3):
            single_channel_result = lanczos.lanczos_grid_sample(img[:, c:c + 1], grid)
            assert torch.allclose(result[:, c:c + 1], single_channel_result, atol=1e-10)

    def test_dtype_preservation(self, simple_image):
        """Test that data type is preserved."""
        # Test float32
        img_32 = simple_image.float()
        grid = torch.randn(1, 4, 4, 2).float()
        result_32 = lanczos.lanczos_grid_sample(img_32, grid)
        assert result_32.dtype == torch.float32

        # Test float64
        img_64 = simple_image.double()
        grid_64 = grid.double()
        result_64 = lanczos.lanczos_grid_sample(img_64, grid_64)
        assert result_64.dtype == torch.float64


class TestScipyComparison:
    """Test comparison with scipy implementation."""

    def scipy_reference(self, source_array, grid_array):
        """Reference implementation using scipy."""
        H, W = source_array.shape
        H_out, W_out, _ = grid_array.shape

        # Convert grid from [-1, 1] to pixel coordinates
        coords_x = ((grid_array[..., 0] + 1) / 2) * (W - 1)
        coords_y = ((grid_array[..., 1] + 1) / 2) * (H - 1)

        # scipy expects coordinates as (2, H_out, W_out) - note y,x order
        coordinates = np.array([coords_y.flatten(), coords_x.flatten()])

        # Use map_coordinates with cubic spline (order=3)
        result = ndimage.map_coordinates(
            source_array,
            coordinates,
            order=3,
            cval=0.0,
            prefilter=True
        )

        return result.reshape(H_out, W_out)

    def test_comparison_with_scipy(self):
        """Compare LANCZOS results with scipy cubic interpolation."""
        # Create test image
        np.random.seed(42)
        source_np = np.random.randn(24, 24).astype(np.float64)

        # Add smooth features
        x, y = np.meshgrid(np.linspace(0, 2 * np.pi, 24), np.linspace(0, 2 * np.pi, 24))
        source_np += 0.5 * np.sin(x) * np.cos(y)

        # Convert to torch
        source_torch = torch.from_numpy(source_np).unsqueeze(0).unsqueeze(0)

        # Create test grid
        H_out, W_out = 18, 18
        i, j = torch.meshgrid(
            torch.linspace(-0.8, 0.8, H_out),
            torch.linspace(-0.8, 0.8, W_out),
            indexing='ij'
        )
        grid = torch.stack([j, i], dim=-1).unsqueeze(0).double()
        grid_np = grid[0].numpy()

        # Get results
        result_lanczos = lanczos.lanczos_grid_sample(source_torch, grid)[0, 0].numpy()
        result_scipy = self.scipy_reference(source_np, grid_np)

        # Compare (they won't be identical due to different algorithms)
        diff = np.abs(result_lanczos - result_scipy)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        # Should be reasonably close (different interpolation methods)
        assert max_diff < 0.5, f"Max difference {max_diff} too large"
        assert mean_diff < 0.1, f"Mean difference {mean_diff} too large"

    def test_pytorch_grid_sample_compatibility(self):
        """Test that grid format is compatible with PyTorch."""
        # Create test data
        img = torch.randn(1, 1, 16, 16, dtype=torch.float64)

        # Create identical grids
        H_out, W_out = 12, 12
        i, j = torch.meshgrid(
            torch.linspace(-0.6, 0.6, H_out),
            torch.linspace(-0.6, 0.6, W_out),
            indexing='ij'
        )
        grid = torch.stack([j, i], dim=-1).unsqueeze(0).double()

        # Compare shapes and basic properties
        result_lanczos = lanczos.lanczos_grid_sample(img, grid)
        result_bilinear = torch.nn.functional.grid_sample(
            img, grid, mode='bilinear', align_corners=True, padding_mode='zeros'
        )

        # Should have same shape
        assert result_lanczos.shape == result_bilinear.shape

        # Results should be different (different interpolation) but reasonable
        diff = torch.abs(result_lanczos - result_bilinear)
        assert torch.max(diff) > 0.01  # Should be different
        assert torch.mean(diff) < 1.0  # But not wildly different


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_image(self):
        """Test behavior with minimal image size."""
        img = torch.tensor([[[[1.0]]]], dtype=torch.float64)  # 1x1 image
        grid = torch.tensor([[[[0.0, 0.0]]]], dtype=torch.float64)  # Sample at center

        result = lanczos.lanczos_grid_sample(img, grid)

        # Should work and preserve the single value (approximately)
        assert result.shape == (1, 1, 1, 1)
        assert torch.abs(result[0, 0, 0, 0] - 1.0) < 0.1

    def test_large_grid_values(self):
        """Test with grid values at the boundaries."""
        img = torch.ones(1, 1, 4, 4, dtype=torch.float64)

        # Test extreme grid values
        grid = torch.tensor([[[[1.0, 1.0], [-1.0, -1.0]]]], dtype=torch.float64)

        result = lanczos.lanczos_grid_sample(img, grid)

        # Should not crash and should return reasonable values
        assert result.shape == (1, 1, 1, 2)
        assert torch.all(torch.isfinite(result))

    def test_memory_efficiency_large_chunks(self):
        """Test that large chunk sizes don't cause memory issues."""
        # Create moderately large image
        img = torch.randn(1, 1, 64, 64, dtype=torch.float64)

        # Create output grid
        H_out, W_out = 48, 48
        i, j = torch.meshgrid(
            torch.linspace(-0.9, 0.9, H_out),
            torch.linspace(-0.9, 0.9, W_out),
            indexing='ij'
        )
        grid = torch.stack([j, i], dim=-1).unsqueeze(0).double()

        # Should work with large chunk size
        result = lanczos.lanczos_grid_sample(img, grid, chunk_size=2048)

        assert result.shape == (1, 1, H_out, W_out)
        assert torch.all(torch.isfinite(result))


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])