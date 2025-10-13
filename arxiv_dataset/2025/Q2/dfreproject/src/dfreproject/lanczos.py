import torch
import numpy as np


def lanczos_kernel(x, a=3):
    """
    Lanczos kernel function with parameter a (number of lobes).

    Args:
        x: Distance from center
        a: Number of lobes (typically 3 for Lanczos-3)

    Returns:
        Kernel weights
    """
    x = torch.abs(x)

    # Lanczos kernel: sinc(x) * sinc(x/a) for |x| < a, 0 otherwise
    mask = x < a
    result = torch.zeros_like(x)

    # Avoid division by zero
    x_nonzero = x[mask]
    x_nonzero = torch.where(x_nonzero == 0, torch.tensor(1e-8, dtype=x.dtype, device=x.device), x_nonzero)

    # sinc(x) = sin(π*x) / (π*x)
    sinc_x = torch.sin(np.pi * x_nonzero) / (np.pi * x_nonzero)
    sinc_x_a = torch.sin(np.pi * x_nonzero / a) / (np.pi * x_nonzero / a)

    result[mask] = sinc_x * sinc_x_a

    # Handle x=0 case (sinc(0) = 1)
    result[x == 0] = 1.0

    return result


def _process_chunk_vectorized(img, target_x, target_y, radius, dtype, device):
    """
    Vectorized processing of a chunk of output pixels.

    Args:
        img: Source image tensor (H, W)
        target_x: Target x coordinates (chunk_h, chunk_w)
        target_y: Target y coordinates (chunk_h, chunk_w)
        radius: Lanczos radius
        dtype: Data type
        device: Device

    Returns:
        Interpolated chunk (chunk_h, chunk_w)
    """
    H, W = img.shape
    chunk_h, chunk_w = target_x.shape

    # Initialize accumulators
    result = torch.zeros_like(target_x, dtype=dtype, device=device)
    total_weights = torch.zeros_like(target_x, dtype=dtype, device=device)

    # For each source pixel offset within the kernel support
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            # Source pixel coordinates for entire chunk - THIS IS THE KEY FIX
            # We need to use floor of target coordinates as the base, not round
            src_y = torch.floor(target_y).long() + dy  # (chunk_h, chunk_w)
            src_x = torch.floor(target_x).long() + dx  # (chunk_h, chunk_w)

            # Create bounds mask
            mask = (src_y >= 0) & (src_y < H) & (src_x >= 0) & (src_x < W)

            if not mask.any():
                continue  # No valid pixels for this offset

            # Calculate Lanczos weights for this offset
            # Use actual source pixel coordinates vs target coordinates
            weight_x = lanczos_kernel(src_x.float() - target_x, a=3)
            weight_y = lanczos_kernel(src_y.float() - target_y, a=3)
            weight = weight_x * weight_y

            # Apply bounds mask
            weight_masked = weight * mask.float()

            # Gather source pixel values (only where mask is True)
            valid_indices = mask.nonzero(as_tuple=True)
            if len(valid_indices[0]) > 0:
                # Get values at valid locations
                src_y_valid = src_y[valid_indices]
                src_x_valid = src_x[valid_indices]
                values = img[src_y_valid, src_x_valid]

                # Create full values tensor
                values_full = torch.zeros_like(result)
                values_full[valid_indices] = values

                # Accumulate weighted contribution
                result += weight_masked * values_full
                total_weights += weight_masked

    # Normalize by total weights (avoid division by zero)
    mask_nonzero = total_weights > 0
    result[mask_nonzero] = result[mask_nonzero] / total_weights[mask_nonzero]

    return result


def lanczos_grid_sample(source_image, grid, padding_mode="zeros", chunk_size=1024):
    """
    Memory-efficient vectorized LANCZOS-3 interpolation for large astronomical images.

    Args:
        source_image: Input tensor of shape (N, C, H, W)
        grid: Grid tensor of shape (N, H_out, W_out, 2) with values in [-1, 1]
        padding_mode: Only "zeros" supported for now
        chunk_size: Size of chunks to process at once (for memory management)

    Returns:
        Interpolated tensor of shape (N, C, H_out, W_out)
    """
    N, C, H, W = source_image.shape
    _, H_out, W_out, _ = grid.shape

    device = source_image.device
    dtype = source_image.dtype

    # Convert grid from [-1, 1] to pixel coordinates
    grid_x = ((grid[..., 0] + 1) / 2) * (W - 1)  # Shape: (N, H_out, W_out)
    grid_y = ((grid[..., 1] + 1) / 2) * (H - 1)  # Shape: (N, H_out, W_out)

    # Initialize output
    output = torch.zeros(N, C, H_out, W_out, dtype=dtype, device=device)

    radius = 3

    # Process each batch and channel
    for n in range(N):
        for c in range(C):
            img = source_image[n, c]  # Shape: (H, W)

            # Process output in chunks to manage memory
            for i_start in range(0, H_out, chunk_size):
                i_end = min(i_start + chunk_size, H_out)

                for j_start in range(0, W_out, chunk_size):
                    j_end = min(j_start + chunk_size, W_out)

                    # Get chunk coordinates
                    chunk_x = grid_x[n, i_start:i_end, j_start:j_end]  # (chunk_h, chunk_w)
                    chunk_y = grid_y[n, i_start:i_end, j_start:j_end]  # (chunk_h, chunk_w)

                    chunk_h, chunk_w = chunk_x.shape

                    # Vectorized processing for this chunk
                    chunk_result = _process_chunk_vectorized(
                        img, chunk_x, chunk_y, radius, dtype, device
                    )

                    # Store result
                    output[n, c, i_start:i_end, j_start:j_end] = chunk_result

    return output