# This code is part of the CM4IR project by tirer-lab.
# Repository: https://github.com/tirer-lab/CM4IR
# If you use or modify this code, please provide appropriate credit.

import torch


def median_inpainting(y, mask, in_channels, image_size):
    device = y.device
    y = y.squeeze(0)
    mask = mask.view(1, image_size, image_size).expand(in_channels, -1, -1).to(device)

    y0 = y.clone().float()
    y0[mask == 0] = float('nan')

    win_size = 1
    C, M, N = y0.shape
    bitmap_NaN = torch.isnan(y0[0])

    while torch.count_nonzero(bitmap_NaN) > 0:
        y0_prev = y0.clone()
        win_size += 1
        rows, cols = torch.where(bitmap_NaN)
        for r, c in zip(rows.tolist(), cols.tolist()):
            row_start = max(0, r - win_size)
            row_end = min(M, r + win_size + 1)
            col_start = max(0, c - win_size)
            col_end = min(N, c + win_size + 1)

            patch = y0_prev[:, row_start:row_end, col_start:col_end]
            patch_flat = patch.reshape(C, -1)
            median_vals = torch.nanmedian(patch_flat, dim=1).values
            y0[:, r, c] = median_vals

        bitmap_NaN = torch.isnan(y0[0])

    return y0.view(1, -1)
