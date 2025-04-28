import numpy as np

def coords_to_binned_coords(coords, bin_width=1., eps=1e-6, mode="center"):
    if mode == "center":
        offset = -bin_width * 0.5
    elif mode == "corner":
        offset = 0
    else:
        raise ValueError(f"unknown mode: {mode}")
    bin_idx = np.floor((coords - offset + bin_width * eps) / bin_width).astype(int)
    bin_idx = np.unique(bin_idx, axis=0)
    binned_coords = bin_idx * bin_width
    return binned_coords

