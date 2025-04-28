from scipy import ndimage
import numpy as np


def remove_connected_components(segmentation, l_min=3):
    """Remove small lesions leq than `l_min` voxels from the binary segmentation mask.
    """
    if l_min > 0:
        if segmentation.ndim != 3:
            raise ValueError(f"Mask must have 3 dimensions, got {segmentation.ndim}.")
        struct_el = ndimage.generate_binary_structure(rank=3, connectivity=2)
        labeled_seg, num_labels = ndimage.label(segmentation, structure=struct_el)
        segmentation_tr = np.zeros_like(segmentation)
        for label in range(1, num_labels + 1):
            if np.sum(labeled_seg == label) > l_min:
                segmentation_tr[labeled_seg == label] = 1
        return segmentation_tr
    else:
        return segmentation.copy()


def get_cc_mask(binary_mask):
    """ Get a labeled mask from a binary one """
    struct_el = ndimage.generate_binary_structure(rank=3, connectivity=2)
    return ndimage.label(binary_mask, structure=struct_el)[0]


def process_probs(prob_map, threshold, l_min):
    """ thresholding + removing cc < lmin"""
    binary_mask = prob_map.copy()
    binary_mask[binary_mask >= threshold] = 1.
    binary_mask[binary_mask < threshold] = 0.
    return remove_connected_components(binary_mask, l_min=l_min)