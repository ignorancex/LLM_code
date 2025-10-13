from typing import Union

import einops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def patchify(
    images: Union[np.ndarray, torch.Tensor], patch_size: int = 16, img_size: int = 128
) -> Union[np.ndarray, torch.Tensor]:
    """
    Takes a batch of 2D or 3D images and outputs the patchify version of it.

    Args:
        images: A tensor of shape (N, C, H, W) or (N, C, D, H, W) for 3D images.
            N is the batch size, C is the number of channels, H, W, D are the height, width, and depth.
        patch_size: An integer indicating the size of the patches to be extracted.
                If an integer is provided, square/cubic patches are assumed for 2D/3D images.
        img_size: An integer indicating the size of the images. This is used to determine the number of patches to be extracted.

    Returns:
        A tensor of shape (N, L, D) where L is the number of patches and D is the flattened dimension of each patch.
    """
    assert images.dim() in (4, 5), "images must be either 4D or 5D tensors"

    n_patches_per_axis = img_size // patch_size
    unfolded = einops.rearrange(
        images,
        "b c (gh ph) (gw pw) (gd pd) -> b (gh gw gd) (ph pw pd c)",
        gh=n_patches_per_axis,
        gw=n_patches_per_axis,
        gd=n_patches_per_axis,
        ph=patch_size,
        pw=patch_size,
        pd=patch_size,
    )

    return unfolded


def unpatchify(
    images: Union[np.ndarray, torch.Tensor], patch_size: int = 16, img_size: int = 128
) -> Union[np.ndarray, torch.Tensor]:
    """
    Takes a batch of patchified images and outputs the original images.

    Args:
        images: A tensor of shape (N, L, D) where L is the number of patches and D is the flattened dimension of each patch.
        patch_size: An integer indicating the size of the patches to be extracted.
                If an integer is provided, square/cubic patches are assumed for 2D/3D images.
        img_size: An integer indicating the size of the images. This is used to determine the number of patches to be extracted.

    Returns:
        A tensor of shape (N, C, H, W) or (N, C, D, H, W) for 3D images.
        N is the batch size, C is the number of channels, H, W, D are the height, width, and depth.
    """
    assert images.dim() == 3, "images must be 3D tensors"
    n_patches_per_axis = img_size // patch_size
    images = einops.rearrange(
        images,
        "b (gh gw gd) (ph pw pd c) -> b c (gh ph) (gw pw) (gd pd)",
        gh=n_patches_per_axis,
        gw=n_patches_per_axis,
        gd=n_patches_per_axis,
        ph=patch_size,
        pw=patch_size,
        pd=patch_size,
    )
    return images


def visualize_multimodal_3d(reconstructions, inputs, mask):
    reconstructions = {m: unpatchify(v) for m, v in reconstructions.items()}
    mask = {m: v.detach() for m, v in mask.items()}
    mask = {m: v.unsqueeze(-1).repeat(1, 1, 16 * 16 * 16) for m, v in mask.items()}

    mask = {m: unpatchify(v) for m, v in mask.items()}

    reconstructions = {m: torch.einsum("nchwd->nhwdc", v).detach().cpu()[0] for m, v in reconstructions.items()}

    mask = {m: torch.einsum("nchwd->nhwdc", v).detach().cpu()[0] for m, v in mask.items()}

    inputs = {m: torch.einsum("nchwd->nhwdc", v).detach().cpu()[0] for m, v in inputs.items()}
    im_masked = {m: inputs[m] * (1 - mask[m]) for m in inputs}
    im_paste = {m: inputs[m] * (1 - mask[m]) + reconstructions[m] * mask[m] for m in inputs}

    fig = plt.figure(figsize=(10, 10))
    for i, key in enumerate(im_paste.keys()):
        plt.subplot(len(im_paste), 4, 4 * i + 1)
        plt.imshow(inputs[key][:, :, 64, -1].cpu().numpy())
        plt.axis("off")
        plt.title(f"Original - {key}")

        plt.subplot(4, 4, 4 * i + 2)
        plt.imshow(im_masked[key][:, :, 64, -1].cpu().numpy())
        plt.axis("off")
        plt.title(f"Masked - {key}")

        plt.subplot(4, 4, 4 * i + 3)
        plt.imshow(reconstructions[key][:, :, 64, -1].cpu().numpy())
        plt.axis("off")
        plt.title(f"Recons. - {key}")

        plt.subplot(4, 4, 4 * i + 4)
        plt.imshow(im_paste[key][:, :, 64, -1].cpu().numpy())
        plt.axis("off")
        plt.title(f"Recons. + Original - {key}")

    return fig
