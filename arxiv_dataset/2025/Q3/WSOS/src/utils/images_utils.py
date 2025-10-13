import numpy as np
from typing import Tuple
import plotly.express as px
import torch
from plotly.graph_objs import Figure
from skimage.filters import threshold_otsu
import cv2


def extract_patch(im, patch_size):
    x = torch.randint(0, high=im.shape[1] - patch_size[0], size=(1,)).item()
    y = torch.randint(0, high=im.shape[2] - patch_size[1], size=(1,)).item()
    patch = im[:, x:x + patch_size[0], y:y + patch_size[1]]
    return patch


def transfer_color(image, ref_image):
    new_image = torch.zeros_like(ref_image)
    for i in range(image.shape[0]):
        tmp = new_image[i].flatten()
        indexes = torch.argsort(image[i].flatten())
        values = torch.sort(ref_image[i].flatten())[0]
        tmp[indexes] = values
        new_image[i] = tmp.reshape((image[i].shape))
    return new_image


def otsu_threshold(images_batch):
    """
    Applies Otsu's threshold on each image in a batch of images.
    :param images_batch: tensor of shape (batch_size, channels, height, width) containing the batch of images.
    :return: tensor of shape (batch_size, 1, height, width) containing the thresholded images.
    """
    batch_size, channels, height, width = images_batch.shape
    thresholded_images = torch.zeros((batch_size, 1, height, width), dtype=torch.bool)
    thresholds = torch.empty(batch_size)
    for i in range(batch_size):
        # Convert image to grayscale
        image = images_batch[i]
        if channels == 3:
            image = image.mean(dim=0, keepdim=True)  # Convert to grayscale by taking the mean across color channels

        # Compute Otsu's threshold
        thresholds[i] = threshold_otsu(image.numpy())

        # Apply threshold
        thresholded_images[i] = image > thresholds[i]

    return thresholded_images, thresholds


def fuse(texture, digit, mask):
    """ Blend MNIST digit with a textured background

    Args
    ----
    texture: torch.Tensor
        Gray-scale texture image
    digit: torch.Tensor
        MNIST digit

    Returns
    -------
    patch_img: torch.Tensor
        The MNIST digit blended with the texture patch
    mask: torch.Tensor
        The mask used to do the blending
    """
    # The blending
    background = (1 - mask) * texture
    foreground = mask * digit
    new_img = background + foreground

    return new_img


def repeat_to_match_size(A: torch.Tensor, B: torch.Tensor, B_=None):
    """
    Takes two batches of images and unifty their size by repeating the smaller. A and B should have size of
    batch_size,num_channels,W,D
    """
    if A.shape[0] > B.shape[0]:
        indices = torch.randint(low=0, high=B.shape[0], size=(A.shape[0],))
        B = B[indices]
        if B_ is not None:
            B_ = B_[indices]
    elif A.shape[0] < B.shape[0]:
        indices = torch.randint(low=0, high=A.shape[0], size=(B.shape[0],))
        A = A[indices]
    return A, B, B_


def get_random_closed_shape_mask(size: Tuple[int, int]):
    while True:
        try:
            num_points = np.random.randint(0, 15)
            x = np.random.rand(num_points)
            y = np.random.rand(num_points)
            t = np.linspace(0, 1, num_points)

            curve_x = np.polyval(np.polyfit(t, x, 3), t)
            curve_y = np.polyval(np.polyfit(t, y, 3), t)

            # Close the shape by connecting the first and last points
            curve_x = np.append(curve_x, curve_x[0])
            curve_y = np.append(curve_y, curve_y[0])

            # Create a mask
            mask = np.zeros(size, dtype=np.uint8)
            scale = (size[0], size[1])
            points = np.column_stack((curve_x * scale[0], curve_y * scale[1]))
            points = np.round(points).astype(int)

            cv2.fillPoly(mask, [points], 1)
            break
        except Exception as e:
            pass
            # I find it annoying to see a lot of these logs, so I comment it out. Undo this if necessary
            # print(f'facing an exception during get_random_closed_shape_mask call', e)

    return torch.tensor(mask).unsqueeze(0)


def unify_images_intensities(images: torch.Tensor, thresholds: torch.Tensor, target_min=200,
                             target_max=250) -> torch.Tensor:
    """"
    unify the images intensities within a specific range [target_min, target_max].
    thresholds are used to specify a specific value for the minimum values. Values below or equal to
    thresholds are not changed
    """
    maxs = torch.amax(images, dim=(2, 3), keepdim=True)
    mins = thresholds[..., None, None, None]
    rescaled_image = images.clone()
    rescaled_image = ((rescaled_image - mins) / (maxs - mins)) * (
            target_max - target_min) + target_min
    images = images.float()
    rescaled_image[images <= mins] = images[images <= mins]
    rescaled_image = rescaled_image.to(torch.uint8)
    return rescaled_image


def add_smooth_intensity_to_masks(masks: torch.Tensor, smoothness: torch.Tensor) -> torch.Tensor:
    initials = (1 / (smoothness + 1e-9)) * masks * (torch.randn(size=masks.shape) ** 2)
    return (initials / initials.amax(dim=(2, 3), keepdim=True)) * 255
