import json
import os
import pathlib
import pickle
from typing import Iterable, Tuple, Union, Any

import cv2
import numpy as np
import torch
from PIL import Image
from skimage.transform import resize

from .logging import log_assert, log_debug

EPSILON = 1e-10


def mkdir(path: str) -> None:
    """
    Creates directory if not exists

    Args:
        path: directory path
    """
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def _check_path(path: Union[Iterable[str], str]
                ) -> str:
    """
    Gets final path to the file from string iterable or string

    Args:
        path: string path or iterable of strings, which will be concatenated into path

    Return:
        full path string
    """
    log_assert(isinstance(path, str) or len(path) > 0, "'path_args' be a string or iterable of strings")

    if isinstance(path, str):
        return path
    else:
        return os.path.join(*path)


def write_pickle(obj: object,
                 file_path: Union[Iterable[str], str]
                 ) -> None:
    """
    Writes pickle to the given path

    Args:
        obj: object to write as pickle
        file_path: string path or iterable of strings, which will be concatenated into path
    """

    full_path = _check_path(file_path)

    path, file = os.path.split(full_path)

    mkdir(path)

    with open(full_path, 'wb') as f:
        log_debug(f"Writing to: {full_path}")
        pickle.dump(obj, f)


def read_pickle(file_path: Union[Iterable[str], str]
                ) -> object:
    """
    Reads pickle from the given path

    Args:
        file_path: string path or iterable of strings, which will be concatenated into path

    Returns:
        loaded object
    """

    full_path = _check_path(file_path)

    with open(full_path, 'rb') as f:
        log_debug(f"Reading from: {full_path}")
        return pickle.load(f)


def apply_mask(img: np.ndarray,
               mask: np.ndarray,
               threshold: float = None,
               crop_around_mask: bool = True
               ) -> np.ndarray:
    """
    Apply mask to image

    Args:
        image: image to mask - np.ndarray[H, W, C]
        mask: mask - np.ndarray[H, W]

    Kwargs:
        threshold: values lower than threshold will be set to 0, rest - to 1
        crop_around_mask: apply (or not) cropping around the active mask pixels, i.e., remove excessive black regions

    Returns:
        masked image: np.ndarray[H, W, C]
    """
    mask = normalize_0_to_1(mask)

    img_mask = resize(mask, (img.shape[0], img.shape[1]))

    if threshold:
        img_mask = img_mask >= threshold

    if crop_around_mask:
        a0min, a0max, a1min, a1max = get_mask_bbox(img_mask)  # bbox for non-zero mask values 

    img_mask = np.expand_dims(img_mask, 2)  # 3d-mask
    masked_img = img * img_mask

    if crop_around_mask:
        masked_img = masked_img[a1min:a1max+1, a0min:a0max+1]  # remove excessive blackness

    return masked_img


def add_countours_around_non_black_pixels(image: np.ndarray,
                                          mask_image: np.ndarray,
                                          thickness: int = 5,
                                          countour_color: Tuple[int, int, int] = (0, 255, 0)
                                          ) -> np.ndarray:
    """
   Draw countours around non-black image

    Args:
        image: image to mask - np.ndarray[H, W, C]
        mask_image: mask - np.ndarray[H, W]

    Kwargs:
        thickness (int = 3): countour line thickness

    Returns:
        image with countours: np.ndarray[H, W, C]
    """
    gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, countour_color, thickness)

    return image


def add_countours_around_mask(image: np.ndarray,
                              binary_mask_image: np.ndarray,
                              thickness: int = 5,
                              countour_color: Tuple[int, int, int] = (0, 255, 0)
                              ) -> np.ndarray:
    """
   Draw countours around non-black image

    Args:
        image: image to mask - np.ndarray[H, W, C]
        binary_mask_image: mask - np.ndarray[H, W]

    Kwargs:
        thickness (int = 3): countour line thickness

    Returns:
        image with countours: np.ndarray[H, W, C]
    """
    if binary_mask_image.dtype == bool:
        binary_mask_image = binary_mask_image.astype(np.uint8) * 255

    contours, _ = cv2.findContours(binary_mask_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, countour_color, thickness)

    return image


def get_mask_bbox(mask: np.ndarray
                  ) -> Tuple[int, int, int, int]:
    """
    Get the bounding box of mask's content

    Args:
        mask: mask np.ndarray[H, W]

    Returns:
        bounding box coordinates: Tuple[int, int, int, int]
    """

    def get_first_nonzero_arg(arr: Iterable[float]
                              ) -> int:
        """
        Get the argument of the first non-zero element in array

        Args:
            arr: array - Iterable[float]

        Returns:
            index of the first non-zero element of array: int
        """
        arg = 0
        for i, a in enumerate(arr):
            if a:
                arg = i
                break
        return arg

    a0 = mask.sum(axis=0) > 0
    a1 = mask.sum(axis=1) > 0

    a0min = get_first_nonzero_arg(a0)
    a0max = len(a0) - get_first_nonzero_arg(a0[::-1]) - 1
    a1min = get_first_nonzero_arg(a1)
    a1max = len(a1) - get_first_nonzero_arg(a1[::-1]) - 1

    return a0min, a0max, a1min, a1max


def apply_heatmap(img: np.ndarray,
                  heatmap: np.ndarray,
                  cmap: int = 2  # cv2.COLORMAP_JET
                  ) -> np.ndarray:
    """
    Apply heatmap to image

    Args:
        image: image (np.uint8) - np.ndarray[H, W, C]
        heatmap: float-valued heatmap - np.ndarray[H, W]

    Kwargs:
        cmap: int of cv2 colormap, defaults to 2 == cv2.COLORMAP_JET

    Returns:
        heatmapped image: np.ndarray[H, W, C]
    """
    heatmap = normalize_0_to_1(heatmap)

    heatmap_img = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_img = np.uint8(255 * heatmap_img)
    heatmap_img = cv2.applyColorMap(heatmap_img, cmap)

    superimposed_img = cv2.addWeighted(heatmap_img, 0.5, img, 0.5, 0)
    return superimposed_img[:, :, ::-1]  # Image.fromarray(superimposed_img[:, :, ::-1])



def normalize_0_to_1(array: Union[np.ndarray, torch.Tensor]
                     ) -> Union[np.ndarray, torch.Tensor]:
    """
    Normalize np.ndarray from 0 to 1

    Args:
        array: array to normalize - np.ndarray[...]

    Returns:
        normalized array: np.ndarray[...]
    """
    array = array - array.min()
    return array / (array.max() + EPSILON)



def read_json(json_path: str
              ) -> Any:
    """
    Read from JSON file

    Args:
        json_path: Path to JSON file

    Returns:
        JSON contents
    """    
    log_debug(f"Reading JSON from: {json_path}")

    with open(json_path, 'r') as file:
        content = json.load(file)
        return content


def write_json(obj: Any,
               json_path: str,
               indent: Union[str, int] = None
               ) -> None:
    """
    Write to JSON file

    Args:
        obj: JSON object to write
        json_path: Path to JSON file

    Kwargs:
        indent: indent in JSON file, see the original documentation of json.dump

    Returns:
        dictionary with JSON contents
    """
    log_debug(f"Writing JSON to: {json_path}")

    with open(json_path, 'w') as file:
        json.dump(obj, file, indent=indent)


def blend_imgs(img1: Image,
               img2: Image,
               alpha: float = 0.5
               ) -> Image:
    """
    Blend 2 PIL.Image

    Args:
        img1 (Image): image 1
        img2 (Image): image 2

    Kwargs:
        alpha (float = 0.5): blending strength variable

    Returns:
        (Image) blended image
    """
    img1 = img1.convert("RGB")
    img2 = img2.convert("RGB")
    img2 = img2.resize(img1.size)
    return Image.blend(img1, img2, alpha)


def get_colored_mask(mask: np.ndarray,
                     color_channels: Iterable[int] = [1]
                     ) -> Image:
    """
    converts grayscale binary mask to RGB mask

    Args:
        mask (np.ndarray): binary grayscale mask

    Kwargs:
        color_channels (Iterable[int] = [1]): channels (RGB) to expand mask to

    Returns:
        (Image) mask RGB image
    """
    rgb_img = np.zeros((*mask.shape, 3), dtype=bool)
    for c in color_channels:
        rgb_img[:,:,c] = mask
    return Image.fromarray(rgb_img.astype(np.uint8) * 255)
