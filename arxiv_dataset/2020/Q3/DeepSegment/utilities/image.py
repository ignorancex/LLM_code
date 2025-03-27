# Written by Chun Kit Wong and CIRC under MIT license:
# https://github.com/wong-ck/DeepSegment/blob/master/LICENSE

import numpy as np
import skimage
from skimage import transform

# determine skimage major.minor version and store as float
SKIMAGE_VERSION = float('.'.join(skimage.__version__.split('.')[:2]))


# crop or pad to match specified size
def crop_or_pad_image(img_original, size_target, background_value=None):
    img_target = np.ones(tuple(size_target))

    if background_value is None:
        background_value = np.min(img_original)
    img_target *= background_value

    size_original = img_original.shape

    slice_original = []
    slice_target = []
    for i in range(len(size_target)):
        if size_target[i] == size_original[i]:
            slice_target += [slice(None, None)]
            slice_original += [slice(None, None)]
        elif size_target[i] > size_original[i]:
            start_target = int((size_target[i] - size_original[i]) // 2)
            slice_target += [
                slice(start_target, start_target + size_original[i])
            ]

            slice_original += [slice(None, None)]
        else:
            slice_target += [slice(None, None)]

            start_original = int((size_original[i] - size_target[i]) // 2)
            slice_original += [
                slice(start_original, start_original + size_target[i])
            ]

    img_target[tuple(slice_target)] = img_original[tuple(slice_original)]

    return img_target


# reverse the effect done by crop_or_pad_image
def inverse_crop_or_pad_image(
    img_target, size_original, background_value=None
):
    img_original = np.ones(tuple(size_original))

    if background_value is None:
        background_value = np.min(img_target)
    img_original *= background_value

    size_target = img_target.shape

    slice_original = []
    slice_target = []
    for i in range(len(size_target)):
        if size_target[i] == size_original[i]:
            slice_target += [slice(None, None)]
            slice_original += [slice(None, None)]
        elif size_target[i] > size_original[i]:
            start_target = int((size_target[i] - size_original[i]) // 2)
            slice_target += [
                slice(start_target, start_target + size_original[i])
            ]

            slice_original += [slice(None, None)]
        else:
            slice_target += [slice(None, None)]

            start_original = int((size_original[i] - size_target[i]) // 2)
            slice_original += [
                slice(start_original, start_original + size_target[i])
            ]

    img_original[tuple(slice_original)] = img_target[tuple(slice_target)]

    return img_original


# resample image
def resample_image(
    img_original,
    res_original,
    res_target,
    order=0,
    disable_anti_aliasing=True
):
    # determine original image size
    size_original = img_original.shape

    # check dimensions
    lengths = [len(x) for x in [res_target, res_original, size_original]]
    if len(set(lengths)) != 1:
        msg = "res_target, res_original, and img_original.shape "
        msg += "must have same length"
        raise ValueError(msg)

    # calculate target size
    size_target = []
    for rt, ro, so in zip(res_target, res_original, size_original):
        if rt is None:
            size_target += [so]
        else:
            size_target += [int(1.0 * ro / rt * so)]

    # turn off anti-aliasing if scikit-image version >= 0.14;
    # AA leads to erroneous behavious in some cases
    extra_kwargs = {}
    if SKIMAGE_VERSION >= 0.14:
        extra_kwargs["anti_aliasing"] = not disable_anti_aliasing

    img_target = transform.resize(
        img_original,
        size_target,
        order=order,
        preserve_range=True,
        mode='constant',
        **extra_kwargs
    )

    return img_target


# resize image
def resize_image(
    img_original,
    size_target,
    order=0,
    disable_anti_aliasing=True,
):
    # turn off anti-aliasing if scikit-image version >= 0.14;
    # AA leads to erroneous behavious in some cases
    extra_kwargs = {}
    if SKIMAGE_VERSION >= 0.14:
        extra_kwargs["anti_aliasing"] = not disable_anti_aliasing

    img_target = transform.resize(
        img_original,
        size_target,
        order=order,
        preserve_range=True,
        mode='constant',
        **extra_kwargs
    )

    return img_target
