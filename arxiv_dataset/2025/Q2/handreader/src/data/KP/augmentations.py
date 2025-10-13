import math
import random

import numpy as np
import torch
from albumentations.core.transforms_interface import BasicTransform
from torch.nn import functional as F


class Resample(BasicTransform):
    """
    Stretches or squeezes input data along the time dimension by applying resampling.

    This transform adjusts the time dimension of the input by interpolating frames to
    a new size based on a randomly selected resampling rate within a specified range.

    Parameters
    ----------
    sample_rate : tuple of (float, float), optional, default=(0.8, 1.2)
        The range of resampling rates. The lower and upper bounds of the resampling
        rate must satisfy `0 <= rate_lower <= rate_upper`.
    always_apply : bool, optional, default=False
        If `True`, the transformation is always applied.
    p : float, optional, default=0.5
        The probability of applying the transformation.

    Targets
    -------
    image : numpy.ndarray
        Applies the transformation to image data.

    Image Types
    -----------
    - `float32` arrays of shape `(seq_len, n_landmarks, 3)` or `(seq_len, n_landmarks, 2)`.

    Methods
    -------
    apply(data, sample_rate=1.0, **params)
        Applies the resampling transformation to the input data.
    get_params()
        Returns a dictionary containing a randomly sampled `sample_rate` within the
        specified range (`rate_lower`, `rate_upper`).
    get_transform_init_args_names()
        Returns the names of the initialization arguments (`rate_lower`, `rate_upper`).
    targets
        Property that specifies the applicable targets for this transform.
    """

    def __init__(
        self,
        sample_rate=(0.8, 1.2),
        always_apply=False,
        p=0.5,
    ):
        """
        Initializes the Resample instance.

        Parameters
        ----------
        sample_rate : tuple of (float, float), optional, default=(0.8, 1.2)
            The range of resampling rates. The lower and upper bounds of the resampling
            rate must satisfy `0 <= rate_lower <= rate_upper`.
        always_apply : bool, optional, default=False
            If `True`, the transformation is always applied.
        p : float, optional, default=0.5
            The probability of applying the transformation.

        Raises
        ------
        ValueError
            If the lower bound of the sample rate is greater than the upper bound,
            or if the sample rate values are out of valid range.
        """
        super().__init__(always_apply, p)

        rate_lower = sample_rate[0]
        rate_upper = sample_rate[1]
        if not 0 <= rate_lower <= rate_upper:
            raise ValueError(
                "Invalid combination of rate_lower and rate_upper. Got: {}".format(
                    (rate_lower, rate_upper)
                )
            )

        self.rate_lower = rate_lower
        self.rate_upper = rate_upper

    def apply(self, data, sample_rate=1.0, **params):
        """
        Applies the sample rate adjustment to the input data.

        Parameters
        ----------
        data : torch.Tensor
            The input data tensor of shape `(length, ...)`, where `length` is the size
            of the first dimension.
        sample_rate : float, optional
            The rate at which to resample the input data. Defaults to 1.0.
        **params : dict
            Additional parameters (unused).

        Returns
        -------
        torch.Tensor
            The resampled data with adjusted dimensions.
        """
        length = data.shape[0]
        new_size = max(int(length * sample_rate), 1)
        new_x = F.interpolate(data.permute(1, 2, 0), new_size).permute(2, 0, 1)
        return new_x

    def get_params(self):
        """
        Generates random parameters for the transformation.

        Returns
        -------
        dict
            A dictionary containing the randomly sampled `sample_rate` parameter.
        """
        return {"sample_rate": random.uniform(self.rate_lower, self.rate_upper)}

    def get_transform_init_args_names(self):
        """
        Returns the names of the initialization arguments for the transformation.

        Returns
        -------
        tuple of str
            A tuple containing the argument names: ("rate_lower", "rate_upper").
        """
        return ("rate_lower", "rate_upper")

    @property
    def targets(self):
        """
        Property that defines the transformation targets.

        Returns
        -------
        dict
            A dictionary mapping the "image" key to the `apply` method.
        """
        return {"image": self.apply}


class SpatialMask(BasicTransform):
    """
    Applies a spatial mask to input data, setting values within a specified region to a mask value.

    The mask can be applied using either absolute or relative dimensions, with its size
    and position determined randomly within specified ranges.

    Parameters
    ----------
    size : tuple of (float, float), optional, default=(0.5, 1.0)
        The range for the size of the mask. In "absolute" mode, this defines the
        mask width and height. In "relative" mode, it is a fraction of the spatial extent.
    mask_value : float, optional, default=float("nan")
        The value to assign to the masked region.
    mode : {"absolute", "relative"}, optional, default="absolute"
        Specifies how the mask size is calculated:
        - "absolute": Uses the size directly as width and height.
        - "relative": Uses the size as a fraction of the data's spatial range.
    always_apply : bool, optional, default=False
        If `True`, the transformation is always applied.
    p : float, optional, default=0.5
        The probability of applying the transformation.

    Targets
    -------
    image : numpy.ndarray
        Applies the transformation to image data.

    Image Types
    -----------
    - `float32` arrays of shape `(seq_len, n_landmarks, 3)` or `(seq_len, n_landmarks, 2)`.

    Methods
    -------
    apply(data, mask_size=0.75, offset_x_01=0.2, offset_y_01=0.2, mask_value=float("nan"), **params)
        Applies the spatial mask to the input data.
    get_params()
        Returns a dictionary of randomly sampled parameters, including mask size,
        offsets, and the mask value.
    get_transform_init_args_names()
        Returns the names of the initialization arguments (`size`, `mask_value`, `mode`).
    targets
        Property that specifies the applicable targets for this transform.
    """

    def __init__(
        self,
        size=(0.5, 1.0),
        mask_value=float("nan"),
        mode="absolute",
        always_apply=False,
        p=0.5,
    ):
        """
        Initializes the SpatialMask instance.

        Parameters
        ----------
        size : tuple of (float, float), optional, default=(0.5, 1.0)
            The range for the size of the mask. In "absolute" mode, this defines the
            mask width and height. In "relative" mode, it is a fraction of the spatial extent.
        mask_value : float, optional, default=float("nan")
            The value to assign to the masked region.
        mode : {"absolute", "relative"}, optional, default="absolute"
            Specifies how the mask size is calculated:
            - "absolute": Uses the size directly as width and height.
            - "relative": Uses the size as a fraction of the data's spatial range.
        always_apply : bool, optional, default=False
            If `True`, the transformation is always applied.
        p : float, optional, default=0.5
            The probability of applying the transformation.
        """
        super().__init__(always_apply, p)

        self.size = size
        self.mask_value = mask_value
        self.mode = mode

    def apply(
        self,
        data,
        mask_size=0.75,
        offset_x_01=0.2,
        offset_y_01=0.2,
        mask_value=float("nan"),
        **params
    ):
        """
        Applies the masking transformation to the input data.

        Parameters
        ----------
        data : torch.Tensor
            The input data to which the mask will be applied.
        mask_size : float, optional
            The size of the mask, as a fraction of the data's size. Defaults to 0.75.
        offset_x_01 : float, optional
            The horizontal offset for the mask's position, as a fraction of the data's width.
            Defaults to 0.2.
        offset_y_01 : float, optional
            The vertical offset for the mask's position, as a fraction of the data's height.
            Defaults to 0.2.
        mask_value : float, optional
            The value to fill the masked area with. Defaults to `float("nan")`.
        **params : dict
            Additional parameters (unused).

        Returns
        -------
        torch.Tensor
            The data with the mask applied.
        """
        return data
        # noqa

    def get_params(self):
        """
        Generates random parameters for the transformation.

        Returns
        -------
        dict
            A dictionary containing the randomly sampled parameters:
            - `offset_x_01`: Horizontal offset for the mask's position.
            - `offset_y_01`: Vertical offset for the mask's position.
            - `mask_size`: Mask size as a fraction of the data's size.
            - `mask_value`: Value to use for masking.
        """
        params = {"offset_x_01": random.uniform(0, 1)}
        params["offset_y_01"] = random.uniform(0, 1)
        params["mask_size"] = random.uniform(self.size[0], self.size[1])
        params["mask_value"] = self.mask_value
        return params

    def get_transform_init_args_names(self):
        """
        Returns the names of the initialization arguments for the transformation.

        Returns
        -------
        tuple of str
            A tuple containing the argument names: ("size", "mask_value", "mode").
        """
        return ("size", "mask_value", "mode")

    @property
    def targets(self):
        """
        Property that defines the transformation targets.

        Returns
        -------
        dict
            A dictionary mapping the "image" key to the `apply` method.
        """
        return {"image": self.apply}


def spatial_random_affine(data, scale=None, shear=None, shift=None, degree=None, center=(0, 0)):
    """
    Applies a random affine transformation to 2D or 3D spatial data.

    This function supports scaling, shearing, rotation, and shifting transformations
    on spatial coordinates. For 3D data, the `z` coordinate is excluded from the
    transformation and re-attached after processing.

    Parameters
    ----------
    data : torch.Tensor
        Input tensor of shape `(seq_len, n_landmarks, 2)` for 2D data or
        `(seq_len, n_landmarks, 3)` for 3D data.
    scale : float, optional
        Scaling factor. If provided, multiplies the coordinates by this value.
    shear : tuple of (float, float), optional
        Shear factors for the x and y axes, respectively. Defines the off-diagonal
        elements of the shear matrix.
    shift : tuple of (float, float), optional
        Translation values for the x and y axes, respectively. Shifts the coordinates by this amount.
    degree : float, optional
        Angle in degrees for the rotation. Rotates the coordinates around the `center`.
    center : tuple of (float, float), optional, default=(0, 0)
        Center of rotation. Defaults to the origin.

    Returns
    -------
    torch.Tensor
        Transformed tensor of the same shape as the input.
    """
    data_tmp = None
    data = data.to(torch.float)
    # if input is xyz, split off z and re-attach later
    if data.shape[-1] == 3:
        data_tmp = data[..., 2:]
        data = data[..., :2]

    center = torch.tensor(center)

    if scale is not None:
        data = data * scale

    if shear is not None:
        shear_x, shear_y = shear
        shear_mat = torch.tensor([[1.0, shear_x], [shear_y, 1.0]])
        data = data @ shear_mat
        center = center + torch.tensor([shear_y, shear_x])

    if degree is not None:
        data -= center
        radian = degree / 180 * np.pi
        c = math.cos(radian)
        s = math.sin(radian)

        rotate_mat = torch.tensor([[c, s], [-s, c]])

        data = data @ rotate_mat
        data = data + center

    if shift is not None:
        data = data + shift

    if data_tmp is not None:
        data = torch.cat([data, data_tmp], axis=-1)

    return data


class SpatialAffine(BasicTransform):
    """
    Applies an affine transformation to spatial data by scaling, shearing, rotating, and shifting.

    This class allows applying random affine transformations such as scaling, shearing, rotation,
    and translation to spatial data, either in 2D or 3D space.

    Parameters
    ----------
    scale : tuple of (float, float), optional
        The scaling factor range for the affine transformation. If provided, the data will
        be scaled by a random factor within this range.
    shear : tuple of (float, float), optional
        The shear factor range for the affine transformation. If provided, the data will
        be sheared by random values in this range.
    shift : tuple of (float, float), optional
        The shift (translation) range for the affine transformation. If provided, the data
        will be translated by random values in this range.
    degree : tuple of (float, float), optional
        The rotation angle range in degrees. If provided, the data will be rotated by a
        random angle within this range.
    center_xy : tuple of (float, float), optional, default=(0, 0)
        The center of rotation. The data will be rotated around this point.
    always_apply : bool, optional, default=False
        Whether to always apply the transformation.
    p : float, optional, default=0.5
        Probability of applying the transformation.

    Targets
    -------
    image : callable
        Applies the affine transformation to spatial data (2D or 3D).

    Image Types
    -----------
    - float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)

    Methods
    -------
    apply(data, scale=None, shear=None, shift=None, degree=None, center=(0, 0), **params)
        Applies the affine transformation to the data.

    get_params()
        Returns the random parameters for the affine transformation, including scale, shear,
        shift, degree, and center.

    get_transform_init_args_names()
        Returns the names of the initialization arguments for the transform.

    targets
        Returns the dictionary of targets that this transform can apply to (typically the "image").
    """

    def __init__(
        self,
        scale=None,
        shear=None,
        shift=None,
        degree=None,
        center_xy=(0, 0),
        always_apply=False,
        p=0.5,
    ):
        """
        Initializes the SpatialAffine instance.

        Parameters
        ----------
        scale : tuple of (float, float), optional
            The scaling factor range for the affine transformation. If provided, the data will
            be scaled by a random factor within this range.
        shear : tuple of (float, float), optional
            The shear factor range for the affine transformation. If provided, the data will
            be sheared by random values in this range.
        shift : tuple of (float, float), optional
            The shift (translation) range for the affine transformation. If provided, the data
            will be translated by random values in this range.
        degree : tuple of (float, float), optional
            The rotation angle range in degrees. If provided, the data will be rotated by a
            random angle within this range.
        center_xy : tuple of (float, float), optional, default=(0, 0)
            The center of rotation. The data will be rotated around this point.
        always_apply : bool, optional, default=False
            Whether to always apply the transformation.
        p : float, optional, default=0.5
            Probability of applying the transformation.
        """
        super().__init__(always_apply, p)

        self.scale = scale
        self.shear = shear
        self.shift = shift
        self.degree = degree
        self.center_xy = center_xy

    def apply(
        self, data, scale=None, shear=None, shift=None, degree=None, center=(0, 0), **params
    ):
        """
        Applies the affine transformation to the input data.

        Parameters
        ----------
        data : torch.Tensor
            The input data to which the affine transformation will be applied.
        scale : float, optional
            The scaling factor. Defaults to None.
        shear : tuple of float, optional
            The shearing factors along the x and y axes. Defaults to None.
        shift : float, optional
            The shifting factor. Defaults to None.
        degree : float, optional
            The rotation angle in degrees. Defaults to None.
        center : tuple of int, optional
            The center of transformation as (x, y) coordinates. Defaults to (0, 0).
        **params : dict
            Additional parameters (unused).

        Returns
        -------
        torch.Tensor
            The data with the affine transformation applied.
        """
        new_x = spatial_random_affine(
            data, scale=scale, shear=shear, shift=shift, degree=degree, center=center
        )
        return new_x

    def get_params(self):
        """
        Generates random parameters for the affine transformation.

        Returns
        -------
        dict
            A dictionary containing the randomly sampled parameters:
            - `scale`: Scaling factor.
            - `shear`: Shearing factors along the x and y axes.
            - `shift`: Shifting factor.
            - `degree`: Rotation angle in degrees.
            - `center_xy`: Center of transformation as (x, y) coordinates.
        """
        params = {
            "scale": None,
            "shear": None,
            "shift": None,
            "degree": None,
            "center_xy": self.center_xy,
        }
        if self.scale:
            params["scale"] = random.uniform(self.scale[0], self.scale[1])
        if self.shear:
            shear_x = shear_y = random.uniform(self.shear[0], self.shear[1])
            if random.uniform(0, 1) < 0.5:
                shear_x = 0.0
            else:
                shear_y = 0.0
            params["shear"] = (shear_x, shear_y)
        if self.shift:
            params["shift"] = random.uniform(self.shift[0], self.shift[1])
        if self.degree:
            params["degree"] = random.uniform(self.degree[0], self.degree[1])

        return params

    def get_transform_init_args_names(self):
        """
        Returns the names of the initialization arguments for the transformation.

        Returns
        -------
        tuple of str
            A tuple containing the argument names: ("scale", "shear", "shift", "degree").
        """
        return ("scale", "shear", "shift", "degree")

    @property
    def targets(self):
        """
        Property that defines the transformation targets.

        Returns
        -------
        dict
            A dictionary mapping the "image" key to the `apply` method.
        """
        return {"image": self.apply}


class TemporalMask(BasicTransform):
    """
    Applies temporal masking by randomly masking a portion of the input over the time dimension.

    This class allows applying random temporal masks by modifying a portion of the sequence
    (time dimension) with a specified mask value.

    Args:
        size : tuple of (float, float), optional, default=(0.2, 0.4)
            The range for the size of the temporal mask (as a fraction of the sequence length).
        mask_value : float, optional, default=float("nan")
            The value to use for masking the temporal data. Typically used to fill in the masked portion.
        num_masks : tuple of (int, int), optional, default=(1, 2)
            The range for the number of temporal masks to apply. A random number of masks between
            these values will be applied.
        always_apply : bool, optional, default=False
            Whether to always apply the transformation.
        p : float, optional, default=0.5
            Probability of applying the transformation.

    Targets:
    --------
    image : callable
        Applies the temporal mask transformation to the image.

    Image types:
    ------------
    - float32 (seq_len, n_landmarks, 3) or (seq_len, n_landmarks, 2)

    Methods
    -------
    apply(data, mask_sizes=[0.3], mask_offsets_01=[0.2], mask_value=float("nan"), **params)
        Applies the temporal mask to the data.

    get_params()
        Returns the random parameters for temporal masking, including mask sizes, offsets, and mask value.

    get_transform_init_args_names()
        Returns the names of the initialization arguments for the transform.

    targets
        Returns the dictionary of targets that this transform can apply to (typically the "image").
    """

    def __init__(
        self,
        size=(0.2, 0.4),
        mask_value=float("nan"),
        num_masks=(1, 2),
        always_apply=False,
        p=0.5,
    ):
        """
        Initializes the TemporalMask instance.

        Parameters
        ----------
        size : tuple of (float, float), optional, default=(0.2, 0.4)
            The range for the size of the temporal mask (as a fraction of the sequence length).
        mask_value : float, optional, default=float("nan")
            The value to use for masking the temporal data. Typically used to fill in the masked portion.
        num_masks : tuple of (int, int), optional, default=(1, 2)
            The range for the number of temporal masks to apply. A random number of masks between
            these values will be applied.
        always_apply : bool, optional, default=False
            Whether to always apply the transformation.
        p : float, optional, default=0.5
            Probability of applying the transformation.

        """
        super().__init__(always_apply, p)

        self.size = size
        self.num_masks = num_masks
        self.mask_value = mask_value

    def apply(
        self, data, mask_sizes=[0.3], mask_offsets_01=[0.2], mask_value=float("nan"), **params
    ):
        """
        Applies the masking transformation to the input data.

        Parameters
        ----------
        data : torch.Tensor
            The input data to which the masking transformation will be applied.
        mask_sizes : list of float, optional
            List of mask sizes as fractions of the total sequence length. Defaults to [0.3].
        mask_offsets_01 : list of float, optional
            List of mask offsets as fractions of the total sequence length. Defaults to [0.2].
        mask_value : float, optional
            The value to use for masking. Defaults to NaN.
        **params : dict
            Additional parameters (unused).

        Returns
        -------
        torch.Tensor
            The data with the specified portions replaced by the mask value.
        """
        length = data.shape[0]
        x_new = data.clone()
        for mask_size, mask_offset_01 in zip(mask_sizes, mask_offsets_01):
            mask_size = int(length * mask_size)
            max_mask = np.clip(length - mask_size, 1, length)
            mask_offset = int(mask_offset_01 * max_mask)
            x_new[mask_offset : mask_offset + mask_size] = torch.tensor(mask_value)
        return x_new

    def get_params(self):
        """
        Generates random parameters for the masking transformation.

        Returns
        -------
        dict
            A dictionary containing the randomly sampled parameters:
            - `mask_sizes`: List of mask sizes as fractions of the total sequence length.
            - `mask_offsets_01`: List of mask offsets as fractions of the total sequence length.
            - `mask_value`: The value to use for masking.
        """
        num_masks = np.random.randint(self.num_masks[0], self.num_masks[1])
        mask_size = [random.uniform(self.size[0], self.size[1]) for _ in range(num_masks)]
        mask_offset_01 = [random.uniform(0, 1) for _ in range(num_masks)]
        return {
            "mask_sizes": mask_size,
            "mask_offsets_01": mask_offset_01,
            "mask_value": self.mask_value,
        }

    def get_transform_init_args_names(self):
        """
        Returns the names of the initialization arguments for the transformation.

        Returns
        -------
        tuple of str
            A tuple containing the argument names: ("size", "mask_value", "num_masks").
        """
        return ("size", "mask_value", "num_masks")

    @property
    def targets(self):
        """
        Property that defines the transformation targets.

        Returns
        -------
        dict
            A dictionary mapping the "image" key to the `apply` method.
        """
        return {"image": self.apply}
