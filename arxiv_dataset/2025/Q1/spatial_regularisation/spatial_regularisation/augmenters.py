from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import numpy as np
from torch.nn.functional import grid_sample

from spatial_regularisation.utils import create_transform, aff_to_field, interpolate, add_axis


class SpatialAugmenter(object):
    """Spatial augmenter for training and validation inputs.
    In training, all augmentation parameters are randomly sampled at each mini-batch. For validation,
    these parameters are subject-specific but fixed across time, and thus given as inputs in list_of_xfm_params."""

    def __init__(self,
                 list_of_xfm_params=None,
                 rotation_range=0.,
                 shift_range=0.,
                 scale_range=0.,
                 shear_range=0.,
                 return_affine=False,
                 use_max_values=False,
                 normalise=False):
        """
        :param list_of_xfm_params: list of dict with {"rotation": r, "translation": t}, where r and t are
        length-3 numpy arrays. (used for validation)
        :param rotation_range: maximum rotation angle for augmentation (in degrees)
        :param shift_range: maximum shift for augmentation (in voxels)
        :param return_affine: whether to return the applied rigid transform in homogeneous representation
        :param use_max_values: whether to use the maximum rotation and translation values instead on drawing them from
        uniform distributions
        :param normalise: whether to rescale the inputs in [0,1]
        """

        # initialisation
        self.list_of_xfm_params = list_of_xfm_params  # validation case
        self.rotation_range = rotation_range
        self.shift_range = shift_range
        self.scale_range = scale_range
        self.shear_range = shear_range

        self.return_affine = return_affine
        self.use_max_values = use_max_values

        self.normalise = normalise

    def random_transform(self, *args):
        """Randomly rotate/translate/crop/flip an image tensor of shape [H, W, D, C], and optionally its labels."""

        # get transformation matrix
        if self.rotation_range:
            if self.use_max_values:
                r = self.rotation_range * np.ones(3)
            else:
                r = np.random.uniform(-self.rotation_range, self.rotation_range, 3)
        else:
            r = np.zeros(3)
        if self.shift_range:
            if self.use_max_values:
                t = self.shift_range * np.ones(3)
            else:
                t = np.random.uniform(-self.shift_range, self.shift_range, 3)
        else:
            t = np.zeros(3)
        if self.scale_range:
            if self.use_max_values:
                s = self.scale_range + np.ones(3)
            else:
                s = np.random.uniform(1 - self.scale_range, 1 + self.scale_range, 3)
        else:
            s = np.ones(3)
        if self.shear_range:
            if self.use_max_values:
                sh = self.shear_range * np.ones(6)
            else:
                sh = np.random.uniform(-self.shear_range, self.shear_range, 6)
        else:
            sh = np.zeros(6)
        transform_matrix = create_transform(*r, *t, *s, *sh, ordering='txyzsh')

        return self.perform_transform(transform_matrix, *args)

    def predefined_transform(self, idx, *args):
        """Rotate/translate/crop/flip an image tensor of shape [H, W, D, C], and optionally its labels,
        with parameters given beforehand. Used for validation."""

        # get transformation matrix
        r = self.list_of_xfm_params[idx]["rotation"]
        t = self.list_of_xfm_params[idx]["translation"]
        s = self.list_of_xfm_params[idx]["scale"]
        sh = self.list_of_xfm_params[idx]["shear"]
        transform_matrix = create_transform(*r, *t, *s, *sh, ordering='txyzsh')

        # apply augmentation in pytorch, but input/output are numpy of size [H, W, D, C]
        return self.perform_transform(transform_matrix, *args)

    def perform_transform(self, transform_matrix, *args):

        # apply transform, computation is done with torch but inputs and outputs are numpy
        with torch.no_grad():
            outputs = []
            for vol_idx, x in enumerate(args):
                if vol_idx > 0:
                    method = "nearest"
                    dtype = torch.int32
                else:
                    method = "linear"
                    dtype = torch.float32
                grid = aff_to_field(transform_matrix, x.shape[:3], invert_affine=True)
                x = interpolate(x, grid, method=method, vol_dtype=dtype)
                outputs.append(x)

            if self.normalise:
                outputs[0] = torch.clamp(outputs[0], 0)
                m = torch.min(outputs[0])
                M = torch.max(outputs[0])
                outputs[0] = (outputs[0] - m) / (M - m + 1e-9)
                outputs[0] = outputs[0] / torch.mean(outputs[0][outputs[0] > 0]) * 0.5

            outputs = [out.detach().numpy() for out in outputs]

        # return outputs
        outputs = [outputs]
        if self.return_affine:
            outputs.append(transform_matrix)
        return outputs[0] if len(outputs) == 1 else outputs  # [H, W, D, C]


class IntensityAugmenter(object):
    """Intensity augmenter for training and validation inputs.
    In training, all augmentation parameters are randomly sampled at each mini-batch. For validation,
    these parameters are subject-specific but fixed across time, and thus given as inputs in list_of_xfm_params."""

    def __init__(self,
                 list_of_params=None,
                 max_noise_std=0.,
                 max_bias_std=0.,
                 bias_scale=0.06,
                 gamma_std=0.,
                 use_max_values=False):
        """
        :param list_of_params: list of dict with {"noise_field": noise, "bias_field": bias, "gamma": g},
        where noise is a field of the same shape as x that will be added to it (additive noise)
              bias is a small field, that will be resampled to image size and multiplied to x
              g is a scalar by which all voxels of x will be exponentiated
        :param max_noise_std: maximum standard deviation for the Gaussian noise
        :param max_bias_std: maximum std. dev for the bias filed corruption (higher = stronger corruption)
        :param bias_scale: scale of the bias field (lower = smoother)
        :param gamma_std: std dev for random exponentiation (higher = stronger)
        :param use_max_values: whether to use the maximum rotation and translation values instead on drawing them from
        uniform distributions
        """

        # initialise
        self.list_of_intensity_params = list_of_params
        self.max_noise_std = max_noise_std
        self.max_bias_std = max_bias_std
        self.bias_scale = bias_scale
        self.gamma_std = gamma_std
        self.use_max_values = use_max_values

    def random_transform(self, x):
        """Randomly corrupt an image tensor of shape [H, W, D, C] with noise/bias """

        if self.max_noise_std > 0 or self.max_bias_std > 0 or self.gamma_std > 0:

            # sample noise field
            if self.use_max_values:
                noise_std = self.max_noise_std
            else:
                noise_std = np.random.uniform(high=self.max_noise_std)
            if noise_std > 0:
                noise_field = np.random.normal(0, noise_std, x.shape[:3])
            else:
                noise_field = None

            # sample small bias field
            if self.use_max_values:
                bias_std = self.max_bias_std
            else:
                bias_std = np.random.uniform(high=self.max_bias_std)
            if bias_std > 0:
                bias_sample_size = [math.ceil(size * self.bias_scale) for size in x.shape[:3]]
                bias_field = np.random.normal(0, bias_std, bias_sample_size)
            else:
                bias_field = None

            # sample gamma
            if self.gamma_std > 0:
                if self.use_max_values:
                    gamma = self.gamma_std
                else:
                    gamma = np.random.normal(scale=self.gamma_std)
            else:
                gamma = None

            # apply intensity augmentation in pytorch, but input/output are numpy of size [H, W, D, C]
            x = self.apply_intensity_transform(x, noise_field, bias_field, gamma)

        return x

    def predefined_transform(self, idx, x):
        """Corrupt an image tensor of shape [H, W, D, C] with noise/bias fields computed beforehand.
        Used for validation"""

        # get noise and bias fields
        noise_field = self.list_of_intensity_params[idx]["noise_field"]
        noise_field = np.array(noise_field) if noise_field is not None else noise_field
        bias_field = self.list_of_intensity_params[idx]["bias_field"]
        bias_field = np.array(bias_field) if bias_field is not None else bias_field
        gamma = self.list_of_intensity_params[idx]["gamma"]

        # apply intensity augmentation in pytorch, but input/output are numpy of size [H, W, D, C]
        if noise_field is not None or bias_field is not None or gamma is not None:
            x = self.apply_intensity_transform(x, noise_field, bias_field, gamma)

        return x

    def apply_intensity_transform(self, x, noise_field, bias_field, gamma):
        """apply intensity augmentation and normalisation. Inputs and outputs are numpy, but computations are with torch.
        :param x: input volume, numpy array with shape [H, W, D, C]
        :param noise_field: field of the same shape as x that will be added to it
        :param bias_field: small bias field, that will be resampled to image size and multiplied to x
        :param gamma: power by which all voxels of x will be raised to.
        """

        with torch.no_grad():

            # switch to channel first, add batch size, and convert to tensor
            x = torch.tensor(add_axis(np.rollaxis(x.astype(np.float32), 3, 0)))  # [B, C, H, W, D]
            im_shape = list(x.size())

            # get mask non-zero values
            mask = (x > 0).to(dtype=x.dtype)

            # bias
            if bias_field is not None:
                bias = add_axis(torch.tensor(bias_field, device=x.device, dtype=torch.float32),
                                [0, 0])  # [B, C, H, W, D]
                loc = torch.meshgrid(*[torch.linspace(-1, 1, ss, device=x.device) for ss in im_shape[2:]])
                loc = add_axis(torch.stack(loc, -1))
                bias = grid_sample(bias, loc, align_corners=False)
                x *= torch.exp(bias)

            # rescale from 0 to 1
            m = torch.min(x)
            M = torch.max(x)
            x = (x - m) / (M - m + 1e-9)

            # gamma transform
            if gamma is not None:
                x = torch.pow(x, np.exp(gamma))

            # noise
            if noise_field is not None:
                noise = add_axis(torch.tensor(noise_field, device=x.device, dtype=torch.float32),
                                 [0, 0])  # [B, C, H, W, D]
                x += noise

            # mask output
            x *= mask
            x = torch.clamp(x, min=0)

            # renormalise as we would do at test-time
            m = torch.min(x)
            M = torch.max(x)
            x = (x - m) / (M - m + 1e-9)
            x = x / torch.mean(x[x > 0]) * 0.5

            # convert back to numpy and remove batch size
            x = x.detach().numpy()[0, ...]

        # switch back to channel last
        x = np.rollaxis(x, 0, 4)  # [H, W, D, C]

        return x
