# Copyright (c) OpenMMLab. All rights reserved.
# Modified by Nicola Marinello, 2025
import warnings
from typing import List, Sequence, Union

import numpy as np
import torch

from mmengine.structures.base_data_element import BaseDataElement


class OccupancyData(BaseDataElement):
    """Data structure for occupancy-level annotations or predictions.

    All data items in ``data_fields`` of ``OccupancyData`` meet the following
    requirements:

    - They all have 3 dimensions in orders of width, length and height.
    - They should have the same width, length and height.
    """

    def __setattr__(self, name: str, value: Union[torch.Tensor, np.ndarray]):
        """Set attributes of ``OccupancyData``.

        If the dimension of value is 2 and its shape meet the demand, it
        will automatically expand its channel-dimension.

        Args:
            name (str): The key to access the value, stored in `OccupancyData`.
            value (Union[torch.Tensor, np.ndarray]): The value to store in.
                The type of value must be `torch.Tensor` or `np.ndarray`,
                and its shape must meet the requirements of `OccupancyData`.
        """
        if name in ('_metainfo_fields', '_data_fields'):
            if not hasattr(self, name):
                super().__setattr__(name, value)
            else:
                raise AttributeError(f'{name} has been used as a '
                                     'private attribute, which is immutable.')

        else:
            assert isinstance(value, (torch.Tensor, np.ndarray)), \
                f'Can not set {type(value)}, only support' \
                f' {(torch.Tensor, np.ndarray)}'

            if self.shape:
                assert tuple(value.shape) == self.shape, (
                    'The height and width of '
                    f'values {tuple(value.shape)} is '
                    'not consistent with '
                    'the shape of this '
                    ':obj:`OccupancyData` '
                    f'{self.shape}')
            assert (value.ndim == 3 or value.ndim == 4), f'The dim of value must be 3 or 4, but got {value.ndim}'
            super().__setattr__(name, value)

    # TODO should check that data is torch.Long ?? Not if I want to use it as logits container or if I need to store a mask

    __setitem__ = __setattr__

    @property
    def shape(self):
        """The shape of occupancy data."""
        if len(self._data_fields) > 0:
            return tuple(self.values()[0].shape)
        else:
            return None