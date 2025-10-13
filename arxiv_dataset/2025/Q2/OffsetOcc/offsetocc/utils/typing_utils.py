# Copyright (c) OpenMMLab. All rights reserved.
"""Collecting some commonly used type hint in mmflow."""
from typing import Optional, Union, Sequence
import torch
from mmengine.config import ConfigDict

from mmdet3d.structures import PointData

# Type hint of config data
ConfigType = Union[ConfigDict, dict]
OptConfigType = Optional[ConfigType]

SampleList = Sequence[PointData]
OptSampleList = Optional[SampleList]

ForwardResults = Union[torch.Tensor]
