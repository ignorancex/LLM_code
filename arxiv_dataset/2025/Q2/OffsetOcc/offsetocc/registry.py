# Copyright (c) OpenMMLab. All rights reserved.
# Modified by Nicola Marinello, 2025
"""Each node is a child of the root registry in MMEngine.

More details can be found at
https://mmengine.readthedocs.io/en/latest/tutorials/registry.html.
"""

from mmengine.registry import DATASETS as MMENGINE_DATASETS
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmengine.registry import TRANSFORMS as MMENGINE_TRANSFORMS
from mmengine.registry import METRICS as MMENGINE_METRICS
from mmengine.registry import VISUALIZERS as MMENGINE_VISUALIZERS
from mmengine.registry import HOOKS as MMENGINE_HOOKS
from mmengine.registry import VISBACKENDS as MMENGINE_VISBACKENDS
from mmdet.registry import TASK_UTILS as MMDET_TASK_UTILS
from mmengine.registry import Registry

# manage all kinds of modules inheriting `nn.Module`
MODELS = Registry('model', parent=MMENGINE_MODELS, locations=['offsetocc.models'])
# manage data-related modules
DATASETS = Registry('dataset', parent=MMENGINE_DATASETS, locations=['offsetocc.datasets'])
TRANSFORMS = Registry('transform', parent=MMENGINE_TRANSFORMS, locations=['offsetocc.datasets.transforms'])
# manage all kinds of metrics
METRICS = Registry('metric', parent=MMENGINE_METRICS, locations=['offsetocc.evaluation'])
# manage visualizer
VISUALIZERS = Registry('visualizer', parent=MMENGINE_VISUALIZERS, locations=['offsetocc.visualization'])
# manage all kinds of hooks`
HOOKS = Registry('hook', parent=MMENGINE_HOOKS, locations=['offsetocc.engine.hooks'])
# manage visualizer backend
VISBACKENDS = Registry('vis_backend', parent=MMENGINE_VISBACKENDS, locations=['offsetocc.visualization'])
# manage all kinds of task utils
TASK_UTILS = Registry('task_utils', parent=MMDET_TASK_UTILS, locations=['offsetocc.models.task_modules'])