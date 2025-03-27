from typing import Tuple

import numpy as np
import torch
from avalanche.models import MultiHeadClassifier, IncrementalClassifier
from torch import nn

from models.routing.model import MergingRoutingModel
from models.utils import AvalanceCombinedModel, CustomMultiHeadClassifier, \
    PytorchCombinedModel
from proposal.head_csc import CascadedScalingClassifier


def get_cl_model(
        backbone,
        model_name: str,
        method_name: str,
        input_shape: Tuple[int, int, int],
        is_class_incremental_learning: bool = False,
        cml_out_features: int = None,
        is_stream: bool = False,
        head_classes=None,
        masking=False,
        **kwargs):
    if isinstance(backbone, (MergingRoutingModel)):
        return backbone

    if method_name in ['cope', 'mcml']:
        return backbone

    x = torch.randn((1,) + input_shape)
    o = backbone(x)

    size = np.prod(o.shape)

    if method_name == 'margin':
        classifier = CascadedScalingClassifier(size)
        model = PytorchCombinedModel(backbone, classifier)

        return model

    def heads_generator(i, o):
        class Wrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Sequential(
                    # nn.Dropout(0.5),
                    nn.Linear(i, i),
                    nn.ReLU(),
                    nn.Linear(i, o),
                    # nn.Dropout(0.2)
                )

            def forward(self, x, task_labels=None, **kwargs):
                return self.model(x)

        return Wrapper()

    if head_classes is not None:
        classifier = nn.Sequential(nn.Flatten(1), nn.Linear(size, head_classes))
        return PytorchCombinedModel(backbone, classifier)
    else:
        if is_stream or method_name == 'icarl':
            classifier = IncrementalClassifier(size)
        else:
            if method_name == 'er':
                classifier = CustomMultiHeadClassifier(size, heads_generator)
            else:
                if method_name != 'cml':
                    if is_class_incremental_learning:
                        classifier = IncrementalClassifier(size,
                                                           masking=masking)
                    else:
                        classifier = MultiHeadClassifier(size, masking=masking)
                else:
                    if cml_out_features is None:
                        cml_out_features = 128

                    classifier = CustomMultiHeadClassifier(size,
                                                           heads_generator,
                                                           cml_out_features)

    model = AvalanceCombinedModel(backbone, classifier)

    return model
