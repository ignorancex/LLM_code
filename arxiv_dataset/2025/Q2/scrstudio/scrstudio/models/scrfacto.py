# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SCR implementation that combines many recent advancements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Type

from torch.nn import Parameter

from scrstudio.configs.base_config import InstantiateConfig
from scrstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from scrstudio.model_components.losses import ReproLossConfig
from scrstudio.model_components.mlp import BlockListConfig, InputBlockConfig, ResBlockConfig
from scrstudio.models.base_model import Model, ModelConfig


@dataclass
class ScrfactoConfig(ModelConfig):

    _target: Type = field(default_factory=lambda: Scrfacto)
    backbone: InstantiateConfig = field(default_factory=lambda: BlockListConfig(blocks=[
                                                    (InputBlockConfig(),1),
                                                    (ResBlockConfig(),3),
                                                    ]))
    in_channels: int = 512
    head_channels: int = 512
    mlp_ratio: float = 1.0
    max_num_iterations: int = 100000

    losses: List[InstantiateConfig] = field(default_factory=lambda: [ReproLossConfig()])

class Scrfacto(Model):
    """Nerfacto model

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: ScrfactoConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()
        self.iteration=0

        self.losses=[loss.setup(
            total_iterations=self.config.max_num_iterations,
        ) for loss in self.config.losses]

        self.backbone=self.config.backbone.setup(
                                        in_channels=self.config.in_channels,
                                        head_channels=self.config.head_channels,
                                        mlp_ratio=self.config.mlp_ratio,
                                        metadata=self.metadata,
                                    )

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["head"] = list(self.parameters())
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []

        def set_step(step):
            # https://arxiv.org/pdf/2111.12077.pdf eq. 18
            self.iteration = step

        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=set_step,
            )
        )
        return callbacks

    def get_outputs(self, ray_bundle: Dict):
        return self.backbone(ray_bundle)


    def get_metrics_dict(self, outputs, batch):
        outputs["step"]=self.iteration
        outputs.update(batch)
        for loss in self.losses:
            outputs=loss(outputs)
        return outputs["metrics"]

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        if metrics_dict is None:
            metrics_dict = self.get_metrics_dict(outputs, batch)
        loss_dict = {
            "loss": metrics_dict["loss"],
        }
        
        return loss_dict

