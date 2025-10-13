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
Put all the method implementations in one location.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from typing import Dict

import tyro

from scrstudio.configs.base_config import MachineConfig
from scrstudio.data.datamanagers.glbuffer_datamanager import GLBufferDataManagerConfig
from scrstudio.data.datasets.camloc_dataset import CamLocDatasetConfig
from scrstudio.data.samplers import BatchRandomSamplerConfig, GlobalFeatSamplerConfig
from scrstudio.data.utils.readers import LMDBReaderConfig
from scrstudio.encoders.base_encoder import ImageAugmentConfig
from scrstudio.encoders.dedode_encoder import DedodeEncoderConfig
from scrstudio.encoders.loftr_encoder import LoFTREncoderConfig
from scrstudio.encoders.pca_encoder import PCAEncoderConfig
from scrstudio.engine.optimizers import AdamOptimizerConfig, AdamWOptimizerConfig
from scrstudio.engine.schedulers import OneCycleSchedulerConfig
from scrstudio.engine.trainer import TrainerConfig
from scrstudio.model_components.losses import CoordLossConfig, ReproLossConfig, RobustLossConfig
from scrstudio.model_components.mlp import (
    BlockListConfig,
    InputBlockConfig,
    PositionDecoderConfig,
    PositionEncoderConfig,
    PositionRefinerConfig,
    ResBlockConfig,
)
from scrstudio.models.scrfacto import ScrfactoConfig
from scrstudio.pipelines.base_pipeline import VanillaPipelineConfig
from scrstudio.pipelines.node2vec_pipeline import Node2VecConfig, Node2VecDataManagerConfig, Node2VecPipelineConfig

method_configs: Dict[str, TrainerConfig] = {}
descriptions = {
    "node2vec": "Node2Vec for covisibility graph-based global encoding training.",
    "scrfacto": "Recommended model for large scenes. This model will be continually updated.",
    "depth-scrfacto": "Model with GT depth supervision. This model will be continually updated.",
    "scrfacto-large": "Model with larger network width.",
}

method_configs["node2vec"] = TrainerConfig(
    method_name="node2vec",
    steps_per_eval_all_images=200,
    max_num_iterations=5000,
    pipeline=Node2VecPipelineConfig(
        model=Node2VecConfig(
            graph="pose_overlap.npz",
            edge_threshold=0.2,
        ),
        datamanager=Node2VecDataManagerConfig(
            batch_size=256,
            num_workers=2,
        ),
    ),
    optimizers={
        "node2vec": {"optimizer": AdamOptimizerConfig(lr=0.01), "scheduler": None},
    },
    vis="tensorboard",
)


method_configs["scrfacto"]=TrainerConfig(
    method_name="scrfacto",
    max_num_iterations=100000,
    mixed_precision=True,
    gradient_accumulation_steps=2,
    pipeline=VanillaPipelineConfig(
        datamanager=GLBufferDataManagerConfig(
            training_buffer_size=32000000,
            samples_per_image=5000,
            global_feat=GlobalFeatSamplerConfig(
                train_covis_graph='pose_overlap.npz',
                train_covis_thres=0.2,
                neighbor_ratio=0.5,
            ),
            sampler=BatchRandomSamplerConfig(batch_size=40960),
            encoder=PCAEncoderConfig(
                encoder=DedodeEncoderConfig(
                    detector="L",
                    descriptor="B",
                    k=5000,
                ),
                pca_path='pcad3LB_128.pth'
            ),
            train_dataset=CamLocDatasetConfig(
                augment=ImageAugmentConfig(),
                split='train',
                feat_name='pose_n2c.pt',
                num_decoder_clusters=50,
            ),
            eval_dataset=CamLocDatasetConfig(
                split='val',
            ),
            mixed_precision=True,
        num_data_loader_workers=2,
        ),
        num_pose_workers=3,
        model=ScrfactoConfig(
            in_channels=384,
            head_channels=768,
            backbone=BlockListConfig(
                blocks=[
                    (InputBlockConfig(
                    ),1),
                    (ResBlockConfig(
                    ),3),
                    (PositionDecoderConfig(
                        output_name="sc0",
                    ),1),
                    (PositionEncoderConfig(
                        input_name="sc0",
                        period=2048,num_freqs=16,
                        max_freq_exp=12
                    ),1),
                    (ResBlockConfig(
                    ),2),
                    (PositionRefinerConfig(
                        base_name="sc0",
                        output_name="sc",
                    ),1),
                ]),
            losses=[
                ReproLossConfig(
                    input_name ="sc",
                    output_name="",
                    robust_loss = RobustLossConfig(
                        soft_clamp=25,
                        soft_clamp_min=1,
                    ),
                    coord_loss=CoordLossConfig(
                        target_name="sc0",
                    ),
                ),
                ReproLossConfig(
                    robust_loss = RobustLossConfig(
                        soft_clamp=50,
                        soft_clamp_min=1,
                        robust_type="gm"
                    ),
                    depth_bias_adjust=True,
                    input_name ="sc0",
                    output_name="0",
                    db_std=3,
                    depth_target=10,
                ),

            ],
            mlp_ratio=2,
            max_num_iterations=100000,
        )
    ),
    machine=MachineConfig(num_devices=4),
    optimizers={
        "head": {
            "optimizer": AdamWOptimizerConfig(lr=0.0005),
            "scheduler": OneCycleSchedulerConfig(
                learning_rate_max=0.003,
                pct_start=0.04,
                max_steps=100000,
            )
        },
    },
    vis="tensorboard",
)

depth_scrfacto=deepcopy(method_configs["scrfacto"])
depth_scrfacto.method_name="depth-scrfacto"
depth_scrfacto.pipeline.datamanager.train_dataset.depth=LMDBReaderConfig(data='depth_lmdb',  img_type='depths')
depth_scrfacto.pipeline.model.losses[0].coord_loss=CoordLossConfig()
depth_scrfacto.pipeline.model.losses[1].coord_loss=CoordLossConfig()
method_configs["depth-scrfacto"]=depth_scrfacto

scrfacto_large=deepcopy(method_configs["scrfacto"])
scrfacto_large.method_name="scrfacto-large"
scrfacto_large.pipeline.model.head_channels=1280
method_configs["scrfacto-large"]=scrfacto_large

loftr=deepcopy(method_configs["scrfacto"])
loftr.method_name="loftr"
loftr.pipeline.datamanager.encoder=PCAEncoderConfig(
    encoder=LoFTREncoderConfig(model="loftr_indoor_new"),
    pca_path='pcaloftr_indoor_ds_new_128.pth'
)
loftr.pipeline.datamanager.samples_per_image=1500
method_configs["loftr"]=loftr

loftr_large=deepcopy(method_configs["loftr"])
loftr_large.method_name="loftr-large"
loftr_large.pipeline.model.head_channels=1280
method_configs["loftr-large"]=loftr_large

loftr_outdoor_large=deepcopy(method_configs["loftr-large"])
loftr_outdoor_large.method_name="loftr-outdoor-large"
loftr_outdoor_large.pipeline.datamanager.encoder=PCAEncoderConfig(
    encoder=LoFTREncoderConfig(model="loftr_outdoor"),
    pca_path='pcaloftr_outdoor_128.pth'
)
method_configs["loftr-outdoor-large"]=loftr_outdoor_large

def merge_methods(methods, method_descriptions, new_methods, new_descriptions, overwrite=True):
    """Merge new methods and descriptions into existing methods and descriptions.
    Args:
        methods: Existing methods.
        method_descriptions: Existing descriptions.
        new_methods: New methods to merge in.
        new_descriptions: New descriptions to merge in.
    Returns:
        Merged methods and descriptions.
    """
    methods = OrderedDict(**methods)
    method_descriptions = OrderedDict(**method_descriptions)
    for k, v in new_methods.items():
        if overwrite or k not in methods:
            methods[k] = v
            method_descriptions[k] = new_descriptions.get(k, "")
    return methods, method_descriptions


def sort_methods(methods, method_descriptions):
    """Sort methods and descriptions by method name."""
    methods = OrderedDict(sorted(methods.items(), key=lambda x: x[0]))
    method_descriptions = OrderedDict(sorted(method_descriptions.items(), key=lambda x: x[0]))
    return methods, method_descriptions


all_methods, all_descriptions = method_configs, descriptions
# Add discovered external methods
all_methods, all_descriptions = merge_methods(all_methods, all_descriptions,{},{})
all_methods, all_descriptions = sort_methods(all_methods, all_descriptions)


AnnotatedBaseConfigUnion = tyro.conf.SuppressFixed[  # Don't show unparseable (fixed) arguments in helptext.
    tyro.conf.FlagConversionOff[
        tyro.extras.subcommand_type_from_defaults(defaults=all_methods, descriptions=all_descriptions)
    ]
]
"""Union[] type over config types, annotated with default instances for use with
tyro.cli(). Allows the user to pick between one of several base configurations, and
then override values in it."""
