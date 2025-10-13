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
Abstracts for the Pipeline class.
"""
from __future__ import annotations

import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Type, Union

import numpy as np
import scipy
import torch
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.nn import Node2Vec as pygNode2Vec

from scrstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig
from scrstudio.data.samplers import BatchRandomSamplerConfig
from scrstudio.models.base_model import Model, ModelConfig
from scrstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from scrstudio.utils import profiler
from scrstudio.utils.comms import is_main_process


@dataclass
class Node2VecConfig(ModelConfig):
    _target: Type = field(default_factory=lambda: Node2Vec)
    graph: str = "mt_pose_overlap_sym.npz"
    embedding_dim: int = 256
    p: float = 0.25
    q: float = 4
    walks_per_node: int = 128
    walk_length: int = 20
    context_size: int = 10

    edge_threshold: float = 0.2



class Node2Vec(Model):
    config: Node2VecConfig

    def populate_modules(self):
        super().populate_modules()
        data = self.kwargs["data"]
        graph = scipy.sparse.load_npz(data/ "train"/self.config.graph).tocoo()
        graph.data[graph.data < self.config.edge_threshold] = 0
        graph.eliminate_zeros()
        self.graph = graph
        self.edge_index = torch.tensor(np.stack([graph.row, graph.col]), dtype=torch.long)
        self.model = pygNode2Vec(
            self.edge_index,
            embedding_dim=self.config.embedding_dim,
            p=self.config.p,
            q=self.config.q,
            walks_per_node=self.config.walks_per_node,
            walk_length=self.config.walk_length,
            context_size=self.config.context_size,
        )

    def get_param_groups(self):
        return {"node2vec": list(self.model.parameters())}
    
    def get_outputs(self, ray_bundle):
        return {}

    def get_metrics_dict(self, outputs, batch):
        return {"loss": self.model.loss(batch["positive"], batch["negative"])}

    def get_loss_dict(self, outputs, batch, metrics_dict):
        return {"loss": metrics_dict["loss"]}


@dataclass
class Node2VecDataManagerConfig(DataManagerConfig):

    _target: Type = field(default_factory=lambda: Node2VecDataManager)
    batch_size: int = 256
    num_workers: int = 0


class Node2VecDataManager(DataManager):
    config: Node2VecDataManagerConfig

    def __init__(
        self,
        config: Node2VecDataManagerConfig,
        node2vec: Node2Vec,
        device: Union[torch.device, str] = "cuda",
        **kwargs,
    ):
        self.config = config
        self.device = device
        self.eval_dataset = 1 # Dummy

        super().__init__()
        self.iter=iter(node2vec.model.loader(batch_sampler=BatchRandomSamplerConfig(
                batch_size=self.config.batch_size).setup(
                dataset_size=node2vec.model.num_nodes),
            num_workers=self.config.num_workers,
            persistent_workers=self.config.num_workers > 0,
        ))

    def next_train(self, step):
        pos, neg = next(self.iter)
        return {}, {"positive": pos.to(self.device), "negative": neg.to(self.device)}

    def get_train_batch_size(self) -> int:
        return self.config.batch_size

@dataclass
class Node2VecPipelineConfig(VanillaPipelineConfig):
    _target: Type = field(default_factory=lambda: Node2VecPipeline)
    model: Node2VecConfig = field(default_factory=Node2VecConfig)
    datamanager: Node2VecDataManagerConfig = field(default_factory=Node2VecDataManagerConfig)
    
def precision_recall_curve(
    target: torch.Tensor,
    input: torch.Tensor,
):
    target,input=target.flatten(),input.flatten()
    threshold, indices = input.sort(descending=True)
    mask = torch.nn.functional.pad(threshold.diff(dim=0) != 0, [0, 1], value=1.0)
    num_tp = (target[indices]).cumsum(0)[mask]
    num_fp = (1 - (target[indices]).long()).cumsum(0)[mask]
    precision = (num_tp / (num_tp + num_fp)).flip(0)
    recall = (num_tp / num_tp[-1]).flip(0)
    threshold = threshold[mask].flip(0)

    # The last precision and recall values are 1.0 and 0.0 without a corresponding threshold.
    # This ensures that the graph starts on the y-axis.
    precision = torch.cat([precision, precision.new_ones(1)])
    recall = torch.cat([recall, recall.new_zeros(1)])

    # If recalls are NaNs, set NaNs to 1.0s.
    if torch.isnan(recall[0]):
        recall = torch.nan_to_num(recall, 1.0)

    return precision, recall, threshold
def average_precision_score(
        target: torch.Tensor,
        input: torch.Tensor,
):
    precision, recall, threshold = precision_recall_curve(target, input)
    # area = riemann_integral(recall, precision)
    area = -torch.trapz(precision, recall)
    return area
class Node2VecPipeline(VanillaPipeline):
    model: Node2Vec
    datamanager: Node2VecDataManager
    def __init__(
        self,
        config: Node2VecPipelineConfig,
        device: str,
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):  
        Module.__init__(self)
        self.config = config
        self.world_size = world_size

        self._model = self.config.model.setup(data=self.config.datamanager.data)
        self.model.to(device)
        self.datamanager = self.config.datamanager.setup(node2vec=self.model,device=device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(Model, DDP(self._model, device_ids=[local_rank], find_unused_parameters=False,output_device=local_rank))


    @profiler.time_function
    def get_average_eval_image_metrics(
        self, step: Optional[int] = None, output_path: Optional[Path] = None, get_std: bool = False
    ):
        if not is_main_process():
            return {}
        self.model.edge_index = self.model.edge_index.to(self.datamanager.device)
        graph = self.model.graph
        edge_index = self.model.edge_index
        gt_label =torch.zeros((graph.shape[0], graph.shape[1]), dtype=torch.bool, device='cuda')
        gt_label[edge_index[0], edge_index[1]] = True
        gt_label.fill_diagonal_(False)
        feat = self.model.model()
        dist=torch.cdist(feat.unsqueeze(0),feat.unsqueeze(0)).squeeze(0)
        dist.fill_diagonal_(float('inf'))
        ap=average_precision_score(gt_label, -dist.flatten())
        pos_dist=dist[gt_label].flatten()
        ngt_label=~gt_label
        ngt_label.fill_diagonal_(False)
        neg_dist=dist[ngt_label].flatten()
        median_pos=torch.median(pos_dist)
        median_neg=torch.median(neg_dist)
        std_pos, mean_pos = torch.std_mean(pos_dist)
        std_neg, mean_neg = torch.std_mean(neg_dist)
        return {
            "AP": ap,
            "median_pos_dist": median_pos,
            "median_neg_dist": median_neg,
            "mean_pos_dist": mean_pos,
            "mean_neg_dist": mean_neg,
            "std_pos_dist": std_pos,
            "std_neg_dist": std_neg,
        }


