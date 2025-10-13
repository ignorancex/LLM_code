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
Datamanager.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

from torch import nn
from torch.nn import Parameter
from torch.utils.data import Dataset

from scrstudio.configs.base_config import InstantiateConfig
from scrstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes


@dataclass
class DataManagerConfig(InstantiateConfig):
    """Configuration for data manager instantiation; DataManager is in charge of keeping the train/eval dataparsers;
    After instantiation, data manager holds both train/eval datasets and is in charge of returning unpacked
    train/eval data at each iteration
    """

    _target: Type = field(default_factory=lambda: DataManager)
    """Target class to instantiate."""
    data: Optional[Path] = None
    """Source of data, may not be used by all models."""


class DataManager(nn.Module):
    """Generic data manager's abstract class

    This version of the data manager is designed be a monolithic way to load data and latents,
    especially since this may contain learnable parameters which need to be shared across the train
    and test data managers. The idea is that we have setup methods for train and eval separately and
    this can be a combined train/eval if you want.

    Usage:
    To get data, use the next_train and next_eval functions.
    This data manager's next_train and next_eval methods will return 2 things:

    1. A Raybundle: This will contain the rays we are sampling, with latents and
        conditionals attached (everything needed at inference)
    2. A "batch" of auxiliary information: This will contain the mask, the ground truth
        pixels, etc needed to actually train, score, etc the model

    Rationale:
    Because of this abstraction we've added, we can support more NeRF paradigms beyond the
    vanilla nerf paradigm of single-scene, fixed-images, no-learnt-latents.
    We can now support variable scenes, variable number of images, and arbitrary latents.


    Train Methods:
        setup_train: sets up for being used as train
        iter_train: will be called on __iter__() for the train iterator
        next_train: will be called on __next__() for the training iterator
        get_train_iterable: utility that gets a clean pythonic iterator for your training data

    Eval Methods:
        setup_eval: sets up for being used as eval
        iter_eval: will be called on __iter__() for the eval iterator
        next_eval: will be called on __next__() for the eval iterator
        get_eval_iterable: utility that gets a clean pythonic iterator for your eval data


    Attributes:
        train_count (int): the step number of our train iteration, needs to be incremented manually
        eval_count (int): the step number of our eval iteration, needs to be incremented manually
        train_dataset (Dataset): the dataset for the train dataset
        eval_dataset (Dataset): the dataset for the eval dataset

        Additional attributes specific to each subclass are defined in the setup_train and setup_eval
        functions.

    """

    train_dataset: Optional[Dataset] = None
    eval_dataset: Optional[Dataset] = None

    def __init__(self):
        """Constructor for the DataManager class.

        Subclassed DataManagers will likely need to override this constructor.

        If you aren't manually calling the setup_train and setup_eval functions from an overriden
        constructor, that you call super().__init__() BEFORE you initialize any
        nn.Modules or nn.Parameters, but AFTER you've already set all the attributes you need
        for the setup functions."""
        super().__init__()        

    def forward(self):
        """Blank forward method

        This is an nn.Module, and so requires a forward() method normally, although in our case
        we do not need a forward() method"""
        raise NotImplementedError

    @abstractmethod
    def setup_train(self):
        """Sets up the data manager for training.

        Here you will define any subclass specific object attributes from the attribute"""

    @abstractmethod
    def setup_eval(self):
        """Sets up the data manager for evaluation"""

    @abstractmethod
    def next_train(self, step: int) -> Tuple[Dict, Dict]:
        """Returns the next batch of data from the train data manager.

        Args:
            step: the step number of the eval image to retrieve
        Returns:
            A tuple of the ray bundle for the image, and a dictionary of additional batch information
            such as the groundtruth image.
        """
        raise NotImplementedError


    @abstractmethod
    def get_train_batch_size(self) -> int:
        """Returns the number of rays per batch for training."""
        raise NotImplementedError

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns a list of callbacks to be used during training."""
        return []

    @abstractmethod
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the data manager.

        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        return {}

