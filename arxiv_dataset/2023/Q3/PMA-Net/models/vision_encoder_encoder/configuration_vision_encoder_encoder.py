# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

import copy

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers import AutoConfig


logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)


class VisionEncoderEncoderConfig(PretrainedConfig):
    r"""
    [`VisionEncoderEncoderDecoderConfig`] is the configuration class to store the configuration of a
    [`VisionEncoderEncoderDecoderModel`]. It is used to instantiate a Vision-Encoder-Encoder-Text-Decoder model
    according to the specified arguments, defining the encoder and decoder configs.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        kwargs (*optional*):
            Dictionary of keyword arguments. Notably:

                - **vision_encoder** ([`PretrainedConfig`], *optional*) -- An instance of a configuration object that
                  defines the vision encoder config.
                - **encoder** ([`PretrainedConfig`], *optional*) -- An instance of a configuration object that defines
                  the encoder config.
    """
    model_type = "vision-encoder-encoder"
    is_composition = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "vision_encoder" not in kwargs or "encoder" not in kwargs:
            raise ValueError(
                f"A configuration of type {self.model_type} cannot be instantiated because "
                f"not both `vision_encoder` and `encoder` sub-configurations are passed, but only {kwargs}"
            )

        vision_encoder_config = kwargs.pop("vision_encoder")
        vision_encoder_model_type = vision_encoder_config.pop("model_type")
        encoder_config = kwargs.pop("encoder")
        encoder_model_type = encoder_config.pop("model_type")
        self.freeze_vision_encoder = kwargs.pop("freeze_vision_encoder", True)
        self.freeze_encoder = kwargs.pop("freeze_encoder", False)

        self.vision_encoder = AutoConfig.for_model(vision_encoder_model_type, **vision_encoder_config)
        self.encoder = AutoConfig.for_model(encoder_model_type, **encoder_config)
        self.is_encoder_decoder = False
        self.hidden_size = self.encoder.d_embed

    @classmethod
    def from_vision_encoder_encoder_configs(
        cls, vision_encoder_config: PretrainedConfig, encoder_config: PretrainedConfig, **kwargs
    ) -> PretrainedConfig:
        r"""
        Instantiate a [`VisionEncoderEncoderConfig`] (or a derived class) from a pre-trained encoder model
        configuration and decoder model configuration.

        Returns:
            [`VisionEncoderEncoderConfig`]: An instance of a configuration object
        """
        # logger.info("Setting `config.is_decoder=True` and `config.add_cross_attention=True` for decoder_config")

        return cls(vision_encoder=vision_encoder_config.to_dict(), encoder=encoder_config.to_dict(), **kwargs)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default *to_dict()* from *PretrainedConfig*.

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["vision_encoder"] = self.vision_encoder.to_dict()
        output["encoder"] = self.encoder.to_dict()
        output["model_type"] = self.__class__.model_type
        return output