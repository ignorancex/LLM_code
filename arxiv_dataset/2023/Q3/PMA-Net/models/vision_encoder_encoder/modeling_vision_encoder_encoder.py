# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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
""" Classes to support Vision-Encoder-Encoder architectures"""


from typing import Optional

import torch
from torch import nn

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings, \
    logging
from transformers import AutoConfig, AutoModel
from . import VisionEncoderEncoderConfig

logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)

_CONFIG_FOR_DOC = "VisionEncoderEncoderConfig"

VISION_ENCODER_ENCODER_START_DOCSTRING = r"""
    This class can be used to initialize an image-to-text-sequence model with any pretrained vision autoencoding model
    as the encoder and any pretrained text autoregressive model as the decoder. The encoder is loaded via
    [`~AutoModel.from_pretrained`] function and the decoder is loaded via [`~AutoModelForCausalLM.from_pretrained`]
    function. Cross-attention layers are automatically added to the decoder and should be fine-tuned on a downstream
    generative task, like image captioning.

    The effectiveness of initializing sequence-to-sequence models with pretrained checkpoints for sequence generation
    tasks was shown in [Leveraging Pre-trained Checkpoints for Sequence Generation
    Tasks](https://arxiv.org/abs/1907.12461) by Sascha Rothe, Shashi Narayan, Aliaksei Severyn. Michael Matena, Yanqi
    Zhou, Wei Li, Peter J. Liu.

    Additionally, in [TrOCR: Transformer-based Optical Character Recognition with Pre-trained
    Models](https://arxiv.org/abs/2109.10282) it is shown how leveraging large pretrained vision models for optical
    character recognition (OCR) yields a significant performance improvement.

    After such a Vision-Encoder-Text-Decoder model has been trained/fine-tuned, it can be saved/loaded just like any
    other models (see the examples for more information).

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`VisionEncoderDecoderConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

VISION_ENCODER_ENCODER_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using a feature extractor (e.g. if you use ViT as the encoder,
            you should use [`ViTFeatureExtractor`]). See [`ViTFeatureExtractor.__call__`] for details.
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`PreTrainedTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            For training, `decoder_input_ids` are automatically created by the model by shifting the `labels` to the
            right, replacing -100 by the `pad_token_id` and prepending them with the `decoder_start_token_id`.
        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.
        encoder_outputs (`tuple(torch.FloatTensor)`, *optional*):
            This tuple must consist of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
            `last_hidden_state` (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) is a tensor
            of hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the
            decoder.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
            representation. This is useful if you want more control over how to convert `decoder_input_ids` indices
            into associated vectors than the model's internal embedding lookup matrix.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss for the decoder. Indices should be in `[-100, 0,
            ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            If set to `True`, the model will return a [`~utils.Seq2SeqLMOutput`] instead of a plain tuple.
        kwargs: (*optional*) Remaining dictionary of keyword arguments. Keyword arguments come in two flavors:

            - Without a prefix which will be input as `**encoder_kwargs` for the encoder forward function.
            - With a *decoder_* prefix which will be input as `**decoder_kwargs` for the decoder forward function.
"""


@add_start_docstrings(VISION_ENCODER_ENCODER_START_DOCSTRING)
class VisionEncoderEncoderModel(PreTrainedModel):
    r"""
    [`VisionEncoderEncoderDecoderModel`] is a generic model class that will be instantiated as a transformer
    architecture with one of the base vision model classes of the library as vision encoder, a model class as encoder
    and another one as decoder when created with the :meth*~transformers.AutoModel.from_pretrained* class method for
    the vision encoder and the encoder and :meth*~transformers.AutoModelForCausalLM.from_pretrained* class method for
    the decoder.
    """
    config_class = VisionEncoderEncoderConfig
    base_model_prefix = "vision_encoder_encoder"
    main_input_name = "pixel_values"

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        vision_encoder: Optional[PreTrainedModel] = None,
        encoder: Optional[PreTrainedModel] = None,
    ):
        if config is None and (vision_encoder is None or encoder is None):
            raise ValueError("Either a configuration or a vision encoder and an encoder has to be provided.")
        if config is None:
            config = VisionEncoderEncoderConfig.from_vision_encoder_encoder_configs(vision_encoder.config,
                                                                                    encoder.config)
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(f"Config: {config} has to be of type {self.config_class}")

        # initialize with config
        # make sure input & output embeddings is not tied
        config.tie_word_embeddings = False
        super().__init__(config)

        if vision_encoder is None:
            vision_encoder = AutoModel.from_config(config.vision_encoder)

        if encoder is None:
            encoder = AutoModel.from_config(config.encoder)

        self.vision_encoder = vision_encoder
        self.encoder = encoder
        self.encoder.main_input_name = self.main_input_name

        if self.vision_encoder.config.to_dict() != self.config.vision_encoder.to_dict():
            logger.warning(
                f"Config of the encoder: {self.vision_encoder.__class__} is overwritten by shared encoder config: {self.config.vision_encoder}"
            )
        if self.encoder.config.to_dict() != self.config.encoder.to_dict():
            logger.warning(
                f"Config of the encoder: {self.encoder.__class__} is overwritten by shared encoder config: {self.config.encoder}"
            )

        # make sure that the individual model's config refers to the shared config
        # so that the updates to the config will be synced
        self.vision_encoder.config = self.config.vision_encoder
        self.encoder.config = self.config.encoder

        # vision encoder outputs might need to be projected to different dimension for encoder
        if (
            self.vision_encoder.config.hidden_size != self.encoder.config.hidden_size
            and self.encoder.config.cross_attention_hidden_size is None
        ):
            self.vis_enc_to_enc_proj = nn.Linear(self.vision_encoder.config.hidden_size, self.encoder.config.hidden_size)

        if self.encoder.get_output_embeddings() is not None:
            raise ValueError(
                f"The encoder {self.encoder} should not have a LM Head. Please use a model without LM Head"
            )

    def get_vision_encoder(self):
        return self.vision_encoder
    
    def get_encoder(self):
        return self.encoder

    def get_output_embeddings(self):
        return self.encoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self.encoder.set_output_embeddings(new_embeddings)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # At the moment fast initialization is not supported for composite models
        if kwargs.get("_fast_init", False):
            logger.warning(
                "Fast initialization is currently not supported for VisionEncoderEncoderModel. "
                "Falling back to slow initialization..."
            )
        kwargs["_fast_init"] = False
        return super().from_pretrained(*args, **kwargs)

    @classmethod
    def from_vision_encoder_encoder_pretrained(
        cls,
        vision_encoder_pretrained_model_name_or_path: str = None,
        encoder_pretrained_model_name_or_path: str = None,
        *model_args,
        **kwargs
    ) -> PreTrainedModel:
        r"""
        Instantiate a vision encoder and a encoder from one or two base classes of the library from pretrained model
        checkpoints.


        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
        the model, you need to first set it back in training mode with `model.train()`.

        Params:
            encoder_pretrained_model_name_or_path (`str`, *optional*):
                Information necessary to initiate the image encoder. Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co. An
                      example is `google/vit-base-patch16-224-in21k`.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
                      this case, `from_tf` should be set to `True` and a configuration object should be provided as
                      `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
                      PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            decoder_pretrained_model_name_or_path (`str`, *optional*, defaults to `None`):
                Information necessary to initiate the text decoder. Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                      user or organization name, like `dbmdz/bert-base-german-cased`.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
                      this case, `from_tf` should be set to `True` and a configuration object should be provided as
                      `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
                      PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.

            model_args (remaining positional arguments, *optional*):
                All remaning positional arguments will be passed to the underlying model's `__init__` method.

            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                `output_attentions=True`).

                - To update the encoder configuration, use the prefix *encoder_* for each configuration parameter.
                - To update the decoder configuration, use the prefix *decoder_* for each configuration parameter.
                - To update the parent model configuration, do not use a prefix for each configuration parameter.

                Behaves differently depending on whether a `config` is provided or automatically loaded.

        """

        kwargs_vision_encoder = {
            argument[len("vision_encoder_") :]: value for argument, value in kwargs.items() if argument.startswith("vision_encoder_")
        }

        kwargs_encoder = {
            argument[len("encoder_") :]: value for argument, value in kwargs.items() if argument.startswith("encoder_")
        }

        # remove vision encoder and encoder kwargs from kwargs
        for key in kwargs_vision_encoder.keys():
            del kwargs["vision_encoder_" + key]
        for key in kwargs_encoder.keys():
            del kwargs["encoder_" + key]

        # Load and initialize the vision encoder and encoder
        # The distinction between vision encoder and encoder at the model level is made
        # by the value of the flag `is_decoder` that we need to set correctly.
        vision_encoder = kwargs_vision_encoder.pop("model", None)
        if vision_encoder is None:
            if vision_encoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `vision_encoder_model` is not defined as an argument, a `vision_encoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_vision_encoder:
                vision_encoder_config, kwargs_vision_encoder = AutoConfig.from_pretrained(
                    vision_encoder_pretrained_model_name_or_path, **kwargs_vision_encoder, return_unused_kwargs=True
                )

                if vision_encoder_config.is_decoder is True or vision_encoder_config.add_cross_attention is True:
                    logger.info(
                        f"Initializing {vision_encoder_pretrained_model_name_or_path} as a vision_encoder model "
                        "from a decoder model. Cross-attention and casual mask are disabled."
                    )
                    vision_encoder_config.is_decoder = False
                    vision_encoder_config.add_cross_attention = False

                kwargs_vision_encoder["config"] = vision_encoder_config

            vision_encoder = AutoModel.from_pretrained(vision_encoder_pretrained_model_name_or_path, *model_args, **kwargs_vision_encoder)
        
        encoder = kwargs_encoder.pop("model", None)
        if encoder is None:
            if encoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `encoder_model` is not defined as an argument, a `encoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_encoder:
                encoder_config, kwargs_encoder = AutoConfig.from_pretrained(
                    encoder_pretrained_model_name_or_path, **kwargs_encoder, return_unused_kwargs=True
                )

                if encoder_config.is_decoder is True or encoder_config.add_cross_attention is True:
                    logger.info(
                        f"Initializing {encoder_pretrained_model_name_or_path} as a encoder model "
                        "from a decoder model. Cross-attention and casual mask are disabled."
                    )
                    encoder_config.is_decoder = False
                    encoder_config.add_cross_attention = False

                kwargs_encoder["config"] = encoder_config

            encoder = AutoModel.from_pretrained(encoder_pretrained_model_name_or_path, *model_args, **kwargs_encoder)

        # instantiate config with corresponding kwargs
        config = VisionEncoderEncoderConfig.from_vision_encoder_encoder_configs(vision_encoder.config,
                                                                                encoder.config, **kwargs)

        # make sure input & output embeddings is not tied
        config.tie_word_embeddings = False
        return cls(vision_encoder=vision_encoder, encoder=encoder, config=config)

    @add_start_docstrings_to_model_forward(VISION_ENCODER_ENCODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values=None,
        vision_encoder_outputs=None,
        encoder_outputs=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        """
        Return:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_vision_encoder = {argument: value for argument, value in kwargs.items() if argument.startswith("vision_encoder_")
                                 and argument != 'gts'}
        
        kwargs_encoder = {argument: value for argument, value in kwargs.items() if argument.startswith("encoder_")
                          and argument != 'gts'}

        if vision_encoder_outputs is None:
            if pixel_values is None:
                raise ValueError("You have to specify pixel_values")

            if self.config.freeze_vision_encoder:
                self.vision_encoder.eval()
            with torch.set_grad_enabled(not self.config.freeze_vision_encoder):
                vision_encoder_outputs = self.vision_encoder(
                    pixel_values,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs_vision_encoder,
                )
        elif isinstance(vision_encoder_outputs, tuple):
            vision_encoder_outputs = BaseModelOutput(*vision_encoder_outputs)
        else:
            vision_encoder_outputs = (vision_encoder_outputs, )

        vision_encoder_hidden_states = vision_encoder_outputs[0]

        # optionally project vision_encoder_hidden_states
        if (
            self.vision_encoder.config.hidden_size != self.encoder.config.hidden_size
            and self.encoder.config.cross_attention_hidden_size is None
        ):
            vision_encoder_hidden_states = vision_encoder_hidden_states.type(self.vis_enc_to_enc_proj.weight.dtype)
            vision_encoder_hidden_states = self.vis_enc_to_enc_proj(vision_encoder_hidden_states)

        vision_encoder_attention_mask = None

        if encoder_outputs is None:
            with torch.set_grad_enabled(not self.config.freeze_encoder):
                encoder_outputs = self.encoder(
                    inputs_embeds=vision_encoder_hidden_states,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs_encoder,
            )
        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = BaseModelOutput(*encoder_outputs)

        return encoder_outputs
