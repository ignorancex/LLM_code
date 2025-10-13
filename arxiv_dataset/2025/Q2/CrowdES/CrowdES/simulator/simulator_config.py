import warnings
# from collections import OrderedDict
# from typing import Mapping
# from packaging import version
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class CrowdESSimulatorConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CrowdESSimulatorModel`]. It is used to instantiate an
    SegFormer model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the CrowdES
    [inhwanbae/Crowd-Behavior-Generation](https://github.com/InhwanBae/Crowd-Behavior-Generation) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        num_encoder_blocks (`int`, *optional*, defaults to 4):
            The number of encoder blocks (i.e. stages in the Mix Transformer encoder).
        depths (`List[int]`, *optional*, defaults to `[2, 2, 2, 2]`):
            The number of layers in each encoder block.
        sr_ratios (`List[int]`, *optional*, defaults to `[8, 4, 2, 1]`):
            Sequence reduction ratios in each encoder block.
        hidden_sizes (`List[int]`, *optional*, defaults to `[32, 64, 160, 256]`):
            Dimension of each of the encoder blocks.
        patch_sizes (`List[int]`, *optional*, defaults to `[7, 3, 3, 3]`):
            Patch size before each encoder block.
        strides (`List[int]`, *optional*, defaults to `[4, 2, 2, 2]`):
            Stride before each encoder block.
        num_attention_heads (`List[int]`, *optional*, defaults to `[1, 2, 5, 8]`):
            Number of attention heads for each attention layer in each block of the Transformer encoder.
        mlp_ratios (`List[int]`, *optional*, defaults to `[4, 4, 4, 4]`):
            Ratio of the size of the hidden layer compared to the size of the input layer of the Mix FFNs in the
            encoder blocks.
    Example:

    ```python
    >>> from transformers import SegformerModel, SegformerConfig

    >>> # Initializing a SegFormer nvidia/segformer-b0-finetuned-ade-512-512 style configuration
    >>> configuration = SegformerConfig()

    >>> # Initializing a model from the nvidia/segformer-b0-finetuned-ade-512-512 style configuration
    >>> model = SegformerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = 'crowdes_simulator'

    def __init__(
        self,
        history_length = 10,
        future_length = 10,
        neighbor_max_num = 4,
        neighbor_hidden_dims = [],
        neighbor_embedding_dim = 64,
        env_size = [64, 64],
        env_dim = 8,
        env_hidden_dims = [32, 64, 128],
        env_embedding_dim = 2048,
        env_num_attention_layers = 1,
        traj_encoder_hidden_dims = [256, 512, 1024],
        traj_decoder_hidden_dims = [1024, 512, 256],
        latent_dim = 8,
        latent_hidden_dims = [128, 64],
        hidden_dim = 2048,
        loss_l_scaling = 0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.history_length = history_length
        self.future_length = future_length
        self.neighbor_max_num = neighbor_max_num
        self.neighbor_hidden_dims = neighbor_hidden_dims
        self.neighbor_embedding_dim = neighbor_embedding_dim
        self.env_size = env_size
        self.env_dim = env_dim
        self.env_hidden_dims = env_hidden_dims
        self.env_embedding_dim = env_embedding_dim
        self.env_num_attention_layers = env_num_attention_layers
        self.traj_encoder_hidden_dims = traj_encoder_hidden_dims
        self.traj_decoder_hidden_dims = traj_decoder_hidden_dims
        self.latent_dim = latent_dim
        self.latent_hidden_dims = latent_hidden_dims
        self.hidden_dim = hidden_dim
        self.loss_l_scaling = loss_l_scaling
