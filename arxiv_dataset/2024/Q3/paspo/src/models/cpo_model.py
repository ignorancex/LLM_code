import logging
import numpy as np
import gym

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict
from ray.rllib.models.utils import get_activation_fn


torch, nn = try_import_torch()
import os

logger = logging.getLogger(os.getenv("LOGGER_NAME"))



class CPOModel(TorchModelV2, nn.Module):
    """Generic fully connected network."""

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        assert model_config["vf_share_layers"] == False, "CPO requires vf_share_layers=False"
        
        self.activation_function = get_activation_fn(model_config["fcnet_activation"], "torch")

        self._actor = self._create_mlp(obs_space, num_outputs, model_config)
        self._critic = self._create_mlp(obs_space, 1, model_config)
        self._critic_costs = self._create_mlp(obs_space, 1, model_config)

    def _create_mlp(self, obs_space, num_outputs, model_config):
        layers = []
        prev_layer_size = int(np.product(obs_space.shape))
        for hidden_size in model_config.get("fcnet_hiddens")[:-1]:
            layers.append(torch.nn.Linear(prev_layer_size, hidden_size))
            layers.append(self.activation_function())
            prev_layer_size = hidden_size
            
        layers.append(torch.nn.Linear(prev_layer_size, num_outputs))
        return nn.Sequential(*layers)
    
        

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        obs = input_dict["obs_flat"].float()
        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        logits = self._actor(self._last_flat_in)

        return logits, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._last_flat_in is not None, "must call forward() first"
        
        return self._critic(self._last_flat_in).squeeze(1)

    def cost_value_function(self) -> TensorType:
        assert self._last_flat_in is not None, "must call forward() first"
        
        return self._critic_costs(self._last_flat_in).squeeze(1)
    