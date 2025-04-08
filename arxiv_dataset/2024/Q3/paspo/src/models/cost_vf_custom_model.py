import logging
import os

import numpy as np
import gym

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from models.custom_slim_fc import CustomSlimFC as SlimFC
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.models.modelv2 import ModelV2
from models.lambda_penalty_model import LambdaPenaltyModel

from utils.helper_functions import calculate_action_space_dim

torch, nn = try_import_torch()

# logger = logging.getLogger(__name__)
logger = logging.getLogger(os.getenv("LOGGER_NAME"))


class CostValueFunctionCustomModel(TorchModelV2, nn.Module):
    """Generic fully connected network with multiple heads to estimate the COST value functions"""

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        number_of_constraints: int,
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.number_assets = int(np.product(action_space.shape))

        hiddens = list(model_config.get("fcnet_hiddens", [])) + list(
            model_config.get("post_fcnet_hiddens", [])
        )
        activation = model_config.get("fcnet_activation")
        if not model_config.get("fcnet_hiddens", []):
            activation = model_config.get("post_fcnet_activation")
        no_final_linear = model_config.get("no_final_linear")
        self.vf_share_layers = model_config.get("vf_share_layers")

        layers = []
        prev_layer_size = int(np.product(obs_space.shape))
        self._logits = None

        # Create layers 0 to second-last.
        for size in hiddens[:-1]:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=None,
                    activation_fn=activation,
                )
            )
            prev_layer_size = size

        # The last layer is adjusted to be of size num_outputs, but it's a
        # layer with activation.
        if no_final_linear and num_outputs:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=num_outputs,
                    initializer=None,
                    activation_fn=activation,
                )
            )
            prev_layer_size = num_outputs
        # Finish the layers with the provided sizes (`hiddens`), plus -
        # iff num_outputs > 0 - a last linear layer of size num_outputs.
        else:
            if len(hiddens) > 0:
                layers.append(
                    SlimFC(
                        in_size=prev_layer_size,
                        out_size=hiddens[-1],
                        initializer=None,
                        activation_fn=activation,
                    )
                )
                prev_layer_size = hiddens[-1]
            if num_outputs:
                self._logits = SlimFC(
                    in_size=prev_layer_size,
                    out_size=num_outputs,
                    initializer=None,
                    activation_fn=None,
                )
            else:
                self.num_outputs = ([int(np.product(obs_space.shape))] + hiddens[-1:])[
                    -1
                ]

        self._hidden_layers = nn.Sequential(*layers)

        self._value_branch_separate = None
        if not self.vf_share_layers:
            # Build a parallel set of hidden layers for the value net.
            prev_vf_layer_size = int(np.product(obs_space.shape))
            vf_layers = []
            for size in hiddens:
                vf_layers.append(
                    SlimFC(
                        in_size=prev_vf_layer_size,
                        out_size=size,
                        activation_fn=activation,
                        initializer=None,
                    )
                )
                prev_vf_layer_size = size
            self._value_branch_separate = nn.Sequential(*vf_layers)

        self._value_branch = SlimFC(
            in_size=prev_layer_size,
            out_size=1,
            initializer=None,
            activation_fn=None,
        )
        # Holds the current "base" output (before logits layer).
        self._features = None
        # Holds the last input, in case value branch is separate.
        self._last_flat_in = None

        # COST VALUE FUNCTIONS
        self.amount_constraints = number_of_constraints

        self.cost_vf_share_layers = model_config.get("custom_model_config").get(
            "cost_vf_share_layers"
        )

        self._cost_value_branch_separate_dict = None
        if not self.cost_vf_share_layers:
            self._cost_value_branch_separate_dict = nn.ModuleDict()

            for ctx in range(self.amount_constraints):
                # Build a parallel set of hidden layers for the value net.
                prev_vf_layer_size = int(np.product(obs_space.shape))
                vf_layers = []
                for size in hiddens:
                    vf_layers.append(
                        SlimFC(
                            in_size=prev_vf_layer_size,
                            out_size=size,
                            activation_fn=activation,
                            initializer=None,
                        )
                    )
                    prev_vf_layer_size = size
                self._cost_value_branch_separate_dict[f"constraint_{ctx}"] = (
                    nn.Sequential(*vf_layers)
                )

        self._cost_value_branch_dict = nn.ModuleDict()
        for ctx in range(self.amount_constraints):
            self._cost_value_branch_dict[f"constraint_{ctx}"] = SlimFC(
                in_size=prev_layer_size,
                out_size=1,
                initializer=None,
                activation_fn=None,
            )

        # Added for CUP approach
        config_lambda_model = model_config.get("custom_model_config").get(
            "config_lambda_model", None
        )

        if config_lambda_model is not None:
            self.lambda_penalty_model = LambdaPenaltyModel(
                config_lambda_model=config_lambda_model,
                number_of_constraints=number_of_constraints
            )
            print("CREATED LAMBDA MODEL")
            self.policy_improvement_optimizer = torch.optim.Adam(self.parameters(),
                                                     lr=model_config["custom_model_config"]["config_lambda_model"]["lambda_model_lr"])

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        obs = input_dict["obs_flat"].float()
        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        self._features = self._hidden_layers(self._last_flat_in)
        logits = self._logits(self._features) if self._logits else self._features

        return logits, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        if self._value_branch_separate:
            return self._value_branch(
                self._value_branch_separate(self._last_flat_in)
            ).squeeze(1)
        else:
            return self._value_branch(self._features).squeeze(1)

    def cost_value_function(self, cost_value_function_index) -> TensorType:
        """
        Outputs depending on the cost value function index an cost value estimation
        :param cost_value_function_index:
        :return:
        """
        assert self._features is not None, "must call forward() first"
        if self._cost_value_branch_separate_dict:
            return self._cost_value_branch_dict[
                f"constraint_{cost_value_function_index}"
            ](
                self._cost_value_branch_separate_dict[
                    f"constraint_{cost_value_function_index}"
                ](self._last_flat_in)
            ).squeeze(
                1
            )
        else:
            return self._cost_value_branch_dict[
                f"constraint_{cost_value_function_index}"
            ](self._features).squeeze(1)


def make_cost_model(
    policy: TorchPolicyV2,
    obs_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
    config: Dict,
) -> ModelV2:

    logger.info(
        f"Making cost VF custom model with obs space: {obs_space} and action space: {action_space}"
    )

    num_outputs = calculate_action_space_dim(
        action_space,
        action_space_distribution=config["model"].get("custom_action_dist", None),
    )

    model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=num_outputs,  # action_space.n,
        model_config=config["model"],
        framework=config["framework"],
        # Providing the `model_interface` arg will make the factory
        # wrap the chosen default model with our new model API class
        # (DummyCustomModel). This way, both `forward` and `get_q_values`
        # are available in the returned class.
        model_interface=CostValueFunctionCustomModel,
        name="cost_value_function_model",
        number_of_constraints=policy.A.shape[0],
    )

    return model
