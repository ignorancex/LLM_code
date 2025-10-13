# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import Any, Dict

import torch
from torch import nn

from contextual_sparsity.evaluation.hooks.base import EvaluationHook
from contextual_sparsity.hw_simulator.utils import (
    calculate_footprint,
    get_dimensions_from_model,
    precision_to_bytes,
)
from contextual_sparsity.utils.layer_names import (
    block_id_to_layer_ids,
    get_block_ids,
    has_gate,
    layer_id_to_block_id,
)

WEIGHT_DENSITY = "weight_density"
MLP_DENSITY = "mlp_density"
MEMORY = "memory"
MLP_MEMORY = "mlp_memory"
MB = 1000.0**2


class Memory(EvaluationHook):
    """
    Evaluate the memory usage based on the activation density.
    """

    metric_dims = {WEIGHT_DENSITY: 1, MEMORY: 1, MLP_MEMORY: 1, MLP_DENSITY: 1}

    def __init__(self, model_id: str, precision: Dict[str, int], sequence_length: int):
        super().__init__()
        self.model_id = model_id
        self.precision = precision_to_bytes(precision)
        self.dimensions = get_dimensions_from_model(model_id)
        self.sequence_length = sequence_length
        self.block_ids = get_block_ids(model_id, block_names="all")

        # Static memory in Bytes
        self.head_memory = calculate_footprint(
            layer_types=["lm_head"],
            precision=self.precision,
            dimensions=self.dimensions,
            seq_len=self.sequence_length,
            verbose=False,
        )

        self.attention_memory = calculate_footprint(
            layer_types=["attention", "kv_cache"],
            precision=self.precision,
            dimensions=self.dimensions,
            seq_len=self.sequence_length,
            verbose=False,
        )

        self.W_params = self.dimensions["intermediate_size"] * self.dimensions["hidden_size"]
        if has_gate(model_id):
            mlp_params = self.W_params * 3
        else:
            mlp_params = self.W_params * 2
        self.mlp_n_params = mlp_params
        self.mlp_base_memory = mlp_params * self.precision["mlp"]

        self.predictor_parameters = {}

    def collect_results(self, module, input, kwargs, output):
        return {WEIGHT_DENSITY: output.float().mean(-1).unsqueeze(-1)}

    def _post_process_batch(
        self, batch_stats: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        # Determine the shape and device
        batch = next(iter(batch_stats.values()))[WEIGHT_DENSITY]
        batch_size = batch.shape
        device = batch.device

        total_mlp_memory = 0
        total_mlp_parameters = 0

        for block_id in self.block_ids:
            up_layer_id, down_layer_id, gate_layer_id = block_id_to_layer_ids(
                model_id=self.model_id, block_id=block_id
            )

            # If the up, down or gate layers do not have a density value, we assume density=0
            if down_layer_id not in batch_stats:
                batch_stats[down_layer_id] = {WEIGHT_DENSITY: torch.zeros(batch_size).to(device)}
            if up_layer_id not in batch_stats:
                batch_stats[up_layer_id] = {WEIGHT_DENSITY: torch.zeros(batch_size).to(device)}
            if gate_layer_id is not None:
                if gate_layer_id not in batch_stats:
                    batch_stats[gate_layer_id] = {
                        WEIGHT_DENSITY: torch.zeros(batch_size).to(device)
                    }

            # Compute the Memory usage for the active MLP weights
            mlp_memory = (
                batch_stats[down_layer_id][WEIGHT_DENSITY]
                + batch_stats[up_layer_id][WEIGHT_DENSITY]
            )
            if gate_layer_id is not None:
                mlp_memory = mlp_memory + batch_stats[gate_layer_id][WEIGHT_DENSITY]
            mlp_active_params = mlp_memory * self.W_params
            mlp_memory = mlp_active_params * self.precision["mlp"]

            # Consider the memory for the predictor
            predictor_active_params = self.predictor_parameters[block_id]
            predictor_memory = predictor_active_params * self.precision["predictors"]

            # Compute the memory percentage and total MLP memory and add them to the stats
            mlp_memory = mlp_memory + predictor_memory
            mlp_param = predictor_active_params + mlp_active_params

            mlp_param_percentage = mlp_param / self.mlp_n_params
            batch_stats[block_id] = {
                MLP_DENSITY: mlp_param_percentage,
                MLP_MEMORY: mlp_memory,
            }

            total_mlp_memory = total_mlp_memory + mlp_memory
            total_mlp_parameters = total_mlp_parameters + mlp_param

        # Compute the average MLP density
        mlp_density = total_mlp_parameters / self.mlp_n_params / len(self.block_ids)
        total_memory = total_mlp_memory + self.head_memory + self.attention_memory

        batch_stats["."] = {
            MLP_DENSITY: mlp_density,
            MLP_MEMORY: total_mlp_memory / MB,
            MEMORY: total_memory / MB,
        }
        return batch_stats

    def attach_to(self, model: nn.Module):
        if hasattr(model, "masking_hooks"):
            for masking_hook in model.masking_hooks:
                for layer_id in masking_hook.mask_cols_of:
                    self._attach_to(masking_hook.masking_func, attached_to=layer_id)
                for layer_id in masking_hook.mask_rows_of:
                    self._attach_to(masking_hook.masking_func, attached_to=layer_id)

                # Compute the memory usage for the predictor
                n_params = sum([param.numel() for param in masking_hook.parameters()])
                block_id = layer_id_to_block_id(
                    model_id=self.model_id, submodule_id=masking_hook.input_from
                )
                self.predictor_parameters[block_id] = n_params
