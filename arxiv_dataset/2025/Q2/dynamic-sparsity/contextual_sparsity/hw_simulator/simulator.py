# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from omegaconf import DictConfig

from contextual_sparsity.hw_simulator.cache import cache_strategy_to_class
from contextual_sparsity.hw_simulator.simulator_hooks import SimulatorResetHook
from contextual_sparsity.hw_simulator.utils import (
    HardwareClock,
    calculate_footprint,
    convert_memory_unit,
    get_dimensions_from_model,
    get_layer_key_to_hook_targets,
    get_layer_type_from_layer_key,
    precision_to_bytes,
)
from contextual_sparsity.mask.hooks import MaskingHook
from contextual_sparsity.utils.layer_names import MODEL_MAPS
from contextual_sparsity.utils.misc import pairwise_disjoint
from contextual_sparsity.utils.stats import get_stats_dict_from_array

logger = logging.getLogger(__name__)


class HardwareSimulator:
    """
    A class used to simulate the behavior of hardware during the execution of a model.
    This includes the simulation of memory allocation, prompt encoding, token generation, and memory writing.
    The class is designed to be flexible and can be configured with different precision, dimensions, and model maps.
    It also provides options for verbose logging to enable debugging and analysis.
    """

    def __init__(
        self,
        precision: Union[DictConfig, Dict[str, int]],
        dram: Union[DictConfig, Dict[str, Any]],
        model_id: str,
        sequence_length: int,
        prompt_length: int,
        io_speed: Union[DictConfig, Dict[str, float]],
        cache_strategy: str = "no_cache",
        model: Optional[torch.nn.Module] = None,
        masking_hooks: Optional[List[MaskingHook]] = None,
        allow_mlp_streaming: bool = True,
        allow_static_layers_streaming: bool = False,
        simulate_glu_pruning: bool = False,
        device: Union[str, torch.device] = "cuda",
        verbose: bool = False,
    ):
        self.precision = precision_to_bytes(
            precision
        )  # dict from module type to number of bytes per value
        self.dimensions = get_dimensions_from_model(
            model_id, model=model, masking_hooks=masking_hooks
        )  # dict of dimensions for the model
        self.layer_key_to_hook_targets = get_layer_key_to_hook_targets(
            model=model, masking_hooks=masking_hooks
        )  # dict from hook's target col layer_key to layer data.
        self.model_id = model_id
        self.model_maps = MODEL_MAPS[
            model_id
        ]  # dict of mappings from layer type to its torch layer name
        assert sequence_length is not None and prompt_length is not None, (
            sequence_length,
            prompt_length,
        )
        self.sequence_length = sequence_length
        self.prompt_length = prompt_length
        self.token_generation_length = self.sequence_length - self.prompt_length
        self.dram_io_speed = io_speed.dram
        self.flash_io_speed = io_speed.flash
        assert cache_strategy in cache_strategy_to_class, f"unknown cache strategy {cache_strategy}"
        self.cache_strategy = cache_strategy
        self.allow_mlp_streaming = allow_mlp_streaming
        self.allow_static_layers_streaming = allow_static_layers_streaming
        self.simulate_glu_pruning = simulate_glu_pruning

        # Initialize caches and clock to track KPIs
        self.caches: Dict[str, Any] = dict()  # a dict from layer id to cache block in memory
        self.clock = HardwareClock()

        # Initialize additional flags
        self.device = device
        self.verbose = verbose
        self.verbose_flag = True
        self.flag_print_cache_size = 2
        self.layer_call_counter = defaultdict(int)  # needed for belady oracles
        self.seq_mask = defaultdict(dict)  # needed for belady oracles

        # Initialize counters for current token
        self._current_prompt_encoding = 0
        self._current_token_generation_fixed = 0
        self._current_token_generation_dynamic = 0
        self._current_cache_hit_rates = []
        self._current_cache_hit_rates_per_layer = defaultdict(list)
        self._counter_forward_calls = defaultdict(int)

        # Initialize variables describing DRAM allocation
        self.concurrent_dram_flash_io = dram["concurrent_dram_flash_io"]
        self.layers_static = dram["layers_static"]
        self.layers_dynamic = dram["layers_dynamic"]
        self.layers_streamed_at_prompt_encoding = dram["layers_streamed_at_prompt_encoding"]
        self.layers_streamed_at_token_generation = dram["layers_streamed_at_token_generation"]
        assert pairwise_disjoint([self.layers_static, self.layers_streamed_at_prompt_encoding]), (
            self.layers_static,
            self.layers_streamed_at_prompt_encoding,
        )
        assert pairwise_disjoint(
            [
                self.layers_static,
                self.layers_dynamic,
                self.layers_streamed_at_token_generation,
            ]
        ), (
            self.layers_static,
            self.layers_dynamic,
            self.layers_streamed_at_token_generation,
        )
        logger.info("\tInitializing DRAM allocation:")
        self._dram_capacity_total = dram["capacity"]
        self._dram_capacity_static = calculate_footprint(
            self.layers_static,
            self.precision,
            self.dimensions,
            seq_len=self.sequence_length,
            verbose=self.verbose,
        )
        self._dram_capacity_dynamic = self._dram_capacity_total - self._dram_capacity_static

        # In some experiments (e.g.: ablation at low DRAM) even static layers do not fit in DRAM.
        # This allows streaming of such static layers from Flash to processing unit.
        self.static_layers_streamed = 0
        if self._dram_capacity_static > self._dram_capacity_total:
            assert self.allow_static_layers_streaming, (
                f"Static layers overflow DRAM by "
                f"{self._dram_capacity_static - self._dram_capacity_total} "
                f'and "allow_static_layers_streaming" is False.'
            )
            self.static_layers_streamed += self._dram_capacity_static - self._dram_capacity_total
            self._dram_capacity_static = self._dram_capacity_total
            self._dram_capacity_dynamic = 0
            print(
                f"Streaming {convert_memory_unit(self.static_layers_streamed, 'B', 'MB') :.3f} "
                f"GB of static layers."
            )

        # Prompt encoding for fixed prompt length is constant, so we compute it once during init.
        self.simulate_prompt_encoding_time()

        if self.simulate_glu_pruning:
            # Hardcoded option to simulate that (parts of) Up and Gate are allocated to static DRAM (100% hitrate).
            # The remaining parts are always loaded from Flash, as are the Top-K neurons for the Down matrix
            # (unless there is some space in dynamic DRAM for caching).
            logger.info("\tSimulating GLU Pruning")

            # account for 2/3 of the MLP weights to be statically in DRAM (or streamed from Flash, if not fitting)
            mlp = calculate_footprint(["mlp"], self.precision, self.dimensions)
            self.mlp_dense_static = mlp * ((2.0 / 3.0) if self.dimensions["has_gate_proj"] else 0.5)
            self._dram_capacity_static += self.mlp_dense_static
            self._dram_capacity_dynamic -= self.mlp_dense_static
            logger.info(
                f"\t\t MLP part for simulated GLU Pruning "
                f"{convert_memory_unit(self.mlp_dense_static, 'B', 'GB') :.3f} GB"
            )
            if self._dram_capacity_static > self._dram_capacity_total:
                self.static_layers_streamed += (
                    self._dram_capacity_static - self._dram_capacity_total
                )
                logger.info(
                    f"\t\t MLP GLU Pruning part streamed turing token generation "
                    f"{convert_memory_unit(self.static_layers_streamed, 'B', 'GB') :.3f} GB"
                )
                self._dram_capacity_static = self._dram_capacity_total
                self._dram_capacity_dynamic = 0

            # the remaining 1/3 is computed dynamically, and will be loaded from Flash
            for layer_key in self.layer_key_to_hook_targets.keys():
                self.layer_key_to_hook_targets[layer_key][
                    "n_linears"
                ] = 1  # the Up and Gate layers are fixed in DRAM

        # Token generation time for static modules is constant, so we compute it once during init.
        self.simulate_static_token_generation_time()

        n_masked_linear_layers = sum(
            hook["n_linears"] for hook in self.layer_key_to_hook_targets.values()
        )
        self._dram_capacity_per_linear = self._dram_capacity_dynamic / (
            n_masked_linear_layers if n_masked_linear_layers > 0 else 1
        )
        logger.info(
            f"\t\tStatic capacity: {convert_memory_unit(self._dram_capacity_static, 'B', 'GB') :.3f}"
            f" GB (contains layers: {self.layers_static})"
        )
        logger.info(
            f"\t\tDynamic capacity: {convert_memory_unit(self._dram_capacity_dynamic, 'B', 'GB') :.3f}"
            f" GB (contains layers: {self.layers_dynamic})"
        )
        logger.info(
            f"\t\t\tAllocating : {convert_memory_unit(self._dram_capacity_per_linear, 'B', 'MB') :.3f}"
            f" MB for each of {n_masked_linear_layers} masked linear layers."
        )
        logger.info(
            f"\t\tTotal capacity: {convert_memory_unit(self._dram_capacity_total, 'B', 'GB') :.3f} GB"
        )
        logger.info(
            f"\tLayers streamed at prompt encoding: {self.layers_streamed_at_prompt_encoding}"
        )
        logger.info(
            f"\tLayers streamed at token generation: {self.layers_streamed_at_token_generation}"
        )

        # Hook attached to the forward function to call the Hw Simulator reset method after a sequence is processed.
        self.reset_hook = None
        if model is not None:
            self.reset_hook = SimulatorResetHook(self._reset, model, active=True)

    def simulate_prompt_encoding_time(self):
        # For now prompt_encoding is constant, so no need to recompute it for each batch.
        if self.prompt_length == 0:
            return  # no prompt encoding in this study

        assert self._current_prompt_encoding == 0

        dram_layers = [
            l for l in self.layers_static if l != "predictor"
        ]  # predictors are not used during prompt enc.
        flash_layers = self.layers_streamed_at_prompt_encoding

        if self.verbose:
            logger.info("Prompt Encoding")
            logger.info("\tDRAM I/O")
        dram_io = calculate_footprint(
            dram_layers,
            self.precision,
            self.dimensions,
            seq_len=self.prompt_length,
            verbose=self.verbose,
        )
        if self.verbose:
            logger.info("\tFlash I/O")
        flash_io = calculate_footprint(
            flash_layers, self.precision, self.dimensions, verbose=self.verbose
        )

        dram_io_time = dram_io / self.dram_io_speed
        flash_io_time = flash_io / self.flash_io_speed
        tot_io_time = (
            max(flash_io_time, dram_io_time)
            if self.concurrent_dram_flash_io
            else (flash_io_time + dram_io_time)
        )
        self._current_prompt_encoding += tot_io_time

    def simulate_static_token_generation_time(self):
        # For now token generation for static modules is constant, so no need to recompute it for each batch.
        assert self._current_token_generation_fixed == 0

        if self.verbose:
            logger.info("Token Generation (static)")
            logger.info("\tDRAM I/O")
        average_seq_len = (
            self.prompt_length + self.sequence_length
        ) / 2.0  # simulate read times for growing KV cache.
        dram_io = calculate_footprint(
            self.layers_static,
            self.precision,
            self.dimensions,
            seq_len=average_seq_len,
            verbose=self.verbose,
        )
        if self.simulate_glu_pruning:
            dram_io += self.mlp_dense_static

        if self.verbose:
            logger.info("\tFlash I/O")
        flash_io = calculate_footprint(
            self.layers_streamed_at_token_generation,
            self.precision,
            self.dimensions,
            verbose=self.verbose,
        )
        flash_io += self.static_layers_streamed

        dram_io_time = dram_io / self.dram_io_speed
        flash_io_time = flash_io / self.flash_io_speed
        tot_io_time = (
            max(flash_io_time, dram_io_time)
            if self.concurrent_dram_flash_io
            else (flash_io_time + dram_io_time)
        )
        logger.info(
            f"\tComputed times: dram {dram_io_time}, flash {flash_io_time}, total {tot_io_time}"
        )
        self._current_token_generation_fixed += tot_io_time

    def simulate_dynamic_token_generation_time(
        self, layer_key: str, flash_io: float, dram_io: float
    ):
        layer_type = get_layer_type_from_layer_key(layer_key, model_id=self.model_id)  # e.g.: 'mlp'
        assert layer_type in self.layers_dynamic, (
            f"Unsupported sparse layer with key:{layer_key} and "
            f"type:{layer_type}. Expected type in {self.layers_dynamic}"
        )

        dram_io_time = dram_io / self.dram_io_speed
        flash_io_time = flash_io / self.flash_io_speed
        tot_io_time = (
            max(flash_io_time, dram_io_time)
            if self.concurrent_dram_flash_io
            else (flash_io_time + dram_io_time)
        )

        # When a hook is masking multiple linears, we just multiply the io_time times the number of target linears
        n_linears_per_hook = self.layer_key_to_hook_targets[layer_key]["n_linears"]
        tot_io_time *= n_linears_per_hook

        self._current_token_generation_dynamic += tot_io_time

        if self.verbose and self.verbose_flag:
            logger.info("Token Generation (dynamic, for single token unless hacked the code)")
            logger.info("\tDRAM I/O")
            logger.info(
                f"\t\t{layer_key}: {convert_memory_unit(dram_io * n_linears_per_hook, 'B', 'MB') :.3f} MB"
            )
            logger.info("\tFlash I/O")
            logger.info(
                f"\t\t{layer_key}: {convert_memory_unit(flash_io * n_linears_per_hook, 'B', 'MB') :.3f} MB"
            )
            self.verbose_flag = False

    def write_to_memory(self, layer_key: str, cur_mask: torch.Tensor):
        if len(self.layers_dynamic) == 0:
            return

        self._counter_forward_calls[layer_key] += 1

        if layer_key not in self.caches:
            self.build_new_layer_cache(layer_key)

        # Read current status of memory
        hw_cache = self.caches[layer_key]
        assert (
            cur_mask.shape == hw_cache.cache.shape
        ), f"mask shape changed {(cur_mask.shape, hw_cache.cache.shape)}"

        if self.cache_strategy == "belady":
            # here, we store all the masks for the layer and apply them one-by-one after computing horizons
            # this is to avoid changing sparse linear layers and callback functions. In case, this is too slow, or
            # recording all masks takes too much space, we need to compute horizons in sparse layer itself.
            if self.layer_call_counter[layer_key] == 0:
                # add an all-one mask at sequence end to compute horizon for last appearance of neurons in mask sequence
                self.seq_mask[layer_key] = torch.ones(
                    (
                        self.token_generation_length + 1,
                        self.layer_key_to_hook_targets[layer_key]["size_mask"],
                    ),
                    dtype=torch.bool,
                    device=self.device,
                )
            self.seq_mask[layer_key][self.layer_call_counter[layer_key], :] = cur_mask
            self.layer_call_counter[layer_key] += 1
        else:
            # Simulate reading times from Flash and DRAM for current mask
            flash_io, dram_io = hw_cache.get_current_io_division(cur_mask)
            self.simulate_dynamic_token_generation_time(layer_key, flash_io, dram_io)

            # Compute cache hit rates
            this_cache_hitrate = hw_cache.get_cache_hit_rate(cur_mask)
            n_linears_per_hook = self.layer_key_to_hook_targets[layer_key]["n_linears"]
            self._current_cache_hit_rates += [this_cache_hitrate] * n_linears_per_hook
            self._current_cache_hit_rates_per_layer[layer_key].append(this_cache_hitrate)

            # Update cache
            hw_cache.update(cur_mask)

    def read_from_memory(
        self, layer_key: str
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return (
            (self.caches[layer_key].cache, self.caches[layer_key].slot_counts)
            if layer_key in self.caches
            else (None, None)
        )

    def playback_seq_masking(self, layer_key: str):
        hw_cache = self.caches[layer_key]
        for i in range(self.layer_call_counter[layer_key]):
            horizon = torch.max(self.seq_mask[layer_key][i + 1 :, :], dim=0)[1]
            mask = self.seq_mask[layer_key][i]

            # Simulate reading times from Flash and DRAM for current mask
            flash_io, dram_io = hw_cache.get_current_io_division(mask)
            self.simulate_dynamic_token_generation_time(layer_key, flash_io, dram_io)

            # Compute cache hit rates
            this_cache_hitrate = hw_cache.get_cache_hit_rate(mask)
            n_linears_per_hook = self.layer_key_to_hook_targets[layer_key]["n_linears"]
            self._current_cache_hit_rates += [this_cache_hitrate] * n_linears_per_hook
            self._current_cache_hit_rates_per_layer[layer_key].append(this_cache_hitrate)

            hw_cache.update(mask, slot_count=horizon)

    def playback_belady_all_layers(self, effective_token_generation_length: int):
        for layer_key in self.layer_call_counter.keys():
            assert self.layer_call_counter[layer_key] == effective_token_generation_length, (
                layer_key,
                self.layer_call_counter[layer_key],
                effective_token_generation_length,
            )
            self.seq_mask[layer_key] = self.seq_mask[layer_key][
                : int(effective_token_generation_length) + 1
            ]
            self.playback_seq_masking(layer_key)
            self.layer_call_counter[layer_key] = 0

    def _reset(self):
        """
        Logs the elapsed times (and cache hit rates) in the clock object.
        Then, resets the cache states and the dynamic times computed for this sequence.
        This method should not be called manually. A hook is automatically registered after the model forward
        function to call this method and log statistics for the current sequence. You can deactivate this hook
        (e.g.: for prompt-encoding or iterative generation) using self.reset_hook.set_inactive().
        """
        effective_token_generation_length = self.get_effective_token_generation_length()
        if self.cache_strategy == "belady":
            # In case of Belady algorithm, we first accumulated all predictions for the sequence, and only at the end
            # run a playback of the sequence to find the greedy optimal cache behavior.
            self.playback_belady_all_layers(effective_token_generation_length)

        # Compute dynamic generation time averaged across tokens
        if effective_token_generation_length == 0:
            assert (
                self._current_token_generation_dynamic == 0
            )  # in some tests we reset the Sim without running model.
        else:
            self._current_token_generation_dynamic /= effective_token_generation_length

        assert (
            self._current_token_generation_dynamic > 0 or effective_token_generation_length == 0
        ), (
            "No dynamic time was computed for this batch! Check that the masking hooks are correctly set, or update "
            "conf.hw_simulator.dram to have no layers."
        )

        # Update clock (add times for this sequence)
        self.clock.update(
            self._current_prompt_encoding,
            self._current_token_generation_fixed,
            self._current_token_generation_dynamic,
            self._current_cache_hit_rates,
            self._current_cache_hit_rates_per_layer,
        )
        if self.verbose:
            logger.info(f"Prompt Encoding time: {self._current_prompt_encoding :.4f} s/token")
            logger.info(
                f"Token Generation time (static): {self._current_token_generation_fixed :.4f} s/token"
            )
            logger.info(
                f"Token Generation time (dynamic): "
                f"{self._current_token_generation_dynamic :.4f} s/token"
            )
            self.print_memory_usage("MB")

        # Reset memory and current dynamic time.
        self.caches = dict()
        self._current_token_generation_dynamic = 0
        self._current_cache_hit_rates = []
        self._current_cache_hit_rates_per_layer = defaultdict(list)
        self._counter_forward_calls = defaultdict(int)
        self.layer_call_counter = defaultdict(int)

    def get_effective_token_generation_length(self) -> float:
        """Check that the count of forward calls was the same for all layers and returns it."""
        if len(self._counter_forward_calls.values()) == 0:
            return 0
        count = list(self._counter_forward_calls.values())[0]
        assert count > 0
        assert all(
            v == count for v in self._counter_forward_calls.values()
        ), self._counter_forward_calls
        return float(count)

    def remove_from_caches(self, layer_key: str):
        self.caches.pop(layer_key)

    def layer_capacity_used(self, layer_key: str) -> float:
        usage_in_bytes = self.caches[layer_key].get_usage_in_bytes()
        usage_in_bytes *= self.layer_key_to_hook_targets[layer_key]["n_linears"]

        return usage_in_bytes

    def print_memory_usage(self, units: str = "B"):
        logger.info("Hardware memory usage:")
        for layer in self.caches:
            usage = convert_memory_unit(self.layer_capacity_used(layer), from_u="B", to_u=units)
            logger.info(f"\t layer {layer}: {usage: .2f} {units}")

    def build_new_layer_cache(self, layer_key: str):
        layer_type = get_layer_type_from_layer_key(layer_key, model_id=self.model_id)  # E.g.: 'mlp'
        precision = self.precision[layer_type]
        size_mask = self.layer_key_to_hook_targets[layer_key]["size_mask"]
        size_per_idx = self.layer_key_to_hook_targets[layer_key]["size_per_idx"]
        max_cache_size = int(self._dram_capacity_per_linear / (size_per_idx * precision))
        hardware_cache = cache_strategy_to_class[self.cache_strategy]
        self.caches[layer_key] = hardware_cache(
            size_per_idx=size_per_idx,
            precision=precision,
            max_index=size_mask,
            device=self.device,
            max_cache_size=max_cache_size,
            allow_mlp_streaming=self.allow_mlp_streaming,
        )
        assert len(self.caches) <= len(self.layer_key_to_hook_targets)
        if self.flag_print_cache_size > 0:
            logger.info(
                f"\tCache will fit {max_cache_size}/{size_mask} neurons for layer {layer_key}"
            )
            self.flag_print_cache_size -= 1

    def get_stats_df(self) -> pd.DataFrame:
        results = {
            "ttft": self.ttft,
            "throughput": self.throughput,
            "cache_hit_rate": self.cache_hit_rate,
        }

        df = []
        for quantity, values in results.items():
            stats = get_stats_dict_from_array(values)
            for stat_name, stat_val in stats.items():
                df.append({"quantity": quantity, "stat": stat_name, "value": stat_val})
        return pd.DataFrame(df)

    @property
    def throughput(self) -> float:
        return self.clock.get_throughput(return_mean=False)

    @property
    def ttft(self) -> float:
        return self.clock.get_ttft(return_mean=False)

    @property
    def cache_hit_rate(self) -> float:
        return self.clock.get_cache_hit_rate(return_mean=False)

    def cache_hit_rate_per_layer(self, return_mean: bool = False) -> float:
        return self.clock.get_cache_hit_rate_per_layer(return_mean=return_mean)

    @property
    def current_token_generation_dynamic(self) -> float:
        return self._current_token_generation_dynamic

    @property
    def current_token_generation_fixed(self) -> float:
        return self._current_token_generation_fixed

    @property
    def current_prompt_encoding(self) -> float:
        return self._current_prompt_encoding

    @property
    def current_cache_hit_rate(self) -> float:
        return self._current_cache_hit_rates

    @property
    def current_cache_hit_rate_per_layer(self) -> float:
        return self._current_cache_hit_rates_per_layer
